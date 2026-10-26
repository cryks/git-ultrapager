//! Rich unified hunk のsyntax・intraline装飾と、そのlatest-wins background worker。
//!
//! `DecorationSet` は`LineId`ごとにfull logical-line byte rangeを保持する。Source本文を扱う
//! syntect/intralineのoffsetにはdiff marker 1 byteを加えてから格納し、layout後のtab展開や折り返しは
//! rendererが`VisualRow::rendered_ranges_for_logical`で解決する。Syntaxはsnapshotをstable ID順に
//! decodeし、1つのreader bufferを全hunkで再利用する。Intralineだけはline数・byte数・grapheme数を
//! cap内と確認した1 change blockをowned textへcopyする。Raw input、parser identity、viewportは
//! 所有しない。Interactive時にinstalled full result、次のfull result、priority partial、wrap cacheが
//! 同時に存在しても合計256 MiB以内となるよう、各resultへ固定shareを割り当てる。

use std::fmt::Write as _;
use std::mem::size_of;
use std::ops::Deref;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread;

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use thiserror::Error;
use unicode_segmentation::UnicodeSegmentation;

use crate::bounded_text::{
    BoundedDocumentTextReader, BoundedTextError, BoundedTextRead, MAX_EAGER_TEXT_RAW_BYTES,
};
use crate::cli::WhitespaceChoice;
use crate::derived_memory::{
    DerivedMemoryError, DerivedMemoryLease, DerivedMemoryLedger, DerivedMemoryPool,
    NEXT_DECORATION_MEMORY_BYTES, PRIORITY_DECORATION_MEMORY_BYTES,
};
use crate::document::{
    DocumentError, DocumentLineMetadata, DocumentLineMetadataReader, DocumentRevision,
    DocumentSnapshot, LineDisplayLength, OwnedDocumentLine,
};
use crate::highlight::{
    BudgetedHighlightedLine, HighlightError, HighlightedLine, SyntaxEngine, SyntaxSession,
    SyntaxSpan,
};
use crate::intraline::{
    BudgetedIntralineResult, IntralineResult, LinePairDiff, MAX_BLOCK_BYTES, MAX_BLOCK_GRAPHEMES,
    MAX_CHANGED_LINES, MAX_LINE_GRAPHEMES, compute_refs_with_budget,
};
use crate::model::{ChangeStatus, CommitId, FileId, HunkId, HunkKind, LineId, LineKind};
use crate::structure_store::{
    FileIdRange, StoredFileRef, StoredHunk, StructureMetadataReader, StructureReader,
    StructureStoreError,
};
use crate::text::MAX_SYNTAX_LINE_BYTES;
use crate::theme::{ColorLevel, Theme};

/// Syntax上限はsource本文に適用するため、bounded readerへはdiff marker 1 byteを加えて渡す。
const MAX_SYNTAX_DISPLAY_BYTES: usize = MAX_SYNTAX_LINE_BYTES + 1;
/// Prefix推定は各hunkから同一file・各sideの後続32行・64 KiBまでとし、候補context数との積で
/// 計算量を固定する。
const MAX_CONTINUATION_INFERENCE_LINES: usize = 32;
const MAX_CONTINUATION_INFERENCE_BYTES: usize = 64 * 1024;
/// Intralineのblock byte上限もsource本文だけを数える。1 lineの復元上限にはmarkerを含める。
const MAX_INTRALINE_LINE_DISPLAY_BYTES: usize = MAX_BLOCK_BYTES + 1;
const DERIVED_RESULT_BUDGET_BYTES: usize = NEXT_DECORATION_MEMORY_BYTES;
const DERIVED_PRIORITY_RESULT_BUDGET_BYTES: usize = PRIORITY_DECORATION_MEMORY_BYTES;

/// 複数の同期range計算でsyntect grammar/theme bundleを共有するcalculator。
///
/// Engineは最初のrich hunkを含むcolored requestまでloadしない。`--paging never`は1 fileずつ
/// [`Self::compute_range`]を呼ぶため、この値をstream lifetimeで保持してfileごとのbundle再読込を避ける。
/// Syntax初期化failureも`OnceLock`へ保存し、各resultのwarningへ同じfailureを写す。
pub struct DecorationComputer {
    engine: OnceLock<Result<SyntaxEngine, HighlightError>>,
}

impl DecorationComputer {
    /// Syntax bundleをまだloadしていないcalculatorを作る。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            engine: OnceLock::new(),
        }
    }

    /// Document全体のrich unified hunkを計算する。
    pub fn compute(
        &self,
        document: &DocumentSnapshot,
        theme: &Theme,
        whitespace: WhitespaceChoice,
    ) -> Result<DecorationSet, DecorationComputeError> {
        self.compute_scoped(
            document,
            theme,
            whitespace,
            NEXT_DECORATION_MEMORY_BYTES,
            None,
        )
    }

    /// 指定logical-line範囲と交差するfileだけを装飾する。
    pub fn compute_range(
        &self,
        document: &DocumentSnapshot,
        theme: &Theme,
        whitespace: WhitespaceChoice,
        line_range: Range<usize>,
    ) -> Result<DecorationSet, DecorationComputeError> {
        let start = line_range.start.min(document.line_count());
        let end = line_range.end.min(document.line_count());
        self.compute_scoped(
            document,
            theme,
            whitespace,
            NEXT_DECORATION_MEMORY_BYTES,
            Some(start..end.max(start)),
        )
    }

    fn compute_scoped(
        &self,
        document: &DocumentSnapshot,
        theme: &Theme,
        whitespace: WhitespaceChoice,
        budget: usize,
        line_range: Option<Range<usize>>,
    ) -> Result<DecorationSet, DecorationComputeError> {
        let ledger = DerivedMemoryLedger::new();
        let budget = budget.min(NEXT_DECORATION_MEMORY_BYTES);
        let mut cursor_reader = StructureMetadataReader::new();
        let structure_cursor =
            ScopedStructureCursor::new(document, line_range.as_ref(), &mut cursor_reader)?;
        let scope = DecorationScope {
            budget,
            pool: DerivedMemoryPool::NextDecoration,
            priority_hunk: None,
            only_priority: false,
            line_range,
            structure_cursor: Some(structure_cursor.clone()),
        };
        let has_rich_hunk = scope_has_rich_hunk(document, &structure_cursor, &mut cursor_reader)?;
        if !has_rich_hunk {
            return Ok(DecorationBuilder::new(
                document.revision(),
                whitespace,
                &ledger,
                scope.pool,
                budget,
            )?
            .finish()
            .into_unbudgeted());
        }
        let syntax = if theme.color_level() == ColorLevel::None {
            SyntaxAvailability::Disabled
        } else {
            match self.engine.get_or_init(SyntaxEngine::new) {
                Ok(engine) => SyntaxAvailability::Ready(engine),
                Err(error) => SyntaxAvailability::Failed(error),
            }
        };
        let cancelled = || false;
        let context = DecorationContext {
            document,
            theme,
            whitespace,
            syntax,
            ledger: &ledger,
            cancelled: &cancelled,
        };
        match DecorationSet::compute_internal(context, scope)? {
            Some(result) => Ok(result.into_unbudgeted()),
            // Predicateはfalse固定で、同期pathではcancelされない。型上のNoneだけ空setへ写す。
            None => Ok(DecorationSet::default()),
        }
    }
}

impl Default for DecorationComputer {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshotからlogical line metadataまたはtextを読む処理のfailure。
#[derive(Debug, Error)]
pub enum DecorationComputeError {
    #[error(transparent)]
    Document(#[from] DocumentError),
    #[error(transparent)]
    StructureStore(#[from] StructureStoreError),
    #[error(transparent)]
    BoundedText(#[from] BoundedTextError),
    #[error(transparent)]
    Memory(#[from] DerivedMemoryError),
    #[error("decoration line boundary {boundary} does not fit the structural u64 index")]
    LineBoundaryOverflow { boundary: usize },
    #[error("document snapshot does not contain logical line {line_id}")]
    MissingLine { line_id: LineId },
}

/// 1 logical lineへ適用するdecorations。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LineDecoration {
    syntax: Vec<SyntaxSpan>,
    intraline: Vec<Range<usize>>,
}

impl LineDecoration {
    /// Full logical-line byteへ写したsyntax foreground spanを返す。
    #[must_use]
    pub fn syntax_spans(&self) -> &[SyntaxSpan] {
        &self.syntax
    }

    /// Full logical-line byteへ写したintraline changed spanを返す。
    #[must_use]
    pub fn intraline_ranges(&self) -> &[Range<usize>] {
        &self.intraline
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DecorationEntry {
    line_id: LineId,
    decoration: LineDecoration,
}

/// 1 immutable document snapshotに対応するderived decoration集合。
///
/// Entryは`LineId`昇順のflat `Vec`に置く。Hash tableのbucket/control allocationを見積もる必要がなく、
/// rendererのlookupはbinary searchで`O(log decorated_lines)`となる。このpayloadはCloneできず、
/// interactive経路では[`BudgetedDecorationSet`]または[`AppliedDecorationSet`]だけが所有する。
#[derive(Debug, PartialEq, Eq)]
pub struct DecorationSet {
    revision: Option<DocumentRevision>,
    lines: Vec<DecorationEntry>,
    whitespace: WhitespaceChoice,
    approximate_bytes: usize,
    truncated: bool,
    warning: Option<String>,
    warning_fallback: Option<&'static str>,
}

impl DecorationSet {
    /// Rich unified hunkをdocument順に計算する。
    ///
    /// Old/new syntax stateはhunkごとに独立して作り、context lineは両方へ進めたうえで通常はnew側、
    /// deleted fileだけold側を採用する。Fallback file/hunkにはdecorationsを付けない。Snapshotの
    /// metadataまたはtextを読めなければ[`DecorationComputeError`]を返し、syntax/cap failureは
    /// [`Self::warning`]へ格納する。
    pub fn compute(
        document: &DocumentSnapshot,
        theme: &Theme,
        whitespace: WhitespaceChoice,
    ) -> Result<Self, DecorationComputeError> {
        DecorationComputer::new().compute(document, theme, whitespace)
    }

    /// 指定logical-line範囲と交差するfileだけを装飾する。
    ///
    /// Direct modeが閉じたfile entryを出力する時に使い、既に出力済みのprefixを再decodeしない。
    /// Hunkと範囲が交差する場合はsyntax stateをhunk先頭から進めるため、そのhunk全体を計算する。
    /// 返るsetには範囲外の同一hunkのspanが含まれ得るが、writerは指定rangeだけを参照する。
    pub fn compute_range(
        document: &DocumentSnapshot,
        theme: &Theme,
        whitespace: WhitespaceChoice,
        line_range: Range<usize>,
    ) -> Result<Self, DecorationComputeError> {
        DecorationComputer::new().compute_range(document, theme, whitespace, line_range)
    }

    #[cfg(test)]
    fn compute_with_budget(
        document: &DocumentSnapshot,
        theme: &Theme,
        whitespace: WhitespaceChoice,
        budget: usize,
        line_range: Option<Range<usize>>,
    ) -> Result<Self, DecorationComputeError> {
        DecorationComputer::new().compute_scoped(document, theme, whitespace, budget, line_range)
    }

    fn compute_internal(
        context: DecorationContext<'_>,
        scope: DecorationScope,
    ) -> Result<Option<BudgetedDecorationSet>, DecorationComputeError> {
        let mut builder = DecorationBuilder::new(
            context.document.revision(),
            context.whitespace,
            context.ledger,
            scope.pool,
            scope.budget,
        )?;
        let mut readers = DecorationReaders::new(context.ledger)?;
        let mut structure_reader = StructureReader::new();
        let mut metadata_reader = StructureMetadataReader::new();
        if let Some(hunk_id) = scope.priority_hunk {
            if let Some(hunk) = metadata_reader.read_hunk(context.document.structure(), hunk_id)?
                && let Some(file) =
                    structure_reader.read_file(context.document.structure(), hunk.file_id)?
                && file.mode.is_rich()
                && hunk.kind == HunkKind::Unified
            {
                if let SyntaxAvailability::Failed(error) = context.syntax {
                    builder.record_syntax_error(error);
                }
                match decorate_one_hunk(&mut builder, &context, file, hunk, &mut readers)? {
                    DecorationStep::Continue => {}
                    DecorationStep::Stop => return Ok(Some(builder.finish())),
                    DecorationStep::Cancelled => return Ok(None),
                }
            }
            if scope.only_priority {
                return Ok(Some(builder.finish()));
            }
        } else if scope.only_priority {
            return Ok(Some(builder.finish()));
        }

        let cursor = match scope.structure_cursor {
            Some(cursor) => cursor,
            None => ScopedStructureCursor::new(
                context.document,
                scope.line_range.as_ref(),
                &mut metadata_reader,
            )?,
        };
        let mut syntax_failure_recorded = false;
        // Commit/hunkはfixed metadata reader、file pathだけStructureReaderへ分離する。同じfileのhunkは
        // copied parent metadataで検証するため、path borrowを保持したまま親file recordを再decodeしない。
        'commits: for raw_commit_id in cursor.commits.clone() {
            if (context.cancelled)() {
                return Ok(None);
            }
            let commit_id = CommitId::new(raw_commit_id);
            let Some(commit) =
                metadata_reader.read_commit(context.document.structure(), commit_id)?
            else {
                continue;
            };
            for file_id in cursor.files_in_commit(commit.files) {
                if (context.cancelled)() {
                    return Ok(None);
                }
                let Some(file) = structure_reader.read_file_in_commit(
                    context.document.structure(),
                    file_id,
                    commit,
                )?
                else {
                    continue;
                };
                if !file.mode.is_rich() || file.hunks.count() == 0 {
                    continue;
                }
                if !syntax_failure_recorded
                    && let SyntaxAvailability::Failed(error) = context.syntax
                {
                    builder.record_syntax_error(error);
                    syntax_failure_recorded = true;
                }
                let file_metadata = file.metadata();
                for hunk_id in file.hunks.iter() {
                    if Some(hunk_id) == scope.priority_hunk {
                        continue;
                    }
                    if (context.cancelled)() {
                        return Ok(None);
                    }
                    let Some(hunk) = metadata_reader.read_hunk_in_file(
                        context.document.structure(),
                        hunk_id,
                        file_metadata,
                    )?
                    else {
                        continue;
                    };
                    if hunk.kind != HunkKind::Unified {
                        continue;
                    }
                    match decorate_one_hunk(&mut builder, &context, file, hunk, &mut readers)? {
                        DecorationStep::Continue => {}
                        DecorationStep::Stop => break 'commits,
                        DecorationStep::Cancelled => return Ok(None),
                    }
                }
            }
        }
        Ok(Some(builder.finish()))
    }

    /// Lineにdecorationsがあれば返す。
    #[must_use]
    pub fn line(&self, id: LineId) -> Option<&LineDecoration> {
        self.lines
            .binary_search_by_key(&id, |entry| entry.line_id)
            .ok()
            .and_then(|index| self.lines.get(index))
            .map(|entry| &entry.decoration)
    }

    /// Derived spanを計算したexact document snapshotと一致するかを返す。
    ///
    /// [`Self::default`]はUI初期化用の未結合setなのでfalseになる。同じbuilderの後続snapshotでも、
    /// open file/hunkのfallback化やgutter boundsが変わり得るため一致しない。
    #[must_use]
    pub fn belongs_to(&self, document: &DocumentSnapshot) -> bool {
        self.revision.as_ref() == Some(&document.revision())
    }

    /// Decoration spanを1つも保持していないかを返す。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Intraline rangeを計算したwhitespace modeを返す。
    #[must_use]
    pub const fn whitespace(&self) -> WhitespaceChoice {
        self.whitespace
    }

    /// Flat entry Vec、syntax span、intraline range、warningがresult poolで占めるbyte数を返す。
    #[must_use]
    pub const fn approximate_bytes(&self) -> usize {
        self.approximate_bytes
    }

    /// Budget到達後のlineを省略したかを返す。
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Syntax failure、intraline cap、budget省略のsummaryを返す。
    #[must_use]
    pub fn warning(&self) -> Option<&str> {
        self.warning.as_deref().or(self.warning_fallback)
    }
}

/// Worker builderまたはresponse queueが所有するNext/Priority result。
///
/// Payloadをleaseより先に宣言するため、channel eviction、stale response、panic unwindのいずれでも
/// span Vecを解放してからpool byteを返す。Cloneを実装せず、response送信はownershipを移す。
#[derive(Debug)]
pub struct BudgetedDecorationSet {
    set: DecorationSet,
    lease: DerivedMemoryLease,
}

impl BudgetedDecorationSet {
    /// Resultが占有するpoolを返す。
    #[must_use]
    pub fn pool(&self) -> DerivedMemoryPool {
        self.lease.pool()
    }

    /// Ledgerへ計上したresult capacityを返す。
    #[must_use]
    pub fn leased_bytes(&self) -> usize {
        self.lease.bytes()
    }

    /// Direct/snapshot testのようにsession共有台帳を使わないcallerへpayloadだけを返す。
    fn into_unbudgeted(self) -> DecorationSet {
        let Self { set, lease } = self;
        drop(lease);
        set
    }
}

impl Deref for BudgetedDecorationSet {
    type Target = DecorationSet;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

/// Main rendererへ公開中のApplied result。
///
/// `replace`だけがNext/Priority payloadをAppliedへ移せる。旧payloadはledger lockを保持したままdropし、
/// workerが解放直後のpoolへ新しいbuilderを入れる前に物理allocationも解放する。
#[derive(Debug)]
pub struct AppliedDecorationSet {
    set: DecorationSet,
    lease: DerivedMemoryLease,
}

impl AppliedDecorationSet {
    /// 空payloadと0-byte Applied leaseを作る。
    pub fn empty(ledger: &DerivedMemoryLedger) -> Result<Self, DerivedMemoryError> {
        Ok(Self {
            set: DecorationSet::default(),
            lease: ledger.reserve(DerivedMemoryPool::AppliedDecoration, 0)?,
        })
    }

    /// 現在のpayloadを空へ戻し、payload drop後にApplied byteを返す。
    pub fn clear(&mut self) -> Result<(), DerivedMemoryError> {
        let old = std::mem::take(&mut self.set);
        drop(old);
        let bytes = self.lease.bytes();
        self.lease.release(bytes)
    }

    /// Next/Priority resultをrendererへ適用する。
    ///
    /// Result wrapperを分解する処理はこのmethod内だけに閉じ、旧payload dropとledger transitionの間に
    /// worker reserveが割り込めないよう[`DerivedMemoryLedger::replace_applied_while_charged`]を使う。
    pub fn replace(
        &mut self,
        replacement: BudgetedDecorationSet,
        ledger: &DerivedMemoryLedger,
    ) -> Result<(), DerivedMemoryError> {
        let BudgetedDecorationSet {
            set: replacement_set,
            lease: mut replacement_lease,
        } = replacement;
        let mut current_set = Some(std::mem::take(&mut self.set));
        let transition =
            ledger.replace_applied_while_charged(&mut self.lease, &mut replacement_lease, || {
                if let Some(current) = current_set.take() {
                    drop(current);
                }
            });
        if let Err(error) = transition {
            if let Some(current) = current_set.take() {
                self.set = current;
            }
            drop(replacement_set);
            drop(replacement_lease);
            return Err(error);
        }
        self.set = replacement_set;
        let released = std::mem::replace(&mut self.lease, replacement_lease);
        drop(released);
        Ok(())
    }
}

impl Deref for AppliedDecorationSet {
    type Target = DecorationSet;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

fn decorate_one_hunk(
    builder: &mut DecorationBuilder,
    context: &DecorationContext<'_>,
    file: StoredFileRef<'_>,
    hunk: StoredHunk,
    readers: &mut DecorationReaders,
) -> Result<DecorationStep, DecorationComputeError> {
    if let SyntaxAvailability::Ready(engine) = context.syntax
        && !decorate_syntax_hunk(builder, engine, file, hunk, context, readers)?
    {
        return Ok(if (context.cancelled)() {
            DecorationStep::Cancelled
        } else {
            DecorationStep::Stop
        });
    }
    if !decorate_intraline_hunk(
        builder,
        hunk,
        context.document,
        context.whitespace,
        context.cancelled,
        readers,
    )? {
        return Ok(if (context.cancelled)() {
            DecorationStep::Cancelled
        } else {
            DecorationStep::Stop
        });
    }
    Ok(DecorationStep::Continue)
}

#[derive(Clone, Copy)]
enum DecorationStep {
    Continue,
    Stop,
    Cancelled,
}

struct DecorationReaders {
    text: BoundedDocumentTextReader,
    metadata: DocumentLineMetadataReader,
    text_lease: DerivedMemoryLease,
    eager_accounted: usize,
    streaming_accounted: usize,
}

impl DecorationReaders {
    fn new(ledger: &DerivedMemoryLedger) -> Result<Self, DerivedMemoryError> {
        Ok(Self {
            text: BoundedDocumentTextReader::new(),
            metadata: DocumentLineMetadataReader::new(),
            text_lease: ledger.reserve(DerivedMemoryPool::DecorationWorkspace, 0)?,
            eager_accounted: 0,
            streaming_accounted: 0,
        })
    }

    /// Bounded readerがworker lifetimeで保持する再利用bufferを、raw/display metadataだけで先に計上する。
    ///
    /// Eager pathはraw・ANSI除去・escaped textの3 payload、streaming pathは完成textと64 KiB decoder/
    /// grapheme buffer群を持つ。両readerは別fieldなので各pathの最大値を加算し、次のlineで小さくなっても
    /// leaseを縮めない。`false`ではtext readを開始していない。
    fn reserve_text_read(&mut self, metadata: DocumentLineMetadata) -> bool {
        const STREAMING_FIXED_BYTES: usize = 2 * 1024 * 1024;
        const EAGER_FIXED_BYTES: usize = 64 * 1024;

        let Some(display_bytes) = metadata.display_len.exact() else {
            return false;
        };
        let mut next_eager = self.eager_accounted;
        let mut next_streaming = self.streaming_accounted;
        let raw_exceeds =
            u64::try_from(MAX_EAGER_TEXT_RAW_BYTES).map_or(true, |limit| metadata.raw_len > limit);
        if raw_exceeds {
            let Some(required) = display_bytes
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(STREAMING_FIXED_BYTES))
            else {
                return false;
            };
            next_streaming = next_streaming.max(required);
        } else {
            let Ok(raw_bytes) = usize::try_from(metadata.raw_len) else {
                return false;
            };
            let Some(required) = raw_bytes
                .checked_mul(4)
                .and_then(|bytes| {
                    display_bytes
                        .checked_mul(2)
                        .and_then(|display| bytes.checked_add(display))
                })
                .and_then(|bytes| bytes.checked_add(EAGER_FIXED_BYTES))
            else {
                return false;
            };
            next_eager = next_eager.max(required);
        }
        let Some(required) = next_eager.checked_add(next_streaming) else {
            return false;
        };
        if required > self.text_lease.bytes() {
            let additional = required - self.text_lease.bytes();
            if self.text_lease.reserve(additional).is_err() {
                // High-water markはlease成功後だけcommitする。失敗した巨大lineの候補値を残すと、
                // 後続の小さいlineまで存在しないbuffer capacityとの合算で省略され続ける。
                return false;
            }
        }
        self.eager_accounted = next_eager;
        self.streaming_accounted = next_streaming;
        true
    }
}

/// 1回のdecorations計算で共有するimmutable input。Hunk helperへ個別に渡さず、cancel判定と
/// syntax engineの組を同じsnapshotへ固定する。
struct DecorationContext<'input> {
    document: &'input DocumentSnapshot,
    theme: &'input Theme,
    whitespace: WhitespaceChoice,
    syntax: SyntaxAvailability<'input>,
    ledger: &'input DerivedMemoryLedger,
    cancelled: &'input dyn Fn() -> bool,
}

/// Derived budgetと走査範囲。`only_priority`はpartial response専用で、指定hunkを処理した後に
/// document全体へ進まない。
#[derive(Clone)]
struct DecorationScope {
    budget: usize,
    pool: DerivedMemoryPool,
    priority_hunk: Option<HunkId>,
    only_priority: bool,
    /// `None`はbackground full scan。`Some`はdirect writerが閉じたfile範囲だけを要求する。
    line_range: Option<Range<usize>>,
    /// Synchronous経路でsyntax初期化判定と本計算に同じseek結果を渡す。Worker経路では未計算。
    structure_cursor: Option<ScopedStructureCursor>,
}

/// Logical-line scopeをspill-backed structure tableのcursorへ変換した値。
///
/// Commit/file spanはstable ID順に単調なので、範囲付き計算はtable全体をmaterializeせず、各tableの
/// binary seekで得た半開範囲だけを走査する。File範囲はcommit child範囲との積を取ってから読むため、
/// line scopeと交差しない同一commit内のfile path blobも復元しない。
#[derive(Clone, Debug)]
struct ScopedStructureCursor {
    commits: Range<u64>,
    files: Range<u64>,
}

impl ScopedStructureCursor {
    fn new(
        document: &DocumentSnapshot,
        line_range: Option<&Range<usize>>,
        reader: &mut StructureMetadataReader,
    ) -> Result<Self, DecorationComputeError> {
        let counts = document.structure().visible_counts()?;
        let Some(line_range) = line_range else {
            return Ok(Self {
                commits: 0..counts.commits,
                files: 0..counts.files,
            });
        };
        let start = u64::try_from(line_range.start).map_err(|_| {
            DecorationComputeError::LineBoundaryOverflow {
                boundary: line_range.start,
            }
        })?;
        let end = u64::try_from(line_range.end).map_err(|_| {
            DecorationComputeError::LineBoundaryOverflow {
                boundary: line_range.end,
            }
        })?;
        let commits = reader.commits_intersecting(document.structure(), start..end)?;
        let files = reader.files_intersecting(document.structure(), start..end)?;
        Ok(Self {
            commits,
            files: files.start()..files.end_exclusive(),
        })
    }

    fn files_in_commit(
        &self,
        commit_files: FileIdRange,
    ) -> impl DoubleEndedIterator<Item = FileId> {
        let start = self.files.start.max(commit_files.start());
        let end = self.files.end.min(commit_files.end_exclusive()).max(start);
        (start..end).map(FileId::new)
    }
}

/// Syntax bundleをloadする前に、対象範囲へrich unified候補があるかをmetadataだけで判定する。
///
/// Caller-owned readerを使い、commitをscope内で1回、fileを親metadata付きで1回ずつ読む。Path blobと
/// hunk recordは触らず、最初のrich fileで終了する。
fn scope_has_rich_hunk(
    document: &DocumentSnapshot,
    cursor: &ScopedStructureCursor,
    reader: &mut StructureMetadataReader,
) -> Result<bool, DecorationComputeError> {
    for raw_commit_id in cursor.commits.clone() {
        let Some(commit) =
            reader.read_commit(document.structure(), CommitId::new(raw_commit_id))?
        else {
            continue;
        };
        for file_id in cursor.files_in_commit(commit.files) {
            let Some(file) = reader.read_file_in_commit(document.structure(), file_id, commit)?
            else {
                continue;
            };
            if file.mode.is_rich() && file.hunks.count() > 0 {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[derive(Clone, Copy)]
enum SyntaxAvailability<'engine> {
    Disabled,
    Ready(&'engine SyntaxEngine),
    Failed(&'engine HighlightError),
}

impl Default for DecorationSet {
    fn default() -> Self {
        Self {
            revision: None,
            lines: Vec::new(),
            whitespace: WhitespaceChoice::Ignore,
            approximate_bytes: 0,
            truncated: false,
            warning: None,
            warning_fallback: None,
        }
    }
}

fn decorate_syntax_hunk(
    builder: &mut DecorationBuilder,
    engine: &SyntaxEngine,
    file: StoredFileRef<'_>,
    hunk: StoredHunk,
    context: &DecorationContext<'_>,
    readers: &mut DecorationReaders,
) -> Result<bool, DecorationComputeError> {
    let old_path = file.old_path.or(file.new_path);
    let new_path = file.new_path.or(file.old_path);
    let mut old = None;
    let mut new = None;
    let mut old_opened = false;
    let mut new_opened = false;
    let use_old_context = file.status == ChangeStatus::Deleted;

    if let Some(prefix) = collect_inference_prefix(context, file, hunk, SyntaxSide::Old, readers)? {
        old = open_inferred_session(builder, engine, old_path, &prefix.lines, context.theme);
        old_opened = true;
    }
    if (context.cancelled)() {
        return Ok(false);
    }
    if let Some(prefix) = collect_inference_prefix(context, file, hunk, SyntaxSide::New, readers)? {
        new = open_inferred_session(builder, engine, new_path, &prefix.lines, context.theme);
        new_opened = true;
    }
    if (context.cancelled)() {
        return Ok(false);
    }

    for raw_id in hunk_line_range(context.document, hunk) {
        if (context.cancelled)() {
            return Ok(false);
        }
        let line_id = LineId::new(raw_id);
        let metadata = readers
            .metadata
            .read(context.document, line_id)?
            .ok_or(DecorationComputeError::MissingLine { line_id })?;
        if !is_source(metadata.kind) || metadata.hunk_id != Some(hunk.id) {
            continue;
        }

        if metadata.display_len.exceeds(MAX_SYNTAX_DISPLAY_BYTES) {
            builder.syntax_omitted = true;
            reset_skipped_syntax_line(
                builder,
                engine,
                metadata.kind,
                old_path,
                new_path,
                context.theme,
                &mut old_opened,
                &mut old,
                &mut new_opened,
                &mut new,
            );
            continue;
        }
        if !readers.reserve_text_read(metadata) {
            builder.syntax_omitted = true;
            reset_skipped_syntax_line(
                builder,
                engine,
                metadata.kind,
                old_path,
                new_path,
                context.theme,
                &mut old_opened,
                &mut old,
                &mut new_opened,
                &mut new,
            );
            continue;
        }

        let (metadata, text) = match readers.text.read(
            context.document,
            line_id,
            MAX_SYNTAX_DISPLAY_BYTES,
            context.cancelled,
        )? {
            BoundedTextRead::Available { metadata, text } => (metadata, text),
            BoundedTextRead::Oversized { metadata } => {
                builder.syntax_omitted = true;
                reset_skipped_syntax_line(
                    builder,
                    engine,
                    metadata.kind,
                    old_path,
                    new_path,
                    context.theme,
                    &mut old_opened,
                    &mut old,
                    &mut new_opened,
                    &mut new,
                );
                continue;
            }
            BoundedTextRead::Cancelled => return Ok(false),
        };
        let content = text.get(1..).unwrap_or_default();
        let selected = match metadata.kind {
            LineKind::Deletion => open_and_advance_session(
                builder,
                engine,
                old_path,
                content,
                context.theme,
                &mut old_opened,
                &mut old,
            ),
            LineKind::Addition => open_and_advance_session(
                builder,
                engine,
                new_path,
                content,
                context.theme,
                &mut new_opened,
                &mut new,
            ),
            LineKind::Context => {
                let old_line = open_and_advance_session(
                    builder,
                    engine,
                    old_path,
                    content,
                    context.theme,
                    &mut old_opened,
                    &mut old,
                );
                let new_line = open_and_advance_session(
                    builder,
                    engine,
                    new_path,
                    content,
                    context.theme,
                    &mut new_opened,
                    &mut new,
                );
                if use_old_context { old_line } else { new_line }
            }
            _ => continue,
        };
        if !builder.add_syntax(metadata.id, text.len(), selected) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[derive(Clone, Copy)]
enum SyntaxSide {
    Old,
    New,
}

struct InferencePrefix {
    lines: Vec<String>,
    _lease: DerivedMemoryLease,
}

/// Hunk先頭から同一file・片側のsource列をboundedに復元する。
///
/// 現在hunkの全行が`*`、`>`、`|`の同じleaderを持つ場合だけ、直後hunkの同じleaderが続く範囲まで
/// 読む。これによりJSDocのcloserを次hunkから拾いつつ、通常code hunkがfile後半の無関係なcloserを
/// 根拠に推定されるのを避ける。Hunk間で省略されたsourceはparseせず、推定で選んだstateも現在
/// hunkのsessionだけに使う。上限超過lineを読むと欠けた遷移を既知扱いしてしまうため、そのsideの
/// 推定全体を捨てる。固定leaseはString payloadとVecの最大capacityを復元前に確保し、候補評価が
/// 終わるまでprefixと同じlifetimeで保持する。
fn collect_inference_prefix(
    context: &DecorationContext<'_>,
    file: StoredFileRef<'_>,
    hunk: StoredHunk,
    side: SyntaxSide,
    readers: &mut DecorationReaders,
) -> Result<Option<InferencePrefix>, DecorationComputeError> {
    let reserved_bytes =
        MAX_CONTINUATION_INFERENCE_BYTES + MAX_CONTINUATION_INFERENCE_LINES * size_of::<String>();
    let Ok(lease) = context
        .ledger
        .reserve(DerivedMemoryPool::DecorationWorkspace, reserved_bytes)
    else {
        return Ok(None);
    };
    let mut lines = Vec::with_capacity(MAX_CONTINUATION_INFERENCE_LINES);
    let mut bytes = 0_usize;
    let mut continuation_hunk = None;
    let mut continuation_leader = None;

    let document_end = u64::try_from(context.document.line_count()).unwrap_or(u64::MAX);
    let start = hunk.span.start.get().min(document_end);
    let end = file
        .span
        .end_exclusive
        .unwrap_or(document_end)
        .min(document_end)
        .max(start);
    for raw_id in start..end {
        if (context.cancelled)() {
            return Ok(None);
        }
        let line_id = LineId::new(raw_id);
        let metadata = readers
            .metadata
            .read(context.document, line_id)?
            .ok_or(DecorationComputeError::MissingLine { line_id })?;
        let belongs_to_side = match side {
            SyntaxSide::Old => matches!(metadata.kind, LineKind::Deletion | LineKind::Context),
            SyntaxSide::New => matches!(metadata.kind, LineKind::Addition | LineKind::Context),
        };
        if metadata.file_id != Some(file.id) {
            continue;
        }
        let is_later_hunk = metadata.hunk_id != Some(hunk.id);
        if let Some(line_hunk) = metadata.hunk_id.filter(|line_hunk| *line_hunk != hunk.id) {
            match continuation_hunk {
                Some(current) if current != line_hunk => break,
                Some(_) => {}
                None => {
                    let Some(leader) = shared_continuation_leader(&lines) else {
                        break;
                    };
                    continuation_hunk = Some(line_hunk);
                    continuation_leader = Some(leader);
                }
            }
        }
        if !belongs_to_side {
            continue;
        }
        if lines.len() == MAX_CONTINUATION_INFERENCE_LINES {
            break;
        }
        if metadata.display_len.exceeds(MAX_SYNTAX_DISPLAY_BYTES)
            || !readers.reserve_text_read(metadata)
        {
            return Ok(None);
        }
        let text = match readers.text.read(
            context.document,
            line_id,
            MAX_SYNTAX_DISPLAY_BYTES,
            context.cancelled,
        )? {
            BoundedTextRead::Available { text, .. } => text,
            BoundedTextRead::Oversized { .. } | BoundedTextRead::Cancelled => return Ok(None),
        };
        let content = text.get(1..).unwrap_or_default();
        if is_later_hunk && continuation_leader != continuation_leader_for_line(content) {
            break;
        }
        let Some(next_bytes) = bytes.checked_add(content.len()) else {
            return Ok(None);
        };
        if next_bytes > MAX_CONTINUATION_INFERENCE_BYTES {
            break;
        }
        lines.push(content.to_owned());
        bytes = next_bytes;
    }

    Ok((!lines.is_empty()).then_some(InferencePrefix {
        lines,
        _lease: lease,
    }))
}

fn shared_continuation_leader(lines: &[String]) -> Option<char> {
    let leader = continuation_leader_for_line(lines.first()?)?;
    lines
        .iter()
        .all(|line| continuation_leader_for_line(line) == Some(leader))
        .then_some(leader)
}

fn continuation_leader_for_line(line: &str) -> Option<char> {
    line.trim_start()
        .chars()
        .next()
        .filter(|leader| matches!(leader, '*' | '>' | '|'))
}

#[allow(clippy::too_many_arguments)]
fn reset_skipped_syntax_line<'engine>(
    builder: &mut DecorationBuilder,
    engine: &'engine SyntaxEngine,
    kind: LineKind,
    old_path: Option<&str>,
    new_path: Option<&str>,
    theme: &Theme,
    old_opened: &mut bool,
    old: &mut Option<SyntaxSession<'engine>>,
    new_opened: &mut bool,
    new: &mut Option<SyntaxSession<'engine>>,
) {
    match kind {
        LineKind::Deletion => {
            reset_skipped_session(builder, engine, old_path, theme, old_opened, old);
        }
        LineKind::Addition => {
            reset_skipped_session(builder, engine, new_path, theme, new_opened, new);
        }
        LineKind::Context => {
            reset_skipped_session(builder, engine, old_path, theme, old_opened, old);
            reset_skipped_session(builder, engine, new_path, theme, new_opened, new);
        }
        _ => {}
    }
}

fn reset_skipped_session<'engine>(
    builder: &mut DecorationBuilder,
    engine: &'engine SyntaxEngine,
    path: Option<&str>,
    theme: &Theme,
    opened: &mut bool,
    session: &mut Option<SyntaxSession<'engine>>,
) {
    if !*opened {
        *session = open_session(builder, engine, path, None, theme);
        *opened = true;
    }
    if let Some(active) = session.as_mut() {
        active.reset_after_skipped_line();
    }
}

fn open_and_advance_session<'engine>(
    builder: &mut DecorationBuilder,
    engine: &'engine SyntaxEngine,
    path: Option<&str>,
    line: &str,
    theme: &Theme,
    opened: &mut bool,
    session: &mut Option<SyntaxSession<'engine>>,
) -> BudgetedHighlightedLine {
    if !*opened {
        *session = open_session(builder, engine, path, Some(line), theme);
        *opened = true;
    }
    advance_session(builder, session, line)
}

fn open_session<'engine>(
    builder: &mut DecorationBuilder,
    engine: &'engine SyntaxEngine,
    path: Option<&str>,
    first_line: Option<&str>,
    theme: &Theme,
) -> Option<SyntaxSession<'engine>> {
    match engine.session_with_budget(path, first_line, theme.choice(), builder.ledger) {
        Ok(session) => session,
        Err(HighlightError::WorkspaceBudgetExceeded) => {
            builder.syntax_omitted = true;
            None
        }
        Err(error) => {
            builder.record_syntax_error(&error);
            None
        }
    }
}

fn open_inferred_session<'engine>(
    builder: &mut DecorationBuilder,
    engine: &'engine SyntaxEngine,
    path: Option<&str>,
    lines: &[String],
    theme: &Theme,
) -> Option<SyntaxSession<'engine>> {
    match engine.session_with_inferred_continuation(
        path,
        lines.first().map(String::as_str),
        theme.choice(),
        lines,
        builder.ledger,
    ) {
        Ok(session) => session,
        Err(HighlightError::WorkspaceBudgetExceeded) => {
            builder.syntax_omitted = true;
            None
        }
        Err(error) => {
            builder.record_syntax_error(&error);
            None
        }
    }
}

fn advance_session(
    builder: &mut DecorationBuilder,
    session: &mut Option<SyntaxSession<'_>>,
    line: &str,
) -> BudgetedHighlightedLine {
    let Some(active) = session.as_mut() else {
        return BudgetedHighlightedLine::plain();
    };
    match active.highlight_line_with_budget(line, builder.ledger) {
        Ok(Some(highlighted)) => highlighted,
        Ok(None) => {
            builder.syntax_omitted = true;
            BudgetedHighlightedLine::plain()
        }
        Err(error) => {
            builder.record_syntax_error(&error);
            *session = None;
            BudgetedHighlightedLine::plain()
        }
    }
}

fn decorate_intraline_hunk(
    builder: &mut DecorationBuilder,
    hunk: StoredHunk,
    document: &DocumentSnapshot,
    whitespace: WhitespaceChoice,
    cancelled: &dyn Fn() -> bool,
    readers: &mut DecorationReaders,
) -> Result<bool, DecorationComputeError> {
    let mut block: Option<IntralineBlockIds> = None;
    let mut adding = false;
    for raw_id in hunk_line_range(document, hunk) {
        if cancelled() {
            return Ok(false);
        }
        let line_id = LineId::new(raw_id);
        let metadata = readers
            .metadata
            .read(document, line_id)?
            .ok_or(DecorationComputeError::MissingLine { line_id })?;
        if !is_source(metadata.kind) || metadata.hunk_id != Some(hunk.id) {
            continue;
        }
        match metadata.kind {
            LineKind::Deletion => {
                if adding {
                    let Some(finished) = block.take() else {
                        return Ok(false);
                    };
                    if !decorate_intraline_block(
                        builder, finished, document, whitespace, cancelled, readers,
                    )? {
                        return Ok(false);
                    }
                }
                adding = false;
                block
                    .get_or_insert_with(|| IntralineBlockIds::new(builder.ledger))
                    .push_removed(metadata);
            }
            LineKind::Addition => {
                if let Some(active) = block.as_mut() {
                    adding = true;
                    active.push_added(metadata);
                }
            }
            LineKind::Context => {
                if adding {
                    let Some(finished) = block.take() else {
                        return Ok(false);
                    };
                    if !decorate_intraline_block(
                        builder, finished, document, whitespace, cancelled, readers,
                    )? {
                        return Ok(false);
                    }
                } else {
                    block = None;
                }
                adding = false;
            }
            _ => {}
        }
    }
    if adding
        && let Some(finished) = block
        && !decorate_intraline_block(builder, finished, document, whitespace, cancelled, readers)?
    {
        return Ok(false);
    }
    Ok(true)
}

/// Metadata scan中の1 deletion/addition block。
///
/// Textをdecodeする前に5,000行と64 MiBの上限を適用する。上限を越えたblockはIDも破棄し、後続の
/// source metadataをblock終端まで読み飛ばす。Grapheme上限はtext decode後、owned copy前に別passで
/// 検証する。
struct IntralineBlockIds {
    removed: Vec<LineId>,
    added: Vec<LineId>,
    _lease: Option<DerivedMemoryLease>,
    content_bytes: usize,
    rejected: bool,
}

impl IntralineBlockIds {
    fn new(ledger: &DerivedMemoryLedger) -> Self {
        let bytes = MAX_CHANGED_LINES
            .checked_mul(size_of::<LineId>())
            // removed/addedの2 Vecとallocator growthの2倍を同時に覆う。
            .and_then(|bytes| bytes.checked_mul(4));
        let Some(bytes) = bytes else {
            return Self::rejected();
        };
        let Ok(lease) = ledger.reserve(DerivedMemoryPool::DecorationWorkspace, bytes) else {
            return Self::rejected();
        };
        let mut removed = Vec::new();
        let mut added = Vec::new();
        if removed.try_reserve_exact(MAX_CHANGED_LINES).is_err()
            || added.try_reserve_exact(MAX_CHANGED_LINES).is_err()
        {
            return Self::rejected();
        }
        Self {
            removed,
            added,
            _lease: Some(lease),
            content_bytes: 0,
            rejected: false,
        }
    }

    fn rejected() -> Self {
        Self {
            removed: Vec::new(),
            added: Vec::new(),
            _lease: None,
            content_bytes: 0,
            rejected: true,
        }
    }

    fn push_removed(&mut self, metadata: DocumentLineMetadata) {
        self.push(metadata, false);
    }

    fn push_added(&mut self, metadata: DocumentLineMetadata) {
        self.push(metadata, true);
    }

    fn push(&mut self, metadata: DocumentLineMetadata, added: bool) {
        if self.rejected {
            return;
        }
        if self.removed.len().saturating_add(self.added.len()) >= MAX_CHANGED_LINES {
            self.reject();
            return;
        }
        let LineDisplayLength::Exact(display_len) = metadata.display_len else {
            self.reject();
            return;
        };
        let Ok(display_len) = usize::try_from(display_len) else {
            self.reject();
            return;
        };
        // display_lenはsource marker込みなので、compute_refsへ渡すcontentと同じbyte数へ1を引く。
        let Some(content_bytes) = self
            .content_bytes
            .checked_add(display_len.saturating_sub(1))
        else {
            self.reject();
            return;
        };
        if content_bytes > MAX_BLOCK_BYTES {
            self.reject();
            return;
        }
        self.content_bytes = content_bytes;
        if added {
            self.added.push(metadata.id);
        } else {
            self.removed.push(metadata.id);
        }
    }

    fn reject(&mut self) {
        self.rejected = true;
        self.removed.clear();
        self.added.clear();
    }
}

fn decorate_intraline_block(
    builder: &mut DecorationBuilder,
    block: IntralineBlockIds,
    document: &DocumentSnapshot,
    whitespace: WhitespaceChoice,
    cancelled: &dyn Fn() -> bool,
    readers: &mut DecorationReaders,
) -> Result<bool, DecorationComputeError> {
    if block.rejected {
        builder.intraline_skipped_blocks = builder.intraline_skipped_blocks.saturating_add(1);
        return Ok(true);
    }

    // このpassではbounded readerのborrowed viewだけを使う。Block全体のgrapheme capが確定する
    // 前に`OwnedDocumentLine`を作らず、raw payloadが巨大でも64 KiB stepごとにcancelへ戻る。
    let mut grapheme_total = 0_usize;
    for line_id in block.removed.iter().chain(&block.added).copied() {
        if cancelled() {
            return Ok(false);
        }
        let metadata = readers
            .metadata
            .read(document, line_id)?
            .ok_or(DecorationComputeError::MissingLine { line_id })?;
        if !readers.reserve_text_read(metadata) {
            builder.intraline_skipped_blocks = builder.intraline_skipped_blocks.saturating_add(1);
            return Ok(true);
        }
        let text = match readers.text.read(
            document,
            line_id,
            MAX_INTRALINE_LINE_DISPLAY_BYTES,
            cancelled,
        )? {
            BoundedTextRead::Available { text, .. } => text,
            BoundedTextRead::Oversized { .. } => {
                builder.intraline_skipped_blocks =
                    builder.intraline_skipped_blocks.saturating_add(1);
                return Ok(true);
            }
            BoundedTextRead::Cancelled => return Ok(false),
        };
        let graphemes = text
            .get(1..)
            .unwrap_or_default()
            .graphemes(true)
            .take(MAX_LINE_GRAPHEMES + 1)
            .count();
        if graphemes > MAX_LINE_GRAPHEMES {
            builder.intraline_skipped_blocks = builder.intraline_skipped_blocks.saturating_add(1);
            return Ok(true);
        }
        grapheme_total = grapheme_total.saturating_add(graphemes);
        if grapheme_total > MAX_BLOCK_GRAPHEMES {
            builder.intraline_skipped_blocks = builder.intraline_skipped_blocks.saturating_add(1);
            return Ok(true);
        }
    }

    let line_count = block.removed.len().saturating_add(block.added.len());
    let owned_workspace_bytes = block
        .content_bytes
        .checked_add(line_count)
        .and_then(|bytes| {
            line_count
                .checked_mul(size_of::<OwnedDocumentLine>() + size_of::<&str>())
                .and_then(|metadata| bytes.checked_add(metadata))
        })
        .and_then(|bytes| bytes.checked_mul(2));
    let Some(owned_workspace_bytes) = owned_workspace_bytes else {
        builder.intraline_skipped_blocks = builder.intraline_skipped_blocks.saturating_add(1);
        return Ok(true);
    };
    // このleaseをremoved/added/borrowed refsより先に宣言する。Localは逆順dropなので、owned payload
    // と参照Vecが解放された後にworkspace byteを返す。
    let _owned_workspace = match builder.ledger.reserve(
        DerivedMemoryPool::DecorationWorkspace,
        owned_workspace_bytes,
    ) {
        Ok(lease) => lease,
        Err(_) => {
            builder.intraline_skipped_blocks = builder.intraline_skipped_blocks.saturating_add(1);
            return Ok(true);
        }
    };

    let removed = match own_lines(readers, document, &block.removed, cancelled)? {
        OwnedLinesRead::Available(lines) => lines,
        OwnedLinesRead::Oversized => {
            builder.intraline_skipped_blocks = builder.intraline_skipped_blocks.saturating_add(1);
            return Ok(true);
        }
        OwnedLinesRead::Cancelled => return Ok(false),
    };
    let added = match own_lines(readers, document, &block.added, cancelled)? {
        OwnedLinesRead::Available(lines) => lines,
        OwnedLinesRead::Oversized => {
            builder.intraline_skipped_blocks = builder.intraline_skipped_blocks.saturating_add(1);
            return Ok(true);
        }
        OwnedLinesRead::Cancelled => return Ok(false),
    };
    let removed_refs = removed
        .iter()
        .map(OwnedDocumentLine::content)
        .collect::<Vec<_>>();
    let added_refs = added
        .iter()
        .map(OwnedDocumentLine::content)
        .collect::<Vec<_>>();
    let result = compute_refs_with_budget(&removed_refs, &added_refs, whitespace, builder.ledger);
    match result.result() {
        IntralineResult::Computed(_) => {
            if cancelled() {
                return Ok(false);
            }
            Ok(builder.add_intraline_result(result, &removed, &added))
        }
        IntralineResult::Skipped(_) => {
            builder.intraline_skipped_blocks = builder.intraline_skipped_blocks.saturating_add(1);
            Ok(true)
        }
    }
}

enum OwnedLinesRead {
    Available(Vec<OwnedDocumentLine>),
    Oversized,
    Cancelled,
}

fn own_lines(
    readers: &mut DecorationReaders,
    document: &DocumentSnapshot,
    line_ids: &[LineId],
    cancelled: &dyn Fn() -> bool,
) -> Result<OwnedLinesRead, DecorationComputeError> {
    let mut lines = Vec::new();
    lines
        .try_reserve_exact(line_ids.len())
        .map_err(|_| DocumentError::OwnedLineAllocation {
            requested: line_ids
                .len()
                .saturating_mul(size_of::<OwnedDocumentLine>()),
        })?;
    for line_id in line_ids {
        if cancelled() {
            return Ok(OwnedLinesRead::Cancelled);
        }
        let metadata = readers
            .metadata
            .read(document, *line_id)?
            .ok_or(DecorationComputeError::MissingLine { line_id: *line_id })?;
        if !readers.reserve_text_read(metadata) {
            return Ok(OwnedLinesRead::Oversized);
        }
        match readers.text.read(
            document,
            *line_id,
            MAX_INTRALINE_LINE_DISPLAY_BYTES,
            cancelled,
        )? {
            BoundedTextRead::Available { metadata, text } => {
                let Some(line) = own_bounded_line(metadata, text, cancelled)? else {
                    return Ok(OwnedLinesRead::Cancelled);
                };
                lines.push(line);
            }
            BoundedTextRead::Oversized { .. } => return Ok(OwnedLinesRead::Oversized),
            BoundedTextRead::Cancelled => return Ok(OwnedLinesRead::Cancelled),
        }
    }
    Ok(OwnedLinesRead::Available(lines))
}

fn own_bounded_line(
    metadata: DocumentLineMetadata,
    text: &str,
    cancelled: &dyn Fn() -> bool,
) -> Result<Option<OwnedDocumentLine>, DocumentError> {
    const COPY_STEP_BYTES: usize = 64 * 1024;

    let mut owned_text = String::new();
    owned_text
        .try_reserve_exact(text.len())
        .map_err(|_| DocumentError::OwnedLineAllocation {
            requested: text.len(),
        })?;
    let mut start = 0_usize;
    while start < text.len() {
        if cancelled() {
            return Ok(None);
        }
        let candidate = start.saturating_add(COPY_STEP_BYTES).min(text.len());
        let end = if candidate == text.len() {
            candidate
        } else {
            text.floor_char_boundary(candidate)
        };
        owned_text.push_str(&text[start..end]);
        start = end;
    }
    if cancelled() {
        return Ok(None);
    }
    Ok(Some(OwnedDocumentLine {
        id: metadata.id,
        text: owned_text,
        kind: metadata.kind,
        commit_id: metadata.commit_id,
        file_id: metadata.file_id,
        hunk_id: metadata.hunk_id,
        old_line_number: metadata.old_line_number,
        new_line_number: metadata.new_line_number,
        file_line_bounds: metadata.file_line_bounds,
        terminated: metadata.terminated,
    }))
}

fn hunk_line_range(document: &DocumentSnapshot, hunk: StoredHunk) -> Range<u64> {
    let document_end = u64::try_from(document.line_count()).unwrap_or(u64::MAX);
    let start = hunk.span.start.get().min(document_end);
    let end = hunk
        .span
        .end_exclusive
        .unwrap_or(document_end)
        .min(document_end)
        .max(start);
    start..end
}

fn is_source(kind: LineKind) -> bool {
    matches!(
        kind,
        LineKind::Context | LineKind::Deletion | LineKind::Addition
    )
}

struct DecorationBuilder<'ledger> {
    set: DecorationSet,
    lease: DerivedMemoryLease,
    ledger: &'ledger DerivedMemoryLedger,
    budget: usize,
    syntax_failed: bool,
    syntax_omitted: bool,
    intraline_skipped_blocks: usize,
}

/// Intraline outputをresult poolへ移す直前の所有単位。
///
/// Payload fieldをleaseより先に宣言し、capacity不足・index不整合・cancellationのearly returnでも
/// pair/range Vecを先にdropする。
struct PendingIntralineTransfer {
    pairs: Vec<LinePairDiff>,
    workspace: DerivedMemoryLease,
}

impl<'ledger> DecorationBuilder<'ledger> {
    fn new(
        revision: DocumentRevision,
        whitespace: WhitespaceChoice,
        ledger: &'ledger DerivedMemoryLedger,
        pool: DerivedMemoryPool,
        budget: usize,
    ) -> Result<Self, DerivedMemoryError> {
        debug_assert!(matches!(
            pool,
            DerivedMemoryPool::NextDecoration | DerivedMemoryPool::PriorityDecoration
        ));
        debug_assert!(budget <= pool.limit_bytes());
        Ok(Self {
            set: DecorationSet {
                revision: Some(revision),
                whitespace,
                ..DecorationSet::default()
            },
            lease: ledger.reserve(pool, 0)?,
            ledger,
            budget,
            syntax_failed: false,
            syntax_omitted: false,
            intraline_skipped_blocks: 0,
        })
    }

    fn add_syntax(
        &mut self,
        line_id: LineId,
        line_len: usize,
        highlighted: BudgetedHighlightedLine,
    ) -> bool {
        let (highlighted, workspace) = highlighted.into_parts();
        let HighlightedLine::Styled(spans) = highlighted else {
            return true;
        };
        let Some(mut workspace) = workspace else {
            self.set.truncated = true;
            return false;
        };
        let mut spans = spans;
        spans.retain_mut(|span| match shift_syntax_span(span.clone(), line_len) {
            Some(shifted) => {
                *span = shifted;
                true
            }
            None => false,
        });
        spans.sort_unstable_by_key(|span| (span.byte_range.start, span.byte_range.end));
        if spans.is_empty() {
            return true;
        }
        if !self.ensure_line_capacity(line_id) {
            return false;
        }
        let output_bytes = workspace.bytes();
        if self
            .lease
            .bytes()
            .checked_add(output_bytes)
            .is_none_or(|next| next > self.budget)
            || self
                .ledger
                .transfer(&mut workspace, &mut self.lease, output_bytes)
                .is_err()
        {
            self.set.truncated = true;
            return false;
        }
        self.line_mut(line_id).syntax = spans;
        true
    }

    fn add_intraline_result(
        &mut self,
        result: BudgetedIntralineResult,
        removed: &[OwnedDocumentLine],
        added: &[OwnedDocumentLine],
    ) -> bool {
        let (result, workspace) = result.into_parts();
        let IntralineResult::Computed(pairs) = result else {
            return true;
        };
        let Some(workspace) = workspace else {
            self.set.truncated = true;
            return false;
        };
        let mut pending = PendingIntralineTransfer { pairs, workspace };

        let mut additional_entries = 0_usize;
        for pair in &mut pending.pairs {
            let (Some(old_line), Some(new_line)) =
                (removed.get(pair.old_index), added.get(pair.new_index))
            else {
                self.set.truncated = true;
                return false;
            };
            shift_ranges_in_place(&mut pair.old_changed, old_line.text.len());
            shift_ranges_in_place(&mut pair.new_changed, new_line.text.len());
            if !pair.old_changed.is_empty() && self.set.line(old_line.id).is_none() {
                additional_entries = additional_entries.saturating_add(1);
            }
            if !pair.new_changed.is_empty() && self.set.line(new_line.id).is_none() {
                additional_entries = additional_entries.saturating_add(1);
            }
        }
        if !self.ensure_additional_line_capacity(additional_entries) {
            return false;
        }
        let output_bytes = pending.workspace.bytes();
        if self
            .lease
            .bytes()
            .checked_add(output_bytes)
            .is_none_or(|next| next > self.budget)
            || self
                .ledger
                .transfer(&mut pending.workspace, &mut self.lease, output_bytes)
                .is_err()
        {
            self.set.truncated = true;
            return false;
        }
        let PendingIntralineTransfer { pairs, workspace } = pending;
        drop(workspace);
        for pair in pairs {
            // Indexはtransfer前のvalidation loopで両sideとも検証済みで、その後sliceを変更しない。
            let old_line = &removed[pair.old_index];
            let new_line = &added[pair.new_index];
            if !pair.old_changed.is_empty() {
                self.line_mut(old_line.id).intraline = pair.old_changed;
            }
            if !pair.new_changed.is_empty() {
                self.line_mut(new_line.id).intraline = pair.new_changed;
            }
        }
        true
    }

    fn ensure_line_capacity(&mut self, line_id: LineId) -> bool {
        if self
            .set
            .lines
            .binary_search_by_key(&line_id, |entry| entry.line_id)
            .is_ok()
        {
            return true;
        }
        self.ensure_additional_line_capacity(1)
    }

    fn ensure_additional_line_capacity(&mut self, additional: usize) -> bool {
        if additional == 0 {
            return true;
        }
        let before = self.set.lines.capacity();
        let Some(requested) = additional.checked_mul(size_of::<DecorationEntry>()) else {
            self.set.truncated = true;
            return false;
        };
        let Some(conservative) = requested.checked_mul(2) else {
            self.set.truncated = true;
            return false;
        };
        let Some(allowance) = self.precharge_growth(conservative) else {
            return false;
        };
        if self.set.lines.try_reserve_exact(additional).is_err() {
            self.release_growth_allowance(allowance, 0);
            self.set.truncated = true;
            return false;
        }
        let actual = self
            .set
            .lines
            .capacity()
            .saturating_sub(before)
            .saturating_mul(size_of::<DecorationEntry>());
        if actual > allowance {
            // 2倍boundを越えるallocator growthはpublishしない。まだentryをinsertしていないため、以前の
            // capacityまで縮小を要求し、残ったcapacityだけをfull allowance内で計上する。
            self.set.lines.shrink_to(before);
            let retained = self
                .set
                .lines
                .capacity()
                .saturating_sub(before)
                .saturating_mul(size_of::<DecorationEntry>());
            if retained > allowance {
                // Allocatorがshrink要求も満たさない場合はentryとそのspan payloadを全体ごと捨てる。
                // 空Vecへ戻した後はresult所有byteが残らないため、それまでのleaseも全て返す。
                let entries = std::mem::take(&mut self.set.lines);
                drop(entries);
                let held = self.lease.bytes();
                let _ = self.lease.release(held);
            } else {
                self.release_growth_allowance(allowance, retained);
            }
            self.set.truncated = true;
            return false;
        }
        self.release_growth_allowance(allowance, actual);
        true
    }

    /// Allocation前にresult poolの残量を一時leaseし、Vec/String allocatorのcapacity丸めも覆う。
    ///
    /// `minimum`の2倍などcallerが定めたconservative boundが残量へ収まらない場合はallocationを開始
    /// しない。Allocation後は実capacity deltaだけを残し、未使用allowanceを返す。
    fn precharge_growth(&mut self, minimum: usize) -> Option<usize> {
        let remaining = self.budget.checked_sub(self.lease.bytes())?;
        if minimum > remaining || self.lease.reserve(remaining).is_err() {
            self.set.truncated = true;
            return None;
        }
        Some(remaining)
    }

    fn release_growth_allowance(&mut self, allowance: usize, used: usize) {
        if used > allowance {
            self.set.truncated = true;
            return;
        }
        let _ = self.lease.release(allowance - used);
    }

    fn line_mut(&mut self, line_id: LineId) -> &mut LineDecoration {
        let index = match self
            .set
            .lines
            .binary_search_by_key(&line_id, |entry| entry.line_id)
        {
            Ok(index) => index,
            Err(index) => {
                self.set.lines.insert(
                    index,
                    DecorationEntry {
                        line_id,
                        decoration: LineDecoration::default(),
                    },
                );
                index
            }
        };
        &mut self.set.lines[index].decoration
    }

    fn record_syntax_error(&mut self, error: &HighlightError) {
        let _ = error;
        self.syntax_failed = true;
    }

    fn finish(mut self) -> BudgetedDecorationSet {
        let has_warning = self.set.truncated
            || self.syntax_failed
            || self.syntax_omitted
            || self.intraline_skipped_blocks > 0;
        if has_warning && let Some(allowance) = self.precharge_growth(2 * 1024) {
            let mut warning = String::new();
            if warning.try_reserve_exact(1024).is_ok() {
                let mut separator = "";
                if self.set.truncated {
                    warning.push_str("decoration budget exceeded; later spans were omitted");
                    separator = "; ";
                }
                if self.syntax_failed {
                    warning.push_str(separator);
                    warning.push_str("syntax highlighting failed");
                    separator = "; ";
                }
                if self.syntax_omitted {
                    warning.push_str(separator);
                    warning.push_str("syntax omitted for some lines");
                    separator = "; ";
                }
                if self.intraline_skipped_blocks > 0 {
                    warning.push_str(separator);
                    let noun = if self.intraline_skipped_blocks == 1 {
                        "block"
                    } else {
                        "blocks"
                    };
                    let _ = write!(
                        warning,
                        "intraline omitted for {} {noun}",
                        self.intraline_skipped_blocks,
                    );
                }
                let used = warning.capacity();
                self.release_growth_allowance(allowance, used);
                self.set.warning = Some(warning);
            } else {
                self.release_growth_allowance(allowance, 0);
            }
        }
        if has_warning && self.set.warning.is_none() {
            self.set.warning_fallback = Some(if self.set.truncated {
                "decoration budget exceeded; later spans were omitted"
            } else {
                "syntax or intraline decoration was skipped"
            });
        }
        self.set.approximate_bytes = self.lease.bytes();
        BudgetedDecorationSet {
            set: self.set,
            lease: self.lease,
        }
    }
}

fn shift_syntax_span(mut span: SyntaxSpan, line_len: usize) -> Option<SyntaxSpan> {
    span.byte_range = shift_range(span.byte_range, line_len)?;
    Some(span)
}

fn shift_range(range: Range<usize>, line_len: usize) -> Option<Range<usize>> {
    let start = range.start.checked_add(1)?;
    let end = range.end.checked_add(1)?;
    (start <= end && end <= line_len).then_some(start..end)
}

fn shift_ranges_in_place(ranges: &mut Vec<Range<usize>>, line_len: usize) {
    ranges.retain_mut(|range| match shift_range(range.clone(), line_len) {
        Some(shifted) => {
            *range = shifted;
            true
        }
        None => false,
    });
}

/// Background request。`generation`はcallerのdocument/whitespace世代をそのままresponseへ返す。
#[derive(Clone, Debug)]
pub struct DecorationRequest {
    /// Callerがdocument/whitespace世代として割り当て、response照合に使うID。
    pub generation: u64,
    /// Worker taskの間だけ保持するimmutable document snapshot。
    pub document: Arc<DocumentSnapshot>,
    /// Built-in palette選択と出力時のcolor capability。
    pub theme: Theme,
    /// このsnapshotのintraline比較mode。
    pub whitespace: WhitespaceChoice,
    /// 最初のpartial responseとfull計算の先頭で処理するvisible hunk。
    pub priority_hunk: Option<HunkId>,
}

/// Latest requestに対応するbackground result。
#[derive(Debug)]
pub struct DecorationResponse {
    /// 対応requestから複写したcaller generation。
    pub generation: u64,
    /// `false`はpriority hunkだけのpartial set、`true`はfull-document計算の完了を表す。
    pub complete: bool,
    /// Task panicまたはsnapshot read failureが`Err`。個別syntax/cap failureはsetのwarningへ格納する。
    pub result: Result<BudgetedDecorationSet, DecorationWorkerError>,
}

/// Decoration workerの起動、snapshot read、task panicのfailure。
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DecorationWorkerError {
    #[error("decoration worker could not start: {0}")]
    WorkerSpawn(String),
    #[error("decoration worker panicked")]
    WorkerPanicked,
    #[error("decoration worker could not read its document snapshot: {0}")]
    Document(String),
}

#[derive(Clone, Debug)]
struct QueuedRequest {
    sequence: u64,
    request: DecorationRequest,
}

/// Pending requestを1件、priority/full responseを合計2件に制限するlatest-wins worker。
pub struct DecorationWorker {
    requests: Sender<QueuedRequest>,
    pending: Receiver<QueuedRequest>,
    latest_sequence: Arc<AtomicU64>,
    submission_gate: Arc<Mutex<()>>,
    next_sequence: AtomicU64,
    /// Priority partialとfull resultをmain loopへ渡すbounded receiver。
    pub responses: Receiver<DecorationResponse>,
}

impl DecorationWorker {
    /// Capacity 1のrequestとcapacity 2のresponse channelを持つworker threadを開始する。
    pub fn spawn(ledger: Arc<DerivedMemoryLedger>) -> Result<Self, DecorationWorkerError> {
        let (requests, worker_requests) = bounded::<QueuedRequest>(1);
        let pending = worker_requests.clone();
        let (response_sender, responses) = bounded::<DecorationResponse>(2);
        let response_eviction = responses.clone();
        let latest_sequence = Arc::new(AtomicU64::new(0));
        let worker_latest = Arc::clone(&latest_sequence);
        let submission_gate = Arc::new(Mutex::new(()));
        let worker_gate = Arc::clone(&submission_gate);
        thread::Builder::new()
            .name("git-ultrapager-decorations".to_owned())
            .spawn(move || {
                decoration_worker_loop(
                    worker_requests,
                    response_sender,
                    response_eviction,
                    &worker_latest,
                    &worker_gate,
                    &ledger,
                );
            })
            .map_err(|error| DecorationWorkerError::WorkerSpawn(error.to_string()))?;
        Ok(Self {
            requests,
            pending,
            latest_sequence,
            submission_gate,
            next_sequence: AtomicU64::new(0),
            responses,
        })
    }

    /// Main threadをblockせず、未開始requestと未取得responseを新しいrequestで置き換える。
    pub fn submit_latest(&self, request: DecorationRequest) -> Result<(), DecorationRequest> {
        let sequence = self
            .next_sequence
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1);
        // Result送信とlatest更新を同じgateに置く。古いtaskが比較を終えた直後に新requestが入っても、
        // submit完了後のresponse queueへ古いgenerationを残さない。
        let _gate = lock_unpoisoned(&self.submission_gate);
        self.latest_sequence.store(sequence, Ordering::Release);
        while self.responses.try_recv().is_ok() {}
        let mut queued = QueuedRequest { sequence, request };

        loop {
            match self.requests.try_send(queued) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Disconnected(returned)) => return Err(returned.request),
                Err(TrySendError::Full(returned)) => {
                    queued = returned;
                    match self.pending.try_recv() {
                        Ok(_) => {}
                        Err(TryRecvError::Empty) => std::hint::spin_loop(),
                        Err(TryRecvError::Disconnected) => return Err(queued.request),
                    }
                }
            }
        }
    }

    /// Latest completed responseを待たずに取り出す。
    pub fn try_recv(&self) -> Result<DecorationResponse, TryRecvError> {
        self.responses.try_recv()
    }

    /// Main loopの`select!`へ登録できるresponse receiverを返す。
    #[must_use]
    pub const fn responses(&self) -> &Receiver<DecorationResponse> {
        &self.responses
    }
}

fn decoration_worker_loop(
    requests: Receiver<QueuedRequest>,
    responses: Sender<DecorationResponse>,
    response_eviction: Receiver<DecorationResponse>,
    latest_sequence: &AtomicU64,
    submission_gate: &Mutex<()>,
    ledger: &DerivedMemoryLedger,
) {
    // SyntaxSet/ThemeSetのloadはworker lifetimeで1回にする。Streaming snapshotやW toggleごとに
    // grammar bundleを再構築すると、latest requestが古いsetup待ちになる。
    let mut engine: Option<Result<SyntaxEngine, HighlightError>> = None;
    while let Ok(queued) = requests.recv() {
        let generation = queued.request.generation;
        if queued.request.theme.color_level() != ColorLevel::None && engine.is_none() {
            match catch_unwind(SyntaxEngine::new) {
                Ok(result) => engine = Some(result),
                Err(_) => {
                    let _gate = lock_unpoisoned(submission_gate);
                    if latest_sequence.load(Ordering::Acquire) == queued.sequence {
                        send_latest_response(
                            &responses,
                            &response_eviction,
                            DecorationResponse {
                                generation,
                                complete: true,
                                result: Err(DecorationWorkerError::WorkerPanicked),
                            },
                        );
                    }
                    continue;
                }
            }
        }
        let syntax = match (queued.request.theme.color_level(), engine.as_ref()) {
            (ColorLevel::None, _) | (_, None) => SyntaxAvailability::Disabled,
            (_, Some(Ok(engine))) => SyntaxAvailability::Ready(engine),
            (_, Some(Err(error))) => SyntaxAvailability::Failed(error),
        };
        let cancelled = || latest_sequence.load(Ordering::Acquire) != queued.sequence;
        if let Some(priority_hunk) = queued.request.priority_hunk {
            let partial = catch_unwind(AssertUnwindSafe(|| {
                let context = DecorationContext {
                    document: &queued.request.document,
                    theme: &queued.request.theme,
                    whitespace: queued.request.whitespace,
                    syntax,
                    ledger,
                    cancelled: &cancelled,
                };
                let scope = DecorationScope {
                    budget: DERIVED_PRIORITY_RESULT_BUDGET_BYTES,
                    pool: DerivedMemoryPool::PriorityDecoration,
                    priority_hunk: Some(priority_hunk),
                    only_priority: true,
                    line_range: None,
                    structure_cursor: None,
                };
                DecorationSet::compute_internal(context, scope)
            }));
            match partial {
                Ok(Ok(Some(result))) if !result.is_empty() => {
                    let _gate = lock_unpoisoned(submission_gate);
                    if latest_sequence.load(Ordering::Acquire) != queued.sequence {
                        continue;
                    }
                    send_latest_response(
                        &responses,
                        &response_eviction,
                        DecorationResponse {
                            generation,
                            complete: false,
                            result: Ok(result),
                        },
                    );
                }
                Ok(Ok(Some(_))) => {}
                Ok(Ok(None)) => continue,
                Ok(Err(error)) => {
                    let _gate = lock_unpoisoned(submission_gate);
                    if latest_sequence.load(Ordering::Acquire) == queued.sequence {
                        send_latest_response(
                            &responses,
                            &response_eviction,
                            DecorationResponse {
                                generation,
                                complete: true,
                                result: Err(DecorationWorkerError::Document(error.to_string())),
                            },
                        );
                    }
                    continue;
                }
                Err(_) => {
                    let _gate = lock_unpoisoned(submission_gate);
                    if latest_sequence.load(Ordering::Acquire) == queued.sequence {
                        send_latest_response(
                            &responses,
                            &response_eviction,
                            DecorationResponse {
                                generation,
                                complete: true,
                                result: Err(DecorationWorkerError::WorkerPanicked),
                            },
                        );
                    }
                    continue;
                }
            }
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let context = DecorationContext {
                document: &queued.request.document,
                theme: &queued.request.theme,
                whitespace: queued.request.whitespace,
                syntax,
                ledger,
                cancelled: &cancelled,
            };
            let scope = DecorationScope {
                budget: DERIVED_RESULT_BUDGET_BYTES,
                pool: DerivedMemoryPool::NextDecoration,
                // Priority hunkは8 MiB partialで既に返している。Full builderはdocument順へ戻し、flat
                // entry Vecをappend中心に構築して先頭側への反復insertを避ける。
                priority_hunk: None,
                only_priority: false,
                line_range: None,
                structure_cursor: None,
            };
            DecorationSet::compute_internal(context, scope)
        }))
        .map_err(|_| DecorationWorkerError::WorkerPanicked);
        let result = match outcome {
            Ok(Ok(Some(result))) => Ok(result),
            Ok(Ok(None)) => continue,
            Ok(Err(error)) => Err(DecorationWorkerError::Document(error.to_string())),
            Err(error) => Err(error),
        };
        let _gate = lock_unpoisoned(submission_gate);
        if latest_sequence.load(Ordering::Acquire) != queued.sequence {
            continue;
        }
        send_latest_response(
            &responses,
            &response_eviction,
            DecorationResponse {
                generation,
                complete: true,
                result,
            },
        );
    }
}

fn lock_unpoisoned(mutex: &Mutex<()>) -> MutexGuard<'_, ()> {
    match mutex.lock() {
        Ok(guard) => guard,
        // Worker panicはresponseへ変換する。Gate自体に保護対象dataはないため、poison後も排他だけを
        // 再利用すればlatest-wins契約を維持できる。
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn send_latest_response(
    responses: &Sender<DecorationResponse>,
    response_eviction: &Receiver<DecorationResponse>,
    mut response: DecorationResponse,
) {
    loop {
        match responses.try_send(response) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => return,
            Err(TrySendError::Full(returned)) => {
                response = returned;
                let _ = response_eviction.try_recv();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::cli::ThemeChoice;
    use crate::document::DocumentLineReader;
    use crate::document_store::CHUNKED_LINE_RAW_BYTES;
    use crate::highlight::syntax_spans_equivalent;
    use crate::input::AppendStore;
    use crate::parser::{GitParser, parse_bytes as parse_snapshot};
    use crate::storage::{StorageConfig, StructuralStorage};
    use crate::structure_store::StoredFileMetadata;
    use crate::theme::ColorLevel;

    /// Inline fixtureの全件materializeは小さい単体testだけに限定する。Productionの範囲走査は
    /// [`ScopedStructureCursor`]を通り、この上限やVecへ依存しない。
    const MAX_MATERIALIZED_TEST_STRUCTURES: u64 = 4_096;

    fn parse_bytes(input: &[u8]) -> DocumentSnapshot {
        parse_snapshot(input).unwrap()
    }

    fn parse_bytes_with_storage(input: &[u8], config: StorageConfig) -> DocumentSnapshot {
        let raw = AppendStore::new();
        let storage = StructuralStorage::with_config(config).unwrap();
        let mut parser = GitParser::new(raw.clone(), &storage).unwrap();
        let start = raw.append(input).unwrap();
        let _ = parser.push(start, input).unwrap();
        let _ = raw.seal().unwrap();
        parser.into_document().unwrap()
    }

    fn materialize_file_metadata(document: &DocumentSnapshot) -> Vec<StoredFileMetadata> {
        let counts = document.structure().visible_counts().unwrap();
        assert!(counts.files <= MAX_MATERIALIZED_TEST_STRUCTURES);
        let mut reader = StructureMetadataReader::new();
        (0..counts.files)
            .map(|raw_id| {
                reader
                    .read_file(document.structure(), FileId::new(raw_id))
                    .unwrap()
                    .unwrap()
            })
            .collect()
    }

    fn materialize_hunks(document: &DocumentSnapshot) -> Vec<StoredHunk> {
        let counts = document.structure().visible_counts().unwrap();
        assert!(counts.hunks <= MAX_MATERIALIZED_TEST_STRUCTURES);
        let mut reader = StructureMetadataReader::new();
        (0..counts.hunks)
            .map(|raw_id| {
                reader
                    .read_hunk(document.structure(), HunkId::new(raw_id))
                    .unwrap()
                    .unwrap()
            })
            .collect()
    }

    fn read_lines(document: &DocumentSnapshot) -> Vec<OwnedDocumentLine> {
        let mut reader = DocumentLineReader::new();
        (0..document.line_count())
            .map(|index| {
                let line_id = document.line_id_at(index).unwrap();
                reader.read_owned(document, line_id).unwrap().unwrap()
            })
            .collect()
    }

    fn line_with_kind(document: &DocumentSnapshot, kind: LineKind) -> OwnedDocumentLine {
        read_lines(document)
            .into_iter()
            .find(|line| line.kind == kind)
            .unwrap()
    }

    fn metadata_with_kind(document: &DocumentSnapshot, kind: LineKind) -> DocumentLineMetadata {
        let mut reader = DocumentLineMetadataReader::new();
        (0..document.line_count())
            .filter_map(|index| {
                let line_id = document.line_id_at(index)?;
                reader.read(document, line_id).ok().flatten()
            })
            .find(|metadata| metadata.kind == kind)
            .unwrap()
    }

    fn push_hidden_osc_source_line(
        input: &mut Vec<u8>,
        marker: u8,
        payload_len: usize,
        content: &[u8],
    ) {
        input.push(marker);
        input.extend_from_slice(b"\x1b]title;");
        input.extend(std::iter::repeat_n(b'x', payload_len));
        input.push(b'\x07');
        input.extend_from_slice(content);
        input.push(b'\n');
    }

    fn many_rust_files(count: usize) -> String {
        let mut input = String::new();
        for index in 0..count {
            let _ = write!(
                input,
                "diff --git a/file{index}.rs b/file{index}.rs\n--- a/file{index}.rs\n+++ b/file{index}.rs\n@@ -1 +1 @@\n-let value = {index};\n+let value = {};\n",
                index + 1,
            );
        }
        input
    }

    fn block_metadata(id: usize, kind: LineKind, display_len: u32) -> DocumentLineMetadata {
        DocumentLineMetadata {
            id: LineId::new(id as u64),
            kind,
            commit_id: None,
            file_id: None,
            hunk_id: None,
            old_line_number: None,
            new_line_number: None,
            file_line_bounds: None,
            terminated: true,
            raw_len: u64::from(display_len),
            display_len: LineDisplayLength::Exact(display_len),
        }
    }

    fn theme() -> Theme {
        Theme::new(ThemeChoice::Dark, ColorLevel::TrueColor)
    }

    fn budgeted_set(
        ledger: &DerivedMemoryLedger,
        pool: DerivedMemoryPool,
        bytes: usize,
        line_id: u64,
    ) -> BudgetedDecorationSet {
        // Test fixtureもpayload allocation前にleaseし、wrapperと同じdrop順を通す。
        let lease = ledger.reserve(pool, bytes).unwrap();
        let mut set = DecorationSet::default();
        set.lines.push(DecorationEntry {
            line_id: LineId::new(line_id),
            decoration: LineDecoration::default(),
        });
        set.approximate_bytes = bytes;
        BudgetedDecorationSet { set, lease }
    }

    #[test]
    fn rename_uses_independent_old_and_new_syntax_paths() {
        let document = parse_bytes(
            b"diff --git a/src/lib.rs b/src/lib.py\nsimilarity index 80%\nrename from src/lib.rs\nrename to src/lib.py\n--- a/src/lib.rs\n+++ b/src/lib.py\n@@ -1 +1 @@\n-fn old() {}\n+def new():\n",
        );
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();
        let deletion = line_with_kind(&document, LineKind::Deletion);
        let addition = line_with_kind(&document, LineKind::Addition);
        assert!(!decorations.line(deletion.id).unwrap().syntax.is_empty());
        assert!(!decorations.line(addition.id).unwrap().syntax.is_empty());

        let engine = SyntaxEngine::new().unwrap();
        let expected_old = engine
            .session(Some("src/lib.rs"), Some("fn old() {}"), ThemeChoice::Dark)
            .unwrap()
            .unwrap()
            .highlight_line("fn old() {}")
            .unwrap();
        let expected_new = engine
            .session(Some("src/lib.py"), Some("def new():"), ThemeChoice::Dark)
            .unwrap()
            .unwrap()
            .highlight_line("def new():")
            .unwrap();
        let HighlightedLine::Styled(expected_old) = expected_old else {
            panic!("expected old syntax");
        };
        let HighlightedLine::Styled(expected_new) = expected_new else {
            panic!("expected new syntax");
        };
        assert_eq!(
            decorations.line(deletion.id).unwrap().syntax,
            expected_old
                .into_iter()
                .filter_map(|span| shift_syntax_span(span, deletion.text.len()))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            decorations.line(addition.id).unwrap().syntax,
            expected_new
                .into_iter()
                .filter_map(|span| shift_syntax_span(span, addition.text.len()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rename_context_prefers_new_side_syntax() {
        let document = parse_bytes(
            b"diff --git a/src/notes.rs b/src/notes.md\nsimilarity index 100%\nrename from src/notes.rs\nrename to src/notes.md\n--- a/src/notes.rs\n+++ b/src/notes.md\n@@ -1 +1 @@\n # heading\n",
        );
        let context = line_with_kind(&document, LineKind::Context);
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();

        let engine = SyntaxEngine::new().unwrap();
        let expected = engine
            .session(Some("src/notes.md"), Some("# heading"), ThemeChoice::Dark)
            .unwrap()
            .unwrap()
            .highlight_line("# heading")
            .unwrap();
        let HighlightedLine::Styled(expected) = expected else {
            panic!("expected markdown syntax");
        };
        assert_eq!(
            decorations.line(context.id).unwrap().syntax,
            expected
                .into_iter()
                .filter_map(|span| shift_syntax_span(span, context.text.len()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn deleted_file_context_keeps_old_side_state() {
        let document = parse_bytes(
            b"diff --git a/src/old.rs b/src/old.rs\ndeleted file mode 100644\n--- a/src/old.rs\n+++ /dev/null\n@@ -1,2 +1 @@\n-/*\n let value = 1;\n",
        );
        let context = line_with_kind(&document, LineKind::Context);
        assert_eq!(
            materialize_file_metadata(&document)[0].status,
            ChangeStatus::Deleted
        );
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();

        let engine = SyntaxEngine::new().unwrap();
        let mut old = engine
            .session(Some("src/old.rs"), Some("/*"), ThemeChoice::Dark)
            .unwrap()
            .unwrap();
        let _ = old.highlight_line("/*").unwrap();
        let expected = old.highlight_line("let value = 1;").unwrap();
        let HighlightedLine::Styled(expected) = expected else {
            panic!("expected old-side syntax");
        };
        assert_eq!(
            decorations.line(context.id).unwrap().syntax,
            expected
                .into_iter()
                .filter_map(|span| shift_syntax_span(span, context.text.len()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syntax_state_resets_at_each_hunk() {
        let document = parse_bytes(
            b"diff --git a/src/main.rs b/src/main.rs\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-old\n+/*\n@@ -10 +10 @@\n let value = 1;\n",
        );
        let context = line_with_kind(&document, LineKind::Context);
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();

        let engine = SyntaxEngine::new().unwrap();
        let expected = engine
            .session(
                Some("src/main.rs"),
                Some("let value = 1;"),
                ThemeChoice::Dark,
            )
            .unwrap()
            .unwrap()
            .highlight_line("let value = 1;")
            .unwrap();
        let HighlightedLine::Styled(expected) = expected else {
            panic!("expected reset syntax");
        };
        assert_eq!(
            decorations.line(context.id).unwrap().syntax,
            expected
                .into_iter()
                .filter_map(|span| shift_syntax_span(span, context.text.len()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn hunk_prefix_infers_a_multiline_syntax_state_from_its_closer() {
        let document = parse_bytes(
            b"diff --git a/src/main.rs b/src/main.rs\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -20,4 +20,5 @@\n  * existing contract.\n+ * added contract.\n  */\n pub fn value() -> usize {\n     1\n",
        );
        let added = read_lines(&document)
            .into_iter()
            .find(|line| line.text == "+ * added contract.")
            .unwrap();
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();

        let engine = SyntaxEngine::new().unwrap();
        let mut expected_session = engine
            .session(
                Some("src/main.rs"),
                Some("/* hidden before the hunk"),
                ThemeChoice::Dark,
            )
            .unwrap()
            .unwrap();
        let _ = expected_session
            .highlight_line("/* hidden before the hunk")
            .unwrap();
        let _ = expected_session
            .highlight_line(" * existing contract.")
            .unwrap();
        let expected = expected_session
            .highlight_line(" * added contract.")
            .unwrap();
        let HighlightedLine::Styled(expected) = expected else {
            panic!("expected inferred comment syntax");
        };

        let expected = expected
            .into_iter()
            .filter_map(|span| shift_syntax_span(span, added.text.len()))
            .collect::<Vec<_>>();
        let actual = &decorations.line(added.id).unwrap().syntax;
        assert!(
            syntax_spans_equivalent(actual, &expected),
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn hunk_prefix_infers_typescript_jsdoc_across_next_hunk_and_embedded_slash() {
        let document = parse_bytes(
            b"diff --git a/src/use-simulator.ts b/src/use-simulator.ts\n--- a/src/use-simulator.ts\n+++ b/src/use-simulator.ts\n@@ -45,3 +45,4 @@ import type {\n  * existing contract.\n+ * added contract.\n  * device.offline/online remains inside the comment.\n  * still inside the comment.\n@@ -63,2 +64,3 @@ import type {\n  * later contract.\n+ * later addition.\n  */\n",
        );
        let added = read_lines(&document)
            .into_iter()
            .find(|line| line.text == "+ * added contract.")
            .unwrap();
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();

        let engine = SyntaxEngine::new().unwrap();
        let mut expected_session = engine
            .session(
                Some("src/use-simulator.ts"),
                Some("/** hidden before the hunk"),
                ThemeChoice::Dark,
            )
            .unwrap()
            .unwrap();
        let _ = expected_session
            .highlight_line("/** hidden before the hunk")
            .unwrap();
        let _ = expected_session
            .highlight_line(" * existing contract.")
            .unwrap();
        let expected = expected_session
            .highlight_line(" * added contract.")
            .unwrap();
        let HighlightedLine::Styled(expected) = expected else {
            panic!("expected inferred JSDoc syntax");
        };

        let expected = expected
            .into_iter()
            .filter_map(|span| shift_syntax_span(span, added.text.len()))
            .collect::<Vec<_>>();
        let actual = &decorations.line(added.id).unwrap().syntax;
        assert!(
            syntax_spans_equivalent(actual, &expected),
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn hunk_prefix_prefers_jsdoc_context_for_star_led_markup() {
        let document = parse_bytes(
            b"diff --git a/server/src/lambda/shared/position-rate.ts b/server/src/lambda/shared/position-rate.ts\n--- a/server/src/lambda/shared/position-rate.ts\n+++ b/server/src/lambda/shared/position-rate.ts\n@@ -27,5 +28,9 @@ import {\n  * existing contract.\n  * gate acquisition can fail.\n  *\n+ * gate shares the `device_state` row.\n+ * conflicts are retried by the caller.\n+ * release uses {@link POSITION_RATE_GATE_RELEASE_RETRY_DELAYS_MS}.\n+ *\n  * external behavior remains unchanged.\n  */\n",
        );
        let added = read_lines(&document)
            .into_iter()
            .find(|line| {
                line.text == "+ * release uses {@link POSITION_RATE_GATE_RELEASE_RETRY_DELAYS_MS}."
            })
            .unwrap();
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();

        let engine = SyntaxEngine::new().unwrap();
        let mut expected_session = engine
            .session(
                Some("server/src/lambda/shared/position-rate.ts"),
                Some("/** hidden before the hunk"),
                ThemeChoice::Dark,
            )
            .unwrap()
            .unwrap();
        for line in [
            "/** hidden before the hunk",
            " * existing contract.",
            " * gate acquisition can fail.",
            " *",
            " * gate shares the `device_state` row.",
            " * conflicts are retried by the caller.",
        ] {
            let _ = expected_session.highlight_line(line).unwrap();
        }
        let expected = expected_session
            .highlight_line(" * release uses {@link POSITION_RATE_GATE_RELEASE_RETRY_DELAYS_MS}.")
            .unwrap();
        let HighlightedLine::Styled(expected) = expected else {
            panic!("expected inferred JSDoc syntax");
        };
        let expected = expected
            .into_iter()
            .filter_map(|span| shift_syntax_span(span, added.text.len()))
            .collect::<Vec<_>>();
        let actual = decorations
            .line(added.id)
            .map(|decoration| decoration.syntax.clone())
            .unwrap_or_default();
        assert!(
            syntax_spans_equivalent(&actual, &expected),
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn hunk_prefix_infers_unclosed_typescript_jsdoc_from_star_leaders() {
        let document = parse_bytes(
            b"diff --git a/src/use-simulator.ts b/src/use-simulator.ts\n--- a/src/use-simulator.ts\n+++ b/src/use-simulator.ts\n@@ -58,4 +58,5 @@ import type {\n  * speed uses the route.\n  * repeatMode controls the terminal behavior.\n- * old fold wording.\n+ * foldRepeat uses fold.direction.\n+ * startFromEnd mirrors the route.\n  * device.offline/online stays serialized.\n",
        );
        let added = read_lines(&document)
            .into_iter()
            .find(|line| line.text == "+ * foldRepeat uses fold.direction.")
            .unwrap();
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();

        let engine = SyntaxEngine::new().unwrap();
        let mut expected_session = engine
            .session(
                Some("src/use-simulator.ts"),
                Some("/** hidden before the hunk"),
                ThemeChoice::Dark,
            )
            .unwrap()
            .unwrap();
        let _ = expected_session
            .highlight_line("/** hidden before the hunk")
            .unwrap();
        let _ = expected_session
            .highlight_line(" * speed uses the route.")
            .unwrap();
        let _ = expected_session
            .highlight_line(" * repeatMode controls the terminal behavior.")
            .unwrap();
        let expected = expected_session
            .highlight_line(" * foldRepeat uses fold.direction.")
            .unwrap();
        let HighlightedLine::Styled(expected) = expected else {
            panic!("expected inferred JSDoc syntax");
        };

        let expected = expected
            .into_iter()
            .filter_map(|span| shift_syntax_span(span, added.text.len()))
            .collect::<Vec<_>>();
        let actual = decorations
            .line(added.id)
            .map(|decoration| decoration.syntax.clone())
            .unwrap_or_default();
        assert!(
            syntax_spans_equivalent(&actual, &expected),
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn hunk_prefix_does_not_seed_regex_before_a_typescript_line_comment() {
        let document = parse_bytes(
            b"diff --git a/server/test/lambda-data/positions-api.test.ts b/server/test/lambda-data/positions-api.test.ts\n--- a/server/test/lambda-data/positions-api.test.ts\n+++ b/server/test/lambda-data/positions-api.test.ts\n@@ -540,4 +544,9 @@ describe(\"positions-api\", () => {\n     }\n   });\n \n+  // gate shares the device_state row with the write pipeline.\n+  // retry a transient transaction conflict.\n+  it(\"retries gate acquisition\", async () => {\n+    expect(true).toBe(true);\n+  });\n });\n",
        );
        let added = read_lines(&document)
            .into_iter()
            .find(|line| {
                line.text == "+  // gate shares the device_state row with the write pipeline."
            })
            .unwrap();
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();

        let engine = SyntaxEngine::new().unwrap();
        let content = added.text.get(1..).unwrap();
        let expected = engine
            .session(
                Some("server/test/lambda-data/positions-api.test.ts"),
                Some(content),
                ThemeChoice::Dark,
            )
            .unwrap()
            .unwrap()
            .highlight_line(content)
            .unwrap();
        let HighlightedLine::Styled(expected) = expected else {
            panic!("expected standalone TypeScript syntax");
        };
        let expected = expected
            .into_iter()
            .filter_map(|span| shift_syntax_span(span, added.text.len()))
            .collect::<Vec<_>>();
        let actual = decorations
            .line(added.id)
            .map(|decoration| decoration.syntax.clone())
            .unwrap_or_default();
        assert!(
            syntax_spans_equivalent(&actual, &expected),
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn hunk_prefix_does_not_use_a_block_closer_inside_a_rust_line_comment() {
        let document = parse_bytes(
            b"diff --git a/src/highlight.rs b/src/highlight.rs\n--- a/src/highlight.rs\n+++ b/src/highlight.rs\n@@ -625,4 +680,9 @@ impl SyntaxEngine {\n             ) else {\n                 continue;\n             };\n+            // A `/` exit embedded in `//` or `*/` belongs to another context.\n+            // Keep the whole line in its visible single-line comment context.\n+            if exit_is_partial_comment_delimiter(line, exit_evidence) {\n+                continue;\n+            }\n             let score = ContinuationScore {\n",
        );
        let lines = read_lines(&document);
        let targets = [
            lines
                .iter()
                .find(|line| line.text.trim() == "continue;")
                .unwrap(),
            lines
                .iter()
                .find(|line| line.text.contains("embedded in `//` or `*/`"))
                .unwrap(),
        ];
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();
        let engine = SyntaxEngine::new().unwrap();

        for target in targets {
            let content = target.text.get(1..).unwrap();
            let expected = engine
                .session(Some("src/highlight.rs"), Some(content), ThemeChoice::Dark)
                .unwrap()
                .unwrap()
                .highlight_line(content)
                .unwrap();
            let HighlightedLine::Styled(expected) = expected else {
                panic!("expected standalone Rust syntax");
            };
            let expected = expected
                .into_iter()
                .filter_map(|span| shift_syntax_span(span, target.text.len()))
                .collect::<Vec<_>>();
            let actual = decorations
                .line(target.id)
                .map(|decoration| decoration.syntax.clone())
                .unwrap_or_default();
            assert!(
                syntax_spans_equivalent(&actual, &expected),
                "line={content:?} actual={actual:?} expected={expected:?}"
            );
        }
    }

    #[test]
    fn hunk_prefix_does_not_seed_typescript_comment_before_visible_opener() {
        let document = parse_bytes(
            b"diff --git a/src/domain/types.ts b/src/domain/types.ts\n--- a/src/domain/types.ts\n+++ b/src/domain/types.ts\n@@ -165,7 +172,7 @@ export interface RuntimeDeviceState {\n   accessTokenExpiresAt: string | null\n   /**\n    * current progress.\n-   * old terminal wording.\n+   * new terminal wording.\n    */\n   progressRatio: number | null\n   /** current direction. */\n@@ -217,8 +224,10 @@ export interface RunningDevice {\n   lastTickAt: number\n   distanceMeters: number\n   repeatMode: RepeatMode\n+  /** mirror the route. */\n+  startFromEnd: boolean\n   /**\n-   * old lap wording.\n+   * new lap wording.\n    * detect a crossed leg.\n    * null immediately after start.\n    */\n",
        );
        let lines = read_lines(&document);
        let targets = [
            lines
                .iter()
                .find(|line| line.text == "   accessTokenExpiresAt: string | null")
                .unwrap(),
            lines
                .iter()
                .find(|line| line.text == "   lastTickAt: number")
                .unwrap(),
        ];
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();
        let engine = SyntaxEngine::new().unwrap();

        for target in targets {
            let content = target.text.get(1..).unwrap();
            let expected = engine
                .session(
                    Some("src/domain/types.ts"),
                    Some(content),
                    ThemeChoice::Dark,
                )
                .unwrap()
                .unwrap()
                .highlight_line(content)
                .unwrap();
            let HighlightedLine::Styled(expected) = expected else {
                panic!("expected standalone TypeScript syntax");
            };
            let expected = expected
                .into_iter()
                .filter_map(|span| shift_syntax_span(span, target.text.len()))
                .collect::<Vec<_>>();
            let actual = decorations
                .line(target.id)
                .map(|decoration| decoration.syntax.clone())
                .unwrap_or_default();
            assert!(
                syntax_spans_equivalent(&actual, &expected),
                "line={content:?} actual={actual:?} expected={expected:?}"
            );
        }
    }

    #[test]
    fn hunk_prefix_does_not_infer_from_a_first_line_closer() {
        let document = parse_bytes(
            b"diff --git a/src/main.rs b/src/main.rs\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -20,3 +20,3 @@\n-*/\n+*/\n let value = 1;\n drop(value);\n",
        );
        let addition = line_with_kind(&document, LineKind::Addition);
        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();

        let engine = SyntaxEngine::new().unwrap();
        let expected = engine
            .session(Some("src/main.rs"), Some("*/"), ThemeChoice::Dark)
            .unwrap()
            .unwrap()
            .highlight_line("*/")
            .unwrap();
        let HighlightedLine::Styled(expected) = expected else {
            panic!("expected standalone syntax");
        };
        let expected = expected
            .into_iter()
            .filter_map(|span| shift_syntax_span(span, addition.text.len()))
            .collect::<Vec<_>>();

        assert_eq!(decorations.line(addition.id).unwrap().syntax, expected);
    }

    #[test]
    fn raw_huge_display_short_source_lines_keep_syntax_and_intraline() {
        let mut input = b"diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n".to_vec();
        let payload_len = MAX_SYNTAX_LINE_BYTES + CHUNKED_LINE_RAW_BYTES;
        push_hidden_osc_source_line(&mut input, b'-', payload_len, b"let old = 1;");
        push_hidden_osc_source_line(&mut input, b'+', payload_len, b"let new = 2;");
        let document = parse_bytes(&input);
        let deletion = metadata_with_kind(&document, LineKind::Deletion);
        let addition = metadata_with_kind(&document, LineKind::Addition);
        assert!(deletion.raw_len > MAX_SYNTAX_DISPLAY_BYTES as u64);
        assert_eq!(deletion.display_len.exact(), Some("-let old = 1;".len()));

        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Show).unwrap();
        for metadata in [deletion, addition] {
            let line = decorations.line(metadata.id).unwrap();
            assert!(!line.syntax.is_empty());
            assert!(!line.intraline.is_empty());
        }
        assert!(decorations.warning().is_none());
    }

    #[test]
    fn syntax_streaming_decode_observes_cancellation_between_raw_chunks() {
        let mut input = b"diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n".to_vec();
        let payload_len = MAX_SYNTAX_LINE_BYTES + 8 * CHUNKED_LINE_RAW_BYTES;
        push_hidden_osc_source_line(&mut input, b'-', payload_len, b"let old = 1;");
        input.extend_from_slice(b"+let new = 2;\n");
        let document = parse_bytes(&input);
        let deletion = metadata_with_kind(&document, LineKind::Deletion);
        let engine = SyntaxEngine::new().unwrap();
        let theme = theme();
        let checks = Cell::new(0_usize);
        let cancelled = || {
            let current = checks.get();
            checks.set(current + 1);
            current >= 5
        };
        let ledger = DerivedMemoryLedger::new();
        let context = DecorationContext {
            document: &document,
            theme: &theme,
            whitespace: WhitespaceChoice::Ignore,
            syntax: SyntaxAvailability::Ready(&engine),
            ledger: &ledger,
            cancelled: &cancelled,
        };
        let mut builder = DecorationBuilder::new(
            document.revision(),
            WhitespaceChoice::Ignore,
            &ledger,
            DerivedMemoryPool::NextDecoration,
            DERIVED_RESULT_BUDGET_BYTES,
        )
        .unwrap();
        let mut readers = DecorationReaders::new(&ledger).unwrap();
        let mut structure_reader = StructureReader::new();
        let file = structure_reader
            .read_file(document.structure(), FileId::new(0))
            .unwrap()
            .unwrap();
        let mut hunk_reader = StructureMetadataReader::new();
        let hunk = hunk_reader
            .read_hunk_in_file(document.structure(), HunkId::new(0), file.metadata())
            .unwrap()
            .unwrap();

        assert!(
            !decorate_syntax_hunk(&mut builder, &engine, file, hunk, &context, &mut readers,)
                .unwrap()
        );
        let raw_bytes_read = readers.text.large_line_stats().raw_bytes_read;
        assert!(raw_bytes_read > 0);
        assert!(raw_bytes_read <= (3 * CHUNKED_LINE_RAW_BYTES) as u64);
        assert!(raw_bytes_read < deletion.raw_len);
    }

    #[test]
    fn oversized_deletion_resets_only_the_old_syntax_side() {
        let mut input =
            b"diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1,2 +1,2 @@\n+/*\n-".to_vec();
        input.extend(std::iter::repeat_n(b'x', MAX_SYNTAX_LINE_BYTES + 1));
        input.extend_from_slice(b"\n let value = 1;\n");
        let document = parse_bytes(&input);
        let context_line = metadata_with_kind(&document, LineKind::Context);

        let decorations =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();
        assert!(
            decorations
                .warning()
                .unwrap()
                .contains("syntax omitted for some lines")
        );

        let engine = SyntaxEngine::new().unwrap();
        let mut new = engine
            .session(Some("a.rs"), Some("/*"), ThemeChoice::Dark)
            .unwrap()
            .unwrap();
        let _ = new.highlight_line("/*").unwrap();
        let HighlightedLine::Styled(expected) = new.highlight_line("let value = 1;").unwrap()
        else {
            panic!("expected new-side syntax");
        };
        assert_eq!(
            decorations.line(context_line.id).unwrap().syntax,
            expected
                .into_iter()
                .filter_map(|span| shift_syntax_span(span, " let value = 1;".len()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn intraline_ignore_and_show_keep_full_line_offsets() {
        let document = parse_bytes(
            b"diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-foo\tbar\n+foo   bar\n",
        );
        let deletion = line_with_kind(&document, LineKind::Deletion);
        let ignored =
            DecorationSet::compute(&document, &theme(), WhitespaceChoice::Ignore).unwrap();
        assert!(
            ignored
                .line(deletion.id)
                .is_none_or(|line| line.intraline.is_empty())
        );

        let shown = DecorationSet::compute(&document, &theme(), WhitespaceChoice::Show).unwrap();
        let ranges = &shown.line(deletion.id).unwrap().intraline;
        assert!(
            ranges
                .iter()
                .any(|range| &deletion.text[range.clone()] == "\t")
        );
        assert!(ranges.iter().all(|range| range.start >= 1));
    }

    #[test]
    fn small_budget_marks_result_truncated() {
        let document = parse_bytes(
            b"diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-let old = 1;\n+let new = 2;\n",
        );
        let result = DecorationSet::compute_with_budget(
            &document,
            &theme(),
            WhitespaceChoice::Show,
            1,
            None,
        )
        .unwrap();
        assert!(result.is_truncated());
        assert!(result.warning().unwrap().contains("budget"));
    }

    #[test]
    fn range_compute_does_not_decorate_previously_emitted_files() {
        let document = parse_bytes(
            b"diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-let a = 1;\n+let a = 2;\ndiff --git a/b.rs b/b.rs\n--- a/b.rs\n+++ b/b.rs\n@@ -1 +1 @@\n-let b = 1;\n+let b = 2;\n",
        );
        let files = materialize_file_metadata(&document);
        let first = files[0];
        let first_range = usize::try_from(first.span.start.get()).unwrap()
            ..usize::try_from(first.span.end_exclusive.unwrap()).unwrap();
        let result = DecorationSet::compute_range(
            &document,
            &theme(),
            WhitespaceChoice::Ignore,
            first_range,
        )
        .unwrap();
        let lines = read_lines(&document);
        let first_source = lines
            .iter()
            .find(|line| line.file_id == Some(first.id) && line.kind == LineKind::Addition)
            .unwrap();
        let second = files[1];
        let second_source = lines
            .iter()
            .find(|line| line.file_id == Some(second.id) && line.kind == LineKind::Addition)
            .unwrap();

        assert!(result.line(first_source.id).is_some());
        assert!(result.line(second_source.id).is_none());
    }

    #[test]
    fn spill_backed_range_compute_decorates_only_the_intersecting_file() {
        // Directory reserveとresident limitを同値にしてencoded payloadのRAM枠を0にする。Table/blobは
        // unlinked tempfileへ置かれ、decoratorは同じcursor/read APIだけで中央fileを処理する。
        let config = StorageConfig::new(64 * 1024, 64 * 1024, 64, 256).unwrap();
        let document = parse_bytes_with_storage(many_rust_files(48).as_bytes(), config);
        let files = materialize_file_metadata(&document);
        let selected = files[24];
        let selected_range = usize::try_from(selected.span.start.get()).unwrap()
            ..usize::try_from(selected.span.end_exclusive.unwrap()).unwrap();

        let colorless = Theme::new(ThemeChoice::Dark, ColorLevel::None);
        let result = DecorationSet::compute_range(
            &document,
            &colorless,
            WhitespaceChoice::Show,
            selected_range,
        )
        .unwrap();
        let lines = read_lines(&document);
        let addition_for = |file_id| {
            lines
                .iter()
                .find(|line| line.file_id == Some(file_id) && line.kind == LineKind::Addition)
                .unwrap()
                .id
        };

        assert!(result.line(addition_for(selected.id)).is_some());
        assert!(result.line(addition_for(files[23].id)).is_none());
        assert!(result.line(addition_for(files[25].id)).is_none());
    }

    #[test]
    fn scoped_hunk_scan_does_not_redecode_parent_records() {
        let document = parse_bytes(many_rust_files(6).as_bytes());
        let mut cursor_reader = StructureMetadataReader::new();
        let cursor = ScopedStructureCursor::new(&document, None, &mut cursor_reader).unwrap();
        let mut path_reader = StructureReader::new();
        let mut metadata_reader = StructureMetadataReader::new();
        let mut visited_commits = 0_u64;
        let mut visited_hunks = 0_u64;

        for raw_commit_id in cursor.commits.clone() {
            let commit = metadata_reader
                .read_commit(document.structure(), CommitId::new(raw_commit_id))
                .unwrap()
                .unwrap();
            visited_commits += 1;
            for file_id in cursor.files_in_commit(commit.files) {
                let file = path_reader
                    .read_file_in_commit(document.structure(), file_id, commit)
                    .unwrap()
                    .unwrap();
                let file_metadata = file.metadata();
                for hunk_id in file.hunks.iter() {
                    let _ = metadata_reader
                        .read_hunk_in_file(document.structure(), hunk_id, file_metadata)
                        .unwrap()
                        .unwrap();
                    visited_hunks += 1;
                }
            }
        }

        let stats = metadata_reader.statistics();
        assert_eq!(stats.hunk_records, visited_hunks);
        assert_eq!(stats.file_records, 0);
        assert_eq!(stats.commit_records, visited_commits);
    }

    #[test]
    fn range_computer_loads_syntax_bundle_once_and_skips_plain_segments() {
        let computer = DecorationComputer::new();
        let plain = parse_bytes(b"metadata only\n");
        computer
            .compute_range(
                &plain,
                &theme(),
                WhitespaceChoice::Ignore,
                0..plain.line_count(),
            )
            .unwrap();
        assert!(computer.engine.get().is_none());

        let rich = parse_bytes(
            b"diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-let a = 1;\n+let a = 2;\n",
        );
        for _ in 0..2 {
            computer
                .compute_range(
                    &rich,
                    &theme(),
                    WhitespaceChoice::Ignore,
                    0..rich.line_count(),
                )
                .unwrap();
        }
        assert!(computer.engine.get().is_some());
    }

    #[test]
    fn budget_never_publishes_only_one_side_of_intraline_block() {
        let document = parse_bytes(
            b"diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old value\n+new value\n",
        );
        let colorless = Theme::new(ThemeChoice::Dark, ColorLevel::None);
        let complete = DecorationSet::compute_with_budget(
            &document,
            &colorless,
            WhitespaceChoice::Show,
            usize::MAX,
            None,
        )
        .unwrap();
        let constrained = DecorationSet::compute_with_budget(
            &document,
            &colorless,
            WhitespaceChoice::Show,
            complete.approximate_bytes().saturating_sub(1),
            None,
        )
        .unwrap();
        assert!(constrained.is_truncated());
        for line in read_lines(&document)
            .into_iter()
            .filter(|line| matches!(line.kind, LineKind::Deletion | LineKind::Addition))
        {
            assert!(
                constrained
                    .line(line.id)
                    .is_none_or(|decoration| decoration.intraline.is_empty())
            );
        }
    }

    #[test]
    fn no_color_skips_syntax_but_keeps_intraline_data() {
        let document = parse_bytes(
            b"diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-let old = 1;\n+let new = 2;\n",
        );
        let colorless = Theme::new(ThemeChoice::Dark, ColorLevel::None);
        let result = DecorationSet::compute(&document, &colorless, WhitespaceChoice::Show).unwrap();
        for line in read_lines(&document)
            .into_iter()
            .filter(|line| is_source(line.kind))
        {
            let decoration = result.line(line.id).unwrap();
            assert!(decoration.syntax.is_empty());
            assert!(!decoration.intraline.is_empty());
        }
    }

    #[test]
    fn intraline_block_releases_ids_when_line_cap_is_exceeded() {
        let ledger = DerivedMemoryLedger::new();
        let mut block = IntralineBlockIds::new(&ledger);
        for id in 0..=MAX_CHANGED_LINES {
            block.push_removed(block_metadata(id, LineKind::Deletion, 2));
        }

        assert!(block.rejected);
        assert!(block.removed.is_empty());
        assert!(block.added.is_empty());
    }

    #[test]
    fn intraline_block_rejects_byte_cap_before_owning_text() {
        let ledger = DerivedMemoryLedger::new();
        let mut block = IntralineBlockIds::new(&ledger);
        let display_len = u32::try_from(MAX_BLOCK_BYTES + 2).unwrap();
        block.push_added(block_metadata(0, LineKind::Addition, display_len));

        assert!(block.rejected);
        assert!(block.removed.is_empty());
        assert!(block.added.is_empty());
    }

    #[test]
    fn bounded_owned_line_preserves_snapshot_file_line_bounds() {
        let mut metadata = block_metadata(0, LineKind::Deletion, 4);
        metadata.file_line_bounds = Some(crate::structure_store::StructureFileLineBounds {
            max_old_line_number: Some(42),
            max_new_line_number: Some(84),
        });

        let line = own_bounded_line(metadata, "-old", &|| false)
            .unwrap()
            .unwrap();
        assert_eq!(line.file_line_bounds, metadata.file_line_bounds);
    }

    #[test]
    fn worker_replaces_queued_stale_response() {
        let worker = DecorationWorker::spawn(Arc::new(DerivedMemoryLedger::new())).unwrap();
        let document = Arc::new(parse_bytes(b"plain\n"));
        worker
            .submit_latest(DecorationRequest {
                generation: 1,
                document: Arc::clone(&document),
                theme: theme(),
                whitespace: WhitespaceChoice::Ignore,
                priority_hunk: None,
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while worker.responses.is_empty() && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(!worker.responses.is_empty());

        worker
            .submit_latest(DecorationRequest {
                generation: 2,
                document,
                theme: theme(),
                whitespace: WhitespaceChoice::Show,
                priority_hunk: None,
            })
            .unwrap();
        let response = worker
            .responses
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        assert_eq!(response.generation, 2);
        assert_eq!(
            response.result.unwrap().whitespace(),
            WhitespaceChoice::Show
        );
        assert!(worker.responses.try_recv().is_err());
    }

    #[test]
    fn worker_publishes_only_the_latest_back_to_back_request() {
        let worker = DecorationWorker::spawn(Arc::new(DerivedMemoryLedger::new())).unwrap();
        let document = Arc::new(parse_bytes(b"plain\n"));
        worker
            .submit_latest(DecorationRequest {
                generation: 10,
                document: Arc::clone(&document),
                theme: theme(),
                whitespace: WhitespaceChoice::Ignore,
                priority_hunk: None,
            })
            .unwrap();
        worker
            .submit_latest(DecorationRequest {
                generation: 11,
                document,
                theme: theme(),
                whitespace: WhitespaceChoice::Show,
                priority_hunk: None,
            })
            .unwrap();

        let response = worker
            .responses
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        assert_eq!(response.generation, 11);
        assert!(worker.responses.try_recv().is_err());
    }

    #[test]
    fn compute_abandons_a_stale_snapshot_between_lines() {
        let mut input = String::from(
            "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1,50 +1,50 @@\n",
        );
        for index in 0..50 {
            input.push_str(&format!(" line {index}\n"));
        }
        let document = parse_bytes(input.as_bytes());
        let checks = Cell::new(0_usize);
        let cancelled = || {
            let current = checks.get();
            checks.set(current + 1);
            current >= 8
        };
        let theme = Theme::new(ThemeChoice::Dark, ColorLevel::None);
        let ledger = DerivedMemoryLedger::new();
        let context = DecorationContext {
            document: &document,
            theme: &theme,
            whitespace: WhitespaceChoice::Ignore,
            syntax: SyntaxAvailability::Disabled,
            ledger: &ledger,
            cancelled: &cancelled,
        };
        let scope = DecorationScope {
            budget: DERIVED_RESULT_BUDGET_BYTES,
            pool: DerivedMemoryPool::NextDecoration,
            priority_hunk: None,
            only_priority: false,
            line_range: None,
            structure_cursor: None,
        };
        let result = DecorationSet::compute_internal(context, scope);
        assert!(result.unwrap().is_none());
        assert!(checks.get() < 50);
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }

    #[test]
    fn worker_publishes_priority_hunk_before_full_set() {
        let document = Arc::new(parse_bytes(
            b"diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-let first = 1;\n+let first = 2;\n@@ -10 +10 @@\n-let second = 1;\n+let second = 2;\n",
        ));
        let hunks = materialize_hunks(&document);
        let priority = hunks[1].id;
        let lines = read_lines(&document);
        let first_line = lines
            .iter()
            .find(|line| line.hunk_id == Some(hunks[0].id) && is_source(line.kind))
            .unwrap()
            .id;
        let priority_line = lines
            .iter()
            .find(|line| line.hunk_id == Some(priority) && is_source(line.kind))
            .unwrap()
            .id;
        let ledger = Arc::new(DerivedMemoryLedger::new());
        let worker = DecorationWorker::spawn(Arc::clone(&ledger)).unwrap();
        worker
            .submit_latest(DecorationRequest {
                generation: 20,
                document,
                theme: theme(),
                whitespace: WhitespaceChoice::Show,
                priority_hunk: Some(priority),
            })
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        while worker.responses.len() < 2 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(worker.responses.len(), 2);
        let coexistence = ledger.snapshot();
        assert!(
            coexistence.reserved_bytes(DerivedMemoryPool::PriorityDecoration) > 0,
            "priority response must own its dedicated pool while queued"
        );
        assert!(
            coexistence.reserved_bytes(DerivedMemoryPool::NextDecoration) > 0,
            "full response must own Next while the priority response is still queued"
        );

        let partial = worker
            .responses
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        assert!(!partial.complete);
        let partial = partial.result.unwrap();
        assert!(partial.line(priority_line).is_some());
        assert!(partial.line(first_line).is_none());
        drop(partial);
        assert_eq!(
            ledger
                .snapshot()
                .reserved_bytes(DerivedMemoryPool::PriorityDecoration),
            0
        );
        assert!(
            ledger
                .snapshot()
                .reserved_bytes(DerivedMemoryPool::NextDecoration)
                > 0
        );

        let complete = worker
            .responses
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        assert!(complete.complete);
        let complete = complete.result.unwrap();
        assert!(complete.line(priority_line).is_some());
        assert!(complete.line(first_line).is_some());
        drop(complete);
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }

    #[test]
    fn applied_replacement_reclassifies_priority_and_next_without_overlap() {
        let ledger = DerivedMemoryLedger::new();
        let mut applied = AppliedDecorationSet::empty(&ledger).unwrap();

        applied
            .replace(
                budgeted_set(&ledger, DerivedMemoryPool::PriorityDecoration, 4096, 10),
                &ledger,
            )
            .unwrap();
        let priority_applied = ledger.snapshot();
        assert_eq!(
            priority_applied.reserved_bytes(DerivedMemoryPool::AppliedDecoration),
            4096
        );
        assert_eq!(
            priority_applied.reserved_bytes(DerivedMemoryPool::PriorityDecoration),
            0
        );
        assert!(applied.line(LineId::new(10)).is_some());

        applied
            .replace(
                budgeted_set(&ledger, DerivedMemoryPool::NextDecoration, 6144, 20),
                &ledger,
            )
            .unwrap();
        let next_applied = ledger.snapshot();
        assert_eq!(
            next_applied.reserved_bytes(DerivedMemoryPool::AppliedDecoration),
            6144
        );
        assert_eq!(
            next_applied.reserved_bytes(DerivedMemoryPool::NextDecoration),
            0
        );
        assert!(applied.line(LineId::new(10)).is_none());
        assert!(applied.line(LineId::new(20)).is_some());

        drop(applied);
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }

    #[test]
    fn applied_replacement_restores_payload_after_foreign_or_invalid_lease() {
        let ledger = DerivedMemoryLedger::new();
        let foreign = DerivedMemoryLedger::new();
        let mut applied = AppliedDecorationSet::empty(&ledger).unwrap();
        applied
            .replace(
                budgeted_set(&ledger, DerivedMemoryPool::NextDecoration, 4096, 10),
                &ledger,
            )
            .unwrap();

        assert_eq!(
            applied.replace(
                budgeted_set(&foreign, DerivedMemoryPool::NextDecoration, 2048, 20,),
                &ledger,
            ),
            Err(DerivedMemoryError::ForeignLease)
        );
        assert!(applied.line(LineId::new(10)).is_some());
        assert!(applied.line(LineId::new(20)).is_none());
        assert_eq!(foreign.snapshot().total_reserved_bytes(), Some(0));

        assert_eq!(
            applied.replace(
                budgeted_set(&ledger, DerivedMemoryPool::AppliedDecoration, 2048, 30,),
                &ledger,
            ),
            Err(DerivedMemoryError::InvalidAppliedReplacement {
                current: DerivedMemoryPool::AppliedDecoration,
                replacement: DerivedMemoryPool::AppliedDecoration,
            })
        );
        assert!(applied.line(LineId::new(10)).is_some());
        assert!(applied.line(LineId::new(30)).is_none());
        assert_eq!(
            ledger
                .snapshot()
                .reserved_bytes(DerivedMemoryPool::AppliedDecoration),
            4096
        );

        drop(applied);
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }

    #[test]
    fn stale_response_eviction_releases_the_replaced_result_lease() {
        let ledger = DerivedMemoryLedger::new();
        let (sender, responses) = bounded::<DecorationResponse>(1);
        let eviction = responses.clone();
        send_latest_response(
            &sender,
            &eviction,
            DecorationResponse {
                generation: 1,
                complete: true,
                result: Ok(budgeted_set(
                    &ledger,
                    DerivedMemoryPool::NextDecoration,
                    4096,
                    10,
                )),
            },
        );
        assert_eq!(
            ledger
                .snapshot()
                .reserved_bytes(DerivedMemoryPool::NextDecoration),
            4096
        );

        send_latest_response(
            &sender,
            &eviction,
            DecorationResponse {
                generation: 2,
                complete: true,
                result: Ok(budgeted_set(
                    &ledger,
                    DerivedMemoryPool::NextDecoration,
                    6144,
                    20,
                )),
            },
        );
        assert_eq!(
            ledger
                .snapshot()
                .reserved_bytes(DerivedMemoryPool::NextDecoration),
            6144
        );
        assert_eq!(responses.recv().unwrap().generation, 2);
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }

    #[test]
    fn budgeted_result_unwind_releases_its_pool() {
        let ledger = DerivedMemoryLedger::new();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _result = budgeted_set(&ledger, DerivedMemoryPool::NextDecoration, 4096, 10);
            panic!("synthetic decoration task panic");
        }));
        assert!(outcome.is_err());
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }

    #[test]
    fn tiny_result_budget_allocates_no_entry_and_keeps_static_warning() {
        let document = parse_bytes(b"plain\n");
        let ledger = DerivedMemoryLedger::new();
        let budget = 2 * size_of::<DecorationEntry>() - 1;
        let mut builder = DecorationBuilder::new(
            document.revision(),
            WhitespaceChoice::Ignore,
            &ledger,
            DerivedMemoryPool::NextDecoration,
            budget,
        )
        .unwrap();

        assert!(!builder.ensure_line_capacity(LineId::new(1)));
        assert_eq!(builder.set.lines.capacity(), 0);
        let result = builder.finish();
        assert_eq!(result.leased_bytes(), 0);
        assert!(result.is_truncated());
        assert!(result.warning().unwrap().contains("budget exceeded"));
        drop(result);
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }

    #[test]
    fn reader_workspace_keeps_separate_eager_and_streaming_high_water_marks() {
        let ledger = DerivedMemoryLedger::new();
        let mut readers = DecorationReaders::new(&ledger).unwrap();
        let eager = block_metadata(1, LineKind::Context, 32);
        assert!(readers.reserve_text_read(eager));
        let eager_bytes = readers.eager_accounted;

        let mut rejected = block_metadata(
            99,
            LineKind::Deletion,
            u32::try_from(48 * 1024 * 1024).unwrap(),
        );
        rejected.raw_len = u64::try_from(MAX_EAGER_TEXT_RAW_BYTES).unwrap() + 1;
        assert!(!readers.reserve_text_read(rejected));
        assert_eq!(readers.eager_accounted, eager_bytes);
        assert_eq!(readers.streaming_accounted, 0);
        assert_eq!(readers.text_lease.bytes(), eager_bytes);

        let mut streaming = block_metadata(2, LineKind::Context, 64);
        streaming.raw_len = u64::try_from(MAX_EAGER_TEXT_RAW_BYTES).unwrap() + 1;
        assert!(readers.reserve_text_read(streaming));
        let streaming_bytes = readers.streaming_accounted;
        assert_eq!(readers.text_lease.bytes(), eager_bytes + streaming_bytes);

        assert!(readers.reserve_text_read(block_metadata(3, LineKind::Context, 1)));
        assert_eq!(readers.eager_accounted, eager_bytes);
        assert_eq!(readers.streaming_accounted, streaming_bytes);
        assert_eq!(readers.text_lease.bytes(), eager_bytes + streaming_bytes);
        drop(readers);
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }

    #[test]
    fn failed_huge_hunk_reservation_does_not_poison_a_later_rust_hunk() {
        let document = parse_bytes(
            b"diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-let old = 1;\n+let new = 2;\n",
        );
        let theme = theme();
        let engine = SyntaxEngine::new().unwrap();
        let ledger = DerivedMemoryLedger::new();
        let mut readers = DecorationReaders::new(&ledger).unwrap();

        // 先行hunkのstreaming lineが48 MiBなら、完成textとdecoderの上限はWorkspace 96 MiBを
        // 超える。このmetadataだけでallocation前の失敗を再現し、48 MiB payloadをtest内に作らない。
        let mut huge = block_metadata(
            100,
            LineKind::Deletion,
            u32::try_from(48 * 1024 * 1024).unwrap(),
        );
        huge.raw_len = u64::try_from(MAX_EAGER_TEXT_RAW_BYTES).unwrap() + 1;
        assert!(!readers.reserve_text_read(huge));
        assert_eq!(readers.eager_accounted, 0);
        assert_eq!(readers.streaming_accounted, 0);
        assert_eq!(readers.text_lease.bytes(), 0);

        let cancelled = || false;
        let context = DecorationContext {
            document: &document,
            theme: &theme,
            whitespace: WhitespaceChoice::Ignore,
            syntax: SyntaxAvailability::Ready(&engine),
            ledger: &ledger,
            cancelled: &cancelled,
        };
        let mut builder = DecorationBuilder::new(
            document.revision(),
            WhitespaceChoice::Ignore,
            &ledger,
            DerivedMemoryPool::NextDecoration,
            DERIVED_RESULT_BUDGET_BYTES,
        )
        .unwrap();
        let mut structure_reader = StructureReader::new();
        let file = structure_reader
            .read_file(document.structure(), FileId::new(0))
            .unwrap()
            .unwrap();
        let mut hunk_reader = StructureMetadataReader::new();
        let hunk = hunk_reader
            .read_hunk_in_file(document.structure(), HunkId::new(0), file.metadata())
            .unwrap()
            .unwrap();
        assert!(
            decorate_syntax_hunk(&mut builder, &engine, file, hunk, &context, &mut readers,)
                .unwrap()
        );

        let addition = metadata_with_kind(&document, LineKind::Addition);
        let result = builder.finish();
        assert!(!result.line(addition.id).unwrap().syntax.is_empty());
        drop(result);
        drop(readers);
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }

    #[test]
    fn flat_result_entries_stay_sorted_and_within_the_result_lease() {
        let document = parse_bytes(b"plain\n");
        let ledger = DerivedMemoryLedger::new();
        let budget = 16 * 1024;
        let mut builder = DecorationBuilder::new(
            document.revision(),
            WhitespaceChoice::Ignore,
            &ledger,
            DerivedMemoryPool::NextDecoration,
            budget,
        )
        .unwrap();
        for raw_id in [9, 1, 5] {
            let line_id = LineId::new(raw_id);
            assert!(builder.ensure_line_capacity(line_id));
            let _ = builder.line_mut(line_id);
        }

        let result = builder.finish();
        assert!(
            result
                .set
                .lines
                .windows(2)
                .all(|entries| entries[0].line_id < entries[1].line_id)
        );
        for raw_id in [1, 5, 9] {
            assert!(result.line(LineId::new(raw_id)).is_some());
        }
        assert_eq!(result.approximate_bytes(), result.leased_bytes());
        assert!(result.leased_bytes() <= budget);
        drop(result);
        assert_eq!(ledger.snapshot().total_reserved_bytes(), Some(0));
    }
}
