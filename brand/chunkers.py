"""按文档类型切块。

思路：先按 Markdown 结构切（章节 → 段落 / 列表 / 表格 → 句子），长度只做兜底；
每个 chunk 的第一行是标题面包屑，单独检索出来也能看懂它属于哪里。

    qa       一问一答一个 chunk，答案太长时分段并在每段重复问题
    product  按产品（H1）和章节切，规格表单独成块，H1 之间绝不合并
    brand    按章节切，小章节合并，段落边界
    article  同 brand，另外相邻 chunk 重叠完整的句子
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .md_parser import LIST_ITEM_RE, Block, ParsedDoc, Section, render_body

logger = logging.getLogger(__name__)

DOC_TYPES = ("qa", "product", "brand", "article")


@dataclass(frozen=True)
class ChunkParams:
    max_chars: int  # chunk 最大字符数（含面包屑）
    min_chars: int  # 小于这个长度的碎片会尝试并入相邻 chunk
    overlap: int  # 相邻 chunk 重叠的句子数


TYPE_CONFIG: Dict[str, ChunkParams] = {
    "qa": ChunkParams(max_chars=800, min_chars=0, overlap=0),
    "product": ChunkParams(max_chars=600, min_chars=80, overlap=0),
    "brand": ChunkParams(max_chars=800, min_chars=150, overlap=0),
    "article": ChunkParams(max_chars=1000, min_chars=150, overlap=1),
}

OVERLAP_MAX_CHARS = 150  # 重叠部分最多占多少字符
QA_RESIDUE_MIN_CHARS = 30  # qa 文档里不属于问答的文字，短于这个长度直接丢弃


class ChunkingError(ValueError):
    """文档内容和声明的类型对不上，没法切。"""


@dataclass
class Chunk:
    content: str
    chunk_type: str  # qa | section | table | list | text
    heading_path: List[str]
    metadata: Dict[str, object] = field(default_factory=dict)


def chunk_document(doc: ParsedDoc, doc_type: str, params: Optional[ChunkParams] = None) -> List[Chunk]:
    if doc_type not in CHUNKERS:
        raise ValueError(f"不支持的文档类型：{doc_type}（可选：{', '.join(DOC_TYPES)}）")
    return CHUNKERS[doc_type](doc, params or TYPE_CONFIG[doc_type])


# ---------------------------------------------------------------------------
# 文本拆分工具
# ---------------------------------------------------------------------------

_ENDERS = "。！？!?；;…"
_CLOSERS = "”’」』）)》\"'"


def split_sentences(text: str) -> List[str]:
    """按句切分。返回的句子首尾相接正好等于原文，一个字符都不丢。"""
    out: List[str] = []
    start, i, n = 0, 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\n" or ch in _ENDERS or (ch == "." and (i + 1 == n or text[i + 1].isspace())):
            if ch != "\n":
                while i + 1 < n and (text[i + 1] in _ENDERS or text[i + 1] in _CLOSERS):
                    i += 1
            out.append(text[start:i + 1])
            start = i + 1
        i += 1
    if start < n:
        out.append(text[start:])
    return out


def _pack(units: Sequence[str], budget: int, sep: str) -> List[str]:
    """按顺序贪心装箱，每箱不超过 budget；单个 unit 永远不拆。"""
    out: List[str] = []
    buf: List[str] = []
    size = 0
    for unit in units:
        extra = len(unit) + (len(sep) if buf else 0)
        if buf and size + extra > budget:
            out.append(sep.join(buf))
            buf, size = [unit], len(unit)
        else:
            buf.append(unit)
            size += extra
    if buf:
        out.append(sep.join(buf))
    return out


def _split_prose(text: str, budget: int) -> List[str]:
    if len(text) <= budget:
        return [text]
    units: List[str] = []
    for sentence in split_sentences(text):
        while len(sentence) > budget:  # 一句话本身就超长，只能硬切
            units.append(sentence[:budget])
            sentence = sentence[budget:]
        if sentence:
            units.append(sentence)
    return [p.strip() for p in _pack(units, budget, "") if p.strip()]


def _split_table(text: str, budget: int) -> List[str]:
    """按行拆表格，每一片都重复表头。单行永远不拆。"""
    rows = text.split("\n")
    header = "\n".join(rows[:2])
    room = budget - len(header) - 1
    if room <= 0 or len(rows) <= 3:
        return [text]
    return [header + "\n" + group for group in _pack(rows[2:], room, "\n")]


def _list_items(text: str) -> List[str]:
    """把列表拆成顶层条目，子列表和续行跟着所属条目走。"""
    lines = text.split("\n")
    first = LIST_ITEM_RE.match(lines[0])
    base = len(first.group(1).expandtabs(4)) if first else 0
    items: List[List[str]] = []
    for line in lines:
        m = LIST_ITEM_RE.match(line)
        if not items or (m and len(m.group(1).expandtabs(4)) <= base):
            items.append([line])
        else:
            items[-1].append(line)
    return ["\n".join(item).rstrip() for item in items]


def _split_list(text: str, budget: int) -> List[str]:
    units: List[str] = []
    for item in _list_items(text):
        units.extend(_split_prose(item, budget) if len(item) > budget else [item])
    return _pack(units, budget, "\n")


def _split_block(block: Block, budget: int) -> List[Tuple[str, str]]:
    """把超长的块拆成不超过 budget 的片段（代码块除外），返回 (kind, text)。"""
    kind, text = block.kind, block.text
    if len(text) <= budget or kind == "code":
        return [(kind, text)]
    if kind == "table":
        parts = _split_table(text, budget)
    elif kind == "list":
        parts = _split_list(text, budget)
    else:
        parts = _split_prose(text, budget)
    return [(kind, p) for p in parts]


# ---------------------------------------------------------------------------
# 按章节切（product / brand / article 共用）
# ---------------------------------------------------------------------------


def _common_len(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _add_kinds(kinds: Dict[str, int], more: Dict[str, int]) -> Dict[str, int]:
    out = dict(kinds)
    for k, v in more.items():
        out[k] = out.get(k, 0) + v
    return out


def _kind_counts(blocks: Iterable[Block]) -> Dict[str, int]:
    kinds: Dict[str, int] = {}
    for b in blocks:
        kinds[b.kind] = kinds.get(b.kind, 0) + len(b.text)
    return kinds


def _chunk_type(kinds: Dict[str, int]) -> str:
    total = sum(kinds.values()) or 1
    for kind in ("table", "list"):
        if kinds.get(kind, 0) / total >= 0.6:
            return kind
    return "section"


@dataclass
class _Piece:
    """切块过程中的中间结果。合并后可能包含多个章节的内容。"""

    parts: List[Tuple[Tuple[str, ...], str]]  # (章节路径, 正文)
    kinds: Dict[str, int]

    @property
    def path(self) -> Tuple[str, ...]:
        first = self.parts[0][0]
        n = len(first)
        for p, _ in self.parts[1:]:
            n = min(n, _common_len(first, p))
        return first[:n]

    def body(self) -> str:
        """渲染正文；比公共路径更深的部分补上小标题，合并后也分得清。"""
        base = self.path
        out: List[str] = []
        prev = base
        for path, text in self.parts:
            for depth in range(max(len(base), _common_len(prev, path)), len(path)):
                out.append("#" * min(depth + 1, 6) + " " + path[depth])
            out.append(text)
            prev = path
        return "\n\n".join(out)

    def merged(self, other: "_Piece") -> "_Piece":
        return _Piece(self.parts + other.parts, _add_kinds(self.kinds, other.kinds))


class _Ctx:
    def __init__(self, params: ChunkParams, prefix: Optional[str] = None, isolate_tables: bool = False):
        self.params = params
        self.prefix = prefix  # 面包屑前缀（产品名 / 文档标题），路径里没有时补上
        self.isolate_tables = isolate_tables
        self.overlap_limit = min(OVERLAP_MAX_CHARS, params.max_chars // 5) if params.overlap else 0

    def header(self, path: Sequence[str]) -> str:
        crumbs = list(path)
        if self.prefix and (not crumbs or crumbs[0] != self.prefix):
            crumbs.insert(0, self.prefix)
        return " > ".join(crumbs)

    def compose(self, path: Sequence[str], body: str) -> str:
        header = self.header(path)
        return f"{header}\n\n{body}" if header else body

    def length(self, piece: _Piece) -> int:
        return len(self.compose(piece.path, piece.body()))

    def pack_budget(self, path: Sequence[str]) -> int:
        """正文能用的字符数：扣掉面包屑，需要重叠时再预留重叠的位置。"""
        header = self.header(path)
        budget = self.params.max_chars - (len(header) + 2 if header else 0)
        if self.overlap_limit:
            budget -= self.overlap_limit + 2
        return max(budget, self.params.max_chars // 2)


def _pack_blocks(blocks: Sequence[Block], path: Sequence[str], ctx: _Ctx) -> List[_Piece]:
    budget = ctx.pack_budget(path)
    key = tuple(path)
    pieces: List[_Piece] = []
    buf: List[str] = []
    kinds: Dict[str, int] = {}
    size = 0

    def flush() -> None:
        nonlocal buf, kinds, size
        if buf:
            pieces.append(_Piece([(key, "\n\n".join(buf))], kinds))
        buf, kinds, size = [], {}, 0

    for block in blocks:
        for kind, frag in _split_block(block, budget):
            isolated = ctx.isolate_tables and kind == "table"
            # 表格单独成块，但前面一句很短的引导语（如「参数如下：」）跟着表格走
            if buf and (size + 2 + len(frag) > budget or (isolated and size >= ctx.params.min_chars)):
                flush()
            size += len(frag) + (2 if buf else 0)
            buf.append(frag)
            kinds[kind] = kinds.get(kind, 0) + len(frag)
            if isolated:
                flush()
    flush()
    return pieces


def _chunk_section(section: Section, ctx: _Ctx) -> List[_Piece]:
    """整个章节放得下就整块保留（子章节也一起）；放不下再往下一层切。

    文档根和 H1 下面有子章节时总是往下切：H2 才是一个话题的自然单位，
    否则短文档会整篇变成一个 chunk，检索时不够精确。
    """
    body = render_body(section)
    if not body.strip():
        return []
    keep_whole = (section.level >= 2 or not section.children) and \
        len(ctx.compose(section.path, body)) <= ctx.params.max_chars
    if keep_whole and ctx.isolate_tables:
        kinds = {b.kind for b in section.iter_blocks()}
        keep_whole = "table" not in kinds or kinds == {"table"}
    if keep_whole:
        return [_Piece([(tuple(section.path), body)], _kind_counts(section.iter_blocks()))]
    pieces = _pack_blocks(section.blocks, section.path, ctx)
    for child in section.children:
        pieces.extend(_chunk_section(child, ctx))
    return pieces


def _can_merge(a: _Piece, b: _Piece, ctx: _Ctx) -> bool:
    """相邻的 a、b 至少有一个是碎片时，判断能不能并：

    - 同一章节拆出来的前后两段：可以；
    - 父章节的引导文字 + 紧跟的子章节：可以；
    - 兄弟章节：两个都是碎片才并（如「使命 / 愿景 / 价值观」），
      一个完整的小章节不会被并进不相关的大章节。
    """
    if ctx.isolate_tables and "table" in (_chunk_type(a.kinds), _chunk_type(b.kinds)):
        return False
    min_chars = ctx.params.min_chars
    small_a, small_b = ctx.length(a) < min_chars, ctx.length(b) < min_chars
    if not (small_a or small_b):
        return False
    pa, pb = a.path, b.path
    if pa == pb or pb[:len(pa)] == pa:
        return True
    is_sibling = len(pa) == len(pb) and _common_len(pa, pb) == len(pa) - 1
    return is_sibling and small_a and small_b


def _merge_small(pieces: List[_Piece], ctx: _Ctx) -> List[_Piece]:
    if ctx.params.min_chars <= 0:
        return pieces
    out: List[_Piece] = []
    for piece in pieces:
        if out and _can_merge(out[-1], piece, ctx):
            merged = out[-1].merged(piece)
            if ctx.length(merged) <= ctx.params.max_chars:
                out[-1] = merged
                continue
        out.append(piece)
    return out


def _tail_sentences(body: str, count: int, limit: int) -> str:
    """取正文最后 count 句作为下一个 chunk 的重叠；结尾是表格、代码或标题时不重叠。"""
    last = body.rsplit("\n\n", 1)[-1].strip()
    if not last or last.startswith(("|", "```", "~~~", "#")):
        return ""
    sentences = [s for s in split_sentences(last) if s.strip()]
    for k in range(min(count, len(sentences)), 0, -1):
        tail = "".join(sentences[-k:]).strip()
        if len(tail) <= limit:
            return tail
    return ""


def _finalize(pieces: List[_Piece], ctx: _Ctx, chunk_type: Optional[str] = None,
              metadata: Optional[Dict[str, object]] = None) -> List[Chunk]:
    chunks: List[Chunk] = []
    prev_path: Optional[Tuple[str, ...]] = None
    prev_body = ""
    for piece in pieces:
        path, body = piece.path, piece.body()
        meta = dict(metadata or {})
        text = body
        if ctx.overlap_limit and path == prev_path:
            tail = _tail_sentences(prev_body, ctx.params.overlap, ctx.overlap_limit)
            candidate = f"{tail}\n\n{body}"
            if tail and len(ctx.compose(path, candidate)) <= ctx.params.max_chars:
                text = candidate
                meta["overlap_chars"] = len(tail)
        prev_path, prev_body = path, body
        chunks.append(Chunk(ctx.compose(path, text), chunk_type or _chunk_type(piece.kinds), list(path), meta))
    return chunks


def _chunk_tree(section: Section, ctx: _Ctx, metadata: Optional[Dict[str, object]] = None) -> List[Chunk]:
    return _finalize(_merge_small(_chunk_section(section, ctx), ctx), ctx, metadata=metadata)


def chunk_by_structure(doc: ParsedDoc, params: ChunkParams) -> List[Chunk]:
    """brand / article：按标题层级切。两者的区别在参数（article 有句子重叠、chunk 更长）。"""
    return _chunk_tree(doc.root, _Ctx(params, prefix=doc.front_matter.get("title")))


def chunk_product(doc: ParsedDoc, params: ChunkParams) -> List[Chunk]:
    """每个 H1 当作一个产品单独切，产品之间绝不合并；产品名写进每个 chunk。"""
    root = doc.root
    fm_product = doc.front_matter.get("product")
    h1s = [c for c in root.children if c.level == 1]
    default = fm_product or (h1s[0].title if len(h1s) == 1 else doc.front_matter.get("title"))
    groups = [(default, Section("", 0, [], list(root.blocks), [c for c in root.children if c.level != 1]))]
    groups.extend((fm_product or h1.title, h1) for h1 in h1s)

    chunks: List[Chunk] = []
    for product, section in groups:
        ctx = _Ctx(params, prefix=product, isolate_tables=True)
        chunks.extend(_chunk_tree(section, ctx, {"product": product} if product else None))
    return chunks


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------


def _marker_re(words: str) -> re.Pattern:
    # 例：Q: / Q1： / Q1. / 问： / 问题2： / **Q:** / - Q：
    return re.compile(
        r"^\s*(?:[-*+]\s+)?(?:\*\*|__)?\s*" + words
        + r"\s*(?:\d+\s*[:：.、]|[:：])\s*(?:\*\*|__)?\s*",
        re.I,
    )


_Q_RE = _marker_re(r"(?:Q|问题|问|Question)")
_A_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?:\*\*|__)?\s*(?:A|答案|答复|回答|答|Answer)\s*\d*\s*[:：]\s*(?:\*\*|__)?\s*", re.I)
_BOLD_LINE_RE = re.compile(r"^\s*(\*\*|__)(.+?)\1\s*$")
_NUMBER_PREFIX_RE = re.compile(r"^\s*\d{1,3}\s*[.、)）](?!\d)\s*")
_QUESTION_END_RE = re.compile(r"[？?]$")
_ANSWER_LEAD_RE = re.compile(r"^(?:[-*+]\s|\d+[.)]\s|\||```|~~~|>)")


def _clean_question(text: str) -> str:
    text = text.strip().strip("*_").strip()
    m = _Q_RE.match(text)
    if m:
        text = text[m.end():].strip().strip("*_").strip()
    return _NUMBER_PREFIX_RE.sub("", text, count=1).strip()


def _is_question_heading(title: str) -> bool:
    return bool(_Q_RE.match(title) or _QUESTION_END_RE.search(_clean_question(title)))


@dataclass
class _QAPair:
    question: str
    category: Optional[str]
    path: List[str]
    answer: List[Block] = field(default_factory=list)


@dataclass
class _Residue:
    """qa 文档里不属于任何问答对的内容。"""

    path: List[str]
    blocks: List[Block]


_QAItem = Union[_QAPair, _Residue]


class _QAScanner:
    """在一个章节自己的块里找「标记式」（Q:/A:）和「加粗式」（**问题？**）问答对。"""

    def __init__(self, path: List[str], category: Optional[str], items: List[_QAItem]):
        self.path, self.category, self.items = list(path), category, items
        self.pair: Optional[_QAPair] = None
        self.residue: List[Block] = []
        self.lines: List[str] = []  # 行模式下还没提交的回答（或残余）行
        self.in_question = False  # 行模式：后续行是否还属于问题（多行问题）

    def feed(self, block: Block) -> None:
        lines = block.text.split("\n")
        if block.kind in ("paragraph", "list") and any(_Q_RE.match(l) or _A_RE.match(l) for l in lines):
            for line in lines:
                self._feed_line(line)
            self._commit_lines()
            return

        bold = _BOLD_LINE_RE.match(lines[0]) if block.kind == "paragraph" else None
        if bold and _QUESTION_END_RE.search(_clean_question(bold.group(2))):
            self._start(_clean_question(bold.group(2)))
            rest = "\n".join(lines[1:]).strip()
            if rest:
                self.pair.answer.append(Block("paragraph", rest))
            return

        if self.pair is not None:
            self.in_question = False
            self.pair.answer.append(block)
        else:
            self.residue.append(block)

    def finish(self) -> None:
        self._commit_lines()
        self._close()

    def _feed_line(self, line: str) -> None:
        mq = _Q_RE.match(line)
        if mq:
            self._start(_clean_question(line[mq.end():]))
            self.in_question = not _QUESTION_END_RE.search(self.pair.question)
            return
        ma = _A_RE.match(line)
        if ma:
            if self.pair is None:
                logger.warning("发现没有对应问题的回答，按普通文本处理：%s", line.strip()[:40])
                self.lines.append(line)
                return
            self.in_question = False
            rest = line[ma.end():].strip()
            if rest:
                self.lines.append(rest)
            return
        if self.pair is not None and self.in_question:
            text = line.strip()
            q = self.pair.question
            sep = " " if q[-1:].isascii() and q[-1:].isalnum() and text[:1].isascii() else ""
            self.pair.question = (q + sep + text).strip()
            self.in_question = not _QUESTION_END_RE.search(self.pair.question)
            return
        self.lines.append(line)

    def _commit_lines(self) -> None:
        text = "\n".join(self.lines).strip()
        self.lines = []
        if text:
            target = self.pair.answer if self.pair is not None else self.residue
            target.append(Block("paragraph", text))

    def _start(self, question: str) -> None:
        self._commit_lines()
        self._close()
        self.pair = _QAPair(question, self.category, self.path)
        self.in_question = False

    def _close(self) -> None:
        if self.pair is not None:
            self.items.append(self.pair)
            self.pair = None
        if self.residue:
            self.items.append(_Residue(self.path, self.residue))
            self.residue = []


def _walk_qa(section: Section, category: Optional[str], items: List[_QAItem]) -> None:
    if section.level and _is_question_heading(section.title):
        # 标题即问题：本章节正文 + 非问题的子章节都是答案
        pair = _QAPair(_clean_question(section.title), category, section.path[:-1], list(section.blocks))
        sub_questions = []
        for child in section.children:
            if _is_question_heading(child.title):
                sub_questions.append(child)
            else:
                child_body = render_body(child)
                heading = "#" * child.level + " " + child.title
                pair.answer.append(Block("paragraph", heading + ("\n\n" + child_body if child_body else "")))
        items.append(pair)
        for child in sub_questions:
            _walk_qa(child, category, items)
        return

    category = section.title or category
    scanner = _QAScanner(section.path, category, items)
    for block in section.blocks:
        scanner.feed(block)
    scanner.finish()
    for child in section.children:
        _walk_qa(child, category, items)


def _answer_lead(text: str) -> str:
    """答案以列表、表格等开头时换行，否则紧跟在「回答：」后面。"""
    return "\n" if _ANSWER_LEAD_RE.match(text) else ""


def _qa_chunks(pair: _QAPair, params: ChunkParams) -> List[Chunk]:
    blocks = [b for b in pair.answer if b.text.strip()]
    head = f"问题：{pair.question}\n回答："
    answer = "\n\n".join(b.text.strip() for b in blocks)
    meta: Dict[str, object] = {"question": pair.question}
    if pair.category:
        meta["category"] = pair.category
    if len(head) + 1 + len(answer) <= params.max_chars:
        return [Chunk(head + _answer_lead(answer) + answer, "qa", list(pair.path), meta)]

    # 答案太长：拆成多段，每段都带上问题
    budget = max(params.max_chars - len(head) - 1, params.max_chars // 2)
    frags = [text for b in blocks for _, text in _split_block(Block(b.kind, b.text.strip()), budget)]
    parts = _pack(frags, budget, "\n\n")
    return [
        Chunk(head + _answer_lead(part) + part, "qa", list(pair.path), {**meta, "part": f"{i}/{len(parts)}"})
        for i, part in enumerate(parts, 1)
    ]


def chunk_qa(doc: ParsedDoc, params: ChunkParams) -> List[Chunk]:
    items: List[_QAItem] = []
    _walk_qa(doc.root, None, items)

    residue_ctx = _Ctx(ChunkParams(params.max_chars, 0, 0), prefix=doc.front_matter.get("title"))
    chunks: List[Chunk] = []
    seen = set()
    pairs = 0
    for item in items:
        if isinstance(item, _Residue):
            pieces = [p for p in _pack_blocks(item.blocks, item.path, residue_ctx)
                      if len(p.body().strip()) >= QA_RESIDUE_MIN_CHARS]
            chunks.extend(_finalize(pieces, residue_ctx, chunk_type="text"))
            continue
        if not item.question:
            logger.warning("有一个问题标记后面是空的，已跳过")
            continue
        if not any(b.text.strip() for b in item.answer):
            logger.warning("问题没有找到回答，已跳过：%s", item.question)
            continue
        if item.question in seen:
            logger.warning("问题重复出现：%s", item.question)
        seen.add(item.question)
        chunks.extend(_qa_chunks(item, params))
        pairs += 1

    if not pairs:
        raise ChunkingError(
            "没有识别到任何问答对。支持的写法：Q:/A:（问：/答：）标记、以问号结尾的标题、"
            "单独一行的 **加粗问题？**。如果这不是问答文档，请改用 --type article"
        )
    return chunks


CHUNKERS: Dict[str, Callable[[ParsedDoc, ChunkParams], List[Chunk]]] = {
    "qa": chunk_qa,
    "product": chunk_product,
    "brand": chunk_by_structure,
    "article": chunk_by_structure,
}
