"""Markdown 解析：front matter、块（Block）和章节树（Section）。

只覆盖知识库切块用得到的结构（标题、段落、列表、表格、代码块、引用），
不追求完整兼容 CommonMark。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
# "## 标题" 是标准写法；"##标题" 这种漏了空格的也认，但 "#话题#" 这种井号话题不算标题
_ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.+?)(?:[ \t]+#+)?|([^#\s][^#]*?))[ \t]*$")
_SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HR_RE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_QUOTE_RE = re.compile(r"^ {0,3}>")
_TABLE_SEP_RE = re.compile(r"^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+(?:[ \t]*:?-+:?[ \t]*)?$")
_EMPHASIS_WRAP_RE = re.compile(r"^(\*\*|__)(.+)\1$")
LIST_ITEM_RE = re.compile(r"^([ \t]*)(?:[-*+]|\d{1,9}[.)])[ \t]+")


@dataclass
class Block:
    """一个不可再按结构拆分的内容块。"""

    kind: str  # paragraph | list | table | code | quote
    text: str


@dataclass
class Section:
    """标题划出来的章节。level 0 是文档根（第一个标题之前的内容）。"""

    title: str
    level: int
    path: List[str]  # 面包屑：从顶层标题到本章节标题
    blocks: List[Block] = field(default_factory=list)
    children: List["Section"] = field(default_factory=list)

    def iter_blocks(self) -> Iterator[Block]:
        """按文档顺序遍历本章节及所有子章节的块。"""
        yield from self.blocks
        for child in self.children:
            yield from child.iter_blocks()


@dataclass
class ParsedDoc:
    front_matter: Dict[str, str]
    root: Section

    def first_title(self) -> Optional[str]:
        """第一个 H1；没有 H1 时取第一个标题。"""
        for child in self.root.children:
            if child.level == 1:
                return child.title
        return self.root.children[0].title if self.root.children else None


def parse_markdown(text: str) -> ParsedDoc:
    text = normalize(text)
    front_matter, body = split_front_matter(text)
    return ParsedDoc(front_matter, _build_tree(_lex(body.split("\n"))))


def normalize(text: str) -> str:
    """统一换行符，去掉 BOM 和 HTML 注释，压缩多余空行。"""
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def split_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    """解析开头 --- 包住的 key: value。不像 YAML 的就当普通正文，原样返回。"""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta: Dict[str, str] = {}
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            return {}, text
        key, _, value = line.partition(":")
        if key.strip():
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[m.end():]


def render_body(section: Section) -> str:
    """把章节正文（含子章节及其标题）还原成 Markdown，不含本章节自己的标题。"""
    parts = [b.text for b in section.blocks]
    for child in section.children:
        child_body = render_body(child)
        heading = "#" * child.level + " " + child.title
        parts.append(heading + ("\n\n" + child_body if child_body else ""))
    return "\n\n".join(p for p in parts if p)


def _clean_heading(title: str) -> str:
    title = title.strip()
    m = _EMPHASIS_WRAP_RE.match(title)
    return m.group(2).strip() if m else title


def _lex(lines: List[str]) -> List[tuple]:
    """逐行扫描，产出 ("heading", level, title) 或 ("block", Block)。"""
    tokens: List[tuple] = []
    para: List[str] = []

    def flush_para() -> None:
        if para:
            tokens.append(("block", Block("paragraph", "\n".join(para).strip())))
            para.clear()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            flush_para()
            i += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            flush_para()
            marker = fence.group(1)
            closing = re.compile(r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*$")
            buf = [line]
            i += 1
            while i < n:
                buf.append(lines[i])
                i += 1
                if closing.match(buf[-1]):
                    break
            tokens.append(("block", Block("code", "\n".join(buf))))
            continue

        heading = _ATX_HEADING_RE.match(line)
        if heading:
            flush_para()
            title = _clean_heading(heading.group(2) or heading.group(3))
            tokens.append(("heading", len(heading.group(1)), title))
            i += 1
            continue

        if len(para) == 1 and _SETEXT_UNDERLINE_RE.match(line):
            level = 1 if line.strip()[0] == "=" else 2
            tokens.append(("heading", level, _clean_heading(para.pop())))
            i += 1
            continue

        if _HR_RE.match(line):
            flush_para()
            i += 1
            continue

        if "|" in line and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            flush_para()
            buf = [line, lines[i + 1]]
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                buf.append(lines[i])
                i += 1
            tokens.append(("block", Block("table", "\n".join(row.strip() for row in buf))))
            continue

        if _QUOTE_RE.match(line):
            flush_para()
            buf = []
            while i < n and _QUOTE_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            tokens.append(("block", Block("quote", "\n".join(buf))))
            continue

        if LIST_ITEM_RE.match(line):
            flush_para()
            buf = [line]
            i += 1
            while i < n:
                cur = lines[i]
                if not cur.strip():
                    # 空行之后如果还是列表项或缩进内容，算同一个列表（松散列表）
                    j = i
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n and (LIST_ITEM_RE.match(lines[j]) or lines[j].startswith(("  ", "\t"))):
                        buf.extend(lines[i:j])
                        i = j
                        continue
                    break
                if LIST_ITEM_RE.match(cur) or cur.startswith(("  ", "\t")):
                    buf.append(cur)
                    i += 1
                    continue
                if _ATX_HEADING_RE.match(cur) or _FENCE_RE.match(cur) or _HR_RE.match(cur) or _QUOTE_RE.match(cur):
                    break
                buf.append(cur)  # 紧跟在列表项后面、没有缩进的续行
                i += 1
            tokens.append(("block", Block("list", "\n".join(buf).rstrip())))
            continue

        para.append(line)
        i += 1

    flush_para()
    return tokens


def _build_tree(tokens: List[tuple]) -> Section:
    root = Section("", 0, [])
    stack = [root]
    for token in tokens:
        if token[0] == "heading":
            _, level, title = token
            while stack[-1].level >= level:
                stack.pop()
            parent = stack[-1]
            section = Section(title, level, parent.path + [title])
            parent.children.append(section)
            stack.append(section)
        else:
            stack[-1].blocks.append(token[1])
    return root
