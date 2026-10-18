"""品牌知识库构建（GEO 全流程第一步）。

把品牌方提交的 Markdown 资料按文档类型切块，写入 data/kb/<品牌>/ 下的
documents.json 和 chunks.json。文档类型必须手动指定。

用法（在项目根目录执行）：
    python3 -m brand.build_kb add raw_doc/faq.md --type qa --brand 青禾
    python3 -m brand.build_kb add raw_doc/products/ --type product --brand 青禾 --dry-run
    python3 -m brand.build_kb list --brand 青禾
    python3 -m brand.build_kb remove <doc_id> --brand 青禾
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import info
from .chunkers import DOC_TYPES, TYPE_CONFIG, ChunkingError, ChunkParams, chunk_document
from .md_parser import parse_markdown
from .store import KnowledgeStore, StoreError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_DIR = PROJECT_ROOT / "data" / "kb"
MD_SUFFIXES = (".md", ".markdown")

class CliError(Exception):
    pass


def make_doc_id(filename: str) -> str:
    """同一品牌下按文件名确定 doc_id，重复导入同名文件会覆盖而不是新增。"""
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]


def build_document(text: str, filename: str, doc_type: str, brand: str, params: ChunkParams,
                   metadata: Optional[Dict[str, str]] = None,
                   source_path: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """解析并切块，返回 (文档信息, chunk 列表)，不落盘。"""
    doc = parse_markdown(text)
    doc_id = make_doc_id(filename)

    records: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunk_document(doc, doc_type, params)):
        records.append({
            "chunk_id": f"{doc_id}-{i:04d}",
            "doc_id": doc_id,
            "doc_type": doc_type,
            "chunk_index": i,
            "chunk_type": chunk.chunk_type,
            "content": chunk.content,
            "char_count": len(chunk.content),
            "heading_path": chunk.heading_path,
            "metadata": {"filename": filename, "brand": brand, **chunk.metadata},
        })

    products: List[str] = []
    for r in records:
        product = r["metadata"].get("product")
        if product and product not in products:
            products.append(product)

    now = datetime.now().isoformat(timespec="seconds")
    doc_info = {
        "doc_id": doc_id,
        "filename": filename,
        "source_path": source_path,
        "doc_type": doc_type,
        "title": doc.front_matter.get("title") or doc.first_title() or Path(filename).stem,
        "brand": brand,
        "products": products,
        "content_hash": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "content_length": len(text),
        "chunk_count": len(records),
        "chunk_params": dataclasses.asdict(params),
        "created_at": now,
        "updated_at": now,
        "metadata": {**doc.front_matter, **(metadata or {})},
    }
    return doc_info, records


def resolve_params(doc_type: str, max_chars: Optional[int], min_chars: Optional[int],
                   overlap: Optional[int]) -> ChunkParams:
    base = TYPE_CONFIG[doc_type]
    params = ChunkParams(
        max_chars=base.max_chars if max_chars is None else max_chars,
        min_chars=base.min_chars if min_chars is None else min_chars,
        overlap=base.overlap if overlap is None else overlap,
    )
    if params.max_chars < 100:
        raise CliError("--max-chars 不能小于 100")
    if not 0 <= params.min_chars < params.max_chars:
        raise CliError("--min-chars 必须 >= 0 且小于 --max-chars")
    if params.overlap < 0:
        raise CliError("--overlap 不能是负数")
    return params


def collect_files(paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in MD_SUFFIXES)
            if not found:
                raise CliError(f"目录里没有 Markdown 文件：{p}")
            files.extend(found)
        elif p.is_file():
            if p.suffix.lower() not in MD_SUFFIXES:
                raise CliError(f"只支持 Markdown 文件（.md / .markdown）：{p}")
            files.append(p)
        else:
            raise CliError(f"找不到文件：{p}")

    names: Dict[str, Path] = {}
    for f in files:
        if f.name in names and names[f.name].resolve() != f.resolve():
            raise CliError(f"同一批里有同名文件，会互相覆盖：{names[f.name]} 和 {f}")
        names[f.name] = f
    return list(dict.fromkeys(f.resolve() for f in files))


def parse_meta(items: Optional[Sequence[str]]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise CliError(f"--meta 的格式应该是 key=value：{item}")
        meta[key.strip()] = value.strip()
    return meta


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _summary(chunks: List[Dict[str, Any]]) -> str:
    types: Dict[str, int] = {}
    for c in chunks:
        types[c["chunk_type"]] = types.get(c["chunk_type"], 0) + 1
    sizes = [c["char_count"] for c in chunks]
    type_str = "，".join(f"{k}×{v}" for k, v in types.items())
    return f"{type_str}；字数 最小 {min(sizes)} / 平均 {sum(sizes) // len(sizes)} / 最大 {max(sizes)}"


def _print_preview(filename: str, doc_type: str, chunks: List[Dict[str, Any]]) -> None:
    print(f"── {filename}（{doc_type}）共 {len(chunks)} 个 chunk：{_summary(chunks)}")
    for c in chunks:
        crumbs = " > ".join(c["heading_path"]) or "-"
        text = c["content"]
        if c["chunk_type"] != "qa" and "\n\n" in text:
            text = text.split("\n\n", 1)[1]  # 去掉面包屑，前面已经单独显示了
        snippet = " ".join(text.split())[:80]
        print(f"  #{c['chunk_index']:04d} {c['chunk_type']:<7} {c['char_count']:>4}字  [{crumbs}]  {snippet}")


def cmd_add(args: argparse.Namespace) -> int:
    params = resolve_params(args.doc_type, args.max_chars, args.min_chars, args.overlap)
    meta = parse_meta(args.meta)
    files = collect_files(args.paths)
    store = KnowledgeStore(args.kb_dir, args.brand)

    ok = True
    written = 0
    for path in files:
        name = path.name
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            print(f"[失败] {name}：不是 UTF-8 编码，请先转成 UTF-8")
            ok = False
            continue
        try:
            doc_info, chunks = build_document(text, name, args.doc_type, args.brand, params, meta,
                                              source_path=_display_path(path))
        except ChunkingError as e:
            print(f"[失败] {name}：{e}")
            ok = False
            continue
        if not chunks:
            print(f"[失败] {name}：没有切出任何内容（文件是空的？）")
            ok = False
            continue

        if args.dry_run:
            _print_preview(name, args.doc_type, chunks)
            continue
        if not args.force and store.is_unchanged(doc_info):
            print(f"[跳过] {name}：内容和切块参数都没变（加 --force 强制重建）")
            continue

        replaced = store.upsert(doc_info, chunks)
        if replaced is None:
            print(f"[新增] {name} → doc_id={doc_info['doc_id']}，{len(chunks)} 个 chunk（{_summary(chunks)}）")
        else:
            print(f"[覆盖] {name} → doc_id={doc_info['doc_id']}，{len(chunks)} 个 chunk"
                  f"（替换了旧的 {replaced} 个；{_summary(chunks)}）")
        written += 1

    if written:
        store.save()
        stats = store.stats()
        print(f"已写入 {_display_path(store.root)}/：共 {stats['total_documents']} 个文档，"
              f"{stats['total_chunks']} 个 chunk")
    return 0 if ok else 1


def cmd_list(args: argparse.Namespace) -> int:
    store = KnowledgeStore(args.kb_dir, args.brand)
    if not store.documents:
        print(f"品牌「{args.brand}」的知识库是空的（{_display_path(store.root)}）")
        return 0
    for doc in sorted(store.documents.values(), key=lambda d: d["updated_at"]):
        print(f"{doc['doc_id']}  {doc['doc_type']:<8} {doc['chunk_count']:>4} 个 chunk  "
              f"{doc['updated_at']}  {doc['filename']}")
    stats = store.stats()
    print(f"共 {stats['total_documents']} 个文档，{stats['total_chunks']} 个 chunk；"
          f"文档类型 {stats['document_types']}；chunk 类型 {stats['chunk_types']}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    store = KnowledgeStore(args.kb_dir, args.brand)
    doc = store.documents.get(args.doc_id)
    if not store.remove(args.doc_id):
        print(f"没有找到 doc_id={args.doc_id}（用 list 查看已有文档）")
        return 1
    store.save()
    print(f"已删除 {doc['filename']}（doc_id={args.doc_id}，{doc['chunk_count']} 个 chunk）")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m brand.build_kb", description="品牌知识库构建：Markdown → JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--brand", required=True, help="品牌名，每个品牌单独一个知识库目录")
        p.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR), help="知识库根目录（默认：项目下的 data/kb）")
        p.add_argument("--target-language", default="Chinese", choices=("English", "Chinese"),
                       help="目标语言（默认 Chinese）")
        p.add_argument("--brand-domain", default=None, help="官网域名（默认空）")
        p.add_argument("--brand-type", default="brand", choices=("brand", "product"),
                       help="推广对象是品牌还是产品（默认 brand）")

    add = sub.add_parser("add", help="导入 Markdown 文档")
    add.add_argument("paths", nargs="+", help="Markdown 文件或目录（目录下所有 .md 按同一类型导入）")
    add.add_argument("--type", dest="doc_type", required=True, choices=DOC_TYPES, help="文档类型")
    add_common(add)
    add.add_argument("--meta", action="append", metavar="KEY=VALUE", help="附加到文档的元数据，可重复")
    add.add_argument("--max-chars", type=int, help="chunk 最大字符数（默认按类型）")
    add.add_argument("--min-chars", type=int, help="小于这个长度的碎片会并入相邻 chunk（默认按类型）")
    add.add_argument("--overlap", type=int, help="相邻 chunk 重叠的句子数（默认按类型）")
    add.add_argument("--force", action="store_true", help="内容没变也重建")
    add.add_argument("--dry-run", action="store_true", help="只预览切块结果，不写文件")
    add.set_defaults(func=cmd_add)

    ls = sub.add_parser("list", help="列出品牌知识库里的文档")
    add_common(ls)
    ls.set_defaults(func=cmd_list)

    rm = sub.add_parser("remove", help="删除一个文档及其 chunk")
    rm.add_argument("doc_id")
    add_common(rm)
    rm.set_defaults(func=cmd_remove)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="  [警告] %(message)s")
    args = build_parser().parse_args(argv)
    info.write(args.brand, args.target_language, args.brand_domain, args.brand_type)
    try:
        return args.func(args)
    except (CliError, StoreError) as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

