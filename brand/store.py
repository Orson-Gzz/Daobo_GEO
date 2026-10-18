"""品牌知识库的 JSON 存储。

每个品牌一个目录：
    documents.json  文档注册表 {doc_id: 文档信息}
    chunks.json     chunk 列表 [chunk, ...]
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_UNSAFE_CHARS_RE = re.compile(r'[\\/:*?"<>|\s]+')


class StoreError(RuntimeError):
    pass


def brand_dirname(brand: str) -> str:
    """品牌名转成目录名：去掉路径分隔符等不安全字符，中文保留。"""
    name = _UNSAFE_CHARS_RE.sub("_", brand.strip()).strip("._")
    if not name:
        raise StoreError(f"品牌名不能用作目录名：{brand!r}")
    return name


class KnowledgeStore:
    def __init__(self, kb_root: os.PathLike, brand: str):
        self.brand = brand
        self.root = Path(kb_root) / brand_dirname(brand)
        self.documents_file = self.root / "documents.json"
        self.chunks_file = self.root / "chunks.json"
        self.documents: Dict[str, Dict[str, Any]] = _load_json(self.documents_file, {})
        self.chunks: List[Dict[str, Any]] = _load_json(self.chunks_file, [])

    def is_unchanged(self, doc_info: Dict[str, Any]) -> bool:
        """内容、类型、切块参数都和库里一样，就没必要重建。"""
        old = self.documents.get(doc_info["doc_id"])
        return bool(old) and all(old.get(k) == doc_info.get(k) for k in ("content_hash", "doc_type", "chunk_params"))

    def upsert(self, doc_info: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Optional[int]:
        """写入一个文档。新文档返回 None；已存在则整体替换，返回被替换掉的旧 chunk 数。"""
        doc_id = doc_info["doc_id"]
        old = self.documents.get(doc_id)
        replaced = None
        if old is not None:
            doc_info = {**doc_info, "created_at": old.get("created_at", doc_info.get("created_at"))}
            before = len(self.chunks)
            self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
            replaced = before - len(self.chunks)
        self.documents[doc_id] = doc_info
        self.chunks.extend(chunks)
        return replaced

    def remove(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        return True

    def stats(self) -> Dict[str, Any]:
        doc_types: Dict[str, int] = {}
        for doc in self.documents.values():
            doc_types[doc["doc_type"]] = doc_types.get(doc["doc_type"], 0) + 1
        chunk_types: Dict[str, int] = {}
        for chunk in self.chunks:
            chunk_types[chunk["chunk_type"]] = chunk_types.get(chunk["chunk_type"], 0) + 1
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "document_types": doc_types,
            "chunk_types": chunk_types,
        }

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.chunks_file, self.chunks)
        _atomic_write_json(self.documents_file, self.documents)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        # 不能当成空库继续：下一次保存会把原有数据全部覆盖掉
        raise StoreError(f"{path} 不是合法的 JSON（{e}），请先修复或备份后删除") from e


def _atomic_write_json(path: Path, data: Any) -> None:
    """先写临时文件再替换，写到一半崩溃也不会留下损坏的 JSON。"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
