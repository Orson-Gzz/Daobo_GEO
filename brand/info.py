"""把品牌信息写成 data/kb/<品牌名>/info.json。"""
from __future__ import annotations

import json
from pathlib import Path


def write(brandname, target_language="Chinese", brand_domain=None, brand_type="brand"):
    brand_info = {
        "brand_name": brandname,
        "target_language": target_language,
        "brand_domain": brand_domain,
        "brand_type": brand_type,
    }
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "kb" / brandname / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(brand_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
