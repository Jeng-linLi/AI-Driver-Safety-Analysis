"""
共用工具（Week 4 / 5 都需要）
============================
- resolve_data_yaml：把 dataset.yaml 的相對 path 解析成絕對路徑，
  寫成臨時 yaml 再回傳路徑。原因見下方 docstring（ultralytics 相容性）。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml


def resolve_data_yaml(path: str) -> str:
    """讀取 dataset.yaml，把相對 path 解析成絕對路徑，寫成「臨時 yaml」並回傳其路徑。

    為什麼要這樣做：
    1) ultralytics 預設會把 dataset.yaml 裡的相對 path 當成 settings 中
       datasets_dir 下的子路徑，導致找不到我們放在 data/ 下的資料集。
       → 所以這裡改成「相對於 dataset.yaml 所在目錄」解析成絕對路徑。
    2) ultralytics 的 model.train(data=...) / model.val(data=...) 只接受
       「yaml 檔路徑字串」，不接受 dict。→ 所以寫成一個臨時 yaml 再傳路徑。
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str((p.parent / cfg["path"]).resolve())
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name
