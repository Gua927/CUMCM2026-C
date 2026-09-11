#!/usr/bin/env python3
"""把官方空白结果模板安全复制到 output/results/。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from cumcm_c.export import RESULT_FILES, prepare_result_workbook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", metavar="RESULT_FILE")
    parser.add_argument("--all", action="store_true", help="准备全部五个结果模板")
    parser.add_argument("--overwrite", action="store_true", help="显式覆盖已有输出")
    args = parser.parse_args()
    names = RESULT_FILES if args.all else tuple(args.names)
    if not names:
        parser.error("请给出结果文件名，或使用 --all")
    for name in names:
        if name not in RESULT_FILES:
            parser.error(f"未知结果文件：{name}")
        target = prepare_result_workbook(name, overwrite=args.overwrite)
        print(f"Prepared: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
