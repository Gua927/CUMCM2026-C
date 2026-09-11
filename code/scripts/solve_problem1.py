#!/usr/bin/env python3
"""Solve and publish an independently validated, versioned Q1 bundle."""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
with warnings.catch_warnings(record=True) as import_warnings:
    warnings.simplefilter("default")
    from cumcm_c.problem1_workflow import run
for item in import_warnings:
    warnings.showwarning(item.message,item.category,item.filename,item.lineno)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-convention",choices=["start","end"],default=None)
    args=parser.parse_args()
    folder,summary=run(args.time_convention,environment_warnings=[str(w.message) for w in import_warnings])
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(f"Validated run: {folder}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
