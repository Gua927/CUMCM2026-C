#!/usr/bin/env python3
"""Validate the latest archived Q1 solution without optimizing again."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code/src"))
from cumcm_c.models.common import Battery
from cumcm_c.models.evaluation import audit
from cumcm_c.export import validate_problem1_workbook


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path)
    args=parser.parse_args()
    folder=args.run or ROOT/json.loads((ROOT/"output/runs/problem1/latest.json").read_text())["path"]
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":raise ValueError("Run is not complete")
    for relative,expected in manifest["artifacts_sha256"].items():
        if hashlib.sha256((folder/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Archived artifact changed: {relative}")
    config=json.loads((folder/"config.json").read_text())
    schedule=pd.read_csv(folder/"optimized_schedule.csv")
    result=audit(schedule,Battery(**config["battery"]),config["time_step_minutes"]/60)
    result["workbook"]=validate_problem1_workbook(folder/"result1.xlsx",schedule)
    log=ROOT/"output/logs";log.mkdir(exist_ok=True)
    (log/"problem1-validation.json").write_text(json.dumps({"run":str(folder),**result},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"PASS: archived hashes, CSV physics, costs, and Excel template ({manifest['run_id']})")
    return 0


if __name__=="__main__":raise SystemExit(main())
