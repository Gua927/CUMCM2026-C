#!/usr/bin/env python3
"""Validate saved Q2 results independently, without reoptimizing."""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.paths import PATHS
from cumcm_c.models.common import Battery
from cumcm_c.models.problem2_evaluation import audit
from cumcm_c.problem2_export import validate_workbook
from cumcm_c.problem1_workflow import sha256,save_json


def main():
    latest=json.loads((PATHS.output_dir/"runs/problem2/latest.json").read_text())
    folder=ROOT/latest["path"];manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":raise ValueError("Run incomplete")
    for name,expected in manifest["artifacts"].items():
        if sha256(folder/name)!=expected:raise ValueError(f"Artifact changed: {name}")
    config=json.loads((folder/"config.json").read_text());b=Battery(**config["battery"])
    results={}
    for source in (folder/"strategies").glob("*.csv"):
        if source.stem.endswith("_daily") or source.stem.startswith("scenarios_"):continue
        f=pd.read_csv(source)
        if len(f)!=334*144:raise ValueError("Incomplete official horizon")
        results[source.stem]=audit(f,b,config["time_step_minutes"]/60,config["pricing"]["emergency_multiplier"],initial=b.initial_soc_kwh)
    frame=pd.read_csv(folder/"strategies/risk_policy.csv")
    results["workbook"]=validate_workbook(folder/"result2.xlsx",frame)
    if sha256(PATHS.result_dir/"result2.xlsx")!=sha256(folder/"result2.xlsx"):raise ValueError("Published workbook differs")
    save_json(PATHS.output_dir/"logs/problem2-validation.json",results)
    print(f"PASS: Q2 immutable hashes, 5 causal trajectories, physical/cost audits, official workbook ({folder.name})")


if __name__=="__main__":main()
