#!/usr/bin/env python3
"""Run and publish the causally validated Q2 experiment."""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.problem2_workflow import run

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reuse-calibration",type=Path,default=None)
    args=parser.parse_args()
    run(args.reuse_calibration)
