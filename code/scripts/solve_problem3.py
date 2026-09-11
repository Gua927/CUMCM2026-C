#!/usr/bin/env python3
"""Run and publish the causally validated Q3 experiment."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.problem3_workflow import run

if __name__=="__main__":run()
