#!/usr/bin/env python3
"""Run and publish the causally validated Q4 experiment (real-time prices)."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.problem4_workflow import run

if __name__=="__main__":run()
