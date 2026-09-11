#!/usr/bin/env python3
"""Rebuild presentation from pinned validated results; do not rerun experiments."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from vis.build import build

if __name__=="__main__":build()
