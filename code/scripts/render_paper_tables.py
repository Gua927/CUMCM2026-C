#!/usr/bin/env python3
"""Reformat wide table headers for print without changing numerical cells."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from cumcm_c.paths import PATHS


def render():
    folder = PATHS.output_dir / 'tables'
    source = (folder / 'q3_horizon.tex').read_text(encoding='utf-8')
    formatted = source
    for old, new in {
        '次日：版本覆盖范围内': '次日覆盖内',
        '次日：覆盖范围外': '次日覆盖外',
        '新：0.7/0.8覆盖': '新：下界覆盖',
        '新：0.9覆盖': '新：上界覆盖',
        '旧：0.7/0.8覆盖': '旧：下界覆盖',
    }.items():
        formatted = formatted.replace(old, new)
    start = formatted.index(r'\caption{')
    end = formatted.index('\n\\label', start)
    caption = r'\caption{新、旧信息链的风险覆盖率（\%）与光伏MAE（kW）。按发布时刻及当日剩余、次日版本覆盖内外分段；当日下、上界名义覆盖率为70\%、90\%，次日仅用80\%轮廓，列于下界列。}'
    formatted = formatted[:start] + caption + formatted[end:]
    def values(text):
        return [re.findall(r'\d+(?:\.\d+)?', line) for line in text.splitlines()
                if line.startswith(('6:00', '12:00', '18:00'))]
    if values(source) != values(formatted):
        raise ValueError('Table formatting changed numerical cells')
    target = folder / 'q3_horizon_layout.tex'
    if not target.exists() or target.read_text(encoding='utf-8') != formatted:
        target.write_text(formatted, encoding='utf-8')


if __name__ == '__main__':
    render()
