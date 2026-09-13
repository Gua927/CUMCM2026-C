#!/usr/bin/env python3
"""Prepare print tables, preserving cells and allowing date-group page breaks."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from cumcm_c.paths import PATHS


def write_if_changed(target: Path, content: str):
    if not target.exists() or target.read_text(encoding='utf-8') != content:
        target.write_text(content, encoding='utf-8')


def render_date_table(source: str) -> str:
    """Turn a four-date float into a longtable; keep each day's rows together."""
    caption = re.search(r'\\caption\{([^\n]+)\}', source).group(1)
    label = re.search(r'\\label\{([^}]+)\}', source).group(1)
    columns = re.search(r'\\begin\{tabular\}\{([^}]+)\}', source).group(1)
    contents = source.split(r'\toprule', 1)[1].split(r'\bottomrule', 1)[0]
    header, body = contents.split(r'\midrule', 1)
    groups = [group.strip() for group in body.split(r'\midrule')]
    if len(groups) != 4 or not all(re.match(r'\\multicolumn\{6\}\{c\}\{2025-', g) for g in groups):
        raise ValueError(f'Unexpected date grouping in {label}')
    protected = []
    for group in groups:
        rows = group.splitlines()
        if not all(row.endswith(r'\\') for row in rows):
            raise ValueError(f'Unexpected table row in {label}')
        protected.append('\n'.join(row + '*' for row in rows[:-1]) + '\n' + rows[-1])
    formatted_body = ('\n\\addlinespace[3pt]\n').join(protected)
    # Only page-break controls and rule spacing may change in the data body.
    normalized = formatted_body.replace(r'\\*', r'\\').replace(r'\addlinespace[3pt]', r'\midrule')
    if re.sub(r'\s+', '', normalized) != re.sub(r'\s+', '', body):
        raise ValueError(f'Table formatting changed data cells in {label}')
    return (
        '\\begingroup\\small\n'
        '\\renewcommand{\\arraystretch}{1.15}\n'
        '\\setlength{\\LTpre}{8pt}\n\\setlength{\\LTpost}{8pt}\n'
        f'\\begin{{longtable}}{{{columns}}}\n'
        f'\\caption{{{caption}}}\\label{{{label}}}\\\\\n'
        '\\toprule\n' + header.strip() + '\n\\midrule\\endfirsthead\n'
        '\\multicolumn{6}{c}{表~\\thetable（续）}\\\\\n'
        '\\toprule\n' + header.strip() + '\n\\midrule\\endhead\n'
        '\\midrule\\multicolumn{6}{r}{续下页}\\\\\n\\endfoot\n'
        '\\bottomrule\\endlastfoot\n'
        + formatted_body + '\n\\end{longtable}\n\\endgroup\n'
    )


def polish_table_words(source: str) -> str:
    """Use the prose's plain-language labels, preserving every numeric data row."""
    replacements = {
        '主策略：三时点调整（新信息链）': '主策略：三次调整',
        '主策略：三时点调整（两项改进均开启）': '主策略：三次调整',
        '冻结旧版信息链': '两项改进均关闭',
        '冻结旧版同参': '改进前同参数',
        '新信息链': '两项改进均开启',
        '仅同构残差校准': '仅重算历史误差',
        '仅同构残差': '仅重算历史误差',
        '同构残差': '误差重算',
        '贪心缓冲器': '贪心充放电',
        '风险轮廓': '风险需求曲线',
        '代理规划': '近似规划',
        '影子检验': '独立模拟比较',
        '信息链修复': '预报利用与误差重算',
        '信息链对照': '预报与误差对照',
        '均采用两项改进均开启': '均开启两项改进',
        '四组同参对照': '四组同参数对照',
    }
    result = source
    for old, new in replacements.items():
        result = result.replace(old, new)
    def numeric_rows(text):
        return [re.findall(r'-?\d+(?:\.\d+)?', row) for row in text.splitlines() if '&' in row]
    if numeric_rows(source) != numeric_rows(result):
        raise ValueError('Terminology edit changed table numbers')
    return result


def render():
    folder = PATHS.output_dir / 'tables'
    for table in folder.glob('*.tex'):
        if table.stem.endswith('_layout'):
            continue
        source = table.read_text(encoding='utf-8')
        write_if_changed(table, polish_table_words(source))
    # Old exports embedded a terminal state from a superseded battery setup.
    # The current terminal state is given by the generated macro in the prose.
    chain = folder / 'q3_information_chain.tex'
    source = chain.read_text(encoding='utf-8')
    corrected = re.sub(r'；期末电量均为\d+(?:\.\d+)? kWh', '', source)
    write_if_changed(chain, corrected)
    for question in ('q2', 'q3', 'q42', 'q43'):
        for kind in ('purchase', 'storage'):
            name = f'{question}_{kind}'
            source = (folder / f'{name}.tex').read_text(encoding='utf-8')
            write_if_changed(folder / f'{name}_layout.tex', render_date_table(source))
    source = (folder / 'q3_horizon.tex').read_text(encoding='utf-8')
    # This short appendix table should stay with the discussion, not occupy
    # a vertically centered float page before the next appendix section.
    formatted = source.replace(r'\begin{table}[htbp]', r'\begin{table}[H]', 1)
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
    caption = r'\caption{改进前后需求曲线的覆盖率（\%）与光伏MAE（kW）。分别统计当日剩余、次日预报覆盖内外时段；当日下、上界对应70\%、90\%分位数，次日仅取80\%分位数，列于下界列。}'
    formatted = formatted[:start] + caption + formatted[end:]
    def values(text):
        return [re.findall(r'\d+(?:\.\d+)?', line) for line in text.splitlines()
                if line.startswith(('6:00', '12:00', '18:00'))]
    if values(source) != values(formatted):
        raise ValueError('Table formatting changed numerical cells')
    target = folder / 'q3_horizon_layout.tex'
    write_if_changed(target, formatted)


if __name__ == '__main__':
    render()
