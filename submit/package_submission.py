#!/usr/bin/env python3
"""组装 CUMCM 2026 提交文件：论文 PDF 与支撑材料压缩包。

只读取已有产物并复制到 output/submission/，不重新求解、不改动论文源码、
不改动 output/results 中的结果工作簿。最后打印两份提交文件的大小与 MD5。

用法：
    python3 submit/package_submission.py            # 组装并打印校验信息
    python3 submit/package_submission.py --md5-only # 只对已有提交文件重新算 MD5
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DEST = OUT / "submission"
PACK_NAME = "支撑材料"
PAPER_NAME = "论文.pdf"
MAX_MB = 20.0

# 归档中体积过大且可由 make solve-q* 重算的逐时轨迹，不放进 20 MB 的压缩包。
RUN_FILE_MAX_BYTES = 700_000
RUN_SKIP_PREFIX = ("previous_published", "template_mapping")
RUN_SKIP_NAMES = {"dashboard.html"}
RUN_SKIP_SUFFIX = (".xlsx",)  # 与包根目录的 result*.xlsx 完全相同，避免重复
RUN_KEEP_DIRS = {"tables", "sensitivity"}
# 论文最终未采用的 AI 生成示意图草稿，不随支撑材料提交。
FIGURE_SKIP_PREFIX = ("framework_imagegen",)
# --with-traces 时额外附带的主策略全年逐时轨迹（相对各问最新归档），以 gzip 存放。
TRACE_FILES = {
    "problem2": ("strategies/risk_policy.csv", "issued_plans.csv"),
    "problem3": ("strategies/adjust_all.csv",),
    "problem4": ("strategies/q43_forecast.csv",),
}
RUN_SKIP_DIRS = {"strategies", "boundary", "figures", "source", "code", "__pycache__"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def copy_tree(src: Path, dst: Path, *, patterns=None, ignore_dirs=(), skip_prefix=()) -> None:
    for item in sorted(src.rglob("*")):
        if any(part in ignore_dirs for part in item.relative_to(src).parts):
            continue
        if item.is_dir():
            continue
        if patterns and item.suffix.lower() not in patterns:
            continue
        if skip_prefix and item.name.startswith(skip_prefix):
            continue
        target = dst / item.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def copy_run(run_dir: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(run_dir.iterdir()):
        if item.is_dir():
            if item.name in RUN_KEEP_DIRS and item.name not in RUN_SKIP_DIRS:
                copy_tree(item, dst / item.name, ignore_dirs={"__pycache__"})
            continue
        if item.name in RUN_SKIP_NAMES:
            continue
        if item.name.startswith(RUN_SKIP_PREFIX) or item.name.endswith(RUN_SKIP_SUFFIX):
            continue
        if item.stat().st_size > RUN_FILE_MAX_BYTES:
            continue
        shutil.copy2(item, dst / item.name)


def latest_run(problem: str) -> tuple[str, Path] | tuple[None, None]:
    index = OUT / "runs" / problem / "latest.json"
    if not index.exists():
        return None, None
    data = json.loads(index.read_text(encoding="utf-8"))
    rel = data.get("path") or f"output/runs/{problem}/{data.get('run_id', '')}"
    return data.get("run_id") or Path(rel).name, ROOT / rel


def build_package(stage: Path, notes: list[str], with_traces: bool = False) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # 1) 五份提交工作簿
    for name in ("result1.xlsx", "result2.xlsx", "result3.xlsx", "result4-2.xlsx", "result4-3.xlsx"):
        src = OUT / "results" / name
        if src.exists():
            shutil.copy2(src, stage / name)
        else:
            notes.append(f"缺少结果工作簿：{name}")

    # 2) AI 工具使用详情（论文声明使用 AI 时必须提供）
    ai = OUT / "support" / "AI工具使用详情.pdf"
    if ai.exists():
        shutil.copy2(ai, stage / ai.name)
    else:
        notes.append("缺少 AI工具使用详情.pdf，请先执行 make ai-details")

    # 3) 完整源程序与运行入口
    copy_tree(ROOT / "code", stage / "code", ignore_dirs={"__pycache__", ".pytest_cache"})
    shutil.copy2(ROOT / "Makefile", stage / "Makefile")

    # 4) 论文图形（矢量 PDF 与绘图参数）与表格数据
    copy_tree(OUT / "figures", stage / "figures", patterns={".pdf", ".json"},
              skip_prefix=FIGURE_SKIP_PREFIX)
    copy_tree(OUT / "tables", stage / "tables")

    # 5) 离线交互图与图形总览
    for name in ("problem1-dashboard.html", "problem2-dashboard.html", "visualization-gallery.html"):
        src = OUT / name
        if src.exists():
            shutil.copy2(src, stage / name)

    # 6) 各问结果说明
    reports = stage / "reports"
    reports.mkdir()
    for name in ("problem1-report.md", "problem2-report.md", "problem3-report.md",
                 "problem4-report.md", "causality-audit.md", "review-1st-response.md",
                 "research-roadmap.md"):
        src = OUT / name
        if src.exists():
            shutil.copy2(src, reports / name)

    # 7) 四问最新归档的配置、核验与汇总（逐时全年轨迹见工作簿，可由 make solve-q* 重建）
    runs_meta: list[tuple[str, str]] = []
    for problem in ("problem1", "problem2", "problem3", "problem4"):
        run_id, run_dir = latest_run(problem)
        if not run_dir or not run_dir.exists():
            notes.append(f"未找到 {problem} 的最新归档")
            continue
        copy_run(run_dir, stage / "runs" / problem / run_id)
        if with_traces:
            for rel in TRACE_FILES.get(problem, ()):  # 大文件压缩后附带
                src = run_dir / rel
                if not src.exists():
                    continue
                dst = stage / "runs" / problem / run_id / (rel + ".gz")
                dst.parent.mkdir(parents=True, exist_ok=True)
                with src.open("rb") as fin, gzip.open(dst, "wb", compresslevel=9) as fout:
                    shutil.copyfileobj(fin, fout)
        shutil.copy2(OUT / "runs" / problem / "latest.json", stage / "runs" / problem / "latest.json")
        runs_meta.append((problem, run_id))

    # 8) 绘图版本与风格规范
    vis_index = OUT / "runs" / "visuals" / "latest.json"
    if vis_index.exists():
        vis_dir = stage / "runs" / "visuals"
        vis_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vis_index, vis_dir / "latest.json")
        version = json.loads(vis_index.read_text(encoding="utf-8")).get("version", "")
        src = OUT / "runs" / "visuals" / version
        if src.exists():
            copy_tree(src, vis_dir / version, patterns={".json", ".md", ".txt", ".csv"},
                      ignore_dirs={"__pycache__"})

    write_manifest(stage, runs_meta)


def write_manifest(stage: Path, runs_meta: list[tuple[str, str]]) -> None:
    lines = [
        "# 支撑材料说明",
        "",
        f"生成日期：{datetime.now().strftime('%Y-%m-%d')}。",
        "",
        "本压缩包与论文附录“支撑材料结构”对应。原始赛题附件按规定不再重复提供，",
        "复现时请把官方附件放回 `doc/Problem/C题/附件/`，在解压后的目录根部执行 `make` 命令。",
        "",
        "## 目录",
        "",
        "| 路径 | 内容 |",
        "|---|---|",
        "| `result1.xlsx` … `result4-3.xlsx` | 四问提交结果工作簿，与最新运行归档逐单元格一致 |",
        "| `AI工具使用详情.pdf` | AI 工具名称、用途、提示方式与人工核验记录 |",
        "| `code/`、`Makefile` | 全部可运行源程序、参数配置、独立核验脚本与测试 |",
        "| `figures/` | 正文与附录图形的矢量 PDF 及绘图参数 |",
        "| `tables/` | 论文表格的数据源 |",
        "| `reports/` | 各问方法与结果说明、因果性审计、审阅处理记录 |",
        "| `runs/problemN/` | 对应问题最新一次运行的配置、独立核验、逐日汇总与指定日期逐时结果 |",
        "| `runs/visuals/` | 绘图版本、风格规范与数值保护哈希 |",
        "| `*.html` | 可离线打开的交互调度图与图形总览 |",
        "",
        "## 运行归档对应关系",
        "",
    ]
    for problem, run_id in runs_meta:
        lines.append(f"- `runs/{problem}/{run_id}/`")
    lines += [
        "",
        "## 关于全年逐时轨迹",
        "",
        "各策略的全年逐十分钟轨迹（`strategies/`、`boundary/`、`issued_plans.csv`、",
        "`issued_adjustments.csv`）单文件在 16–43 MB 量级，合计远超支撑材料 20 MB 上限，",
        "因此未随包提供。提交结果的全年逐时数据已完整保存在五份 `result*.xlsx` 中；",
        "其余对照轨迹可用同一配置执行 `make solve-q1` 至 `make solve-q4` 原样重建，",
        "归档中的 `config.json`、`input_hashes.json` 与 `manifest.json` 给出重建所需的全部参数与输入哈希。",
        "",
        "## 复现顺序",
        "",
        "```bash",
        "make inspect                 # 检查四个附件与五个结果模板",
        "make check-q1 ... make check-q4   # 对随包归档做独立核验，不重算",
        "make solve-q1 ... make solve-q4   # 需要时按顺序重算",
        "make audit-causality         # 逐日毒化未来数据的因果性审计",
        "make test                    # 手算算例与接口测试",
        "make figures                 # 从归档数值重绘图形",
        "```",
        "",
    ]
    (stage / "支撑材料说明.md").write_text("\n".join(lines), encoding="utf-8")


def make_zip(stage: Path, zip_path: Path) -> None:
    """按固定顺序与固定时间戳打包，使内容不变时重复运行得到同一 MD5。"""
    if zip_path.exists():
        zip_path.unlink()
    files = sorted((p for p in stage.rglob("*") if p.is_file()),
                   key=lambda p: str(p.relative_to(stage)))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for item in files:
            arcname = str(Path(PACK_NAME) / item.relative_to(stage))
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, item.read_bytes())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_mtime(paths) -> float:
    best = 0.0
    for path in paths:
        if path.is_file():
            best = max(best, path.stat().st_mtime)
    return best


def check_paper(notes: list[str]) -> None:
    paper_pdf = OUT / "paper" / "main.pdf"
    if not paper_pdf.exists():
        notes.append(("缺少 output/paper/main.pdf", "make check"))
        return
    sources = list((ROOT / "paper").rglob("*.tex")) + list((ROOT / "paper").rglob("*.sty")) \
        + list((ROOT / "paper").rglob("*.cls")) + list((OUT / "tables").glob("*")) \
        + list((OUT / "figures").glob("*.pdf"))
    if paper_pdf.stat().st_mtime < newest_mtime(sources):
        notes.append(("论文 PDF 比 paper/ 源码、表格或图形旧", "make check"))

    ai_pdf = OUT / "support" / "AI工具使用详情.pdf"
    ai_tex = ROOT / "paper" / "ai_details.tex"
    if not ai_pdf.exists():
        notes.append(("缺少 AI工具使用详情.pdf", "make ai-details"))
    elif ai_tex.exists() and ai_pdf.stat().st_mtime < ai_tex.stat().st_mtime:
        notes.append(("AI工具使用详情.pdf 比 ai_details.tex 旧", "make ai-details"))


def check_figures(notes: list[str]) -> None:
    figs = list((OUT / "figures").glob("*.pdf"))
    if not figs:
        notes.append(("缺少 output/figures 下的图形", "make figures"))
        return
    vis_src = [p for p in (ROOT / "code" / "vis").rglob("*.py")]
    if newest_mtime(figs) < newest_mtime(vis_src):
        notes.append(("图形 PDF 比 code/vis/ 绘图源码旧", "make figures"))


def check_inputs(notes: list[str], run_dir: Path, problem: str) -> None:
    """原始附件被改动会让整条结果链失效，必须拦下。"""
    index = run_dir / "input_hashes.json"
    if not index.exists():
        return
    for rel, digest in json.loads(index.read_text(encoding="utf-8")).items():
        src = ROOT / rel
        if not src.exists():
            notes.append((f"{problem} 所用输入缺失：{rel}", "恢复原始附件后重跑"))
        elif sha256(src) != digest:
            notes.append((f"{problem} 所用输入已被改动：{rel}", "恢复原始附件后重跑该问"))


def check_code_vs_runs(notes: list[str]) -> None:
    """比对归档记录的模型源码哈希与配置，识别“代码改了但结果没重跑”。"""
    current_config = None
    config_path = ROOT / "code" / "config" / "problem.json"
    if config_path.exists():
        current_config = json.loads(config_path.read_text(encoding="utf-8"))

    for problem in ("problem1", "problem2", "problem3", "problem4"):
        run_id, run_dir = latest_run(problem)
        if not run_dir or not run_dir.exists():
            notes.append((f"未找到 {problem} 的最新归档", f"make solve-q{problem[-1]}"))
            continue
        check_inputs(notes, run_dir, problem)

        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        hashes = manifest.get("model_source_hashes")
        stale: list[str] = []
        if hashes:
            for rel, digest in hashes.items():
                src = ROOT / rel
                if not src.exists() or sha256(src) != digest:
                    stale.append(rel)
        else:
            # 问题一归档保存的是整份 code/ 快照，只比对与该问相关的源文件。
            snapshot = run_dir / "code"
            for item in snapshot.rglob("*.py") if snapshot.exists() else []:
                rel = item.relative_to(snapshot).as_posix()
                if rel.startswith("vis/") or any(k in rel for k in ("problem2", "problem3", "problem4")):
                    continue
                cur = ROOT / "code" / rel
                if not cur.exists() or sha256(cur) != sha256(item):
                    stale.append(f"code/{rel}")
        if stale:
            shown = "、".join(stale[:3]) + ("…" if len(stale) > 3 else "")
            notes.append((f"{problem} 归档所用源码与当前代码不一致（{shown}）",
                          f"make solve-q{problem[-1]} && make check-q{problem[-1]}"))

        archived_config = manifest.get("config")
        if archived_config and current_config is not None and archived_config != current_config:
            notes.append((f"{problem} 归档所用参数与 code/config/problem.json 不一致",
                          f"make solve-q{problem[-1]} && make check-q{problem[-1]}"))


def check_results(notes: list[str]) -> None:
    published = {
        "result1.xlsx": "problem1", "result2.xlsx": "problem2", "result3.xlsx": "problem3",
        "result4-2.xlsx": "problem4", "result4-3.xlsx": "problem4",
    }
    for name, problem in published.items():
        src = OUT / "results" / name
        if not src.exists():
            notes.append((f"缺少结果工作簿 {name}", f"make solve-q{problem[-1]}"))
            continue
        _, run_dir = latest_run(problem)
        archived = run_dir / name if run_dir else None
        if archived and archived.exists() and sha256(archived) != sha256(src):
            notes.append((f"{name} 与 {problem} 最新归档不一致",
                          f"make solve-q{problem[-1]} 后重新发布，或确认使用哪一版"))


def preflight(notes: list[str]) -> None:
    check_paper(notes)
    check_figures(notes)
    check_code_vs_runs(notes)
    check_results(notes)
    if not (ROOT / "code" / "config" / "team_private.json").exists():
        notes.append(("未配置 code/config/team_private.json，身份词检查未覆盖真实姓名与学校",
                      "cp code/config/team_private.example.json code/config/team_private.json 并填写后 make check"))


def run_make(target: str) -> bool:
    print(f"==> make {target}")
    proc = subprocess.run(["make", target], cwd=ROOT)
    return proc.returncode == 0


def report(paper: Path, pack: Path) -> None:
    print("\n提交文件：")
    for path in (paper, pack):
        if path.exists():
            mark = "OK " if size_mb(path) <= MAX_MB else "超限"
            print(f"  [{mark}] {path}  {size_mb(path):.2f} MB  MD5 {md5(path)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md5-only", action="store_true", help="只对已有提交文件重新计算 MD5")
    parser.add_argument("--with-traces", action="store_true",
                        help="额外附带主策略全年逐时轨迹（gzip），注意 20 MB 上限")
    parser.add_argument("--rebuild", action="store_true",
                        help="打包前先执行 make ai-details 与 make check（不重跑求解）")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    paper_out = DEST / PAPER_NAME
    zip_out = DEST / f"{PACK_NAME}.zip"

    if args.md5_only:
        report(paper_out, zip_out)
        return 0

    if args.rebuild:
        for target in ("ai-details", "check"):
            if not run_make(target):
                print(f"\nmake {target} 失败，已中止打包。")
                return 2
        print()

    notes: list[tuple[str, str]] = []
    preflight(notes)

    src_paper = OUT / "paper" / "main.pdf"
    if src_paper.exists():
        shutil.copy2(src_paper, paper_out)

    stage = DEST / PACK_NAME
    build_package(stage, notes, with_traces=args.with_traces)
    make_zip(stage, zip_out)

    total_files = sum(1 for p in stage.rglob("*") if p.is_file())
    print(f"支撑材料条目：{total_files} 个文件，解压后 {size_mb(stage) if stage.is_file() else sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())/1024/1024:.2f} MB")
    report(paper_out, zip_out)

    if notes:
        print("\n打包完成，但以下问题会让提交物与当前代码/论文不一致：")
        for issue, fix in notes:
            print(f"  - {issue}")
            print(f"    处理：{fix}")
        print("\n结论：请先处理上述问题并重新打包，再提交 MD5。")
        return 1

    print("\n结论：论文、图表、结果与当前代码一致，可按上面的 MD5 提交。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
