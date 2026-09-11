#!/usr/bin/env python3
"""对 CUMCM 论文 PDF 做可自动化的交卷前检查。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "code" / "config" / "validation.json"
PRIVATE_CONFIG_PATH = ROOT / "code" / "config" / "team_private.json"


def run(*args: str) -> str:
    try:
        proc = subprocess.run(args, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"缺少外部命令：{args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or f"命令失败：{' '.join(args)}") from exc
    return proc.stdout


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    forbidden = list(config.get("forbidden_terms", []))
    if PRIVATE_CONFIG_PATH.exists():
        private = json.loads(PRIVATE_CONFIG_PATH.read_text(encoding="utf-8"))
        forbidden.extend(private.get("forbidden_terms", []))
    config["forbidden_terms"] = [x for x in forbidden if x]
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        action="store_true",
        help="模板检查模式：占位内容只警告，不作为错误",
    )
    parser.add_argument("paper", type=Path, help="待提交论文 PDF")
    args = parser.parse_args()

    paper = args.paper.resolve()
    config = load_config()
    errors: list[str] = []
    warnings: list[str] = []

    if not paper.exists():
        print(f"ERROR: 文件不存在：{paper}")
        return 2
    if paper.suffix.lower() != ".pdf" or paper.read_bytes()[:4] != b"%PDF":
        errors.append("论文不是有效的 PDF 文件")

    size_mb = paper.stat().st_size / 1024 / 1024
    if size_mb > float(config["max_paper_mb"]):
        errors.append(f"论文大小 {size_mb:.2f} MB，超过 20 MB")

    try:
        info = run("pdfinfo", str(paper))
        text = run("pdftotext", str(paper), "-")
        first_page = run("pdftotext", "-f", "1", "-l", "1", str(paper), "-")
    except RuntimeError as exc:
        errors.append(str(exc))
        info = text = first_page = ""

    page_size = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info)
    if page_size:
        width, height = map(float, page_size.groups())
        if abs(width - 595.28) > 2 or abs(height - 841.89) > 2:
            errors.append(f"页面不是 A4：{width:.2f} x {height:.2f} pt")
    else:
        warnings.append("无法从 pdfinfo 确认页面尺寸")

    # 中文标题在 PDF 文本层中可能被排成“摘 要”或“关 键 词”，
    # 检查前先去除排版产生的空白，避免把合规首页误判为缺失字段。
    first_page_compact = re.sub(r"\s+", "", first_page)
    if "摘要" not in first_page_compact or "关键词" not in first_page_compact:
        errors.append("PDF 第一页未同时检测到“摘要”和“关键词”")
    if "承诺书" in first_page_compact or "编号专用页" in first_page_compact:
        errors.append("电子版第一页疑似包含承诺书或编号专用页")
    if "目录" in first_page_compact:
        warnings.append("第一页出现“目录”，请确认正文没有目录")

    for term in config.get("placeholder_terms", []):
        if term in text:
            message = f"论文仍含占位符：{term}"
            (warnings if args.template else errors).append(message)
    for term in config.get("forbidden_terms", []):
        if term in text or term in info or term in paper.name:
            errors.append(f"检测到可能泄露身份的信息：{term}")

    if "AI工具使用声明" not in text.replace(" ", ""):
        errors.append("未检测到“AI工具使用声明”")
    if "本参赛队在竞赛过程中使用了AI工具" in text.replace(" ", ""):
        warnings.append("论文声明使用 AI；支撑材料必须包含 AI工具使用详情.pdf")

    print(f"File: {paper}")
    print(f"Size: {size_mb:.2f} MB")
    pages = re.search(r"Pages:\s+(\d+)", info)
    if pages:
        print(f"Pages: {pages.group(1)}")
    for item in warnings:
        print(f"WARN: {item}")
    for item in errors:
        print(f"ERROR: {item}")
    if errors:
        print(f"Result: FAIL ({len(errors)} error(s), {len(warnings)} warning(s))")
        return 1
    print(f"Result: PASS ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
