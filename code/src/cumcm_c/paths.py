"""集中定义仓库路径，避免脚本中散落相对路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def attachment_dir(self) -> Path:
        return self.root / "doc" / "Problem" / "C题" / "附件"

    @property
    def result_template_dir(self) -> Path:
        return self.attachment_dir / "附件5"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def result_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def config_file(self) -> Path:
        return self.root / "code" / "config" / "problem.json"


PATHS = ProjectPaths(Path(__file__).resolve().parents[3])
