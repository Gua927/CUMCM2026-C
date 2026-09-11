"""附件 1-4 的只读加载与最小结构校验。"""

from __future__ import annotations

from pathlib import Path
from datetime import time

import pandas as pd
import numpy as np

from .paths import PATHS


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(map(str, frame.columns))
    if missing:
        raise ValueError(f"{source.name} 缺少列：{sorted(missing)}")


def load_attachment1() -> pd.DataFrame:
    """读取固定日的电价、负载和光伏预测，保留原始单位。"""
    source = PATHS.attachment_dir / "附件1.xlsx"
    frame = pd.read_excel(source)
    _require_columns(frame, {"时间", "电价", "小区负载", "光伏发电预测功率"}, source)
    return frame


def load_attachment2() -> dict[str, pd.DataFrame]:
    """读取全年 10 分钟粒度的负载与光伏实际功率。"""
    source = PATHS.attachment_dir / "附件2.xlsx"
    sheets = pd.read_excel(source, sheet_name=None)
    expected = {"小区负载", "光伏发电实际功率"}
    if not expected.issubset(sheets):
        raise ValueError(f"{source.name} 缺少工作表：{sorted(expected.difference(sheets))}")
    return sheets


def load_attachment3() -> pd.DataFrame:
    """读取每日四次发布的未来 24 小时光伏整点预报。"""
    source = PATHS.attachment_dir / "附件3.xlsx"
    frame = pd.read_excel(source)
    _require_columns(frame, {"日期", "预报时刻", "预报1小时", "预报24小时"}, source)
    return frame


def load_attachment4() -> pd.DataFrame:
    """读取全年 10 分钟粒度的实时电价。"""
    source = PATHS.attachment_dir / "附件4.xlsx"
    return pd.read_excel(source)


def load_all() -> dict[str, object]:
    return {
        "attachment1": load_attachment1(),
        "attachment2": load_attachment2(),
        "attachment3": load_attachment3(),
        "attachment4": load_attachment4(),
    }


def time_minutes(value) -> int:
    """Parse mixed Excel time/string cells, preserving the explicit +1 day."""
    if isinstance(value, time):
        if value.second or value.microsecond:
            raise ValueError(f"Non-minute time: {value}")
        return value.hour * 60 + value.minute
    text = str(value).strip()
    day = 1440 if text.endswith("+1") else 0
    if day:
        text = text[:-2]
    parts = text.split(":")
    if len(parts) not in (2, 3) or (len(parts) == 3 and int(parts[2]) != 0):
        raise ValueError(f"Unknown time: {value}")
    hour, minute = map(int, parts[:2])
    if not (0 <= hour <= 24 and 0 <= minute < 60) or (hour == 24 and minute):
        raise ValueError(f"Invalid time: {value}")
    return day + hour * 60 + minute


def problem1_intervals(convention: str = "start") -> pd.DataFrame:
    """Canonical [00:00,24:00) rows; preserve source rows for both interpretations.

    start: follow official template left endpoints; the last sample is cyclic 00:00.
    end: sample at 00:10 describes 00:00--00:10.
    No timestamp or template label is silently overwritten.
    """
    if convention not in ("start", "end"):
        raise ValueError("convention must be start or end")
    frame = load_attachment1()
    ends = frame["时间"].map(time_minutes).to_numpy()
    if not np.array_equal(ends, np.arange(10, 1441, 10)):
        raise ValueError("Attachment 1 must contain exactly 00:10,...,24:00")
    numeric = frame[["电价", "小区负载", "光伏发电预测功率"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 1:] < 0).any():
        raise ValueError("Missing, non-finite, or negative power data")
    if (numeric[:, 0] <= 0).any():
        raise ValueError("Q1 positive-price simplification requires p > 0")
    order = np.roll(np.arange(144), 1) if convention == "start" else np.arange(144)
    numeric = numeric[order]
    out = pd.DataFrame({
        "slot": np.arange(144), "start_minute": np.arange(0, 1440, 10),
        "end_minute": np.arange(10, 1441, 10), "source_excel_row": order + 2,
        "source_time": frame["时间"].map(str).to_numpy()[order],
        "price_yuan_per_kwh": numeric[:, 0], "load_kw": numeric[:, 1],
        "pv_kw": numeric[:, 2], "load_kwh": numeric[:, 1] / 6,
        "pv_kwh": numeric[:, 2] / 6,
    })
    out["interval"] = [f"{a//60:02d}:{a%60:02d}-{b//60:02d}:{b%60:02d}"
                       for a, b in zip(out.start_minute, out.end_minute)]
    return out


def problem2_daily(convention="daily_cyclic_start"):
    """Return dates and (day, slot, load/PV) kWh with auditable time mapping."""
    sheets = load_attachment2()
    arrays=[];dates=None
    for name in ("小区负载", "光伏发电实际功率"):
        frame=sheets[name]
        current=pd.DatetimeIndex(pd.to_datetime(frame.iloc[:,0]))
        if not current.equals(pd.date_range("2025-01-01", "2025-12-31")):
            raise ValueError("Unexpected annual dates")
        if dates is not None and not dates.equals(current):
            raise ValueError("Load/PV dates disagree")
        dates=current
        if not np.array_equal([time_minutes(x) for x in frame.columns[1:]], np.arange(10,1441,10)):
            raise ValueError("Unexpected annual time labels")
        values=frame.iloc[:,1:].to_numpy(float)
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError("Missing or negative annual power")
        if convention == "daily_cyclic_start":
            values=np.roll(values,1,axis=1)
        elif convention == "chronological_start":
            # First 00:00 is absent. It is only imputed for historical pretraining;
            # January is unscored and storage remains idle throughout that month.
            values=np.column_stack([np.r_[values[0,0],values[:-1,-1]],values[:,:-1]])
        else:
            raise ValueError(convention)
        arrays.append(values/6)
    return dates,np.stack(arrays,axis=-1)


def problem3_forecasts():
    """Return (dates, issue_slots, hourly[day, issue, hour] kW) from attachment 3.

    Row k of each date is the forecast released at issue_slots[k]*10 minutes; column
    "预报h小时" is the PV power h hours after release. The alignment against attachment 2
    (best fit: the 10-minute slot starting at release+h) is checked in problem3 tests.
    """
    frame = load_attachment3().copy()
    frame["日期"] = pd.to_datetime(frame["日期"].ffill())
    frame["预报时刻"] = frame["预报时刻"].map(lambda x: time_minutes(x) if not isinstance(x, str) else time_minutes(x.strip()))
    dates = pd.DatetimeIndex(sorted(frame["日期"].unique()))
    if not dates.equals(pd.date_range("2025-01-01", "2025-12-31")):
        raise ValueError("Attachment 3 must cover every day of 2025")
    issue_minutes = (0, 360, 720, 1080)
    columns = [f"预报{h}小时" for h in range(1, 25)]
    hourly = np.full((len(dates), len(issue_minutes), 24), np.nan)
    for (date, minute), group in frame.groupby(["日期", "预报时刻"]):
        if minute not in issue_minutes or len(group) != 1:
            raise ValueError(f"Unexpected forecast release {date} {minute}")
        hourly[dates.get_loc(date), issue_minutes.index(minute)] = group[columns].to_numpy(float)[0]
    if not np.isfinite(hourly).all() or (hourly < 0).any():
        raise ValueError("Missing or negative PV forecast")
    return dates, tuple(m // 10 for m in issue_minutes), hourly
