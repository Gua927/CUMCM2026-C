"""Shared energy units, battery parameters, and canonical dispatch schema."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from cumcm_c.paths import PATHS


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    initial_soc_kwh: float
    max_charge_power_kw: float
    max_discharge_power_kw: float
    charge_efficiency: float
    discharge_efficiency: float

    def validate(self) -> None:
        if not all(np.isfinite(x) for x in asdict(self).values()):
            raise ValueError("Battery parameters must be finite")
        if not 0 <= self.soc_min_kwh <= self.initial_soc_kwh <= self.soc_max_kwh <= self.capacity_kwh:
            raise ValueError("Invalid battery energy limits")
        if min(self.max_charge_power_kw, self.max_discharge_power_kw) <= 0:
            raise ValueError("Power limits must be positive")
        if not all(0 < x <= 1 for x in (self.charge_efficiency, self.discharge_efficiency)):
            raise ValueError("Efficiencies must be in (0, 1]")


def load_parameters():
    config = json.loads(PATHS.config_file.read_text(encoding="utf-8"))
    battery = Battery(**config["battery"])
    battery.validate()
    dt = config["time_step_minutes"] / 60
    if dt <= 0:
        raise ValueError("Time step must be positive")
    return battery, dt, config


def clock_label(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def make_schedule(data, battery, dt, grid, charge, discharge, curtailed, energy):
    """All flows are bus-side kWh; E is internal battery kWh, with n+1 edges."""
    out = data.copy()
    n = len(out)
    arrays = (grid, charge, discharge, curtailed)
    if any(len(x) != n for x in arrays) or len(energy) != n + 1:
        raise ValueError("Dispatch dimensions disagree")
    for name, value in zip(
        ("planned_grid_kwh", "charge_kwh", "discharge_kwh", "pv_curtailed_kwh"), arrays
    ):
        out[name] = np.asarray(value, dtype=float)
    out["actual_grid_kwh"] = out.planned_grid_kwh
    out["emergency_grid_kwh"] = 0.0
    out["plan_unused_kwh"] = 0.0
    out["soc_start_kwh"] = np.asarray(energy[:-1])
    out["soc_end_kwh"] = np.asarray(energy[1:])
    out["cost_yuan"] = out.price_yuan_per_kwh * out.planned_grid_kwh
    out["conversion_loss_kwh"] = (
        (1 - battery.charge_efficiency) * out.charge_kwh
        + (1 / battery.discharge_efficiency - 1) * out.discharge_kwh
    )
    out["grid_kw"] = out.planned_grid_kwh / dt
    return out
