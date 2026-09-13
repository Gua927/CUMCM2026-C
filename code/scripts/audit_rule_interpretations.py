#!/usr/bin/env python3
"""Read-only audit of template semantics and alternative adjustment accounting.

Alternative bills hold every issued contract and executed action fixed; they are
not optimized policies under the alternative rule. Immutable runs are not edited.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/reviews/full-model"


def latest(question):
    meta = json.loads((ROOT / f"output/runs/problem{question}/latest.json").read_text())
    return ROOT / meta["path"]


def adjustment_bills(question, strategy):
    folder = latest(question)
    frame = pd.read_csv(folder / f"strategies/{strategy}.csv")
    records = pd.read_csv(folder / "issued_adjustments.csv")
    records = records.loc[records.lead_days == 1]
    rows = []
    for date, day in frame.groupby("date", sort=True):
        day = day.sort_values("slot")
        original = day.planned_grid_kwh.to_numpy()
        current = original.copy()
        price = day.price_yuan_per_kwh.to_numpy()
        variation = np.zeros(144)
        sequential = 0.0
        versions = records.loc[records.date == date]
        for release, part in versions.groupby("release_slot", sort=True):
            part = part.sort_values("slot")
            slots = part.slot.to_numpy(int)
            assert np.array_equal(slots, np.arange(int(release), 144))
            delta = part.contract_kwh.to_numpy() - current[slots]
            variation[slots] += np.abs(delta)
            sequential += float(price[slots] @ (1.5*np.maximum(delta, 0)-0.5*np.maximum(-delta, 0)))
            current[slots] = part.contract_kwh.to_numpy()
        np.testing.assert_allclose(current, day.adjusted_grid_kwh, atol=2e-5, rtol=0)
        net_delta = current-original
        final_adjustment = float(price @ (1.5*np.maximum(net_delta, 0)-0.5*np.maximum(-net_delta, 0)))
        np.testing.assert_allclose(final_adjustment, day.adjustment_cost_yuan.sum(), atol=2e-5, rtol=0)
        extra = sequential-final_adjustment
        # At fixed delivery prices, the difference is precisely the reversal fee.
        identity = float(0.5*price @ (variation-np.abs(net_delta)))
        np.testing.assert_allclose(extra, identity, atol=2e-5, rtol=0)
        assert extra >= -2e-5
        rows.append({"date": date, "published_cash_yuan": day.cost_yuan.sum(),
                     "final_net_adjustment_yuan": final_adjustment,
                     "sequential_adjustment_yuan": sequential,
                     "extra_reversal_fee_yuan": extra,
                     "reversed_slots": int((variation-np.abs(net_delta) > 2e-5).sum()),
                     "sequential_same_path_cash_yuan": day.cost_yuan.sum()+extra})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"q{question}_sequential_settlement.csv", index=False)
    return {"run": str(folder.relative_to(ROOT)),
            "published_cash_yuan": float(result.published_cash_yuan.sum()),
            "same_path_sequential_cash_yuan": float(result.sequential_same_path_cash_yuan.sum()),
            "extra_reversal_fee_yuan": float(result.extra_reversal_fee_yuan.sum()),
            "days_with_reversal": int((result.extra_reversal_fee_yuan > 2e-5).sum()),
            "reversed_slots": int(result.reversed_slots.sum())}


def calendar_examples():
    rows = []
    for question, strategy in ((2, "risk_policy"), (3, "adjust_all"), (4, "q43_forecast")):
        folder = latest(question)
        frame = pd.read_csv(folder / f"strategies/{strategy}.csv")
        zero = frame.loc[frame.slot == 0].set_index("date")
        for date in ("2025-02-01", "2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
            next_date = str((pd.Timestamp(date)+pd.Timedelta(days=1)).date())
            rows.append({"question": question, "template_date": date,
                         "template_label": "0:00-0:10+1",
                         "declared_delivery_date": date, "literal_delivery_date": next_date,
                         "current_cell_kwh": float(zero.loc[date, "planned_grid_kwh"]),
                         "next_day_issued_midnight_kwh": float(zero.loc[next_date, "planned_grid_kwh"])})
    pd.DataFrame(rows).to_csv(OUT / "midnight_semantics_examples.csv", index=False)
    return {"examples": len(rows), "note": "Next-day issued quantities illustrate different dates; copying them back would reveal a later decision and is NOT a proposed fix."}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"q3": adjustment_bills(3, "adjust_all"),
               "q43": adjustment_bills(4, "q43_forecast"),
               "calendar": calendar_examples(),
               "scope": "Same issued/executed paths; delivery-time prices; alternative accounting only, not alternative-rule optimization."}
    (OUT / "rule_interpretations.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
