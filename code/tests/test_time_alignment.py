"""Calendar boundaries and raw-observation causality, before forecasting."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
from cumcm_c.data import align_daily_samples, problem4_prices


class TimeAlignmentTests(unittest.TestCase):
    def test_midnight_uses_previous_date_and_never_wraps_year(self):
        # Each value identifies its raw row and column independently.
        raw = np.arange(3 * 144, dtype=float).reshape(3, 144)
        result = align_daily_samples(raw, "chronological_start")
        self.assertEqual(result[0, 0], raw[0, 0])  # explicit warm-up imputation
        self.assertEqual(result[1, 0], 143)
        self.assertEqual(result[2, 0], 287)
        self.assertEqual(result[-1, -1], 430)
        np.testing.assert_array_equal(result[:, 1:], raw[:, :-1])
        self.assertNotIn(raw[-1, -1], result)

    def test_future_midnight_does_not_change_current_day(self):
        raw = np.arange(3 * 144, dtype=float).reshape(3, 144)
        changed = raw.copy()
        changed[1, -1] += 10000  # tomorrow 00:00, not today's 00:00
        a = align_daily_samples(raw, "chronological_start")
        b = align_daily_samples(changed, "chronological_start")
        np.testing.assert_array_equal(a[:2], b[:2])
        self.assertEqual(b[2, 0] - a[2, 0], 10000)

    def test_legacy_results_and_source_are_preserved(self):
        raw = np.arange(288, dtype=float).reshape(2, 144)
        original = raw.copy()
        np.testing.assert_array_equal(align_daily_samples(raw), np.roll(raw, 1, axis=1))
        align_daily_samples(raw, "chronological_start")
        np.testing.assert_array_equal(raw, original)

    def test_prices_use_same_calendar_mapping_across_months(self):
        dates = pd.date_range("2025-01-01", "2025-12-31")
        columns = [f"{m//60:02d}:{m%60:02d}" for m in range(10, 1440, 10)] + ["0:00+1"]
        raw = np.arange(365 * 144, dtype=float).reshape(365, 144) + 1
        frame = pd.DataFrame(raw, columns=columns)
        frame.insert(0, "日期", dates)
        with patch("cumcm_c.data.load_attachment4", return_value=frame):
            actual_dates, prices = problem4_prices("chronological_start")
        self.assertTrue(actual_dates.equals(dates))
        for date in ("2025-02-01", "2025-03-01", "2025-12-31"):
            day = dates.get_loc(date)
            self.assertEqual(prices[day, 0], raw[day-1, -1])
        np.testing.assert_array_equal(prices[:, 1:], raw[:, :-1])

    def test_invalid_inputs_fail_explicitly(self):
        for raw in (np.zeros((2, 143)), np.zeros((0, 144)), np.full((2, 144), np.nan)):
            with self.assertRaises(ValueError):
                align_daily_samples(raw)
        with self.assertRaises(ValueError):
            align_daily_samples(np.zeros((2, 144)), "typo")


if __name__ == "__main__":
    unittest.main()
