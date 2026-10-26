import copy
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from study.adaptive import METHODS, release_path, replay
from study.equity import (ALPHA, BUCKETS, CAPS, PAIRS, REQUIRED, SCOPES, build_records,
                    evaluate_record, fixed_schedule_capacity, load_session, run, sessions, summarize)


def minute_rows():
    rows = []
    for minute in range(390):
        clock = 570 + minute
        rows.append({"Date": "20210104", "Ticker": "EBAY", "TimeBarStart": f"{clock // 60:02d}:{clock % 60:02d}",
                     "FirstTradeTime": "09:30:00.100000000", "FirstTradePrice": "100",
                     "Volume": "10", "VolumeWeightPrice": "100", "FinraVolume": "20",
                     "FinraVolumeWeightPrice": "103", "TotalVolume": "30", "TotalVolumeWeightPrice": "102"})
    return rows


def fixture(day, multiplier=1):
    total = np.linspace(1000, 5000, BUCKETS) * multiplier
    return {"asset": "EBAY", "date": day, "arrival_price": 100,
            "volumes": {"total": total, "exchange": total * 0.6, "finra": total * 0.4},
            "prices": {scope: np.linspace(99, 101, BUCKETS) for scope in PAIRS}}


class EquityDataTests(unittest.TestCase):
    def test_missing_data_preserves_existing_results_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing_results"
            existing.mkdir()
            sentinels = {name: b"previous verified result\n" for name in
                         ("equity_results.json", "equity_daily.csv", "equity_capacity.png")}
            for name, content in sentinels.items():
                (existing / name).write_bytes(content)
            new_output = root / "new_results"
            for output in (existing, new_output):
                with self.subTest(output=output.name):
                    with self.assertRaisesRegex(ValueError, "No usable equity evaluation data.*No output files were written"):
                        run(root / "missing_data", output)
            self.assertEqual({path.name: path.read_bytes() for path in existing.iterdir()}, sentinels)
            self.assertFalse(new_output.exists())

    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "EBAY_20210104.csv"
            fields = sorted(REQUIRED) + [f"unused_{i}" for i in range(90 - len(REQUIRED))]
            with path.open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            return load_session(path, "EBAY", sessions()[0])

    def test_fixed_sessions_have_no_holidays_and_twenty_evaluation_days(self):
        dates = sessions()
        self.assertEqual(len(dates), 40)
        self.assertEqual(str(dates[20]), "2021-02-02")
        self.assertEqual(str(dates[-1]), "2021-03-02")
        self.assertNotIn("2021-01-18", list(map(str, dates)))
        self.assertNotIn("2021-02-15", list(map(str, dates)))

    def test_field_pairing_and_zero_minute_are_not_close_price_proxies(self):
        rows = minute_rows()
        rows[0].update(Volume="0", VolumeWeightPrice="", FinraVolume="20",
                       TotalVolume="20", TotalVolumeWeightPrice="103")
        item, report = self.load(rows)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["zero_volume_minutes"]["exchange"], 1)
        self.assertEqual(item["volumes"]["exchange"][0], 40)
        self.assertEqual(item["volumes"]["total"][0], 140)
        self.assertAlmostEqual(item["prices"]["total"][0], (20 * 103 + 120 * 102) / 140)
        self.assertEqual(item["prices"]["exchange"][0], 100)
        self.assertEqual(item["prices"]["finra"][0], 103)
        self.assertEqual(item["arrival_price"], 100)

    def test_missing_duplicate_zero_bucket_and_volume_identity_are_excluded(self):
        cases = []
        missing = minute_rows(); missing.pop(20)
        cases.append((missing, "missing_minutes"))
        duplicate = minute_rows(); duplicate.append(duplicate[20].copy())
        cases.append((duplicate, "duplicate_minutes"))
        zero = minute_rows()
        for row in zero[:5]:
            row.update(Volume="0", VolumeWeightPrice="", FinraVolume="30",
                       FinraVolumeWeightPrice="102")
        cases.append((zero, "nonpositive_exchange_buckets"))
        identity = minute_rows(); identity[2]["TotalVolume"] = "31"
        cases.append((identity, "volume_identity_failed"))
        for rows, reason in cases:
            with self.subTest(reason=reason):
                item, report = self.load(rows)
                self.assertIsNone(item)
                self.assertIn(reason, report["reasons"])

    def test_outside_minutes_count_but_do_not_expand_selected_window(self):
        rows = minute_rows()
        rows.append({**rows[0], "TimeBarStart": "16:00"})
        item, report = self.load(rows)
        self.assertEqual(report["outside_window_rows"], 1)
        self.assertEqual(len(item["prices"]["total"]), 78)


class EquityExecutionTests(unittest.TestCase):
    def test_fixed_schedule_limit_matches_hand_calculated_suffix_constraint(self):
        quantity, bucket = fixed_schedule_capacity([0.4, 0.7, 1], [10, 100, 10], 0.1)
        self.assertAlmostEqual(quantity, 10 / 3)
        self.assertEqual(bucket, 2)

    def test_fixed_schedule_limit_is_the_completion_boundary(self):
        fractions, volume = np.array([0.4, 0.7, 1]), np.array([10, 100, 10])
        limit, _ = fixed_schedule_capacity(fractions, volume, 0.1)
        for multiplier, completed in ((1 - 1e-6, True), (1, True), (1 + 1e-6, False)):
            quantity = limit * multiplier
            result = replay(fractions * quantity, [100, 110, 120], volume, quantity, 0.1, 100)
            self.assertEqual(result["fully_completed"], completed)

    def test_fixed_schedule_limit_obeys_capacity_bound_and_skips_zero_remaining_fraction(self):
        volume = np.array([10, 100, 10])
        for fractions in ([1, 1, 1], [0, 0, 1], [0.5, 1, 1]):
            limit, _ = fixed_schedule_capacity(fractions, volume, 0.1)
            self.assertLessEqual(limit, 0.1 * volume.sum())
        immediate, bucket = fixed_schedule_capacity([1, 1, 1], volume, 0.1)
        self.assertEqual(immediate, 12)
        self.assertEqual(bucket, 0)
        self.assertEqual(fixed_schedule_capacity([1, 1, 1], volume, 0)[0], 0)

    def test_twenty_past_sessions_and_no_future_information(self):
        days = [fixture(day) for day in sessions()]
        before, skips = build_records(days)
        self.assertFalse(skips)
        self.assertEqual(len(before), 20)
        changed = [fixture(day, 10 if index >= 20 else 1) for index, day in enumerate(sessions())]
        after, _ = build_records(changed)
        np.testing.assert_allclose(before[0]["mean_volumes"], after[0]["mean_volumes"])
        self.assertNotEqual(before[1]["mean_volumes"][0], after[1]["mean_volumes"][0])
        self.assertTrue(all(d < before[0]["date"].isoformat() for d in before[0]["history_dates"]))
        reduced, skipped = build_records(days[1:])
        self.assertEqual(len(reduced), 19)
        self.assertEqual(skipped[0]["prior_valid_sessions"], 19)
        self.assertEqual(skipped[0]["date"], str(sessions()[20]))

    def test_78_bucket_quantity_conservation_and_current_volume_causality(self):
        record = build_records([fixture(day) for day in sessions()])[0][0]
        for method in METHODS:
            releases = release_path(record["weights"], record["volumes"]["total"], record["mean_volumes"],
                                    record["quantity"], 0.05, method, ALPHA)
            for scope in SCOPES:
                result = replay(releases, record["prices"]["total"], record["volumes"][scope],
                                record["quantity"], 0.05, record["arrival_price"])
                self.assertAlmostEqual(result["fills"].sum() + result["unfilled_base_quantity"], record["quantity"])
                self.assertAlmostEqual(result["suffix_backlog"], result["unfilled_base_quantity"])
                self.assertTrue(np.all(result["fills"] <= 0.05 * record["volumes"][scope] + 1e-12))
            later = record["volumes"]["total"].copy(); later[30:] *= 100
            alternative = release_path(record["weights"], later, record["mean_volumes"], record["quantity"], 0.05, method, ALPHA)
            np.testing.assert_allclose(releases[:31], alternative[:31])

    def test_scope_changes_capacity_not_release_or_total_price_benchmark(self):
        record = build_records([fixture(day) for day in sessions()])[0][0]
        record["quantity"] = 10
        record["volumes"]["exchange"] = record["volumes"]["total"] * np.linspace(0.1, 0.9, BUCKETS)
        rows = evaluate_record(record, 0.05)
        expected = record["volumes"]["total"] @ record["prices"]["total"] / record["volumes"]["total"].sum()
        for method in METHODS:
            total, exchange = [r for r in rows if r["method"] == method]
            self.assertEqual(total["release_sha256"], exchange["release_sha256"])
            for row in (total, exchange):
                self.assertAlmostEqual(row["market_vwap"], expected)
                self.assertAlmostEqual(row["full_order_deviation_bps"], 10_000 * (row["reference_filled_price"] / expected - 1))

    def test_price_intersection_excludes_one_incomplete_scope_but_retains_completion(self):
        record = build_records([fixture(day) for day in sessions()])[0][0]
        record["quantity"] = 10
        rows = evaluate_record(record, 0.05)
        second = copy.deepcopy(rows)
        for row in second:
            row["date"] = str(sessions()[21])
        second[0].update(fully_completed=False, unfilled_fraction=0.5,
                         full_order_deviation_bps=None, full_order_absolute_deviation_bps=None)
        result = summarize(rows + second)["EBAY"]["0.05"]
        self.assertEqual(result["evaluation_days"], 2)
        self.assertEqual(result["common_complete_days"], 1)
        self.assertEqual(result["common_complete_dates"], [str(sessions()[20])])
        self.assertEqual(result["scopes"]["total"]["twap"]["days_completed"], 1)
        self.assertEqual(result["scopes"]["exchange"]["twap"]["days_completed"], 2)


if __name__ == "__main__":
    unittest.main()
