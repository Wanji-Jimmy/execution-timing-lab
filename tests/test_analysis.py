import unittest

import numpy as np

from study.analysis import capped_replay, evaluate, moving_block_interval, schedule
from study.adaptive import release_path, replay


class ExecutionAnalysisTests(unittest.TestCase):
    def test_hand_calculated_cash_benchmarks_and_attribution(self):
        result = evaluate([0.5, 0.5], [100, 104], [10_000, 30_000], 100, 10_000)
        self.assertAlmostEqual(result["scheduled_base_quantity"], 100)
        self.assertAlmostEqual(result["reference_price"] * 100, 10_200)
        self.assertAlmostEqual(result["market_vwap"], 103)
        self.assertAlmostEqual(result["signed_deviation_bps"], -10_000 / 103)
        self.assertAlmostEqual(result["absolute_deviation_bps"], 10_000 / 103)
        self.assertAlmostEqual(result["arrival_shortfall_bps"], 200)
        self.assertAlmostEqual(result["volume_share_rmse"], 0.25)
        self.assertAlmostEqual(np.sum(result["attribution_bps"]), -10_000 / 103)
        self.assertAlmostEqual(result["attribution_residual"], 0)

    def test_schedule_uses_twenty_past_days_and_finite_ewma_weights(self):
        uniform = np.full(96, 1 / 96)
        recent = np.tile(uniform, (20, 1))
        recent[-1] = 0
        recent[-1, 0] = 1
        earlier = np.zeros((4, 96))
        earlier[:, -1] = 1
        history = np.vstack([earlier, recent])
        np.testing.assert_allclose(schedule(history, "historical"), recent.mean(axis=0))
        last_weight = 0.2 / (1 - 0.8**20)
        expected = uniform * (1 - last_weight)
        expected[0] += last_weight
        np.testing.assert_allclose(schedule(history, "ewma"), expected)
        for method in ("historical", "ewma"):
            np.testing.assert_allclose(schedule(history, method), schedule(recent, method))
        np.testing.assert_allclose(schedule(np.empty((0, 96)), "twap"), uniform)


    def test_constant_prices_and_oracle_schedule_have_zero_tracking_error(self):
        flat = evaluate([0.1, 0.3, 0.6], [123, 123, 123], [1, 2, 7], 123)
        self.assertAlmostEqual(flat["absolute_deviation_bps"], 0)
        self.assertAlmostEqual(flat["arrival_shortfall_bps"], 0)
        # Realised volume is an oracle for this identity test, never a trading rule.
        oracle = evaluate([0.1, 0.2, 0.7], [97, 108, 101], [10, 20, 70], 100)
        self.assertAlmostEqual(oracle["absolute_deviation_bps"], 0)
        self.assertAlmostEqual(oracle["volume_share_rmse"], 0)

    def test_better_volume_forecast_can_have_worse_price_tracking(self):
        prices, volumes = [100, 101, 110], [10, 20, 70]
        closer = evaluate([0.1, 0.25, 0.65], prices, volumes, 100)
        farther = evaluate([0.2, 0.1, 0.7], prices, volumes, 100)
        self.assertLess(closer["volume_share_rmse"], farther["volume_share_rmse"])
        self.assertGreater(closer["absolute_deviation_bps"], farther["absolute_deviation_bps"])
        self.assertAlmostEqual(closer["reference_price"] - closer["market_vwap"], -0.45)
        self.assertAlmostEqual(farther["reference_price"] - farther["market_vwap"], -0.1)

    def test_size_scales_quantity_and_participation_not_price_deviation(self):
        args = ([0.2, 0.3, 0.5], [100, 102, 99], [100, 400, 500], 100)
        small, large = evaluate(*args, notional=1_000), evaluate(*args, notional=10_000)
        for key in ("reference_price", "market_vwap", "signed_deviation_bps", "arrival_shortfall_bps"):
            self.assertAlmostEqual(small[key], large[key])
        self.assertAlmostEqual(large["scheduled_base_quantity"], 10 * small["scheduled_base_quantity"])
        self.assertAlmostEqual(large["max_participation"], 10 * small["max_participation"])
        scaled_volume = evaluate(args[0], args[1], np.array(args[2]) * 10, args[3], 1_000)
        self.assertAlmostEqual(scaled_volume["market_vwap"], small["market_vwap"])
        self.assertAlmostEqual(scaled_volume["max_participation"], small["max_participation"] / 10)

    def test_participation_thresholds_mean_strictly_above(self):
        result = evaluate([0.5, 0.5], [100, 104], [1_000, 5_000], 100, 10_000)
        self.assertAlmostEqual(result["max_participation"], 0.05)
        self.assertEqual(result["fraction_above_1pct"], 0.5)
        self.assertEqual(result["fraction_above_5pct"], 0)

    def test_invalid_inputs_are_rejected_instead_of_repaired(self):
        valid = ([0.5, 0.5], [100, 101], [10, 20], 100)
        bad_cases = [
            ([1, 1], *valid[1:]),
            ([-0.1, 1.1], *valid[1:]),
            ([0, 0], *valid[1:]),
            ([np.nan, 0.5], *valid[1:]),
            (valid[0], [0, 101], valid[2], valid[3]),
            (valid[0], [np.inf, 101], valid[2], valid[3]),
            (valid[0], valid[1], [0, 20], valid[3]),
            (valid[0], valid[1], [-1, 20], valid[3]),
            (valid[0], valid[1], [10], valid[3]),
            (*valid[:3], 0),
            ([], [], [], 100),
        ]
        for args in bad_cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluate(*args)
        for notional in (0, -1, np.nan):
            with self.subTest(notional=notional), self.assertRaises(ValueError):
                evaluate(*valid, notional=notional)
        history = np.full((20, 96), 1 / 96)
        for method in ("historical", "ewma"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                schedule(history[:19], method)
        bad_history = history.copy()
        bad_history[0, 0] = -1
        with self.assertRaises(ValueError):
            schedule(bad_history, "historical")
        with self.assertRaises(ValueError):
            schedule(history * 2, "historical")

    def test_block_interval_preserves_constant_samples_and_is_reproducible(self):
        np.testing.assert_allclose(moving_block_interval(np.full(28, -2.5)), [-2.5, -2.5])
        observations = np.arange(28, dtype=float) - 14
        first = moving_block_interval(observations, draws=200, seed=3)
        self.assertEqual(first, moving_block_interval(observations, draws=200, seed=3))
        self.assertLessEqual(first[0], first[1])
        for invalid in (np.arange(6), [np.nan] * 10, np.ones((7, 2))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                moving_block_interval(invalid)

    def test_capped_replay_carries_backlog_and_conserves_quantity(self):
        result = capped_replay([0.5, 0.3, 0.2], [100, 110, 120], [10, 100, 10], 100, 0.1, 1_000)
        # Bucket fills are 1, 7 and 1; their cash total is 990.
        self.assertAlmostEqual(result["scheduled_base_quantity"], 10)
        self.assertAlmostEqual(result["filled_base_quantity"], 9)
        self.assertAlmostEqual(result["unfilled_base_quantity"], 1)
        self.assertAlmostEqual(result["completion_fraction"], 0.9)
        self.assertAlmostEqual(result["reference_filled_price"] * result["filled_base_quantity"], 990)
        self.assertAlmostEqual(
            result["filled_base_quantity"] + result["unfilled_base_quantity"],
            result["scheduled_base_quantity"],
        )
        late = capped_replay([0, 0, 1], [100, 110, 120], [1_000, 1_000, 1], 100, 0.1, 1_000)
        self.assertAlmostEqual(late["filled_base_quantity"], 0.1)
        self.assertAlmostEqual(late["unfilled_base_quantity"], 9.9)
        self.assertAlmostEqual(late["reference_filled_price"], 120)

    def test_partial_fills_have_no_full_order_price_comparison(self):
        args = ([0.5, 0.3, 0.2], [100, 110, 120], [10, 100, 10], 100)
        partial = capped_replay(*args, participation_cap=0.1, notional=1_000)
        self.assertFalse(partial["fully_completed"])
        self.assertIsNone(partial["full_order_deviation_bps"])
        full = capped_replay(*args, participation_cap=0.5, notional=1_000)
        self.assertTrue(full["fully_completed"])
        self.assertAlmostEqual(full["completion_fraction"], 1)
        self.assertAlmostEqual(full["unfilled_base_quantity"], 0)
        self.assertAlmostEqual(full["reference_filled_price"], 107)
        self.assertAlmostEqual(full["full_order_deviation_bps"], (107 - 110) / 110 * 10_000)
        zero = capped_replay(*args, participation_cap=0, notional=1_000)
        self.assertEqual(zero["filled_base_quantity"], 0)
        self.assertIsNone(zero["reference_filled_price"])
        self.assertIsNone(zero["full_order_deviation_bps"])

    def test_relaxing_participation_cap_cannot_reduce_completion(self):
        caps = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
        results = [capped_replay([0.5, 0.3, 0.2], [100, 110, 120], [10, 100, 10],
                                 100, cap, 1_000) for cap in caps]
        fractions = [result["completion_fraction"] for result in results]
        np.testing.assert_allclose(fractions, [0.12, 0.6, 0.9, 1, 1, 1])
        self.assertTrue(np.all(np.diff(fractions) >= 0))
        for invalid_cap in (-0.1, 1.1, np.nan):
            with self.subTest(cap=invalid_cap), self.assertRaises(ValueError):
                capped_replay([0.5, 0.5], [100, 101], [10, 20], 100, invalid_cap)


class FeedbackReleaseTests(unittest.TestCase):
    def test_sufficient_daily_capacity_can_still_leave_a_late_order_unfilled(self):
        late = replay([50, 100], [100, 110], [900, 100], 100, 0.1, 100)
        early = replay([100, 100], [100, 110], [900, 100], 100, 0.1, 100)
        np.testing.assert_allclose(late["fills"], [50, 10])
        np.testing.assert_allclose(early["fills"], [90, 10])
        self.assertAlmostEqual(late["unfilled_base_quantity"], 40)
        self.assertAlmostEqual(late["total_shortage_fraction"], 0)
        self.assertAlmostEqual(late["timing_shortage_fraction"], 0.4)
        self.assertTrue(early["fully_completed"])

    def test_hand_calculated_fills_cash_and_mixed_shortage(self):
        result = replay([4, 7, 10], [100, 110, 120], [10, 100, 10], 10, 0.1, 100)
        np.testing.assert_allclose(result["fills"], [1, 6, 1])
        np.testing.assert_allclose(result["cumulative_fills"], [1, 7, 8])
        np.testing.assert_allclose(result["backlog"], [3, 0, 2])
        np.testing.assert_allclose(result["release_increments"], [4, 3, 3])
        self.assertAlmostEqual(result["reference_filled_price"] * 8, 880)
        self.assertAlmostEqual(result["unfilled_fraction"], 0.2)
        self.assertIsNone(result["full_order_deviation_bps"])
        self.assertIsNone(result["arrival_shortfall_bps"])
        mixed = replay([1, 2, 10], [100, 110, 120], [10, 30, 10], 10, 0.1, 100)
        np.testing.assert_allclose(mixed["fills"], [1, 1, 1])
        self.assertAlmostEqual(mixed["unfilled_fraction"], 0.7)
        self.assertAlmostEqual(mixed["total_shortage_fraction"], 0.5)
        self.assertAlmostEqual(mixed["timing_shortage_fraction"], 0.2)

    def test_suffix_backlog_identity_and_exact_completion_boundary(self):
        # The last three units need 30% of the last bucket's ten-unit volume.
        below = replay([4, 7, 10], [100, 110, 120], [10, 100, 10], 10, 0.299, 100)
        at = replay([4, 7, 10], [100, 110, 120], [10, 100, 10], 10, 0.3, 100)
        self.assertAlmostEqual(below["suffix_backlog"], 0.01)
        self.assertFalse(below["fully_completed"])
        self.assertTrue(at["fully_completed"])
        rng = np.random.default_rng(104)
        for _ in range(30):
            increments = rng.dirichlet(np.ones(8)) * 10
            volumes = rng.uniform(1, 30, 8)
            result = replay(np.cumsum(increments), np.full(8, 100), volumes, 10, 0.1, 100)
            self.assertAlmostEqual(result["suffix_backlog"], result["unfilled_base_quantity"])
            self.assertLess(result["max_suffix_identity_error"], 1e-10)
            self.assertAlmostEqual(result["fills"].sum() + result["unfilled_base_quantity"], 10)
            self.assertTrue(np.all(result["fills"] <= 0.1 * volumes + 1e-12))

    def test_immediate_release_attains_total_capacity_bound(self):
        for volumes, cap in (([10, 20, 30], 0.1), ([100, 20, 30], 0.1), ([1, 2, 3], 0)):
            with self.subTest(volumes=volumes, cap=cap):
                result = replay([10, 10, 10], [100, 101, 102], volumes, 10, cap, 100)
                self.assertAlmostEqual(result["fills"].sum(), min(10, cap * sum(volumes)))
                self.assertAlmostEqual(result["timing_shortage_fraction"], 0)
        zero = replay([10, 10], [100, 101], [10, 10], 10, 0, 100)
        self.assertIsNone(zero["reference_filled_price"])
        self.assertIsNone(zero["full_order_absolute_deviation_bps"])

    def test_feedback_uses_only_preceding_volume_and_keeps_releases_monotone(self):
        weights, mean = np.full(4, 0.25), np.full(4, 100)
        actual = np.array([20, 100, 100, 100])
        baseline = release_path(weights, actual, mean, 100, 0.1, "feedback")
        np.testing.assert_allclose(baseline, [70, 88, 92, 100])
        for first_changed in range(4):
            changed = actual.copy()
            changed[first_changed:] = [1000, 1, 500, 2][:4 - first_changed]
            observed = release_path(weights, changed, mean, 100, 0.1, "feedback")
            np.testing.assert_allclose(observed[:first_changed + 1], baseline[:first_changed + 1])
            self.assertTrue(np.all(np.diff(observed) >= 0))
            self.assertTrue(np.all((observed >= 0) & (observed <= 100)))
            self.assertAlmostEqual(observed[-1], 100)

    def test_feedback_ablation_removes_current_day_information(self):
        weights, mean = np.full(4, 0.25), np.full(4, 100)
        no_information = release_path(weights, [20, 100, 100, 100], mean, 100, 0.1, "no_feedback")
        np.testing.assert_allclose(no_information, [70, 80, 90, 100])
        np.testing.assert_allclose(
            release_path(weights, [900, 1, 900, 1], mean, 100, 0.1, "no_feedback"),
            no_information,
        )
        np.testing.assert_allclose(
            release_path(weights, mean, mean, 100, 0.1, "feedback"), no_information,
        )

    def test_fixed_early_endpoints_and_full_order_price(self):
        args = ([0.2, 0.3, 0.5], [100, 100, 100], [100, 100, 100], 10, 0.1)
        historical = release_path(*args, "historical")
        np.testing.assert_allclose(historical, [2, 5, 10])
        np.testing.assert_allclose(release_path(*args, "fixed_early", alpha=0), historical)
        np.testing.assert_allclose(release_path(*args, "fixed_early", alpha=1), [10, 10, 10])
        np.testing.assert_allclose(release_path(*args, "fixed_early", alpha=0.5), [6, 7.5, 10])
        result = replay([4, 7, 10], [100, 110, 120], [100, 100, 100], 10, 0.1, 100)
        self.assertTrue(result["fully_completed"])
        self.assertAlmostEqual(result["reference_filled_price"], 109)
        self.assertAlmostEqual(result["full_order_deviation_bps"], -10_000 / 110)
        self.assertAlmostEqual(result["arrival_shortfall_bps"], 900)


    def test_invalid_release_paths_are_rejected(self):
        for releases in ([4, 3, 10], [4, 7, 9], [-1, 7, 10], [4, np.nan, 10]):
            with self.subTest(releases=releases), self.assertRaises(ValueError):
                replay(releases, [100, 100, 100], [10, 10, 10], 10, 0.1, 100)


if __name__ == "__main__":
    unittest.main()
