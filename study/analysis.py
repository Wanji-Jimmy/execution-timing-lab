"""Compare fixed execution schedules using historical bucket reference prices."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np


METHODS = ("twap", "historical", "ewma")
START, END = date(2026, 7, 1), date(2026, 9, 30)
BUCKETS, LOOKBACK = 96, 20


def schedule(history, method):
    """Use only previously completed daily volume-share profiles."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or history.shape[1] != BUCKETS:
        raise ValueError("history must have shape (days, 96)")
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("historical shares must be finite and nonnegative")
    if not np.allclose(history.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("each historical profile must sum to one")
    if method == "twap":
        return np.full(BUCKETS, 1 / BUCKETS)
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if len(history) < LOOKBACK:
        raise ValueError("historical and EWMA schedules require 20 prior days")
    recent = history[-LOOKBACK:]
    if method == "historical":
        weights = recent.mean(axis=0)
    else:
        day_weights = 0.8 ** np.arange(LOOKBACK - 1, -1, -1)
        weights = np.average(recent, axis=0, weights=day_weights)
    return weights / weights.sum()


def evaluate(weights, prices, volumes, arrival_price, notional=1_000_000):
    """Evaluate an already-fixed schedule; prices are historical bucket VWAPs."""
    weights, prices, volumes = (
        np.asarray(values, dtype=float) for values in (weights, prices, volumes)
    )
    if any(values.ndim != 1 for values in (weights, prices, volumes)):
        raise ValueError("weights, prices and volumes must be one-dimensional")
    if not len(weights) or weights.shape != prices.shape or prices.shape != volumes.shape:
        raise ValueError("weights, prices and volumes must have equal nonzero length")
    if not all(np.isfinite(values).all() for values in (weights, prices, volumes)):
        raise ValueError("inputs must be finite")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("weights must be nonnegative and sum to one")
    if (prices <= 0).any() or (volumes <= 0).any():
        raise ValueError("prices and volumes must be positive")
    if not np.isfinite([arrival_price, notional]).all() or min(arrival_price, notional) <= 0:
        raise ValueError("arrival price and notional must be finite and positive")
    market_shares = volumes / volumes.sum()
    reference_price = float(weights @ prices)
    market_vwap = float(market_shares @ prices)
    price_difference = reference_price - market_vwap
    attribution = float((weights - market_shares) @ prices)
    residual = attribution - price_difference
    if abs(residual) > 1e-10 * float(prices.max()):
        raise ArithmeticError("price-deviation attribution failed")
    quantity = notional / arrival_price
    participation = quantity * weights / volumes
    signed_bps = 10_000 * price_difference / market_vwap
    return {
        "reference_price": reference_price,
        "market_vwap": market_vwap,
        "signed_deviation_bps": signed_bps,
        "absolute_deviation_bps": abs(signed_bps),
        "arrival_shortfall_bps": 10_000 * (reference_price / arrival_price - 1),
        "volume_share_rmse": float(np.sqrt(np.mean((weights - market_shares) ** 2))),
        "attribution_bps": 10_000 * attribution / market_vwap,
        "attribution_residual": residual,
        "scheduled_base_quantity": quantity,
        "max_participation": float(participation.max()),
        "fraction_above_1pct": float(np.mean(participation > 0.01)),
        "fraction_above_5pct": float(np.mean(participation > 0.05)),
    }


def capped_replay(weights, prices, volumes, arrival_price, participation_cap, notional=1_000_000):
    """Exploratory volume-cap fill assumption; unfilled quantity rolls forward."""
    baseline = evaluate(weights, prices, volumes, arrival_price, notional)
    if not np.isfinite(participation_cap) or not 0 <= participation_cap <= 1:
        raise ValueError("participation cap must be finite and in [0, 1]")
    quantity = baseline["scheduled_base_quantity"]
    backlog, filled_quantity, filled_cost = 0.0, 0.0, 0.0
    for weight, price, volume in zip(weights, prices, volumes):
        attempt = quantity * float(weight) + backlog
        filled = min(attempt, participation_cap * float(volume))
        backlog = max(0.0, attempt - filled)
        filled_quantity += filled
        filled_cost += filled * float(price)
    complete = bool(backlog <= quantity * 1e-12)
    filled_price = filled_cost / filled_quantity if filled_quantity else None
    return {
        "scheduled_base_quantity": quantity,
        "filled_base_quantity": filled_quantity,
        "completion_fraction": min(1.0, max(0.0, filled_quantity / quantity)),
        "unfilled_base_quantity": backlog,
        "reference_filled_price": filled_price,
        "fully_completed": complete,
        "full_order_deviation_bps": (
            10_000 * (filled_price / baseline["market_vwap"] - 1) if complete else None
        ),
    }


def moving_block_interval(differences, block=7, draws=2000, seed=20261025):
    """Percentile interval for a mean using contiguous, noncircular blocks."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not np.isfinite(differences).all():
        raise ValueError("differences must be a finite one-dimensional array")
    if not isinstance(block, (int, np.integer)) or block < 1 or len(differences) < block:
        raise ValueError("block must be a positive integer no larger than sample length")
    if not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(differences) - block + 1,
                         size=(draws, (len(differences) + block - 1) // block))
    indices = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)
    means = differences[indices[:, :len(differences)]].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def load_days(path):
    groups = defaultdict(dict)
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        required = {"asset", "date", "bucket", "open", "base_volume", "quote_volume"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("bucket CSV is missing required columns")
        for row in reader:
            key = (row["asset"], date.fromisoformat(row["date"]))
            bucket = int(row["bucket"])
            if bucket in groups[key] or not 0 <= bucket < BUCKETS:
                raise ValueError(f"duplicate or invalid bucket: {key}, {bucket}")
            values = np.array([float(row[name]) for name in
                               ("open", "base_volume", "quote_volume")])
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"invalid bucket data: {key}, {bucket}")
            groups[key][bucket] = values
    days = defaultdict(list)
    for (asset, day), buckets in sorted(groups.items()):
        if len(buckets) != BUCKETS:
            raise ValueError(f"incomplete day in validated CSV: {asset}, {day}")
        values = np.array([buckets[i] for i in range(BUCKETS)])
        days[asset].append((day, values[:, 0], values[:, 1], values[:, 2]))
    if set(days) != {"BTCUSDT", "ETHUSDT"}:
        raise ValueError("expected exactly BTCUSDT and ETHUSDT")
    return days


def run(data_path, output_dir):
    rows, skipped = [], []
    for asset, days in sorted(load_days(data_path).items()):
        history = []
        for day, opens, base_volume, quote_volume in days:
            if START <= day <= END:
                if len(history) < LOOKBACK:
                    skipped.append({"asset": asset, "date": day.isoformat(),
                                    "reason": "fewer than 20 prior valid days"})
                else:
                    for method in METHODS:
                        weights = schedule(np.asarray(history[-LOOKBACK:]), method)
                        result = evaluate(weights, quote_volume / base_volume,
                                          base_volume, opens[0])
                        for label, cap in (("1pct", 0.01), ("5pct", 0.05)):
                            capped = capped_replay(weights, quote_volume / base_volume,
                                                   base_volume, opens[0], cap)
                            result.update({f"cap_{label}_{key}": value for key, value in capped.items()
                                           if key != "scheduled_base_quantity"})
                        rows.append({"asset": asset, "date": day.isoformat(),
                                     "method": method, **result})
            # The current profile becomes available only after every decision above.
            history.append(base_volume / base_volume.sum())
    if not rows:
        raise ValueError("no eligible evaluation days")
    summary = {
        "protocol": "docs/protocol.md",
        "evaluation_start": START.isoformat(), "evaluation_end": END.isoformat(),
        "notional_usdt": 1_000_000, "lookback_valid_days": LOOKBACK, "ewma_alpha": 0.2,
        "liquidity_stress": {
            "status": "Exploratory extension added after inspecting the initial participation diagnostics.",
            "caps": [0.01, 0.05],
            "assumption": "Fill up to cap times historical bucket volume at bucket VWAP; carry backlog; leave deadline residual unfilled.",
            "limitation": "Counterfactual volume-cap fills, not evidence of actual queue access or executable capacity. Partial-order prices are not ranked.",
        },
        "bootstrap": {"block_days": 7, "draws": 2000, "seed": 20261025,
                      "interval": "95% descriptive noncircular moving-block percentile"},
        "interpretation": {
            "signed_deviation": "Positive means a higher reference purchase price than market VWAP; negative means lower.",
            "absolute_deviation": "Closeness to market VWAP, not profit or net cost saving.",
            "arrival_shortfall": "Positive means a higher reference purchase price than the 08:00 UTC open.",
            "participation": "Scheduled base quantity / historical bucket base volume; no demonstrated fill capacity.",
            "limitations": "Historical bucket VWAP completion assumed. No spread, fee, impact, queue or routing model.",
        },
        "skipped_warmup": skipped, "assets": {},
    }
    for asset in sorted({row["asset"] for row in rows}):
        methods = {method: [r for r in rows if r["asset"] == asset and r["method"] == method]
                   for method in METHODS}
        asset_result = {"days": len(methods["twap"]), "methods": {}, "comparisons_vs_twap": {}}
        asset_result["liquidity_stress"] = {label: {} for label in ("1pct", "5pct")}
        for method, records in methods.items():
            metrics = {"mean_" + key: float(np.mean([r[key] for r in records])) for key in
                       ("absolute_deviation_bps", "signed_deviation_bps", "arrival_shortfall_bps",
                        "volume_share_rmse", "fraction_above_1pct", "fraction_above_5pct")}
            metrics.update({
                "max_participation": max(r["max_participation"] for r in records),
                "max_absolute_attribution_residual": max(abs(r["attribution_residual"]) for r in records),
            })
            asset_result["methods"][method] = metrics
            for label in ("1pct", "5pct"):
                prefix = f"cap_{label}_"
                completion = [r[prefix + "completion_fraction"] for r in records]
                asset_result["liquidity_stress"][label][method] = {
                    "days_fully_completed": sum(r[prefix + "fully_completed"] for r in records),
                    "mean_completion_fraction": float(np.mean(completion)),
                    "worst_completion_fraction": min(completion),
                }
        for method in METHODS[1:]:
            reference, records = methods["twap"], methods[method]
            if [r["date"] for r in reference] != [r["date"] for r in records]:
                raise ValueError("unpaired dates")
            differences = [r["absolute_deviation_bps"] - t["absolute_deviation_bps"]
                           for r, t in zip(records, reference)]
            reversal = [(r, t) for r, t in zip(records, reference)
                        if r["volume_share_rmse"] < t["volume_share_rmse"]
                        and r["absolute_deviation_bps"] > t["absolute_deviation_bps"]]
            comparison = {
                "mean_absolute_error_difference_bps": float(np.mean(differences)),
                "descriptive_interval_bps": moving_block_interval(differences),
                "days_lower_absolute_deviation": sum(value < 0 for value in differences),
                "days_better_volume_forecast_but_worse_tracking": len(reversal),
            }
            if reversal:
                r, t = reversal[0]
                comparison["first_counterexample"] = {
                    "date": r["date"],
                    "method_volume_rmse": r["volume_share_rmse"],
                    "twap_volume_rmse": t["volume_share_rmse"],
                    "method_absolute_deviation_bps": r["absolute_deviation_bps"],
                    "twap_absolute_deviation_bps": t["absolute_deviation_bps"],
                }
            asset_result["comparisons_vs_twap"][method] = comparison
        summary["assets"][asset] = asset_result
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "daily_results.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "buckets.csv")
    parser.add_argument("--output", type=Path, default=root / "results/crypto")
    args = parser.parse_args()
    result = run(args.data, args.output)
    for asset, metrics in result["assets"].items():
        print(f"{asset}: {metrics['days']} paired days; outputs saved to {args.output}")
