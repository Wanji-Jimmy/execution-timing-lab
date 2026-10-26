"""Test whether volume feedback adds value beyond releasing orders earlier."""

import argparse
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.analysis import BUCKETS, LOOKBACK, load_days, moving_block_interval


METHODS = ("twap", "historical", "fixed_early", "feedback")
CAPS = (0.01, 0.05)
PERIODS = {
    "calibration": (date(2026, 7, 1), date(2026, 8, 31)),
    "seen_check": (date(2026, 9, 1), date(2026, 9, 30)),
    "validation": (date(2026, 10, 1), date(2026, 10, 23)),
}


def _vectors(*values):
    arrays = [np.asarray(value, dtype=float) for value in values]
    if any(a.ndim != 1 or not len(a) for a in arrays):
        raise ValueError("inputs must be nonempty one-dimensional arrays")
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("inputs must have equal lengths and finite values")
    return arrays


def _order(quantity, cap):
    if not np.isfinite([quantity, cap]).all() or quantity <= 0 or not 0 <= cap <= 1:
        raise ValueError("quantity must be positive and cap must be in [0, 1]")


def release_path(weights, volumes, mean_volumes, quantity, cap, method, alpha=0.0):
    """Cumulative eligible quantity; bucket t sees only volumes before t."""
    weights, volumes, means = _vectors(weights, volumes, mean_volumes)
    _order(quantity, cap)
    if (weights < 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("weights must be nonnegative and sum to one")
    if (volumes < 0).any() or (means <= 0).any():
        raise ValueError("observed volumes must be nonnegative; historical means positive")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    cumulative = np.cumsum(weights)
    if method == "twap":
        releases = quantity * np.arange(1, len(weights) + 1) / len(weights)
    elif method == "historical":
        releases = quantity * cumulative
    elif method == "fixed_early":
        releases = quantity * (alpha + (1 - alpha) * cumulative)
    elif method in ("feedback", "no_feedback"):
        activity = np.ones(len(weights))
        if method == "feedback" and len(weights) > 1:
            activity[1:] = (1 + np.cumsum(volumes[:-1]) / np.cumsum(means[:-1])) / 2
        later_mean_volume = np.r_[np.cumsum(means[:0:-1])[::-1], 0.0]
        required_release = quantity - cap * activity * later_mean_volume
        releases = np.maximum.accumulate(np.maximum(quantity * cumulative, required_release))
    else:
        raise ValueError(f"unknown release method: {method}")
    releases = np.clip(releases, 0, quantity)
    releases[-1] = quantity
    return releases


def replay(releases, prices, volumes, quantity, cap, arrival_price):
    """Historical bucket-VWAP fills under an assumed participation cap."""
    releases, prices, volumes = _vectors(releases, prices, volumes)
    _order(quantity, cap)
    tolerance = quantity * 1e-10
    if (prices <= 0).any() or (volumes <= 0).any():
        raise ValueError("replay prices and volumes must be positive")
    if not np.isfinite(arrival_price) or arrival_price <= 0:
        raise ValueError("arrival price must be finite and positive")
    increments = np.diff(np.r_[0.0, releases])
    if (increments < -tolerance).any() or (releases > quantity + tolerance).any():
        raise ValueError("cumulative releases must increase within the order quantity")
    if abs(releases[-1] - quantity) > tolerance:
        raise ValueError("the full order must be released by the final bucket")
    fills = np.empty(len(prices))
    filled = 0.0
    for t in range(len(prices)):
        fills[t] = min(max(0.0, releases[t] - filled), cap * volumes[t])
        filled += fills[t]
    cumulative_fills = np.cumsum(fills)
    backlog = np.maximum(0, releases - cumulative_fills)
    unfilled = max(0.0, quantity - filled)
    suffix = max(0.0, float(np.cumsum((increments - cap * volumes)[::-1]).max()))
    unavoidable = max(0.0, quantity - cap * volumes.sum())
    identity_error = abs(unfilled - suffix)
    if identity_error > tolerance or unfilled < unavoidable - tolerance:
        raise ArithmeticError("suffix backlog or immediate-release capacity bound failed")
    if (fills < -tolerance).any() or (fills > cap * volumes + tolerance).any():
        raise ArithmeticError("fill capacity violated")
    if abs(filled + unfilled - quantity) > tolerance:
        raise ArithmeticError("order quantity conservation failed")
    complete = bool(unfilled <= quantity * 1e-12)
    reference = float(fills @ prices / filled) if filled else None
    vwap = float(volumes @ prices / volumes.sum())
    signed = 10_000 * (reference / vwap - 1) if complete else None
    return {
        "fills": fills, "cumulative_fills": cumulative_fills, "backlog": backlog,
        "release_increments": increments,
        "scheduled_base_quantity": float(quantity), "filled_base_quantity": float(filled),
        "unfilled_base_quantity": float(unfilled), "unfilled_fraction": float(unfilled / quantity),
        "total_shortage_fraction": float(unavoidable / quantity),
        "timing_shortage_fraction": float(max(0.0, unfilled - unavoidable) / quantity),
        "fully_completed": complete, "suffix_backlog": suffix,
        "max_suffix_identity_error": identity_error,
        "market_vwap": vwap, "reference_filled_price": reference,
        "full_order_deviation_bps": signed,
        "full_order_absolute_deviation_bps": abs(signed) if complete else None,
        "arrival_shortfall_bps": 10_000 * (reference / arrival_price - 1) if complete else None,
        "weighted_participation": float(fills @ (fills / volumes) / filled) if filled else None,
        "max_participation": float(np.max(fills / volumes)),
    }


def fit_alpha(records):
    """Match cumulative advance using July-August only, without price outcomes."""
    numerator = denominator = 0.0
    for record in records:
        day = date.fromisoformat(record["date"]) if isinstance(record["date"], str) else record["date"]
        if not PERIODS["calibration"][0] <= day <= PERIODS["calibration"][1]:
            continue
        cumulative = np.cumsum(record["weights"])
        releases = release_path(*(record[k] for k in
                                  ("weights", "volumes", "mean_volumes", "quantity", "cap")),
                                method="feedback")
        numerator += float(np.sum(releases / record["quantity"] - cumulative))
        denominator += float(np.sum(1 - cumulative))
    if denominator <= 0:
        raise ValueError("matching alpha needs eligible July-August records")
    alpha = numerator / denominator
    if not -1e-10 <= alpha <= 1 + 1e-10:
        raise ArithmeticError("matched alpha outside [0, 1]")
    return float(np.clip(alpha, 0, 1))


def _dates(start, end):
    return {start + timedelta(days=i) for i in range((end - start).days + 1)}


def _records(data_path, validation_path):
    old, new = load_days(data_path), load_days(validation_path)
    expected = _dates(*PERIODS["validation"])
    manifest_path = Path(validation_path).with_name("validation_manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest_path.exists():
        if manifest.get("status") != "complete" or manifest.get("errors"):
            raise ValueError("validation download manifest is incomplete or has errors")
        sources = manifest.get("sources", [])
        source_dates = [(s["asset"], date.fromisoformat(s["date"])) for s in sources]
        required_sources = {(asset, day) for asset in old for day in expected}
        if len(source_dates) != len(required_sources) or set(source_dates) != required_sources:
            raise ValueError("validation manifest must cover every required daily archive once")
        if any(s.get("status") != "complete" or s.get("checksum_verified") is not True for s in sources):
            raise ValueError("validation source checksums or source status not verified")
        expected_hash = manifest.get("buckets_sha256")
        if expected_hash and expected_hash != hashlib.sha256(Path(validation_path).read_bytes()).hexdigest():
            raise ValueError("validation CSV differs from its manifest hash")
    exclusions = [item for source in manifest.get("sources", [])
                  for item in source.get("excluded_days", [])]
    exclusion_keys = [(item["asset"], date.fromisoformat(item["date"])) for item in exclusions]
    if len(exclusion_keys) != len(set(exclusion_keys)) or any(asset not in old or day not in expected for asset, day in exclusion_keys):
        raise ValueError("duplicate or unexpected validation exclusions")
    records = []
    for asset in sorted(old):
        if any(not date(2026, 6, 1) <= day <= date(2026, 9, 30) for day, *_ in old[asset]):
            raise ValueError("historical input contains dates outside June-September")
        observed = {day for day, *_ in new[asset]}
        excluded = {date.fromisoformat(item["date"]) for item in exclusions if item["asset"] == asset}
        if observed & excluded or observed | excluded != expected:
            raise ValueError(f"validation dates missing, unexpected or inconsistently excluded: {asset}")
        history = []
        for day, opens, volume, quote in old[asset] + new[asset]:
            if day >= PERIODS["calibration"][0]:
                if len(history) < LOOKBACK:
                    raise ValueError("fewer than 20 past valid days")
                recent = np.asarray(history[-LOOKBACK:])
                weights = (recent / recent.sum(axis=1, keepdims=True)).mean(axis=0)
                weights /= weights.sum()
                records.append({"asset": asset, "date": day, "weights": weights,
                                "mean_volumes": recent.mean(axis=0), "volumes": volume,
                                "prices": quote / volume, "quantity": 1_000_000 / opens[0],
                                "arrival_price": opens[0]})
            history.append(volume)
    return records, exclusions


def _summary(rows):
    output = {}
    for period in PERIODS:
        output[period] = {}
        for asset in sorted({r["asset"] for r in rows}):
            output[period][asset] = {}
            for cap in CAPS:
                subset = [r for r in rows if (r["period"], r["asset"], r["cap"]) == (period, asset, cap)]
                by_method = {method: [r for r in subset if r["method"] == method]
                             for method in (*METHODS, "no_feedback")}
                dates = [r["date"] for r in by_method["historical"]]
                if not dates:
                    raise ValueError("an evaluation group has no valid dates")
                common = set(dates)
                metrics = {}
                for method, observations in by_method.items():
                    if [r["date"] for r in observations] != dates:
                        raise ValueError("unpaired method dates")
                    if method in METHODS:
                        common &= {r["date"] for r in observations if r["fully_completed"]}
                    metrics[method] = {
                        "days_completed": sum(r["fully_completed"] for r in observations),
                        "mean_unfilled_pct": 100 * float(np.mean([r["unfilled_fraction"] for r in observations])),
                        "worst_unfilled_pct": 100 * max(r["unfilled_fraction"] for r in observations),
                        "total_capacity_shortage_days": sum(r["total_shortage_fraction"] > 1e-12 for r in observations),
                        "timing_only_failure_days": sum(not r["fully_completed"] and r["total_shortage_fraction"] <= 1e-12 for r in observations),
                        "mean_timing_shortage_pct": 100 * float(np.mean([r["timing_shortage_fraction"] for r in observations])),
                        "mean_cumulative_advance_pct": 100 * float(np.mean([r["mean_cumulative_advance"] for r in observations])),
                    }
                for method in METHODS:
                    conditional = [r for r in by_method[method] if r["date"] in common]
                    metrics[method]["common_complete_prices"] = {
                        key: float(np.mean([r[key] for r in conditional])) if conditional else None
                        for key in ("full_order_deviation_bps", "full_order_absolute_deviation_bps",
                                    "arrival_shortfall_bps", "weighted_participation")}
                feedback, fixed, historical, ablation = [by_method[m] for m in
                                                         ("feedback", "fixed_early", "historical", "no_feedback")]
                differences = [100 * (a["unfilled_fraction"] - b["unfilled_fraction"])
                               for a, b in zip(feedback, fixed)]
                consecutive = all(date.fromisoformat(b) - date.fromisoformat(a) == timedelta(days=1)
                                  for a, b in zip(dates, dates[1:]))
                interval = moving_block_interval(differences) if consecutive and len(dates) >= 7 else None
                diagnostics = {
                    "trigger_days": sum(r["triggered"] for r in feedback),
                    "rescued_vs_historical_days": sum(f["fully_completed"] and not h["fully_completed"] for f, h in zip(feedback, historical)),
                    "false_alarm_days": sum(f["triggered"] and h["fully_completed"] for f, h in zip(feedback, historical)),
                    "triggered_but_incomplete_days": sum(f["triggered"] and not f["fully_completed"] for f in feedback),
                    "untriggered_timing_failure_days": sum(not f["triggered"] and not f["fully_completed"] and f["total_shortage_fraction"] <= 1e-12 for f in feedback),
                    "triggered_timing_failure_days": sum(f["triggered"] and not f["fully_completed"] and f["total_shortage_fraction"] <= 1e-12 for f in feedback),
                    "feedback_complete_fixed_incomplete_days": sum(f["fully_completed"] and not b["fully_completed"] for f, b in zip(feedback, fixed)),
                    "fixed_complete_feedback_incomplete_days": sum(b["fully_completed"] and not f["fully_completed"] for f, b in zip(feedback, fixed)),
                    "feedback_complete_ablation_incomplete_days": sum(f["fully_completed"] and not b["fully_completed"] for f, b in zip(feedback, ablation)),
                    "ablation_complete_feedback_incomplete_days": sum(b["fully_completed"] and not f["fully_completed"] for f, b in zip(feedback, ablation)),
                }
                output[period][asset][str(cap)] = {
                    "days": len(dates), "common_complete_days": len(common), "common_complete_dates": sorted(common),
                    "methods": metrics, "feedback_diagnostics": diagnostics,
                    "feedback_minus_fixed_mean_unfilled_percentage_points": float(np.mean(differences)),
                    "feedback_minus_ablation_mean_unfilled_percentage_points": 100 * float(np.mean([
                        f["unfilled_fraction"] - b["unfilled_fraction"] for f, b in zip(feedback, ablation)])),
                    "feedback_minus_fixed_mean_advance_percentage_points": metrics["feedback"]["mean_cumulative_advance_pct"] - metrics["fixed_early"]["mean_cumulative_advance_pct"],
                    "descriptive_95pct_interval_percentage_points": interval,
                    "interval_note": "Paired calendar-day blocks, separately by asset." if interval is not None else "Not computed: fewer than seven days or calendar gaps.",
                }
    return output


def run(data_path, validation_path, output_dir):
    records, exclusions = _records(data_path, validation_path)
    alphas, rows = {}, []
    for asset in sorted({r["asset"] for r in records}):
        alphas[asset] = {}
        for cap in CAPS:
            contexts = [dict(record, cap=cap) for record in records if record["asset"] == asset]
            alpha = fit_alpha(contexts)
            alphas[asset][str(cap)] = alpha
            for record in contexts:
                day = record["date"]
                period = next(name for name, (start, end) in PERIODS.items() if start <= day <= end)
                baseline = np.cumsum(record["weights"]) * record["quantity"]
                for method in (*METHODS, "no_feedback"):
                    releases = release_path(*(record[k] for k in
                                              ("weights", "volumes", "mean_volumes", "quantity", "cap")),
                                            method, alpha)
                    result = replay(releases, record["prices"], record["volumes"],
                                    record["quantity"], cap, record["arrival_price"])
                    advance = (releases - baseline) / record["quantity"]
                    triggers = np.flatnonzero(advance > 1e-10)
                    scalar = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
                    rows.append({"asset": asset, "date": day.isoformat(), "period": period,
                                 "cap": cap, "method": method, "fixed_alpha": alpha,
                                 "mean_cumulative_advance": float(advance.mean()),
                                 "triggered": bool(len(triggers)),
                                 "first_trigger_bucket": int(triggers[0]) if len(triggers) else None,
                                 **scalar})
    paths = {"data/buckets.csv": Path(data_path), "data/validation_buckets.csv": Path(validation_path),
             "docs/protocol.md": Path(__file__).resolve().parents[1] / "docs/protocol.md",
             "study/adaptive.py": Path(__file__)}
    for label, path in (("data/manifest.json", Path(data_path).with_name("manifest.json")),
                        ("data/validation_manifest.json", Path(validation_path).with_name("validation_manifest.json"))):
        if path.exists():
            paths[label] = path
    period_results = _summary(rows)
    summary = {
        "protocol": "docs/protocol.md: Feedback release study", "notional_usdt": 1_000_000,
        "lookback_valid_days": LOOKBACK, "buckets_per_day": BUCKETS, "activity_smoothing": 0.5,
        "periods": {name: [a.isoformat(), b.isoformat()] for name, (a, b) in PERIODS.items()},
        "fixed_alpha": alphas, "validation_exclusions": exclusions,
        "calibration_advance_matching_error_percentage_points": {
            asset: {cap: group["feedback_minus_fixed_mean_advance_percentage_points"]
                    for cap, group in groups.items()}
            for asset, groups in period_results["calibration"].items()},
        "sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in paths.items()},
        "bootstrap": {"calendar_block_days": 7, "draws": 2000, "seed": 20261025,
                      "scope": "Descriptive, paired per asset; assets are never pooled as independent days."},
        "limitations": "Historical bucket-VWAP fills and exogenous participation caps are assumptions. No actual impact, spread, fees, queue access or profit is measured. Prices use only common-complete dates. July-September informed design; October is newly reserved historical validation, not a live trial.",
        "diagnostic_definitions": {
            "trigger": "Cumulative release exceeds historical release by more than 1e-10 of Q.",
            "false_alarm": "Triggered on a day historical already completes; does not imply a worse price.",
            "timing_failure": "Incomplete despite sufficient full-window capacity; a trigger may have come too late or been too small.",
            "weighted_participation": "Sum(fill * fill / historical_volume) / sum(fill); concentration proxy, not actual cost.",
        },
        "period_results": period_results,
        "max_suffix_identity_error": max(r["max_suffix_identity_error"] for r in rows),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "adaptive_daily.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "adaptive_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data/buckets.csv")
    parser.add_argument("--validation", type=Path, default=root / "data/validation_buckets.csv")
    parser.add_argument("--output", type=Path, default=root / "results/crypto")
    args = parser.parse_args()
    if not args.validation.with_name("validation_manifest.json").exists():
        parser.error("validation_manifest.json is required for the research CLI")
    results = run(args.data, args.validation, args.output)
    print("October validation: completed days (TWAP / historical / fixed early / feedback)")
    for asset, groups in results["period_results"]["validation"].items():
        for cap, group in groups.items():
            completed = " / ".join(str(group["methods"][method]["days_completed"]) for method in METHODS)
            difference = group["feedback_minus_fixed_mean_unfilled_percentage_points"]
            triggers = group["feedback_diagnostics"]["trigger_days"]
            print(f"{asset} cap={float(cap):.0%}: {completed} of {group['days']}; "
                  f"feedback-minus-fixed unfilled={difference:+.4f} pp; triggers={triggers}")
    print("See docs/protocol.md for the earlier crypto study; outputs: adaptive_results.json and adaptive_daily.csv.")
