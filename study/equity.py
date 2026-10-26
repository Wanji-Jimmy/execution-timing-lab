"""Compare fixed release paths under two reported-volume capacity definitions."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.adaptive import METHODS, release_path, replay


ASSETS = ("AAPL", "MSFT", "AMZN", "INTC")
CAPS, SCOPES = (0.01, 0.05), ("total", "exchange")
LOOKBACK, BUCKETS, NOTIONAL, ALPHA = 20, 78, 1_000_000, 0.10
PAIRS = {"exchange": ("Volume", "VolumeWeightPrice"),
         "finra": ("FinraVolume", "FinraVolumeWeightPrice"),
         "total": ("TotalVolume", "TotalVolumeWeightPrice")}
REQUIRED = {"Date", "Ticker", "TimeBarStart", "FirstTradeTime", "FirstTradePrice"}
REQUIRED.update(field for pair in PAIRS.values() for field in pair)


def sessions():
    start, end = date(2021, 4, 1), date(2021, 6, 30)
    holidays = {date(2021, 4, 2), date(2021, 5, 31)}
    return [day for i in range((end - start).days + 1)
            if (day := start + timedelta(days=i)).weekday() < 5 and day not in holidays]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_session(path, asset, day):
    """Require every selected minute; zero-volume minutes contribute zero turnover."""
    path = Path(path)
    report = {"asset": asset, "date": day.isoformat(), "file": path.name,
              "outside_window_rows": 0, "zero_volume_minutes": dict.fromkeys(PAIRS, 0)}
    reasons, minutes = Counter(), {}
    if not path.exists():
        return None, {**report, "status": "excluded", "reasons": {"missing_file": 1}}
    report["sha256"] = sha256(path)
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        if len(fields) != 90 or len(set(fields)) != 90 or not REQUIRED.issubset(fields):
            return None, {**report, "status": "excluded", "reasons": {"invalid_schema": 1}}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                reasons["wrong_column_count"] += 1
                continue
            if row["Ticker"] != asset or row["Date"] != day.strftime("%Y%m%d"):
                reasons["wrong_asset_or_date"] += 1
                continue
            try:
                hour, minute = map(int, row["TimeBarStart"].split(":"))
                if not (0 <= hour < 24 and 0 <= minute < 60):
                    raise ValueError
            except ValueError:
                reasons["invalid_minute_time"] += 1
                continue
            slot = hour * 60 + minute - 570
            if not 0 <= slot < 390:
                report["outside_window_rows"] += 1
                continue
            if slot in minutes:
                reasons["duplicate_minutes"] += 1
                continue
            minutes[slot] = None
            try:
                pairs = {}
                for scope, (volume_field, price_field) in PAIRS.items():
                    volume = float(row[volume_field])
                    if not np.isfinite(volume) or volume < 0 or not volume.is_integer():
                        raise ValueError("invalid_volume")
                    raw_price = row[price_field]
                    price = float(raw_price) if raw_price else 0.0
                    if not np.isfinite(price) or price < 0 or (volume > 0 and price <= 0):
                        raise ValueError("invalid_vwap")
                    report["zero_volume_minutes"][scope] += int(volume == 0)
                    pairs[scope] = (volume, volume * price)
                if pairs["total"][0] != pairs["exchange"][0] + pairs["finra"][0]:
                    raise ValueError("volume_identity_failed")
                minutes[slot] = pairs
                if slot == 0:
                    arrival = float(row["FirstTradePrice"])
                    if not np.isfinite(arrival) or arrival <= 0 or not row["FirstTradeTime"].startswith("09:30:"):
                        raise ValueError("invalid_arrival_trade")
            except (ValueError, OverflowError) as exc:
                reason = str(exc)
                reasons[reason if reason in {"invalid_volume", "invalid_vwap", "volume_identity_failed",
                                            "invalid_arrival_trade"} else "invalid_number"] += 1
    missing = sorted(set(range(390)) - minutes.keys())
    report.update(observed_minutes=len(minutes), missing_minute_slots=missing)
    if missing:
        reasons["missing_minutes"] = len(missing)
    if reasons:
        return None, {**report, "status": "excluded", "reasons": dict(reasons)}
    volumes, prices = {}, {}
    for scope in PAIRS:
        pairs = np.asarray([minutes[i][scope] for i in range(390)]).reshape(BUCKETS, 5, 2).sum(axis=1)
        volumes[scope] = pairs[:, 0]
        prices[scope] = np.divide(pairs[:, 1], pairs[:, 0], out=np.zeros(BUCKETS), where=pairs[:, 0] > 0)
        if scope in SCOPES and (volumes[scope] <= 0).any():
            reasons[f"nonpositive_{scope}_buckets"] = int((volumes[scope] <= 0).sum())
    if reasons:
        return None, {**report, "status": "excluded", "reasons": dict(reasons)}
    result = {"asset": asset, "date": day, "arrival_price": arrival,
              "volumes": volumes, "prices": prices}
    return result, {**report, "status": "valid", "reasons": {},
                    "total_volume": float(volumes["total"].sum()),
                    "exchange_volume": float(volumes["exchange"].sum())}


def build_records(observations):
    """Warm-up stays on its fixed dates; each day's history excludes that day."""
    records, skipped = [], []
    evaluation = set(sessions()[LOOKBACK:])
    for asset in ASSETS:
        history, history_dates = [], []
        days = sorted((x for x in observations if x["asset"] == asset), key=lambda x: x["date"])
        if len({x["date"] for x in days}) != len(days):
            raise ValueError("duplicate asset session")
        for item in days:
            day, volume = item["date"], item["volumes"]["total"]
            if day not in sessions():
                raise ValueError("session outside the frozen sample")
            if day in evaluation:
                if len(history) < LOOKBACK:
                    skipped.append({"asset": asset, "date": day.isoformat(), "prior_valid_sessions": len(history),
                                    "reason": "fewer_than_20_prior_valid_sessions"})
                else:
                    recent = np.asarray(history[-LOOKBACK:])
                    weights = (recent / recent.sum(axis=1, keepdims=True)).mean(axis=0)
                    records.append({**item, "weights": weights / weights.sum(),
                                    "mean_volumes": recent.mean(axis=0),
                                    "quantity": NOTIONAL / item["arrival_price"],
                                    "history_dates": [d.isoformat() for d in history_dates[-LOOKBACK:]]})
            history.append(volume)
            history_dates.append(day)
    return records, skipped


def fixed_schedule_capacity(fractions, volumes, cap):
    """Ex-post deadline limit, only for release fractions independent of order size."""
    fractions, volumes = np.asarray(fractions, dtype=float), np.asarray(volumes, dtype=float)
    if (fractions.ndim != 1 or not len(fractions) or fractions.shape != volumes.shape
            or not np.isfinite(fractions).all() or not np.isfinite(volumes).all()
            or (volumes < 0).any() or (fractions < 0).any() or (fractions > 1).any()
            or (np.diff(fractions) < 0).any() or abs(fractions[-1] - 1) > 1e-12
            or not np.isfinite(cap) or not 0 <= cap <= 1):
        raise ValueError("invalid fixed schedule or capacity")
    remaining = 1 - np.r_[0.0, fractions[:-1]]
    limits = np.divide(cap * np.cumsum(volumes[::-1])[::-1], remaining,
                       out=np.full(len(volumes), np.inf), where=remaining > 0)
    bottleneck = int(np.argmin(limits))
    return float(limits[bottleneck]), bottleneck


def evaluate_record(record, cap):
    total = record["volumes"]["total"]
    prices, quantity = record["prices"]["total"], record["quantity"]
    benchmark = float(total @ prices / total.sum())
    rows = []
    for method in METHODS:
        releases = release_path(record["weights"], total, record["mean_volumes"], quantity,
                                cap, method, alpha=ALPHA)
        release_hash = hashlib.sha256(releases.astype("<f8").tobytes()).hexdigest()
        for scope in SCOPES:
            result = replay(releases, prices, record["volumes"][scope], quantity, cap, record["arrival_price"])
            limit, bottleneck = (fixed_schedule_capacity(releases / quantity, record["volumes"][scope], cap)
                                 if method != "feedback" else (None, None))
            # Only capacity changes: both runs keep the Total-field price benchmark.
            deviation = 10_000 * (result["reference_filled_price"] / benchmark - 1) if result["fully_completed"] else None
            result.update(market_vwap=benchmark, full_order_deviation_bps=deviation,
                          full_order_absolute_deviation_bps=abs(deviation) if deviation is not None else None)
            scalar = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
            rows.append({"asset": record["asset"], "date": record["date"].isoformat(), "cap": cap,
                         "method": method, "capacity_scope": scope, "fixed_alpha": ALPHA,
                         "release_sha256": release_hash,
                         "fixed_schedule_max_notional_usd": limit * record["arrival_price"] if limit is not None else None,
                         "total_capacity_notional_usd": float(cap * record["volumes"][scope].sum() * record["arrival_price"]),
                         "bottleneck_start_bucket": bottleneck, **scalar})
    return rows


def summarize(rows):
    output = {}
    for asset in ASSETS:
        output[asset] = {}
        for cap in CAPS:
            groups = {(method, scope): [r for r in rows if (r["asset"], r["cap"], r["method"], r["capacity_scope"])
                                        == (asset, cap, method, scope)] for method in METHODS for scope in SCOPES}
            dates = [r["date"] for r in groups[(METHODS[0], SCOPES[0])]]
            if len(set(dates)) != len(dates) or any([r["date"] for r in group] != dates for group in groups.values()):
                raise ValueError("unpaired or duplicate policy/scope dates")
            common = set(dates)
            for group in groups.values():
                common &= {r["date"] for r in group if r["fully_completed"]}
            metrics = {scope: {} for scope in SCOPES}
            for (method, scope), group in groups.items():
                conditional = [r for r in group if r["date"] in common]
                limits = [r["fixed_schedule_max_notional_usd"] for r in group
                          if r["fixed_schedule_max_notional_usd"] is not None]
                metrics[scope][method] = {
                    "days_completed": sum(r["fully_completed"] for r in group),
                    "mean_unfilled_pct": 100 * float(np.mean([r["unfilled_fraction"] for r in group])) if group else None,
                    "worst_unfilled_pct": 100 * max(r["unfilled_fraction"] for r in group) if group else None,
                    "mean_timing_shortage_pct": 100 * float(np.mean([r["timing_shortage_fraction"] for r in group])) if group else None,
                    "capacity_shortage_days": sum(r["total_shortage_fraction"] > 1e-12 for r in group),
                    "common_complete_mean_absolute_deviation_bps": float(np.mean([r["full_order_absolute_deviation_bps"] for r in conditional])) if conditional else None,
                    "common_complete_mean_signed_deviation_bps": float(np.mean([r["full_order_deviation_bps"] for r in conditional])) if conditional else None,
                    "fixed_schedule_max_notional_usd_min": min(limits) if limits else None,
                    "fixed_schedule_max_notional_usd_median": float(np.median(limits)) if limits else None,
                    "fixed_schedule_headroom_multiple_min": min(limits) / NOTIONAL if limits else None,
                    "fixed_schedule_headroom_multiple_median": float(np.median(limits)) / NOTIONAL if limits else None,
                }
            output[asset][str(cap)] = {"evaluation_days": len(dates), "common_complete_days": len(common),
                                       "common_complete_dates": sorted(common), "scopes": metrics}
    return output


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ASSETS), 2, figsize=(12, 3 * len(ASSETS) + 1),
                             squeeze=False, constrained_layout=True)
    labels = ("TWAP", "Historical", "Fixed early", "Feedback")
    for i, asset in enumerate(ASSETS):
        group = summary[asset]["0.01"]
        for j, methods in enumerate((METHODS[:3], METHODS)):
            ax = axes[i, j]
            for offset, scope, color in ((-0.2, "total", "#286483"), (0.2, "exchange", "#cf8941")):
                key = "fixed_schedule_max_notional_usd_median" if j == 0 else "common_complete_mean_absolute_deviation_bps"
                values = [group["scopes"][scope][m][key] for m in methods]
                if all(v is not None for v in values):
                    values = np.asarray(values) / (1_000_000 if j == 0 else 1)
                    errors = [values - np.asarray([group["scopes"][scope][m]["fixed_schedule_max_notional_usd_min"] for m in methods]) / 1_000_000,
                              np.zeros(len(methods))] if j == 0 else None
                    ax.bar(np.arange(len(methods)) + offset, values, 0.38, color=color, yerr=errors, capsize=4,
                           label="Total-reported volume" if scope == "total" else "Exchange-reported volume")
            if j == 0:
                ax.axhline(1, color="#333333", linestyle="--", linewidth=1)
            ax.set(title=f'{asset} | 1% cap | {"deadline limit" if j == 0 else "price comparison"}',
                   ylabel="Median maximum notional (USD million)" if j == 0 else "Mean absolute reference deviation (bps)",
                   xticks=np.arange(len(methods)), xticklabels=labels[:len(methods)])
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    evaluation_days = len(sessions()) - LOOKBACK
    all_complete = all(g["evaluation_days"] == evaluation_days and g["common_complete_days"] == evaluation_days
                       for asset in ASSETS for g in summary[asset].values())
    completion = (f"All four rules complete {evaluation_days}/{evaluation_days} sessions under both scopes and caps."
                  if all_complete else "Price panels use only dates complete for all four rules under both scopes.")
    fig.suptitle("Capacity headroom and reference-price differences", fontsize=16)
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False,
               title=completion + "\nLimits use realised future volume; whiskers reach sample minima. Dashed line: USD 1m.")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(data_dir, output_dir):
    data_dir, output = Path(data_dir), Path(output_dir)
    observations, coverage = [], []
    fixed_sessions = sessions()
    for asset in ASSETS:
        for index, day in enumerate(fixed_sessions):
            observation, report = load_session(data_dir / f"{asset}_{day:%Y%m%d}.csv", asset, day)
            report["phase"] = "warmup" if index < LOOKBACK else "evaluation"
            coverage.append(report)
            if observation is not None:
                observations.append(observation)
    records, skipped = build_records(observations)
    rows = [row for record in records for cap in CAPS for row in evaluate_record(record, cap)]
    if not rows:
        raise ValueError(
            "No usable equity evaluation data. Check --data-dir and the stock-file "
            "setup in README.md; each evaluated session needs 20 prior valid sessions. "
            "No output files were written."
        )
    results = summarize(rows)
    root = Path(__file__).resolve().parents[1]
    sources = {name: sha256(root / name) for name in
               ("study/equity.py", "study/adaptive.py", "docs/protocol.md")}
    acquisition = {name: sha256(data_dir / name) for name in
                   ("acquisition_manifest.json", "quality_report.json", "five_minute_coverage.json") if (data_dir / name).exists()}
    summary = {
        "protocol": "docs/protocol.md: Independent equity sample - 26 October 2026", "notional_usd": NOTIONAL,
        "symbols": list(ASSETS), "fixed_alpha": ALPHA, "caps": CAPS, "lookback_valid_sessions": LOOKBACK,
        "data_source": "AlgoSeek public Machine Learning for Trading companion minute bars",
        "source_url": "https://algoseek.com/ml-for-trading/", "raw_data_redistributed": False,
        "session_timezone": "America/New_York", "session_window": "09:30 <= minute start < 16:00",
        "buckets_per_session": BUCKETS, "fixed_sessions": [d.isoformat() for d in fixed_sessions],
        "warmup_sessions": [d.isoformat() for d in fixed_sessions[:LOOKBACK]],
        "evaluation_sessions": [d.isoformat() for d in fixed_sessions[LOOKBACK:]],
        "price_reference": "TotalVolume-weighted TotalVolumeWeightPrice; same selected-window benchmark for both capacities",
        "capacity_scopes": {"total": "TotalVolume", "exchange": "Volume"},
        "field_pairs": PAIRS, "valid_asset_sessions": len(observations), "evaluated_asset_sessions": len(records),
        "coverage": coverage, "history_skips": skipped,
        "history_windows": [{"asset": r["asset"], "date": r["date"].isoformat(), "prior_sessions": r["history_dates"]} for r in records],
        "code_sha256": sources, "acquisition_sha256": acquisition, "results": results,
        "deadline_limit_diagnostic": {
            "status": "Developed after the earlier EBAY/LULU pilot; fixed before evaluating this independent stock sample.",
            "formula": "Q_max = min_j cap * sum(volume[j:]) / (1 - release_fraction[j-1]); previous fraction is 0 at j=0; skip zero denominators.",
            "scope": "Fixed-fraction TWAP, historical and fixed_early only. Feedback fractions depend on Q; no such limit is reported.",
            "headroom_definition": "Maximum fixed-schedule notional divided by the unchanged USD 1m order; not an ex-ante capacity promise.",
        },
        "max_suffix_identity_error": max((r["max_suffix_identity_error"] for r in rows), default=None),
        "corporate_actions": {
            "review": "See the dated sample review and issuer sources in docs/protocol.md.",
            "adjustment": "Provider prices and share volumes are used unchanged. Bundle adjustment flags were not independently certified; no additional adjustment was applied.",
            "sources": ["docs/protocol.md"],
        },
        "limitations": ["Four selected stocks from one historical quarter; fixed alpha=0.10 was not fitted on this sample.",
                        "Selected-window provider bar references exclude records assigned to 16:00; not auction-complete daily VWAP.",
                        "Reported volume coverage is not executable venue access. No routing, queue, spread, fees or own impact model.",
                        "Continuous share quantities and historical reference-price fills are approximations, not actual trades.",
                        "Price summaries use the intersection where all four methods complete under both capacity scopes.",
                        "Missing or invalid sessions are disclosed, not replaced. No profit or cross-market causal claim."],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "equity_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "equity_daily.csv").open("w", newline="") as target:
        if rows:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_results(results, output / "equity_capacity.png")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data/equities")
    parser.add_argument("--output-dir", type=Path, default=root / "results/equities")
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir)
    expected = len(ASSETS) * len(sessions())
    evaluated = len(ASSETS) * (len(sessions()) - LOOKBACK)
    print(f'{result["valid_asset_sessions"]}/{expected} valid asset-sessions; {result["evaluated_asset_sessions"]}/{evaluated} evaluated.')
    print(f'Results saved to {args.output_dir}; provider adjustment flags remain uncertified (see metadata).')
