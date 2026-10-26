"""Optional baseline and diagnostic helpers, separate from the stock run."""

import numpy as np


METHODS = ("twap", "historical", "ewma")
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
