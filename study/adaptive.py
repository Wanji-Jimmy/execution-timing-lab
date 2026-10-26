"""Release schedules and capacity-capped replay."""

import numpy as np


METHODS = ("twap", "historical", "fixed_early", "feedback")


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
