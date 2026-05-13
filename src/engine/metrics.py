"""Extract VBT Portfolio metrics into a flat, JSON-safe dict.

Cardinal rule: no manual sharpe/DD/PF math. Everything via port.stats() / VBT properties.
"""
import math

import pandas as pd


def _to_scalar(v):
    """Convert any VBT/numpy/pandas value to a JSON-safe scalar."""
    if v is None:
        return ""

    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass

    if isinstance(v, bool):
        return v

    if isinstance(v, (int, float)):
        if math.isinf(v) or math.isnan(v):
            return ""
        return v

    if isinstance(v, pd.Timedelta):
        return round(v.total_seconds() / 86400, 6)

    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    if isinstance(v, str):
        return v

    try:
        f = float(v)
        if math.isinf(f) or math.isnan(f):
            return ""
        return f
    except (TypeError, ValueError):
        return str(v)


def extract_from_portfolio(port, extra=None):
    """Extract all VBT Portfolio metrics into a flat dict.

    Uses port.stats() first, then fallback on direct properties.
    """
    metrics = {}

    try:
        stats = port.stats()
        for k, v in stats.items():
            key = (
                k.lower()
                .replace(" [%]", "_pct")
                .replace(" ", "_")
                .replace("[", "")
                .replace("]", "")
                .replace("/", "_")
                .replace(".", "")
            )
            metrics[key] = _to_scalar(v)
    except Exception as e:
        metrics["_stats_error"] = str(e)[:120]

    for attr in (
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown", "total_return", "annualized_return",
        "annualized_volatility",
    ):
        if attr not in metrics:
            try:
                metrics[attr] = _to_scalar(getattr(port, attr))
            except Exception:
                pass

    try:
        tr = port.trades
        for attr in ("win_rate", "profit_factor", "expectancy"):
            key = f"trades_{attr}"
            if key not in metrics:
                try:
                    metrics[key] = _to_scalar(getattr(tr, attr))
                except Exception:
                    pass
        try:
            metrics.setdefault("trades_count", _to_scalar(tr.count()))
        except Exception:
            pass
    except Exception:
        pass

    try:
        dd = port.drawdowns
        if "dd_dur_days" not in metrics:
            try:
                dur = dd.duration.mean()
                if hasattr(dur, "total_seconds"):
                    metrics["dd_dur_days"] = _to_scalar(dur.total_seconds() / 86400)
                else:
                    metrics["dd_dur_days"] = _to_scalar(dur)
            except Exception:
                pass
    except Exception:
        pass

    if extra:
        metrics.update(extra)

    return metrics
