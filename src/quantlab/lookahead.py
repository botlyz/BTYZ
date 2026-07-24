"""Test anti-look-ahead générique (étape 0.3).

Principe : une stratégie causale doit produire les mêmes signaux sur [0, i)
qu'elle voie les données jusqu'à i ou jusqu'à la fin. On échantillonne
n_prefixes tailles de préfixe espacées géométriquement et on compare les
4 séries booléennes après .align(). Attrape un .shift(-1) comme un z-score
calculé sur la série complète.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import config

_MAX_BARS = 50_000          # stratégies lentes : on tronque aux dernières barres
_SERIES = ("long_entries", "long_exits", "short_entries", "short_exits")


def _get(sig, name: str, index: pd.Index) -> pd.Series:
    s = getattr(sig, name)
    if s is None:
        return pd.Series(False, index=index)
    return s


def _prefix_sizes(n: int, n_prefixes: int) -> list[int]:
    lo = min(config.LOOKAHEAD_MIN_PREFIX, max(n // 2, 2))
    sizes = np.geomspace(lo, n, num=n_prefixes)
    sizes = sorted({int(round(s)) for s in sizes if lo <= round(s) < n})
    return sizes


def check(strategy, data: pd.DataFrame, params: dict | None = None,
          n_prefixes: int = config.LOOKAHEAD_N_PREFIXES) -> dict:
    """{"passed": bool, "first_divergence": {timestamp, series, prefix}|None, "detail"}."""
    if len(data) > _MAX_BARS:
        data = data.iloc[-_MAX_BARS:]
    fp = strategy.full_params(params or {})
    ref = strategy.signals(data, fp).align(data.index)

    sizes = _prefix_sizes(len(data), n_prefixes)
    first_div = None
    n_divergent = 0
    for i in sizes:
        sub_data = data.iloc[:i]
        sub = strategy.signals(sub_data, fp).align(sub_data.index)
        for name in _SERIES:
            a = _get(ref, name, data.index).iloc[:i]
            b = _get(sub, name, sub_data.index)
            diff = a.values != b.values
            if diff.any():
                n_divergent += 1
                ts = sub_data.index[int(np.argmax(diff))]
                if first_div is None or ts < first_div["timestamp"]:
                    first_div = {"timestamp": ts, "series": name, "prefix": i}
        if first_div is not None:
            break  # une divergence suffit — inutile de payer les autres préfixes

    return {
        "passed": first_div is None,
        "first_divergence": first_div,
        "detail": {
            "n_bars": len(data),
            "prefixes": sizes,
            "n_prefixes_checked": (sizes.index(first_div["prefix"]) + 1
                                   if first_div else len(sizes)),
            "n_series_divergent": n_divergent,
        },
    }
