"""Per-pair re-opti for §5 portfolio extension (called by marimo notebook).

Worker function picklable pour `ProcessPoolExecutor`. Une instance = une paire.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings


def reopti_pair_worker(args: tuple) -> tuple:
    """Re-opti Optuna une paire. Retourne (pair, status, params_dict_or_None).

    args = (approach_id, pair, tf, anchor_iso, cache_file_str, train_days,
            trials, fees, slippage, base_ohlcv_dir, with_cross_exchange).
    """
    (approach_id, pair, tf, anchor_iso, cache_file_str, train_days,
     trials, fees, slippage, base_ohlcv_dir, with_cross_exchange) = args

    src_root = "/home/devbox/BTYZ/src"
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    warnings.filterwarnings("ignore")

    import pandas as pd

    cache_file = pathlib.Path(cache_file_str)
    if cache_file.exists():
        try:
            return pair, "cache_hit", json.loads(cache_file.read_text())
        except Exception:
            pass  # corrupt cache → re-opti

    # Load OHLCV (1m source, resample target tf)
    fp = pathlib.Path(base_ohlcv_dir) / f"{pair}.csv"
    if not fp.exists():
        return pair, f"ohlcv_missing: {fp}", None
    df = pd.read_csv(
        fp, low_memory=False,
        usecols=["date", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    ohlcv = df.resample(tf, label="left", closed="left").agg(agg).dropna()

    if with_cross_exchange:
        try:
            from engine.data_loader import _attach_hl_funding
            _xe = _attach_hl_funding(ohlcv, pair)
            if _xe is not None and len(_xe) > 0:
                ohlcv = _xe
        except Exception:
            pass
    else:
        try:
            from engine.data_loader import _maybe_attach_funding
            ohlcv = _maybe_attach_funding(ohlcv, pair, tf)
        except Exception:
            pass

    # Train window = [anchor - train_days, anchor]
    anchor_ts = pd.Timestamp(anchor_iso)
    train_start = anchor_ts - pd.Timedelta(days=train_days)
    si = int(ohlcv.index.searchsorted(train_start, side="left"))
    ei = int(ohlcv.index.searchsorted(anchor_ts, side="left"))
    if ei - si < 500:
        return pair, f"train_too_short ({ei - si} bars)", None
    train_df = ohlcv.iloc[si:ei]

    # Strat with target fees/freq/slippage
    from engine.approach_loader import instantiate_strategy
    strat = instantiate_strategy(approach_id)
    mod = sys.modules.get(f"approach.{approach_id}.strategy")
    if mod is not None:
        if hasattr(mod, "_target_fees"):
            mod._target_fees = fees
        if hasattr(mod, "_target_freq"):
            mod._target_freq = tf
        if hasattr(mod, "_target_slippage"):
            mod._target_slippage = slippage

    from engine.tpe_search import run_tpe_fold
    try:
        res = run_tpe_fold(
            train_data=train_df, test_data=train_df,
            param_space_fn=strat.param_space,
            run_backtest_fn=strat.run_backtest,
            score_fn=strat.score,
            trials=trials, min_trades_per_fold=10,
            seed=42, fold_idx=0, n_jobs=1,
        )
    except Exception as e:
        return pair, f"reopti_exception: {type(e).__name__}: {e}", None

    if res is None or not res.get("params"):
        return pair, "reopti_failed", None

    params = res["params"]
    sharpe = float(res.get("train_metrics", {}).get("sharpe_ratio", 0) or 0)
    trades = int(res.get("train_metrics", {}).get("total_trades", 0) or 0)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(params, indent=2, default=str))
    except Exception:
        pass
    return pair, f"done train_sh={sharpe:.2f} trades={trades}", params
