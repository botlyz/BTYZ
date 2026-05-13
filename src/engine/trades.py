"""Extract trades DataFrame from a vbt.Portfolio."""
import pandas as pd


def extract_trades_df(pf, test_start=None) -> pd.DataFrame:
    """Pull trade records from VBT pf.trades and slice to test_start onward."""
    try:
        t = pf.trades.records_readable.copy()
    except Exception:
        return pd.DataFrame()
    if t.empty:
        return t

    if test_start is not None and "Entry Index" in t.columns and pd.api.types.is_datetime64_any_dtype(t["Entry Index"]):
        t = t[t["Entry Index"] >= test_start]

    t = t.rename(columns={
        "Entry Index":     "entry_time",
        "Exit Index":      "exit_time",
        "Avg Entry Price": "entry_price",
        "Avg Exit Price":  "exit_price",
        "Size":            "size",
        "Return":          "return_pct",
        "PnL":             "pnl",
        "Direction":       "side",
        "Entry Fees":      "entry_fees",
        "Exit Fees":       "exit_fees",
        "Status":          "status",
    })

    keep = ["entry_time", "exit_time", "side", "entry_price", "exit_price",
            "size", "return_pct", "pnl", "entry_fees", "exit_fees", "status"]
    keep = [c for c in keep if c in t.columns]
    t = t[keep].copy()

    if "return_pct" in t.columns:
        t["return_pct"] = (t["return_pct"] * 100).round(6)
    for c in t.select_dtypes(include="float").columns:
        t[c] = t[c].round(8)

    return t.reset_index(drop=True)
