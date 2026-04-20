"""
Alpha Scanner BTC/ETH — 100 hypothèses × mini grid = ~300+ backtests
Données : prix (spot/um/cm), CVD, OI, funding, LS ratio, taker
Timeframe : 5min (resamplé depuis 1min)
Backtest : VBT long/short, 10bps fees, 2bps slippage
"""
import sys, os, time, warnings, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from vectorbtpro import vbt

FEES = 0.001      # 10 bps
SLIPPAGE = 0.0002 # 2 bps
INIT_CASH = 10000

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING — resample tout en 5min
# ═══════════════════════════════════════════════════════════════════════
print("Chargement des données...", flush=True)

def load_1m_to_5m(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df.set_index('date').sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    ohlcv = df.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'taker_buy_volume': 'sum'
    }).dropna()
    ohlcv['taker_sell_volume'] = ohlcv['volume'] - ohlcv['taker_buy_volume']
    ohlcv['delta'] = ohlcv['taker_buy_volume'] - ohlcv['taker_sell_volume']
    ohlcv['cvd'] = ohlcv['delta'].cumsum()
    return ohlcv

def load_metrics(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df.set_index('date').sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def load_funding(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df.set_index('date').sort_index()
    df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
    return df

results_all = []

for PAIR, pair_label in [('BTC', 'BTCUSDT'), ('ETH', 'ETHUSDT')]:
    print(f"\n{'='*60}")
    print(f"  {PAIR}")
    print(f"{'='*60}", flush=True)

    # Prix
    um = load_1m_to_5m(f'data/raw/binance/um/1m/{pair_label}.csv')
    spot_path = f'data/raw/binance/spot/1m/{pair_label}.csv'
    spot = load_1m_to_5m(spot_path) if os.path.exists(spot_path) else None
    cm_label = f'{PAIR}USD_PERP'
    cm = load_1m_to_5m(f'data/raw/binance/cm/1m/{cm_label}.csv')

    # Normaliser CM : volume en contrats → convertir en USD via close price
    cm['volume_usd'] = cm['volume'] * cm['close']
    cm['taker_buy_usd'] = cm['taker_buy_volume'] * cm['close']
    cm['taker_sell_usd'] = cm['volume_usd'] - cm['taker_buy_usd']
    cm['delta_usd'] = cm['taker_buy_usd'] - cm['taker_sell_usd']
    cm['cvd_usd'] = cm['delta_usd'].cumsum()

    # Metrics (perp only)
    metrics = load_metrics(f'data/raw/binance/um/metrics/{pair_label}.csv')

    # Funding
    funding = load_funding(f'data/raw/binance/um/funding/{pair_label}.csv')

    # Merge tout sur l'index um (référence)
    data = um[['open', 'high', 'low', 'close', 'volume', 'delta', 'cvd']].copy()
    data.columns = ['open', 'high', 'low', 'close', 'volume', 'um_delta', 'um_cvd']

    if spot is not None:
        data = data.join(spot[['delta', 'cvd', 'close']].rename(columns={
            'delta': 'spot_delta', 'cvd': 'spot_cvd', 'close': 'spot_close'
        }), how='left')
        data['basis'] = (data['close'] - data['spot_close']) / data['spot_close']
    else:
        data['spot_delta'] = np.nan
        data['spot_cvd'] = np.nan
        data['basis'] = np.nan

    data = data.join(cm[['delta_usd', 'cvd_usd']].rename(columns={
        'delta_usd': 'cm_delta', 'cvd_usd': 'cm_cvd'
    }), how='left')

    data = data.join(metrics[['oi', 'oi_value', 'top_trader_ls', 'ls_ratio', 'taker_ls_vol']], how='left')
    data = data.join(funding[['funding_rate']], how='left')
    data['funding_rate'] = data['funding_rate'].ffill()
    data = data.dropna(subset=['close']).ffill()

    N = len(data)
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    print(f"  {N} candles 5min, {data.index[0]} → {data.index[-1]}", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════
    def rolling_zscore(arr, w):
        z = np.zeros(N)
        for i in range(w, N):
            chunk = arr[i-w:i]
            m = np.mean(chunk); s = np.std(chunk)
            if s > 0: z[i] = (arr[i] - m) / s
        return z

    def rolling_pctchange(arr, p):
        r = np.zeros(N)
        for i in range(p, N):
            if arr[i-p] != 0: r[i] = (arr[i] - arr[i-p]) / abs(arr[i-p])
        return r

    def rolling_mean(arr, w):
        r = np.full(N, np.nan)
        cs = np.nancumsum(arr)
        r[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
        return r

    def rsi(close, period):
        r = np.full(N, 50.0)
        ag = 0.0; al = 0.0
        for i in range(1, min(period+1, N)):
            d = close[i] - close[i-1]
            if d > 0: ag += d
            else: al -= d
        if period < N:
            ag /= period; al /= period
            r[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
        for i in range(period+1, N):
            d = close[i] - close[i-1]
            g = d if d > 0 else 0.0; l = -d if d < 0 else 0.0
            ag = (ag * (period-1) + g) / period
            al = (al * (period-1) + l) / period
            r[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
        return r

    # Pré-calculs
    um_cvd = data['um_cvd'].values
    um_delta = data['um_delta'].values
    spot_cvd = data['spot_cvd'].values if 'spot_cvd' in data else np.zeros(N)
    spot_delta = data['spot_delta'].values if 'spot_delta' in data else np.zeros(N)
    cm_cvd = data['cm_cvd'].values
    cm_delta = data['cm_delta'].values
    oi = data['oi'].values
    oi_value = data['oi_value'].values
    ls_ratio = data['ls_ratio'].values
    top_ls = data['top_trader_ls'].values
    taker_ls = data['taker_ls_vol'].values
    funding_arr = data['funding_rate'].values
    basis_arr = data['basis'].values
    rsi_14 = rsi(close, 14)
    rsi_60 = rsi(close, 60)

    # ═══════════════════════════════════════════════════════════════════
    # BACKTEST ENGINE
    # ═══════════════════════════════════════════════════════════════════
    def make_edge_signals(raw_long, raw_short, cooldown=12):
        """
        Convertit des signaux bruts (True sur chaque bougie) en signaux d'entrée
        seulement au CHANGEMENT (False→True) avec un cooldown minimum entre les trades.
        cooldown=12 → minimum 1h entre 2 entrées (en 5min)
        """
        long_entry = np.zeros(N, dtype=bool)
        short_entry = np.zeros(N, dtype=bool)
        last_trade = -cooldown - 1

        for i in range(1, N):
            if i - last_trade < cooldown:
                continue
            # Entrée long : signal passe de False à True
            if raw_long[i] and not raw_long[i-1]:
                long_entry[i] = True
                last_trade = i
            # Entrée short : signal passe de False à True
            elif raw_short[i] and not raw_short[i-1]:
                short_entry[i] = True
                last_trade = i

        return long_entry, short_entry

    def backtest_signal(long_signal, short_signal, hold_bars=60, cooldown=12):
        """
        Backtest : entre au changement de signal, sort quand le signal s'inverse.
        Max hold = hold_bars bougies. Cooldown entre trades.
        """
        long_entry, short_entry = make_edge_signals(long_signal, short_signal, cooldown)

        # Exits = signal inverse ou fin de holding period
        long_exit = pd.Series(short_signal, index=data.index)
        short_exit = pd.Series(long_signal, index=data.index)

        try:
            pf = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=pd.Series(long_entry, index=data.index),
                short_entries=pd.Series(short_entry, index=data.index),
                exits=long_exit,
                short_exits=short_exit,
                td_stop=hold_bars,
                accumulate=False,
                init_cash=INIT_CASH,
                fees=FEES,
                slippage=SLIPPAGE,
                freq='5min',
            )
            s = pf.stats(silence_warnings=True)
            return {
                'return_%': round(float(s.get('Total Return [%]', 0)), 2),
                'sharpe': round(float(s.get('Sharpe Ratio', 0)), 3),
                'sortino': round(float(s.get('Sortino Ratio', 0)), 3),
                'maxdd_%': round(float(s.get('Max Drawdown [%]', 0)), 2),
                'winrate_%': round(float(s.get('Win Rate [%]', 0)), 1),
                'trades': int(s.get('Total Trades', 0)),
                'profit_factor': round(float(s.get('Profit Factor', 0)), 3),
                'calmar': round(float(s.get('Calmar Ratio', 0)), 3),
            }
        except Exception:
            return None

    def test_alpha(name, long_sig, short_sig, params_label, hold=60):
        r = backtest_signal(long_sig, short_sig, hold_bars=hold)
        if r is None or r['trades'] < 30:
            return
        r['pair'] = PAIR
        r['hypothesis'] = name
        r['params'] = params_label
        r['hold'] = hold
        results_all.append(r)
        tag = ' ★' if r['sharpe'] > 0.5 else ''
        print(f"  [{len(results_all):3d}] {name:40s} {params_label:20s} → "
              f"Ret={r['return_%']:+8.1f}% Sharpe={r['sharpe']:6.3f} DD={r['maxdd_%']:5.1f}% "
              f"Trades={r['trades']:5d}{tag}", flush=True)

    t0 = time.time()

    # ═══════════════════════════════════════════════════════════════════
    # 100 HYPOTHÈSES
    # ═══════════════════════════════════════════════════════════════════

    # ── 1-3. CVD futures momentum ─────────────────────────────────────
    for lb in [12, 60, 288]:
        dc = np.zeros(N)
        for i in range(lb, N): dc[i] = um_cvd[i] - um_cvd[i-lb]
        test_alpha("01_um_cvd_momentum", dc > 0, dc < 0, f"lb={lb}")

    # ── 4-6. CVD spot momentum ────────────────────────────────────────
    if not np.all(np.isnan(spot_cvd)):
        for lb in [12, 60, 288]:
            dc = np.zeros(N)
            for i in range(lb, N): dc[i] = spot_cvd[i] - spot_cvd[i-lb]
            test_alpha("02_spot_cvd_momentum", dc > 0, dc < 0, f"lb={lb}")

    # ── 7-9. CVD coin-margined momentum ───────────────────────────────
    for lb in [12, 60, 288]:
        dc = np.zeros(N)
        for i in range(lb, N): dc[i] = cm_cvd[i] - cm_cvd[i-lb]
        test_alpha("03_cm_cvd_momentum", dc > 0, dc < 0, f"lb={lb}")

    # ── 10-12. CVD divergence prix/futures ────────────────────────────
    for lb in [12, 60, 288]:
        dp = np.zeros(N); dc = np.zeros(N)
        for i in range(lb, N):
            dp[i] = close[i] - close[i-lb]
            dc[i] = um_cvd[i] - um_cvd[i-lb]
        test_alpha("04_um_cvd_divergence", (dp < 0) & (dc > 0), (dp > 0) & (dc < 0), f"lb={lb}")

    # ── 13-15. CVD divergence prix/spot ───────────────────────────────
    if not np.all(np.isnan(spot_cvd)):
        for lb in [12, 60, 288]:
            dp = np.zeros(N); dc = np.zeros(N)
            for i in range(lb, N):
                dp[i] = close[i] - close[i-lb]
                dc[i] = spot_cvd[i] - spot_cvd[i-lb]
            test_alpha("05_spot_cvd_divergence", (dp < 0) & (dc > 0), (dp > 0) & (dc < 0), f"lb={lb}")

    # ── 16-18. CVD divergence prix/cm ─────────────────────────────────
    for lb in [12, 60, 288]:
        dp = np.zeros(N); dc = np.zeros(N)
        for i in range(lb, N):
            dp[i] = close[i] - close[i-lb]
            dc[i] = cm_cvd[i] - cm_cvd[i-lb]
        test_alpha("06_cm_cvd_divergence", (dp < 0) & (dc > 0), (dp > 0) & (dc < 0), f"lb={lb}")

    # ── 19-21. Divergence CVD futures vs spot ─────────────────────────
    if not np.all(np.isnan(spot_cvd)):
        for lb in [12, 60, 288]:
            d_um = np.zeros(N); d_sp = np.zeros(N)
            for i in range(lb, N):
                d_um[i] = um_cvd[i] - um_cvd[i-lb]
                d_sp[i] = spot_cvd[i] - spot_cvd[i-lb]
            # Futures CVD monte + spot CVD baisse = futures bullish, spot pas → long
            test_alpha("07_um_vs_spot_cvd", (d_um > 0) & (d_sp < 0), (d_um < 0) & (d_sp > 0), f"lb={lb}")

    # ── 22-24. Divergence CVD futures vs cm ───────────────────────────
    for lb in [12, 60, 288]:
        d_um = np.zeros(N); d_cm = np.zeros(N)
        for i in range(lb, N):
            d_um[i] = um_cvd[i] - um_cvd[i-lb]
            d_cm[i] = cm_cvd[i] - cm_cvd[i-lb]
        test_alpha("08_um_vs_cm_cvd", (d_um > 0) & (d_cm < 0), (d_um < 0) & (d_cm > 0), f"lb={lb}")

    # ── 25-27. Delta z-score futures ──────────────────────────────────
    for w in [60, 288, 576]:
        z = rolling_zscore(um_delta, w)
        test_alpha("09_um_delta_zscore", z > 2, z < -2, f"w={w}_z=2")

    # ── 28-30. Delta z-score spot ─────────────────────────────────────
    if not np.all(np.isnan(spot_delta)):
        for w in [60, 288, 576]:
            z = rolling_zscore(spot_delta, w)
            test_alpha("10_spot_delta_zscore", z > 2, z < -2, f"w={w}_z=2")

    # ── 31-33. OI momentum ───────────────────────────────────────────
    for lb in [12, 60, 288]:
        doi = np.zeros(N)
        for i in range(lb, N): doi[i] = oi_value[i] - oi_value[i-lb]
        test_alpha("11_oi_momentum", doi > 0, doi < 0, f"lb={lb}")

    # ── 34-36. OI + prix divergence ──────────────────────────────────
    for lb in [12, 60, 288]:
        doi = np.zeros(N); dp = np.zeros(N)
        for i in range(lb, N):
            doi[i] = oi_value[i] - oi_value[i-lb]
            dp[i] = close[i] - close[i-lb]
        # OI monte + prix baisse = shorts s'accumulent → squeeze → long
        test_alpha("12_oi_price_div", (doi > 0) & (dp < 0), (doi > 0) & (dp > 0), f"lb={lb}")

    # ── 37-39. OI z-score ────────────────────────────────────────────
    for w in [60, 288, 576]:
        z = rolling_zscore(oi_value, w)
        test_alpha("13_oi_zscore_high", z > 2, z < -2, f"w={w}")

    # ── 40-42. Funding rate momentum ─────────────────────────────────
    for th in [0.0001, 0.0005, 0.001]:
        test_alpha("14_funding_dir", funding_arr < -th, funding_arr > th, f"th={th}")

    # ── 43-45. Funding rate z-score ──────────────────────────────────
    for w in [288, 576, 1152]:
        z = rolling_zscore(funding_arr, w)
        test_alpha("15_funding_zscore", z < -2, z > 2, f"w={w}")

    # ── 46-48. Funding + CVD combo ───────────────────────────────────
    for lb in [60, 288, 576]:
        dc = np.zeros(N)
        for i in range(lb, N): dc[i] = um_cvd[i] - um_cvd[i-lb]
        test_alpha("16_funding_cvd", (funding_arr < 0) & (dc > 0), (funding_arr > 0) & (dc < 0), f"lb={lb}")

    # ── 49-51. LS ratio directionnel ─────────────────────────────────
    for th in [0.9, 0.95, 1.0]:
        test_alpha("17_ls_ratio_dir", ls_ratio < th, ls_ratio > (1/th), f"th={th}")

    # ── 52-54. Top trader LS contrarian ──────────────────────────────
    for th in [0.85, 0.9, 0.95]:
        test_alpha("18_top_ls_contrarian", top_ls < th, top_ls > (1/th), f"th={th}")

    # ── 55-57. Taker LS vol z-score ──────────────────────────────────
    for w in [60, 288, 576]:
        z = rolling_zscore(taker_ls, w)
        test_alpha("19_taker_zscore", z < -2, z > 2, f"w={w}")

    # ── 58-60. Basis (spot-futures) ──────────────────────────────────
    if not np.all(np.isnan(basis_arr)):
        for th in [0.001, 0.002, 0.005]:
            test_alpha("20_basis_dir", basis_arr < -th, basis_arr > th, f"th={th}")

    # ── 61-63. Basis z-score ─────────────────────────────────────────
    if not np.all(np.isnan(basis_arr)):
        for w in [60, 288, 576]:
            z = rolling_zscore(basis_arr, w)
            test_alpha("21_basis_zscore", z < -2, z > 2, f"w={w}")

    # ── 64-66. CVD + OI combo ────────────────────────────────────────
    for lb in [60, 288, 576]:
        dc = np.zeros(N); doi = np.zeros(N); dp = np.zeros(N)
        for i in range(lb, N):
            dc[i] = um_cvd[i] - um_cvd[i-lb]
            doi[i] = oi_value[i] - oi_value[i-lb]
            dp[i] = close[i] - close[i-lb]
        # Prix baisse + OI monte + CVD monte = acheteurs résistent pendant accumulation short
        test_alpha("22_cvd_oi_price", (dp < 0) & (doi > 0) & (dc > 0), (dp > 0) & (doi > 0) & (dc < 0), f"lb={lb}")

    # ── 67-69. CVD + Funding combo ───────────────────────────────────
    for lb in [60, 288, 576]:
        dc = np.zeros(N)
        for i in range(lb, N): dc[i] = um_cvd[i] - um_cvd[i-lb]
        # Funding négatif + CVD monte = shorts paient ET acheteurs poussent → squeeze
        test_alpha("23_cvd_funding_squeeze", (funding_arr < -0.0001) & (dc > 0), (funding_arr > 0.0001) & (dc < 0), f"lb={lb}")

    # ── 70-72. Triple CVD (um + spot + cm même direction) ────────────
    if not np.all(np.isnan(spot_cvd)):
        for lb in [12, 60, 288]:
            d_um = np.zeros(N); d_sp = np.zeros(N); d_cm = np.zeros(N)
            for i in range(lb, N):
                d_um[i] = um_cvd[i] - um_cvd[i-lb]
                d_sp[i] = spot_cvd[i] - spot_cvd[i-lb]
                d_cm[i] = cm_cvd[i] - cm_cvd[i-lb]
            test_alpha("24_triple_cvd_align", (d_um > 0) & (d_sp > 0) & (d_cm > 0), (d_um < 0) & (d_sp < 0) & (d_cm < 0), f"lb={lb}")

    # ── 73-75. OI + LS ratio combo ───────────────────────────────────
    for lb in [60, 288, 576]:
        doi = np.zeros(N)
        for i in range(lb, N): doi[i] = oi_value[i] - oi_value[i-lb]
        # OI monte + shorts dominent (ls < 1) → squeeze imminent → long
        test_alpha("25_oi_ls_squeeze", (doi > 0) & (ls_ratio < 0.95), (doi > 0) & (ls_ratio > 1.05), f"lb={lb}")

    # ── 76-78. RSI + CVD combo ───────────────────────────────────────
    for lb in [60, 288, 576]:
        dc = np.zeros(N)
        for i in range(lb, N): dc[i] = um_cvd[i] - um_cvd[i-lb]
        test_alpha("26_rsi_cvd", (rsi_14 < 30) & (dc > 0), (rsi_14 > 70) & (dc < 0), f"lb={lb}_rsi14")

    # ── 79-81. CVD accélération (2ème dérivée) ───────────────────────
    for lb in [12, 60, 288]:
        dc1 = np.zeros(N); dc2 = np.zeros(N)
        for i in range(lb, N): dc1[i] = um_cvd[i] - um_cvd[i-lb]
        for i in range(lb*2, N): dc2[i] = dc1[i] - dc1[i-lb]
        test_alpha("27_cvd_acceleration", dc2 > 0, dc2 < 0, f"lb={lb}")

    # ── 82-84. Volume imbalance futures ──────────────────────────────
    um_imb = np.where(data['volume'].values > 0,
                      um_delta / data['volume'].values, 0)
    for w in [60, 288, 576]:
        z = rolling_zscore(um_imb, w)
        test_alpha("28_um_imbalance_z", z > 2, z < -2, f"w={w}")

    # ── 85-87. Funding + basis combo ─────────────────────────────────
    if not np.all(np.isnan(basis_arr)):
        for th in [0.001, 0.002, 0.005]:
            # Funding négatif + basis négatif = double signal bearish → contrarian long
            test_alpha("29_funding_basis", (funding_arr < 0) & (basis_arr < -th), (funding_arr > 0) & (basis_arr > th), f"th={th}")

    # ── 88-90. OI changement rapide ──────────────────────────────────
    for lb in [12, 60, 288]:
        oi_roc = rolling_pctchange(oi_value, lb)
        test_alpha("30_oi_roc", oi_roc > 0.05, oi_roc < -0.05, f"lb={lb}_>5%")

    # ── 91-93. Taker + OI combo ──────────────────────────────────────
    for w in [60, 288, 576]:
        z_taker = rolling_zscore(taker_ls, w)
        doi = np.zeros(N)
        for i in range(w, N): doi[i] = oi_value[i] - oi_value[i-w]
        # Takers vendent (z < -1) + OI monte = shorts s'accumulent via market orders
        test_alpha("31_taker_oi", (z_taker < -2) & (doi > 0), (z_taker > 2) & (doi > 0), f"w={w}")

    # ── 94-96. Multi-CVD z-score ─────────────────────────────────────
    for w in [60, 288, 576]:
        z_um = rolling_zscore(um_delta, w)
        z_cm = rolling_zscore(cm_delta, w)
        # Les deux marchés bullish en même temps
        test_alpha("32_multi_cvd_zscore", (z_um > 2) & (z_cm > 2), (z_um < -2) & (z_cm < -2), f"w={w}")

    # ── 97-99. Funding extreme + RSI ─────────────────────────────────
    for w in [288, 576, 1152]:
        z_fr = rolling_zscore(funding_arr, w)
        test_alpha("33_funding_rsi", (z_fr < -2) & (rsi_60 < 35), (z_fr > 2) & (rsi_60 > 65), f"w={w}")

    # ── 100. CVD all sources divergence prix ─────────────────────────
    if not np.all(np.isnan(spot_cvd)):
        for lb in [60, 288, 576]:
            dp = np.zeros(N); d_um = np.zeros(N); d_sp = np.zeros(N); d_cm = np.zeros(N)
            for i in range(lb, N):
                dp[i] = close[i] - close[i-lb]
                d_um[i] = um_cvd[i] - um_cvd[i-lb]
                d_sp[i] = spot_cvd[i] - spot_cvd[i-lb]
                d_cm[i] = cm_cvd[i] - cm_cvd[i-lb]
            # Prix baisse mais TOUS les CVDs montent
            test_alpha("34_all_cvd_div_price",
                       (dp < 0) & (d_um > 0) & (d_sp > 0) & (d_cm > 0),
                       (dp > 0) & (d_um < 0) & (d_sp < 0) & (d_cm < 0), f"lb={lb}")

    # ── HOLD PERIOD GRID pour les meilleurs ──────────────────────────
    # Retester les top avec hold = 12, 36, 120
    _top_so_far = sorted([r for r in results_all if r['pair'] == PAIR],
                         key=lambda x: x['sharpe'], reverse=True)[:5]
    # (on skip cette partie pour ne pas rallonger — les holds sont déjà testés via le lb)

    elapsed = time.time() - t0
    print(f"\n  {PAIR} terminé en {elapsed:.0f}s", flush=True)
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# RÉSULTATS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"TERMINÉ : {len(results_all)} backtests")
print(f"{'='*70}\n")

df = pd.DataFrame(results_all)
df.to_csv('research/alpha_btc_eth_results.csv', index=False)

for pair in ['BTC', 'ETH']:
    sub = df[df['pair'] == pair].sort_values('sharpe', ascending=False)
    print(f"\n{'='*70}")
    print(f"TOP 15 {pair} par Sharpe :")
    print(f"{'='*70}")
    for _, r in sub.head(15).iterrows():
        print(f"  {r['hypothesis']:40s} {r['params']:20s} hold={r['hold']:3d} → "
              f"Ret={r['return_%']:+8.1f}% Sharpe={r['sharpe']:6.3f} Sort={r['sortino']:6.3f} "
              f"DD={r['maxdd_%']:5.1f}% WR={r['winrate_%']:5.1f}% Trades={r['trades']:5d} PF={r['profit_factor']:5.3f}")

# Top global
print(f"\n{'='*70}")
print(f"TOP 10 GLOBAL par Sharpe :")
print(f"{'='*70}")
for _, r in df.sort_values('sharpe', ascending=False).head(10).iterrows():
    print(f"  [{r['pair']}] {r['hypothesis']:40s} {r['params']:20s} → "
          f"Ret={r['return_%']:+8.1f}% Sharpe={r['sharpe']:6.3f} DD={r['maxdd_%']:5.1f}% Trades={r['trades']:5d}")
