"""
Alpha Scanner Autonome — 50 hypotheses × 3 params = 150 backtests
Compare chaque variante à la baseline RAM DCA sur HYPE 5min.
Résultats loggés dans research/alpha_results.csv + résumé console.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from numba import njit
from vectorbtpro import vbt
from src.strategies.ram_dca import ram_dca_nb

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
print("Chargement des données...")

# HYPE Lighter 5min (prix)
price = pd.read_csv('data/raw/lighter/5m/HYPE.csv')
price['date'] = pd.to_datetime(price['date'], unit='ms')
price = price.set_index('date').sort_index()

# Binance um/1m HYPEUSDT (volume taker)
bn = pd.read_csv('data/raw/binance/um/1m/HYPEUSDT.csv')
bn['date'] = pd.to_datetime(bn['date'], unit='ms')
bn = bn.set_index('date').sort_index()
bn_5m = bn.resample('5min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'taker_buy_volume': 'sum'
}).dropna()
bn_5m['taker_sell_volume'] = bn_5m['volume'] - bn_5m['taker_buy_volume']
bn_5m['delta'] = bn_5m['taker_buy_volume'] - bn_5m['taker_sell_volume']
bn_5m['cvd'] = bn_5m['delta'].cumsum()
bn_5m['bn_close'] = bn_5m['close']
bn_5m['bn_volume'] = bn_5m['volume']
bn_5m['bn_high'] = bn_5m['high']
bn_5m['bn_low'] = bn_5m['low']

# Binance metrics 5min (OI, taker_ls_vol, ls_ratio, top_trader_ls)
metrics = pd.read_csv('data/raw/binance/um/metrics/HYPEUSDT.csv')
metrics['date'] = pd.to_datetime(metrics['date'], unit='ms')
metrics = metrics.set_index('date').sort_index()

# Binance funding 8h
funding = pd.read_csv('data/raw/binance/um/funding/HYPEUSDT.csv')
funding['date'] = pd.to_datetime(funding['date'], unit='ms')
funding = funding.set_index('date').sort_index()
funding = funding.reindex(price.index, method='ffill')

# Merge tout
data = price.copy()
data = data.join(bn_5m[['delta', 'cvd', 'bn_close', 'bn_volume', 'taker_buy_volume', 'taker_sell_volume', 'bn_high', 'bn_low']], how='left')
data = data.join(metrics[['oi', 'oi_value', 'top_trader_ls', 'ls_ratio', 'taker_ls_vol']], how='left')
data = data.join(funding[['funding_rate']], how='left')
data = data.ffill().dropna(subset=['close', 'oi'])

print(f"Données: {len(data)} candles, {data.index[0]} -> {data.index[-1]}")

# ═══════════════════════════════════════════════════════════════════════
# PARAMS FIXES + BASELINE
# ═══════════════════════════════════════════════════════════════════════
MA_W = 140; ENV = 0.03; SL = 0.11
FEES = 0.001; SLIP = 0.0002; CASH = 10000
N = len(data)

src = data['close']
ma = vbt.MA.run(src, window=MA_W).ma
upper_envs = np.zeros((1, N)); lower_envs = np.zeros((1, N))
upper_envs[0] = (ma * (1.0 + ENV)).values
lower_envs[0] = (ma * (1.0 - ENV)).values

def run_pf(ts, ep):
    return vbt.Portfolio.from_orders(
        close=data['close'], size=pd.Series(ts, index=data.index),
        price=pd.Series(ep, index=data.index), size_type='TargetPercent',
        init_cash=CASH, fees=FEES, slippage=SLIP, freq='5min')

ts0, ep0 = ram_dca_nb(data['high'].values, data['low'].values, data['close'].values,
                       ma.values, upper_envs, lower_envs, np.array([1.0]), SL)
pf0 = run_pf(ts0, ep0)
s0 = pf0.stats(silence_warnings=True)
BASELINE = {
    'return': s0['Total Return [%]'],
    'sharpe': s0['Sharpe Ratio'],
    'sortino': s0['Sortino Ratio'],
    'maxdd': s0['Max Drawdown [%]'],
    'winrate': s0['Win Rate [%]'],
    'trades': s0['Total Trades'],
    'calmar': s0['Calmar Ratio'],
    'pf': s0['Profit Factor'],
}
print(f"\nBASELINE: Return={BASELINE['return']:.1f}% Sharpe={BASELINE['sharpe']:.3f} "
      f"Sortino={BASELINE['sortino']:.3f} MaxDD={BASELINE['maxdd']:.1f}% "
      f"Trades={BASELINE['trades']} Calmar={BASELINE['calmar']:.3f}\n")

# ═══════════════════════════════════════════════════════════════════════
# NOYAU FILTRÉ (can_long / can_short)
# ═══════════════════════════════════════════════════════════════════════
@njit(cache=False)
def ram_filtered(high, low, close, ma, upper_envs, lower_envs,
                 allocations, sl_pct, can_long, can_short):
    n = len(close); n_levels = len(allocations)
    target_size = np.full(n, np.nan); exec_price = np.full(n, np.nan)
    pos_dir = 0; level_idx = 0; avg_entry = 0.0; qty = 0.0
    sl_cooldown = False; pending_sl = False; pending_sl_px = 0.0
    for i in range(1, n):
        if pending_sl:
            target_size[i] = 0.0; exec_price[i] = pending_sl_px
            pos_dir = 0; level_idx = 0; avg_entry = 0.0; qty = 0.0
            sl_cooldown = True; pending_sl = False; continue
        ma_prev = ma[i - 1]
        if np.isnan(ma_prev): continue
        if sl_cooldown:
            touched = low[i] <= ma_prev <= high[i]
            crossed = min(close[i-1], close[i]) <= ma_prev <= max(close[i-1], close[i])
            if touched or crossed: sl_cooldown = False
            else: continue
        if pos_dir != 0:
            exit_hit = (pos_dir == 1 and high[i] >= ma_prev) or (pos_dir == -1 and low[i] <= ma_prev)
            if exit_hit:
                target_size[i] = 0.0; exec_price[i] = ma_prev
                pos_dir = 0; level_idx = 0; avg_entry = 0.0; qty = 0.0; continue
        tmp_dir = pos_dir; tmp_idx = level_idx; tmp_avg = avg_entry; tmp_qty = qty
        traded = False; wp_sum = 0.0; alloc_sum = 0.0
        if tmp_dir >= 0 and can_long[i]:
            while tmp_idx < n_levels:
                lim = lower_envs[tmp_idx, i-1]; alloc = allocations[tmp_idx]
                if low[i] <= lim:
                    prev_val = tmp_avg * tmp_qty; tmp_qty += alloc
                    tmp_avg = (prev_val + lim * alloc) / tmp_qty
                    wp_sum += lim * alloc; alloc_sum += alloc; tmp_idx += 1; tmp_dir = 1; traded = True
                else: break
        if tmp_dir <= 0 and can_short[i]:
            while tmp_idx < n_levels:
                lim = upper_envs[tmp_idx, i-1]; alloc = allocations[tmp_idx]
                if high[i] >= lim:
                    prev_val = tmp_avg * tmp_qty; tmp_qty += alloc
                    tmp_avg = (prev_val + lim * alloc) / tmp_qty
                    wp_sum += lim * alloc; alloc_sum += alloc; tmp_idx += 1; tmp_dir = -1; traded = True
                else: break
        sl_hit = False; sl_px = 0.0
        if tmp_dir != 0:
            if tmp_dir == 1:
                sl_level = tmp_avg * (1.0 - sl_pct)
                if low[i] <= sl_level: sl_hit = True; sl_px = sl_level
            else:
                sl_level = tmp_avg * (1.0 + sl_pct)
                if high[i] >= sl_level: sl_hit = True; sl_px = sl_level
        if sl_hit and traded and pos_dir == 0:
            pos_dir = tmp_dir; level_idx = tmp_idx; avg_entry = tmp_avg; qty = tmp_qty
            target_size[i] = qty if pos_dir == 1 else -qty
            exec_price[i] = wp_sum / alloc_sum if alloc_sum > 0 else np.nan
            pending_sl = True; pending_sl_px = sl_px
        elif sl_hit:
            target_size[i] = 0.0; exec_price[i] = sl_px
            pos_dir = 0; level_idx = 0; avg_entry = 0.0; qty = 0.0; sl_cooldown = True
        elif traded:
            pos_dir = tmp_dir; level_idx = tmp_idx; avg_entry = tmp_avg; qty = tmp_qty
            target_size[i] = qty if pos_dir == 1 else -qty
            exec_price[i] = wp_sum / alloc_sum if alloc_sum > 0 else np.nan
    return target_size, exec_price

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
close_arr = data['close'].values
high_arr = data['high'].values
low_arr = data['low'].values
ma_arr = ma.values
oi_arr = data['oi'].values
oi_value_arr = data['oi_value'].values
cvd_arr = data['cvd'].values
delta_arr = data['delta'].values
volume_arr = data['volume'].astype(float).values
bn_vol_arr = data['bn_volume'].astype(float).values
taker_buy_arr = data['taker_buy_volume'].astype(float).values
taker_sell_arr = data['taker_sell_volume'].astype(float).values
taker_ls_arr = data['taker_ls_vol'].astype(float).values
ls_ratio_arr = data['ls_ratio'].astype(float).values
top_ls_arr = data['top_trader_ls'].astype(float).values
funding_arr = data['funding_rate'].astype(float).values
bn_close_arr = data['bn_close'].astype(float).values

def rolling_zscore(arr, window):
    z = np.zeros(N)
    for i in range(window, N):
        chunk = arr[i-window:i]
        m = np.mean(chunk); s = np.std(chunk)
        if s > 0: z[i] = (arr[i] - m) / s
    return z

def rolling_pctchange(arr, period):
    r = np.zeros(N)
    for i in range(period, N):
        if arr[i-period] != 0: r[i] = (arr[i] - arr[i-period]) / arr[i-period]
    return r

def rolling_mean(arr, window):
    r = np.full(N, np.nan)
    cs = np.cumsum(arr)
    r[window-1:] = (cs[window-1:] - np.concatenate([[0], cs[:-window]])) / window
    return r

def rolling_std(arr, window):
    r = np.zeros(N)
    for i in range(window, N):
        r[i] = np.std(arr[i-window:i])
    return r

def rolling_max(arr, window):
    r = np.zeros(N)
    for i in range(window, N):
        r[i] = np.max(arr[i-window:i])
    return r

def rolling_min(arr, window):
    r = np.zeros(N)
    for i in range(window, N):
        r[i] = np.min(arr[i-window:i])
    return r

def atr(high, low, close, window):
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[0], tr])
    return rolling_mean(tr, window)

def rsi(close, period):
    r = np.zeros(N)
    for i in range(period, N):
        gains = 0.0; losses = 0.0
        for j in range(i-period+1, i+1):
            d = close[j] - close[j-1]
            if d > 0: gains += d
            else: losses -= d
        if losses == 0: r[i] = 100.0
        elif gains == 0: r[i] = 0.0
        else: r[i] = 100.0 - 100.0 / (1.0 + gains / losses)
    return r

# Pré-calcul ATR
atr_14 = atr(high_arr, low_arr, close_arr, 14)
atr_60 = atr(high_arr, low_arr, close_arr, 60)
atr_288 = atr(high_arr, low_arr, close_arr, 288)
# RSI
rsi_14 = rsi(close_arr, 14)
rsi_28 = rsi(close_arr, 28)

def test_filter(name, param_label, can_long, can_short):
    """Lance le backtest filtré et retourne un dict de résultats."""
    ts, ep = ram_filtered(high_arr, low_arr, close_arr, ma_arr,
                          upper_envs, lower_envs, np.array([1.0]), SL, can_long, can_short)
    pf = run_pf(ts, ep)
    s = pf.stats(silence_warnings=True)
    return {
        'hypothesis': name,
        'params': param_label,
        'return_%': round(s['Total Return [%]'], 2),
        'sharpe': round(s['Sharpe Ratio'], 3),
        'sortino': round(s['Sortino Ratio'], 3),
        'maxdd_%': round(s['Max Drawdown [%]'], 2),
        'winrate_%': round(s['Win Rate [%]'], 1),
        'trades': int(s['Total Trades']),
        'calmar': round(s['Calmar Ratio'], 3),
        'profit_factor': round(s['Profit Factor'], 3),
        'beats_sharpe': s['Sharpe Ratio'] > BASELINE['sharpe'],
        'beats_return': s['Total Return [%]'] > BASELINE['return'],
        'beats_dd': s['Max Drawdown [%]'] < BASELINE['maxdd'],
    }

# ═══════════════════════════════════════════════════════════════════════
# 50 HYPOTHÈSES × 3 PARAMS
# ═══════════════════════════════════════════════════════════════════════
results = []
total = 0
t0 = time.time()

def run(name, param_label, cl, cs):
    global total
    total += 1
    r = test_filter(name, param_label, cl, cs)
    tag = ''
    if r['beats_sharpe'] and r['beats_return']: tag = ' ★★★'
    elif r['beats_sharpe'] or r['beats_dd']: tag = ' ★'
    elapsed = time.time() - t0
    print(f"[{total:3d}/150] {elapsed:5.0f}s | {name} ({param_label}) → "
          f"Ret={r['return_%']:+.1f}% Sharpe={r['sharpe']:.3f} DD={r['maxdd_%']:.1f}% "
          f"Trades={r['trades']}{tag}")
    results.append(r)

ALL_TRUE = np.ones(N, dtype=bool)

# ── 1. Volatilité ATR haute (ne trader que quand vol > median) ──
for w, label in [(14,'14'), (60,'60'), (288,'288')]:
    a = atr(high_arr, low_arr, close_arr, w)
    med = np.nanmedian(a[a>0])
    run("01_vol_haute", f"atr_{label}>med", a > med, a > med)

# ── 2. Volatilité ATR basse ──
for w, label in [(14,'14'), (60,'60'), (288,'288')]:
    a = atr(high_arr, low_arr, close_arr, w)
    med = np.nanmedian(a[a>0])
    run("02_vol_basse", f"atr_{label}<med", a < med, a < med)

# ── 3. ATR z-score extrême (vol élevée = z>1) ──
for w in [60, 144, 288]:
    a = atr(high_arr, low_arr, close_arr, 14)
    z = rolling_zscore(a, w)
    run("03_atr_zscore_high", f"w={w}_z>1", z > 1, z > 1)

# ── 4. ATR z-score bas (vol comprimée = z<-0.5) ──
for w in [60, 144, 288]:
    a = atr(high_arr, low_arr, close_arr, 14)
    z = rolling_zscore(a, w)
    run("04_atr_zscore_low", f"w={w}_z<-0.5", z < -0.5, z < -0.5)

# ── 5. Volume Lighter spike (vol > 2× moyenne) ──
for w in [20, 60, 288]:
    vm = rolling_mean(volume_arr, w)
    run("05_vol_spike", f"w={w}_>2x", volume_arr > 2*vm, volume_arr > 2*vm)

# ── 6. Volume Lighter creux (vol < 0.5× moyenne) ──
for w in [20, 60, 288]:
    vm = rolling_mean(volume_arr, w)
    run("06_vol_creux", f"w={w}_<0.5x", volume_arr < 0.5*vm, volume_arr < 0.5*vm)

# ── 7. RSI oversold/overbought ──
for period in [14, 28, 60]:
    r = rsi(close_arr, period)
    run("07_rsi_extreme", f"p={period}_<30/>70", r < 30, r > 70)

# ── 8. RSI modéré (40-60 = pas de momentum) ──
for period in [14, 28, 60]:
    r = rsi(close_arr, period)
    mid = (r > 40) & (r < 60)
    run("08_rsi_neutre", f"p={period}_40-60", mid, mid)

# ── 9. Momentum épuisé (ROC proche de 0) ──
for lb in [12, 60, 288]:
    roc = rolling_pctchange(close_arr, lb)
    run("09_momentum_epuise", f"lb={lb}_|roc|<0.5%", np.abs(roc) < 0.005, np.abs(roc) < 0.005)

# ── 10. Momentum fort (ROC extrême = ne PAS trader) → inversé : trader quand PAS de momentum ──
for lb in [12, 60, 288]:
    roc = rolling_pctchange(close_arr, lb)
    run("10_no_strong_momentum", f"lb={lb}_|roc|<2%", np.abs(roc) < 0.02, np.abs(roc) < 0.02)

# ── 11. Funding rate négatif (bearish extrême → bon pour long) ──
for th in [-0.0001, -0.0005, -0.001]:
    run("11_funding_neg", f"fr<{th}", funding_arr < th, ALL_TRUE)

# ── 12. Funding rate positif (bullish extrême → bon pour short) ──
for th in [0.0001, 0.0005, 0.001]:
    run("12_funding_pos", f"fr>{th}", ALL_TRUE, funding_arr > th)

# ── 13. Funding rate directionnel (neg→long, pos→short) ──
for th in [0.0, 0.0001, 0.0005]:
    run("13_funding_dir", f"th={th}", funding_arr < -th, funding_arr > th)

# ── 14. LS ratio extrême bas (shorts dominent → long) ──
for th in [0.8, 0.9, 0.95]:
    run("14_ls_ratio_low", f"ls<{th}", ls_ratio_arr < th, ALL_TRUE)

# ── 15. LS ratio extrême haut (longs dominent → short) ──
for th in [1.05, 1.1, 1.2]:
    run("15_ls_ratio_high", f"ls>{th}", ALL_TRUE, ls_ratio_arr > th)

# ── 16. LS ratio directionnel ──
for th in [0.9, 0.95, 1.0]:
    run("16_ls_dir", f"L<{th}_S>{1/th:.2f}", ls_ratio_arr < th, ls_ratio_arr > 1/th)

# ── 17. Top trader LS < 1 (top traders short → contrarian long) ──
for th in [0.8, 0.9, 0.95]:
    run("17_top_ls_low", f"top<{th}", top_ls_arr < th, ALL_TRUE)

# ── 18. Top trader LS > 1 (top traders long → contrarian short) ──
for th in [1.05, 1.1, 1.2]:
    run("18_top_ls_high", f"top>{th}", ALL_TRUE, top_ls_arr > th)

# ── 19. OI value z-score élevé (surexposition → mean reversion) ──
for w in [60, 288, 576]:
    z = rolling_zscore(oi_value_arr, w)
    run("19_oi_zscore_high", f"w={w}_z>1", z > 1, z > 1)

# ── 20. OI value z-score bas ──
for w in [60, 288, 576]:
    z = rolling_zscore(oi_value_arr, w)
    run("20_oi_zscore_low", f"w={w}_z<-1", z < -1, z < -1)

# ── 21. Basis (Lighter - Binance) négatif → long ──
basis = (close_arr - bn_close_arr) / bn_close_arr
for th in [-0.001, -0.002, -0.005]:
    run("21_basis_neg", f"b<{th}", basis < th, ALL_TRUE)

# ── 22. Basis positif → short ──
for th in [0.001, 0.002, 0.005]:
    run("22_basis_pos", f"b>{th}", ALL_TRUE, basis > th)

# ── 23. Basis directionnel ──
for th in [0.001, 0.002, 0.005]:
    run("23_basis_dir", f"th={th}", basis < -th, basis > th)

# ── 24. CVD divergence (prix baisse + CVD monte → long) ──
for lb in [12, 60, 288]:
    dp = np.zeros(N); dc = np.zeros(N)
    for i in range(lb, N):
        dp[i] = close_arr[i] - close_arr[i-lb]
        dc[i] = cvd_arr[i] - cvd_arr[i-lb]
    run("24_cvd_div", f"lb={lb}", (dp < 0) & (dc > 0), (dp > 0) & (dc < 0))

# ── 25. CVD momentum (CVD monte → long, baisse → short) ──
for lb in [12, 60, 288]:
    dc = np.zeros(N)
    for i in range(lb, N): dc[i] = cvd_arr[i] - cvd_arr[i-lb]
    run("25_cvd_mom", f"lb={lb}", dc > 0, dc < 0)

# ── 26. Delta volume spike (|delta| > 2× std) ──
for w in [60, 144, 288]:
    ds = rolling_std(delta_arr, w)
    run("26_delta_spike", f"w={w}", np.abs(delta_arr) > 2*ds, np.abs(delta_arr) > 2*ds)

# ── 27. Delta directionnel (taker buy domine → long) ──
for w in [12, 60, 288]:
    dm = rolling_mean(delta_arr, w)
    run("27_delta_dir", f"w={w}", dm > 0, dm < 0)

# ── 28. Taker LS vol z-score (sells extremes → long) ──
for w in [60, 144, 288]:
    z = rolling_zscore(taker_ls_arr, w)
    run("28_taker_z", f"w={w}_L<-1_S>1", z < -1, z > 1)

# ── 29. Range compression (ATR/close en baisse) ──
atr_norm = atr_14 / close_arr
for w in [60, 144, 288]:
    atr_norm_ma = rolling_mean(atr_norm, w)
    run("29_range_compress", f"w={w}", atr_norm < atr_norm_ma, atr_norm < atr_norm_ma)

# ── 30. Range expansion ──
for w in [60, 144, 288]:
    atr_norm_ma = rolling_mean(atr_norm, w)
    run("30_range_expand", f"w={w}", atr_norm > atr_norm_ma, atr_norm > atr_norm_ma)

# ── 31. Bougies consécutives baissières → long ──
consec_down = np.zeros(N)
consec_up = np.zeros(N)
for i in range(1, N):
    if close_arr[i] < close_arr[i-1]: consec_down[i] = consec_down[i-1] + 1
    else: consec_down[i] = 0
    if close_arr[i] > close_arr[i-1]: consec_up[i] = consec_up[i-1] + 1
    else: consec_up[i] = 0
for n_consec in [3, 5, 7]:
    run("31_consec_down", f"n={n_consec}", consec_down >= n_consec, consec_up >= n_consec)

# ── 32. Heure du jour (UTC) ──
hours = np.array([t.hour for t in data.index])
for h_range, label in [((8,16), '8-16'), ((0,8), '0-8'), ((16,24), '16-24')]:
    mask = (hours >= h_range[0]) & (hours < h_range[1])
    run("32_heure", label, mask, mask)

# ── 33. Jour de la semaine (0=lundi) ──
days = np.array([t.weekday() for t in data.index])
for d_range, label in [((0,5), 'lun-ven'), ((5,7), 'sam-dim'), ((0,3), 'lun-mer')]:
    mask = (days >= d_range[0]) & (days < d_range[1])
    run("33_jour", label, mask, mask)

# ── 34. Distance au plus haut local (prix loin du top → long) ──
for w in [60, 288, 576]:
    hh = rolling_max(close_arr, w)
    dist = (close_arr - hh) / hh
    run("34_dist_high", f"w={w}_<-3%", dist < -0.03, ALL_TRUE)

# ── 35. Distance au plus bas local (prix loin du bottom → short) ──
for w in [60, 288, 576]:
    ll = rolling_min(close_arr, w)
    dist = (close_arr - ll) / ll
    run("35_dist_low", f"w={w}_>3%", ALL_TRUE, dist > 0.03)

# ── 36. Bollinger squeeze (BB width < median) ──
for w in [20, 60, 140]:
    m = rolling_mean(close_arr, w)
    s = rolling_std(close_arr, w)
    bb_width = np.where(m > 0, 2*s/m, 0)
    med = np.nanmedian(bb_width[bb_width > 0])
    run("36_bb_squeeze", f"w={w}", bb_width < med, bb_width < med)

# ── 37. Bollinger expansion ──
for w in [20, 60, 140]:
    m = rolling_mean(close_arr, w)
    s = rolling_std(close_arr, w)
    bb_width = np.where(m > 0, 2*s/m, 0)
    med = np.nanmedian(bb_width[bb_width > 0])
    run("37_bb_expand", f"w={w}", bb_width > med, bb_width > med)

# ── 38. OI + CVD combo (OI monte + CVD diverge) ──
for lb in [12, 60, 288]:
    doi = np.zeros(N); dp = np.zeros(N); dc = np.zeros(N)
    for i in range(lb, N):
        doi[i] = oi_arr[i] - oi_arr[i-lb]
        dp[i] = close_arr[i] - close_arr[i-lb]
        dc[i] = cvd_arr[i] - cvd_arr[i-lb]
    # OI monte + prix baisse + CVD monte = shorts s'accumulent mais acheteurs résistent
    run("38_oi_cvd_combo", f"lb={lb}", (doi > 0) & (dp < 0) & (dc > 0), (doi > 0) & (dp > 0) & (dc < 0))

# ── 39. Funding + LS combo ──
for fr_th, ls_th in [(-0.0001, 0.9), (-0.0005, 0.85), (0.0, 0.95)]:
    run("39_funding_ls", f"fr<{fr_th}_ls<{ls_th}",
        (funding_arr < fr_th) & (ls_ratio_arr < ls_th), ALL_TRUE)

# ── 40. Volume décroissant (exhaustion) ──
for w in [12, 60, 288]:
    vol_roc = rolling_pctchange(volume_arr, w)
    run("40_vol_decay", f"w={w}_roc<-30%", vol_roc < -0.3, vol_roc < -0.3)

# ── 41. Volume croissant ──
for w in [12, 60, 288]:
    vol_roc = rolling_pctchange(volume_arr, w)
    run("41_vol_growth", f"w={w}_roc>50%", vol_roc > 0.5, vol_roc > 0.5)

# ── 42. Close/MA ratio (distance à la MA en %) ──
close_ma_ratio = (close_arr - ma_arr) / ma_arr
for th in [0.01, 0.02, 0.03]:
    # Long seulement si pas trop loin sous la MA, short pas trop loin au-dessus
    run("42_close_ma_dist", f"th={th}",
        close_ma_ratio < -th, close_ma_ratio > th)

# ── 43. Trend filter (MA slope) ──
ma_slope = rolling_pctchange(ma_arr, 60)
for th in [0.0, 0.005, 0.01]:
    # Long en downtrend (contrarian), short en uptrend
    run("43_trend_contrarian", f"slope_th={th}", ma_slope < -th, ma_slope > th)

# ── 44. Trend following (même direction) ──
for th in [0.0, 0.005, 0.01]:
    run("44_trend_follow", f"slope_th={th}", ma_slope > th, ma_slope < -th)

# ── 45. OI per unit price (OI normalisé par le prix) ──
oi_norm = oi_value_arr / close_arr
for w in [60, 288, 576]:
    z = rolling_zscore(oi_norm, w)
    run("45_oi_norm_z", f"w={w}_z>1", z > 1, z > 1)

# ── 46. Taker buy/sell imbalance absolue ──
imbalance = np.where(bn_vol_arr > 0, (taker_buy_arr - taker_sell_arr) / bn_vol_arr, 0)
for w in [12, 60, 288]:
    z = rolling_zscore(imbalance, w)
    run("46_imbalance_z", f"w={w}_L<-1_S>1", z < -1, z > 1)

# ── 47. High-low range vs ATR (inside bar = range < ATR) ──
hl_range = high_arr - low_arr
for mult in [0.5, 0.7, 1.0]:
    inside = hl_range < mult * atr_14
    run("47_inside_bar", f"range<{mult}xATR14", inside, inside)

# ── 48. Multi-timeframe : prix sous MA longue → favoriser long ──
for w in [288, 576, 1152]:
    ma_long = rolling_mean(close_arr, w)
    run("48_mtf_ma", f"ma_{w}", close_arr < ma_long, close_arr > ma_long)

# ── 49. Combo RSI + Vol basse (mean reversion optimal) ──
for rsi_p, vol_w in [(14, 60), (28, 144), (14, 288)]:
    r = rsi(close_arr, rsi_p)
    a = atr(high_arr, low_arr, close_arr, vol_w)
    med_a = np.nanmedian(a[a>0])
    run("49_rsi_lowvol", f"rsi{rsi_p}_atr{vol_w}",
        (r < 35) & (a < med_a), (r > 65) & (a < med_a))

# ── 50. Combo distance + volume spike ──
for dist_w, vol_w in [(60, 60), (288, 288), (576, 144)]:
    hh = rolling_max(close_arr, dist_w)
    ll = rolling_min(close_arr, dist_w)
    dist_high = (close_arr - hh) / hh
    dist_low = (close_arr - ll) / ll
    vm = rolling_mean(volume_arr, vol_w)
    vol_ok = volume_arr > 1.5 * vm
    run("50_dist_volspike", f"dw={dist_w}_vw={vol_w}",
        (dist_high < -0.02) & vol_ok, (dist_low > 0.02) & vol_ok)

# ═══════════════════════════════════════════════════════════════════════
# RÉSULTATS
# ═══════════════════════════════════════════════════════════════════════
elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"TERMINÉ : {total} backtests en {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"{'='*70}")

df = pd.DataFrame(results)
df.to_csv('research/alpha_results.csv', index=False)

# Top par Sharpe
print(f"\nBASELINE: Return={BASELINE['return']:.1f}% Sharpe={BASELINE['sharpe']:.3f} "
      f"Sortino={BASELINE['sortino']:.3f} MaxDD={BASELINE['maxdd']:.1f}%\n")

winners = df[(df['beats_sharpe']) | (df['beats_return']) | (df['beats_dd'])]
if len(winners) > 0:
    print(f"{'='*70}")
    print(f"GAGNANTS ({len(winners)} sur {total} battent la baseline sur au moins 1 critère):")
    print(f"{'='*70}")
    for _, r in winners.sort_values('sharpe', ascending=False).head(30).iterrows():
        tags = []
        if r['beats_sharpe']: tags.append('Sharpe')
        if r['beats_return']: tags.append('Return')
        if r['beats_dd']: tags.append('DD')
        print(f"  {r['hypothesis']:30s} {r['params']:25s} → "
              f"Ret={r['return_%']:+8.1f}% Sharpe={r['sharpe']:6.3f} "
              f"Sort={r['sortino']:6.3f} DD={r['maxdd_%']:5.1f}% "
              f"Trades={r['trades']:4d} | bat: {', '.join(tags)}")
else:
    print("Aucun gagnant.")

# Top améliorateurs de DD
dd_better = df[df['beats_dd']].sort_values('maxdd_%')
if len(dd_better) > 0:
    print(f"\n{'='*70}")
    print(f"TOP DD RÉDUCTEURS (baseline DD={BASELINE['maxdd']:.1f}%):")
    print(f"{'='*70}")
    for _, r in dd_better.head(15).iterrows():
        print(f"  {r['hypothesis']:30s} {r['params']:25s} → "
              f"DD={r['maxdd_%']:5.1f}% Ret={r['return_%']:+8.1f}% "
              f"Sharpe={r['sharpe']:6.3f} Trades={r['trades']:4d}")
