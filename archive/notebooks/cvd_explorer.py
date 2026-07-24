import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _imports():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import os
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    CM_PAIRS = {
        'AAVE','ADA','APT','AVAX','BCH','BNB','BTC','DOGE','DOT',
        'ETC','ETH','FIL','LINK','LTC','NEAR','OP','SOL','SUI',
        'TRX','UNI','WIF','WLD','XLM','XRP',
    }

    SIGNAL_META = {
        'S1_short_buildup':         {'label':'S1 Short build-up → squeeze',       'color':'#ff6d00','tier':'S','bullish':True},
        'S2_short_covering':        {'label':'S2 Short covering (move épuisé)',    'color':'#ffd600','tier':'S','bullish':False},
        'A1_spot_accum':            {'label':'A1 Accum spot / retail short',       'color':'#00e676','tier':'A','bullish':True},
        'A2_retail_long_pros_hedge':{'label':'A2 Retail long / pros hedge',        'color':'#ff1744','tier':'A','bullish':False},
        'A3_bullish_div':           {'label':'A3 Bullish div (CVD UM↑ / Prix↓)',   'color':'#00b0ff','tier':'A','bullish':True},
    }

    def load_ohlcv(path, tf_minutes):
        df = pd.read_csv(path, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df = df.set_index('date').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        for c in ['open','high','low','close','volume','taker_buy_volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if tf_minutes > 1:
            df = df.resample(f'{tf_minutes}min').agg({
                'open':'first','high':'max','low':'min','close':'last',
                'volume':'sum','taker_buy_volume':'sum',
            }).dropna(subset=['close'])
        return df

    def load_metrics(pair, tf_minutes):
        path = f'data/raw/binance/um/metrics/{pair}USDT.csv'
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df = df.set_index('date').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df['oi'] = pd.to_numeric(df['oi'], errors='coerce')
        if tf_minutes > 5:
            df = df.resample(f'{tf_minutes}min').agg({'oi':'last'}).dropna(subset=['oi'])
        return df

    def compute_cvd(df):
        return (2 * df['taker_buy_volume'] - df['volume']).cumsum()

    def detect_signals(close, cvd_um, cvd_spot, cvd_cm, oi, window, price_thr_pct):
        idx = close.index

        def cvd_dir(s):
            return np.sign(s.reindex(idx, method='ffill').diff(window).fillna(0))

        def pct_dir(s):
            chg = s.reindex(idx, method='ffill').pct_change(window).fillna(0) * 100
            return np.where(chg > price_thr_pct, 1, np.where(chg < -price_thr_pct, -1, 0))

        def oi_dir_native(s):
            chg = s.pct_change(max(1, window // 5)).fillna(0) * 100
            d = pd.Series(
                np.where(chg > price_thr_pct, 1, np.where(chg < -price_thr_pct, -1, 0)),
                index=s.index,
            )
            return d.reindex(idx, method='ffill').fillna(0).values

        d_price = pct_dir(close)
        d_um    = cvd_dir(cvd_um).values
        d_spot  = cvd_dir(cvd_spot).values if cvd_spot is not None else np.zeros(len(idx))
        d_oi    = oi_dir_native(oi)         if oi is not None         else np.zeros(len(idx))
        d_cm    = cvd_dir(cvd_cm).values    if cvd_cm is not None     else np.zeros(len(idx))

        sig = pd.DataFrame({
            'S1_short_buildup':         (d_oi == 1)  & (d_um == -1) & (d_price == -1),
            'S2_short_covering':        (d_oi == -1) & (d_price == 1),
            'A1_spot_accum':            (d_spot == 1) & (d_um == -1),
            'A2_retail_long_pros_hedge':(d_um == 1)  & (d_cm == -1),
            'A3_bullish_div':           (d_price == -1) & (d_um == 1),
        }, index=idx)

        if cvd_spot is None:
            sig['A1_spot_accum'] = False
        if cvd_cm is None:
            sig['A2_retail_long_pros_hedge'] = False

        return sig

    def event_study(close_arr, signal_arr, horizon):
        """
        Pour chaque occurrence (rising edge), calcule le return forward à chaque step.
        Retourne (mean_curve, std_curve, n_events, finals) ou (None,None,0,None).
        finals : array des returns finaux par event (pour winrate).
        """
        n = len(close_arr)
        fwd = []
        in_zone = False
        for i in range(n - horizon):
            v = signal_arr[i]
            if v and not in_zone:
                base = close_arr[i]
                if base > 0:
                    fwd.append([(close_arr[i+k] / base - 1) * 100 for k in range(1, horizon + 1)])
            in_zone = bool(v)
        if not fwd:
            return None, None, 0, None
        arr = np.array(fwd)
        return arr.mean(axis=0), arr.std(axis=0), len(fwd), arr[:, -1]

    # ── Paires disponibles ────────────────────────────────────────────────────
    _um_pairs  = set(f.replace('USDT.csv','') for f in os.listdir('data/raw/binance/um/1m')      if f.endswith('USDT.csv'))
    _spot_pairs= set(f.replace('USDT.csv','') for f in os.listdir('data/raw/binance/spot/1m')    if f.endswith('USDT.csv'))
    _met_pairs = set(f.replace('USDT.csv','') for f in os.listdir('data/raw/binance/um/metrics') if f.endswith('USDT.csv'))
    # Requis : um + metrics. Spot optionnel (A1 désactivé si absent)
    AVAILABLE_PAIRS = sorted(_um_pairs & _met_pairs)
    SPOT_PAIRS = _spot_pairs

    return (
        AVAILABLE_PAIRS, CM_PAIRS, SIGNAL_META, SPOT_PAIRS,
        compute_cvd, detect_signals, event_study, go,
        load_metrics, load_ohlcv, make_subplots,
        mo, np, optuna, os, pd,
    )


@app.cell
def _controls(AVAILABLE_PAIRS, mo):
    pair_sel   = mo.ui.dropdown(options={p:p for p in AVAILABLE_PAIRS}, value='BTC', label='Paire')
    tf_sel     = mo.ui.dropdown(
        options={'1m':1,'5m':5,'15m':15,'30m':30,'1h':60,'4h':240},
        value='1m', label='Timeframe',
    )
    n_candles  = mo.ui.slider(200, 30000, value=1000, step=200, label='Nb bougies affichées', show_value=True)
    show_cm    = mo.ui.checkbox(value=True, label='CVD Coin-M')
    show_sigs  = mo.ui.checkbox(value=True, label='Afficher signaux')
    sig_window = mo.ui.slider(3, 500, value=10, step=1,  label='Fenêtre signal (bougies)', show_value=True)
    price_thr  = mo.ui.slider(0.1, 5.0, value=0.3, step=0.1, label='Seuil Prix/OI (%)', show_value=True)

    mo.output.replace(mo.vstack([
        mo.md("## CVD Explorer — Divergences"),
        mo.hstack([pair_sel, tf_sel, n_candles, show_cm], justify="start", gap=3),
        mo.hstack([show_sigs, sig_window, price_thr], justify="start", gap=3),
        mo.md(
            "| | Signal | Tier | Sens |\n|---|---|---|---|\n"
            "| 🟠 | S1 OI↑ CVD UM↓ Prix↓ → squeeze | S | Haussier |\n"
            "| 🟡 | S2 OI↓ Prix↑ → short covering | S | Neutre |\n"
            "| 🟢 | A1 CVD Spot↑ CVD UM↓ → accum spot | A | Haussier |\n"
            "| 🔴 | A2 CVD UM↑ CVD CM↓ → pros hedge | A | Baissier |\n"
            "| 🔵 | A3 Prix↓ CVD UM↑ → bullish div | A | Haussier |\n"
        ),
    ]))
    return n_candles, pair_sel, price_thr, show_cm, show_sigs, sig_window, tf_sel


@app.cell
def _plot(
    CM_PAIRS, SIGNAL_META, SPOT_PAIRS,
    compute_cvd, detect_signals, go, load_metrics, load_ohlcv, make_subplots,
    mo, np, os, pd,
    n_candles, pair_sel, price_thr, show_cm, show_sigs, sig_window, tf_sel,
):
    _pair   = pair_sel.value
    _tf     = tf_sel.value
    _n      = n_candles.value
    _has_cm = _pair in CM_PAIRS and show_cm.value
    _has_spot = _pair in SPOT_PAIRS

    try:
        _um   = load_ohlcv(f'data/raw/binance/um/1m/{_pair}USDT.csv',   _tf).iloc[-_n:]
        _spot = load_ohlcv(f'data/raw/binance/spot/1m/{_pair}USDT.csv', _tf).iloc[-_n:] if _has_spot else None
        _cm_df= load_ohlcv(f'data/raw/binance/cm/1m/{_pair}USD_PERP.csv', _tf).iloc[-_n:] \
                if (_has_cm and os.path.exists(f'data/raw/binance/cm/1m/{_pair}USD_PERP.csv')) else None
        _t0, _t1 = _um.index[0], _um.index[-1]
        _met_raw = load_metrics(_pair, _tf)
        _metrics      = _met_raw.loc[_t0:_t1] if _met_raw is not None else None
        _metrics_wide = _met_raw.iloc[-max(_n, 2000):] if _met_raw is not None else None
    except Exception as _e:
        import traceback as _tb
        mo.stop(True, mo.callout(mo.md(f"Erreur : `{_e}`\n```\n{_tb.format_exc()}\n```"), kind="danger"))

    _cvd_um   = compute_cvd(_um)
    _cvd_spot = compute_cvd(_spot) if _spot is not None else None
    _cvd_cm   = compute_cvd(_cm_df) if _cm_df is not None else None
    _oi       = _metrics['oi']      if (_metrics is not None and 'oi' in _metrics.columns) else None
    _oi_wide  = _metrics_wide['oi'] if (_metrics_wide is not None and 'oi' in _metrics_wide.columns) else None

    _signals = None
    if show_sigs.value:
        _signals = detect_signals(
            _um['close'], _cvd_um, _cvd_spot, _cvd_cm,
            _oi_wide, sig_window.value, price_thr.value,
        )

    # ── Subplots ─────────────────────────────────────────────────────────────
    _has_oi   = _oi is not None and len(_oi) > 0
    _n_rows   = 2 + (1 if _cvd_spot is not None else 0) + (1 if _cvd_cm is not None else 0) + (1 if _has_oi else 0)
    _base_h   = 0.44
    _row_h    = [_base_h] + [(1 - _base_h) / (_n_rows - 1)] * (_n_rows - 1)

    _titles = [f'{_pair} — {tf_sel.value}', 'CVD USDT-M']
    if _cvd_spot is not None: _titles.append('CVD Spot')
    if _cvd_cm   is not None: _titles.append('CVD Coin-M')
    if _has_oi:               _titles.append('Open Interest')

    _fig = make_subplots(rows=_n_rows, cols=1, shared_xaxes=True,
                         vertical_spacing=0.018, row_heights=_row_h,
                         subplot_titles=_titles)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _bands(sig_s, color, row):
        if sig_s is None or not sig_s.any(): return
        _in, _x0 = False, None
        for _ts, _v in sig_s.items():
            if _v and not _in:  _in, _x0 = True, _ts
            elif not _v and _in:
                _in = False
                _fig.add_vrect(x0=_x0, x1=_ts, fillcolor=color, opacity=0.17, line_width=0, row=row, col=1)
        if _in:
            _fig.add_vrect(x0=_x0, x1=sig_s.index[-1], fillcolor=color, opacity=0.17, line_width=0, row=row, col=1)

    # ── Row 1 : Candlestick ───────────────────────────────────────────────────
    _fig.add_trace(go.Candlestick(
        x=_um.index, open=_um['open'], high=_um['high'], low=_um['low'], close=_um['close'],
        name='OHLC',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',  decreasing_fillcolor='#ef5350',
        showlegend=False,
    ), row=1, col=1)

    if _signals is not None:
        for _sn, _sm in SIGNAL_META.items():
            _mask = _signals[_sn]
            if not _mask.any(): continue
            _rise = _mask & ~_mask.shift(1, fill_value=False)
            _xs   = _um.index[_rise]
            _ys   = _um['low'][_rise]*0.9985  if _sm['bullish'] else _um['high'][_rise]*1.0015
            _sym  = 'triangle-up' if _sm['bullish'] else 'triangle-down'
            _fig.add_trace(go.Scatter(
                x=_xs, y=_ys, mode='markers',
                marker=dict(symbol=_sym, size=8, color=_sm['color'], line=dict(width=0.5, color='white')),
                name=_sm['label'],
                hovertemplate=f"<b>{_sm['label']}</b><br>%{{x}}<extra></extra>",
            ), row=1, col=1)
            _bands(_mask, _sm['color'], 1)

    # ── Row 2 : CVD UM ───────────────────────────────────────────────────────
    _cur = 2
    _fig.add_trace(go.Scatter(x=_cvd_um.index, y=_cvd_um, name='CVD UM',
        line=dict(color='#42a5f5', width=1), fill='tozeroy',
        fillcolor='rgba(66,165,245,0.12)', showlegend=False), row=_cur, col=1)
    if _signals is not None:
        for _sn in ['S1_short_buildup','A1_spot_accum','A3_bullish_div','A2_retail_long_pros_hedge']:
            _bands(_signals[_sn], SIGNAL_META[_sn]['color'], _cur)
    _cur += 1

    # ── Row CVD Spot ─────────────────────────────────────────────────────────
    if _cvd_spot is not None:
        _fig.add_trace(go.Scatter(x=_cvd_spot.index, y=_cvd_spot, name='CVD Spot',
            line=dict(color='#ab47bc', width=1), fill='tozeroy',
            fillcolor='rgba(171,71,188,0.12)', showlegend=False), row=_cur, col=1)
        if _signals is not None: _bands(_signals['A1_spot_accum'], SIGNAL_META['A1_spot_accum']['color'], _cur)
        _cur += 1

    # ── Row CVD CM ───────────────────────────────────────────────────────────
    if _cvd_cm is not None:
        _fig.add_trace(go.Scatter(x=_cvd_cm.index, y=_cvd_cm, name='CVD CM',
            line=dict(color='#ffca28', width=1), fill='tozeroy',
            fillcolor='rgba(255,202,40,0.12)', showlegend=False), row=_cur, col=1)
        if _signals is not None: _bands(_signals['A2_retail_long_pros_hedge'], SIGNAL_META['A2_retail_long_pros_hedge']['color'], _cur)
        _cur += 1

    # ── Row OI ───────────────────────────────────────────────────────────────
    if _has_oi:
        _fig.add_trace(go.Scatter(x=_oi.index, y=_oi, name='OI',
            line=dict(color='#ff7043', width=1), fill='tozeroy',
            fillcolor='rgba(255,112,67,0.10)', showlegend=False), row=_cur, col=1)
        if _signals is not None:
            _bands(_signals['S1_short_buildup'], SIGNAL_META['S1_short_buildup']['color'], _cur)
            _bands(_signals['S2_short_covering'], SIGNAL_META['S2_short_covering']['color'], _cur)

    _fig.update_layout(
        template='plotly_dark', height=860,
        margin=dict(l=60, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='#131722', plot_bgcolor='#131722',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
                    font=dict(size=11), bgcolor='rgba(0,0,0,0.4)'),
    )
    for _r in range(1, _n_rows + 1):
        _fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)',
                          zerolinecolor='rgba(255,255,255,0.15)', zerolinewidth=1, row=_r, col=1)
        _fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)',
                          showticklabels=(_r == _n_rows), row=_r, col=1)

    _n_sigs = int(sum(_signals[c].sum() for c in _signals.columns)) if _signals is not None else 0
    _spot_note = '' if _has_spot else ' · ⚠️ Spot absent (A1 désactivé)'
    mo.output.replace(mo.vstack([
        mo.md(f"**{_pair}** · `{tf_sel.value}` · {len(_um):,} bougies · CM:{'✓' if _cvd_cm is not None else '✗'}{_spot_note} · Signaux: **{_n_sigs}**"),
        mo.ui.plotly(_fig),
    ]))
    return


# ═══════════════════════════════════════════════════════════════════════════════
#  EVENT STUDY + OPTIMISATION TPE
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell
def _study_controls(mo):
    study_horizon = mo.ui.slider(10, 300, value=50, step=5, label='Horizon étude (bougies)', show_value=True)
    n_trials_sel  = mo.ui.slider(20, 300, value=100, step=10, label='Trials TPE', show_value=True)
    run_study_btn = mo.ui.button(label='▶ Event Study (params actuels)', value=0, on_click=lambda v: v+1)
    run_opti_btn  = mo.ui.button(label='⚡ Optimiser (TPE sur toutes les données)', value=0, on_click=lambda v: v+1)

    mo.output.replace(mo.vstack([
        mo.md("---\n## Event Study & Optimisation"),
        mo.hstack([study_horizon, n_trials_sel], justify="start", gap=3),
        mo.hstack([run_study_btn, run_opti_btn], justify="start", gap=3),
    ]))
    return n_trials_sel, run_opti_btn, run_study_btn, study_horizon


@app.cell
def _event_study_cell(
    CM_PAIRS, SIGNAL_META, SPOT_PAIRS,
    compute_cvd, detect_signals, event_study, go, load_metrics, load_ohlcv,
    mo, np, os, pd,
    pair_sel, tf_sel, sig_window, price_thr, study_horizon, run_study_btn,
):
    if not run_study_btn.value:
        mo.stop(True, mo.callout(mo.md("Clique **▶ Event Study** pour lancer."), kind="neutral"))

    mo.output.replace(mo.callout(mo.md("Calcul en cours…"), kind="info"))

    _pair = pair_sel.value
    _tf   = tf_sel.value
    _has_spot = _pair in SPOT_PAIRS
    _has_cm   = _pair in CM_PAIRS

    try:
        _um_all   = load_ohlcv(f'data/raw/binance/um/1m/{_pair}USDT.csv', _tf)
        _spot_all = load_ohlcv(f'data/raw/binance/spot/1m/{_pair}USDT.csv', _tf) if _has_spot else None
        _cm_all   = load_ohlcv(f'data/raw/binance/cm/1m/{_pair}USD_PERP.csv', _tf) \
                    if (_has_cm and os.path.exists(f'data/raw/binance/cm/1m/{_pair}USD_PERP.csv')) else None
        _met_all  = load_metrics(_pair, _tf)
        _oi_all   = _met_all['oi'] if _met_all is not None else None
    except Exception as _e:
        mo.stop(True, mo.callout(mo.md(f"Erreur : `{_e}`"), kind="danger"))

    _cvd_um_all   = compute_cvd(_um_all)
    _cvd_spot_all = compute_cvd(_spot_all) if _spot_all is not None else None
    _cvd_cm_all   = compute_cvd(_cm_all)   if _cm_all   is not None else None

    _sigs_all = detect_signals(
        _um_all['close'], _cvd_um_all, _cvd_spot_all, _cvd_cm_all,
        _oi_all, sig_window.value, price_thr.value,
    )
    _close_arr = _um_all['close'].values
    _H = study_horizon.value

    _rows_es = []
    for _sn, _sm in SIGNAL_META.items():
        _mean, _std, _n_ev, _finals = event_study(_close_arr, _sigs_all[_sn].values, _H)
        if _mean is None: continue
        _rows_es.append((_sn, _sm, _mean, _std, _n_ev))

    if not _rows_es:
        mo.stop(True, mo.callout(mo.md("Aucun signal détecté avec ces paramètres."), kind="warn"))

    # ── Plot event study ──────────────────────────────────────────────────────
    _n_sig = len(_rows_es)
    _fig_es = go.Figure()
    _steps = list(range(1, _H + 1))

    for _sn, _sm, _mean, _std, _n_ev in _rows_es:
        _col = _sm['color']
        _fig_es.add_trace(go.Scatter(
            x=_steps, y=_mean,
            name=f"{_sm['label']} (n={_n_ev})",
            line=dict(color=_col, width=2),
        ))
        _fig_es.add_trace(go.Scatter(
            x=_steps + _steps[::-1],
            y=list(_mean + _std) + list((_mean - _std)[::-1]),
            fill='toself',
            fillcolor=f"rgba({int(_col[1:3],16)},{int(_col[3:5],16)},{int(_col[5:7],16)},0.10)",
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))

    _fig_es.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.3)')
    _fig_es.update_layout(
        template='plotly_dark', height=420,
        title=f'Event Study — {_pair} {tf_sel.value} · window={sig_window.value} · thr={price_thr.value}%',
        xaxis_title='Bougies après signal', yaxis_title='Return moyen (%)',
        paper_bgcolor='#131722', plot_bgcolor='#131722',
        legend=dict(font=dict(size=11)),
        margin=dict(l=60, r=20, t=50, b=40),
    )

    _summary_rows = []
    for _sn, _sm, _mean, _std, _n_ev in _rows_es:
        _final   = _mean[-1]
        _mid     = _mean[_H//2]
        _pct_pos = float((_mean > 0).mean() * 100)
        _summary_rows.append({
            'Signal': _sm['label'], 'N événements': _n_ev,
            'Return final (%)': round(_final, 3),
            'Return mi-horizon (%)': round(_mid, 3),
            '% steps positifs': round(_pct_pos, 1),
        })

    mo.output.replace(mo.vstack([
        mo.md(f"### Event Study — {_pair} · {len(_um_all):,} bougies · H={_H}"),
        mo.ui.table(pd.DataFrame(_summary_rows), selection=None),
        mo.ui.plotly(_fig_es),
    ]))
    return


@app.cell
def _optimize_cell(
    CM_PAIRS, SIGNAL_META, SPOT_PAIRS,
    compute_cvd, detect_signals, event_study, go, load_metrics, load_ohlcv,
    mo, np, optuna, os, pd,
    pair_sel, tf_sel, n_trials_sel, run_opti_btn,
):
    if not run_opti_btn.value:
        mo.stop(True, mo.callout(mo.md("Clique **⚡ Optimiser** pour lancer la recherche TPE."), kind="neutral"))

    mo.output.replace(mo.callout(mo.md("Optimisation TPE en cours… (patience)"), kind="info"))

    _pair = pair_sel.value
    _tf   = tf_sel.value
    _has_spot = _pair in SPOT_PAIRS
    _has_cm   = _pair in CM_PAIRS

    try:
        _um_all   = load_ohlcv(f'data/raw/binance/um/1m/{_pair}USDT.csv', _tf)
        _spot_all = load_ohlcv(f'data/raw/binance/spot/1m/{_pair}USDT.csv', _tf) if _has_spot else None
        _cm_all   = load_ohlcv(f'data/raw/binance/cm/1m/{_pair}USD_PERP.csv', _tf) \
                    if (_has_cm and os.path.exists(f'data/raw/binance/cm/1m/{_pair}USD_PERP.csv')) else None
        _met_all  = load_metrics(_pair, _tf)
        _oi_all   = _met_all['oi'] if _met_all is not None else None
    except Exception as _e:
        mo.stop(True, mo.callout(mo.md(f"Erreur chargement : `{_e}`"), kind="danger"))

    _cvd_um_all   = compute_cvd(_um_all)
    _cvd_spot_all = compute_cvd(_spot_all) if _spot_all is not None else None
    _cvd_cm_all   = compute_cvd(_cm_all)   if _cm_all   is not None else None
    _close_arr    = _um_all['close'].values

    # ── TPE par signal — cherche window, thr ET horizon ───────────────────────
    _best_params  = {}
    _studies_info = []

    for _sn, _sm in SIGNAL_META.items():
        def _objective(trial, _sn=_sn, _sm=_sm):
            _w   = trial.suggest_int('window',  10, 500, step=10)
            _thr = trial.suggest_float('thr',   0.5, 5.0, step=0.5)
            _h   = trial.suggest_int('horizon', 10, 300, step=5)
            _sigs = detect_signals(
                _um_all['close'], _cvd_um_all, _cvd_spot_all, _cvd_cm_all,
                _oi_all, _w, _thr,
            )
            _mean, _std, _n_ev, _finals = event_study(_close_arr, _sigs[_sn].values, _h)
            if _mean is None or _n_ev < 5:
                return -999.0
            # Objectif : return_final × winrate  (les deux doivent être élevés)
            _sign    = 1.0 if _sm['bullish'] else -1.0
            _ret     = float(_sign * _mean[-1])
            _winrate = float(np.mean(_sign * _finals > 0))
            return _ret * _winrate

        _study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        _study.optimize(_objective, n_trials=n_trials_sel.value, show_progress_bar=False)

        _bp = _study.best_params
        _bv = _study.best_value
        _best_params[_sn] = {
            'window':  _bp['window'],
            'thr':     _bp['thr'],
            'horizon': _bp['horizon'],
            'score':   _bv,
        }
        _studies_info.append({
            'Signal':      _sm['label'],
            'Tier':        _sm['tier'],
            'window*':     _bp['window'],
            'thr* (%)':    _bp['thr'],
            'horizon*':    _bp['horizon'],
            'score (ret×WR)': round(_bv, 4),
            'N trials':    n_trials_sel.value,
        })

    # ── Event study avec params optimisés (chaque signal à son horizon) ───────
    _fig_opt = go.Figure()
    _opt_summary = []

    for _sn, _sm in SIGNAL_META.items():
        _bp  = _best_params[_sn]
        _H   = _bp['horizon']
        _sigs_opt = detect_signals(
            _um_all['close'], _cvd_um_all, _cvd_spot_all, _cvd_cm_all,
            _oi_all, _bp['window'], _bp['thr'],
        )
        _mean, _std, _n_ev, _finals = event_study(_close_arr, _sigs_opt[_sn].values, _H)
        if _mean is None: continue
        _sign    = 1.0 if _sm['bullish'] else -1.0
        _winrate = float(np.mean(_sign * _finals > 0)) * 100
        _col     = _sm['color']
        _steps   = list(range(1, _H + 1))
        _fig_opt.add_trace(go.Scatter(
            x=_steps, y=_mean,
            name=f"{_sm['label']} (n={_n_ev}, w={_bp['window']}, thr={_bp['thr']}%, H={_H})",
            line=dict(color=_col, width=2),
        ))
        _fig_opt.add_trace(go.Scatter(
            x=_steps + _steps[::-1],
            y=list(_mean + _std) + list((_mean - _std)[::-1]),
            fill='toself',
            fillcolor=f"rgba({int(_col[1:3],16)},{int(_col[3:5],16)},{int(_col[5:7],16)},0.10)",
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))
        _opt_summary.append({
            'Signal':          _sm['label'],
            'N événements':    _n_ev,
            'window*':         _bp['window'],
            'thr* (%)':        _bp['thr'],
            'horizon*':        _H,
            'Return final (%)':round(float(_mean[-1]), 3),
            'Winrate (%)':     round(_winrate, 1),
            'Score (ret×WR)':  round(float(_mean[-1]) * _winrate / 100, 4),
        })

    _fig_opt.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.3)')
    _fig_opt.update_layout(
        template='plotly_dark', height=460,
        title=f'Event Study post-TPE — {_pair} {tf_sel.value} · {n_trials_sel.value} trials (window, thr, horizon)',
        xaxis_title='Bougies après signal', yaxis_title='Return moyen (%)',
        paper_bgcolor='#131722', plot_bgcolor='#131722',
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=55, b=40),
    )

    _df_summary = pd.DataFrame(_opt_summary).sort_values('Score (ret×WR)', ascending=False)

    mo.output.replace(mo.vstack([
        mo.md(f"### Résultats TPE — {_pair} · {n_trials_sel.value} trials · {len(_um_all):,} bougies"),
        mo.callout(
            mo.md("**Objectif** : `return_final × winrate` — maximise à la fois le gain moyen **et** la fiabilité.  \n"
                  "**horizon** est lui aussi optimisé automatiquement par signal."),
            kind="info",
        ),
        mo.md("#### Meilleurs paramètres trouvés"),
        mo.ui.table(pd.DataFrame(_studies_info), selection=None),
        mo.md("#### Event Study post-optimisation (trié par score)"),
        mo.ui.table(_df_summary, selection=None),
        mo.ui.plotly(_fig_opt),
    ]))
    return


if __name__ == "__main__":
    app.run()
