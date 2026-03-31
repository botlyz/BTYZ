import marimo

__generated_with = "0.21.1"
app = marimo.App(width="wide")


@app.cell
def _imports():
    import sys, os, io, base64, pickle, math
    sys.path.insert(0, '.')
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    return base64, io, math, mo, np, os, pd, pickle, plt, mdates, sys


@app.cell
def _controls(mo, os):
    _pkl_dir = 'cache/ml'
    _files   = sorted([f for f in os.listdir(_pkl_dir) if f.endswith('.pkl')])

    pkl_select = mo.ui.dropdown(
        options  = _files,
        value    = 'xgb_swing_30min.pkl' if 'xgb_swing_30min.pkl' in _files else _files[0],
        label    = 'Modèle ML',
    )
    capital_input = mo.ui.number(
        value = 10000, start = 100, stop = 1_000_000, step = 100,
        label = 'Capital initial ($)',
    )
    mo.vstack([
        mo.md("# Evaluate — Backtest des signaux ML"),
        mo.hstack([pkl_select, capital_input]),
    ])
    return capital_input, pkl_select


@app.cell
def _load(mo, os, pd, pickle, pkl_select):
    _path = os.path.join('cache/ml', pkl_select.value)
    with open(_path, 'rb') as _f:
        result = pickle.load(_f)

    preds = result['preds_oos']

    # ── Détection du type de modèle ───────────────────────────────────────────
    if isinstance(preds, pd.DataFrame) and 'rr_pred' in preds.columns:
        model_type   = 'rr'
        signal_vals  = preds['rr_pred']
        config_str   = f"horizon={result.get('horizon','?')}m"
        type_str     = "R/R dynamique"
    else:
        model_type   = 'classification'
        signal_vals  = preds
        cfg          = result.get('config', {})
        config_str   = f"TP={cfg.get('tp',0.015)*100:.1f}% · SL={cfg.get('sl',0.009)*100:.1f}% · max={cfg.get('max_bars',30)}m"
        type_str     = "Classification (Triple Barrier)"

    results_df = result['results_df']

    mo.callout(
        mo.md(f"**{pkl_select.value}** · {type_str} · {len(signal_vals):,} prédictions OOS · {config_str}"),
        kind="success",
    )
    return config_str, model_type, preds, result, results_df, signal_vals, type_str


@app.cell
def _quality(base64, io, math, mo, model_type, np, pd, plt, result, results_df, signal_vals):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))

    # ── Histogramme distribution ──────────────────────────────────────────────
    _ax = _axes[0]
    _vals = signal_vals.dropna().values
    if model_type == 'classification':
        _ax.hist(_vals, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
        _ax.set_xlabel('Probabilité prédite')
        _ax.set_title('Distribution des probabilités OOS')
        _ax.axvline(0.5, color='red', linestyle='--', alpha=0.8, label='seuil=0.5')
        _base = results_df['tp_rate'].mean()
        _ax.axvline(_base, color='orange', linestyle=':', alpha=0.8, label=f'tp_rate={_base:.2f}')
        _ax.legend(fontsize=8)
    else:
        _ax.hist(np.clip(_vals, 0, 8), bins=60, color='darkorange', edgecolor='white', linewidth=0.3)
        _ax.set_xlabel('R/R prédit (clippé à 8)')
        _ax.set_title('Distribution R/R prédit OOS')
        _ax.axvline(1.5, color='red', linestyle='--', alpha=0.8, label='seuil=1.5')
        _ax.legend(fontsize=8)
    _ax.set_ylabel('Fréquence')
    _ax.grid(True, linestyle='--', linewidth=0.4)

    # ── Courbe qualité vs seuil ───────────────────────────────────────────────
    _ax2 = _axes[1]
    if model_type == 'classification':
        _thrs      = np.arange(0.30, 0.85, 0.05)
        _precs, _ns = [], []
        _labels_oos = result['labels_oos']
        for _t in _thrs:
            _mask = signal_vals > _t
            _ns.append(_mask.sum())
            if _mask.sum() > 0:
                _matched = _labels_oos.reindex(signal_vals[_mask].index)
                _precs.append(float(_matched.mean()) if len(_matched) > 0 else 0.0)
            else:
                _precs.append(0.0)
        _ax2.plot(_thrs, _precs, 'steelblue', marker='o', ms=4, label='Précision')
        _base = results_df['tp_rate'].mean()
        _ax2.axhline(_base, color='red', linestyle='--', alpha=0.7, label=f'baseline tp_rate={_base:.3f}')
        _ax2.set_xlabel('Seuil proba')
        _ax2.set_ylabel('Précision', color='steelblue')
        _ax2b = _ax2.twinx()
        _ax2b.bar(_thrs, _ns, width=0.03, alpha=0.25, color='gray')
        _ax2b.set_ylabel('N signaux', color='gray')
        _ax2.legend(fontsize=8)
        _ax2.set_title('Précision & volume signaux vs seuil')
    else:
        _thrs_rr = [1.2, 1.5, 2.0, 2.5, 3.0]
        _lifts = [results_df[f'lift_rr_{t}'].mean() for t in _thrs_rr]
        _nsigs = [results_df[f'n_sig_{t}'].mean()   for t in _thrs_rr]
        _ax2.plot(_thrs_rr, _lifts, 'darkorange', marker='o', ms=5, label='Lift R/R moyen')
        _ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        _ax2.set_xlabel('Seuil R/R')
        _ax2.set_ylabel('Lift R/R moyen', color='darkorange')
        _ax2b = _ax2.twinx()
        _ax2b.bar(_thrs_rr, _nsigs, width=0.12, alpha=0.25, color='gray')
        _ax2b.set_ylabel('Signaux/fold moyen', color='gray')
        _ax2.legend(fontsize=8)
        _ax2.set_title('Lift R/R & volume vs seuil')
    _ax2.grid(True, linestyle='--', linewidth=0.4)

    plt.tight_layout()
    _buf = io.BytesIO()
    _fig.savefig(_buf, format='png', dpi=130, bbox_inches='tight')
    _buf.seek(0)
    _quality_html = mo.Html(f'<img src="data:image/png;base64,{base64.b64encode(_buf.read()).decode()}" style="width:100%;max-width:1400px"/>')
    plt.close(_fig)

    mo.vstack([
        mo.md("### Distribution du signal & qualité par seuil"),
        _quality_html,
    ])
    return


@app.cell
def _threshold_controls(mo, model_type):
    if model_type == 'classification':
        threshold_slider = mo.ui.slider(
            0.30, 0.85, step=0.05, value=0.55,
            label='Seuil proba entrée',
            show_value=True,
        )
        _hint = "Proba > X → signal long · TP/SL fixes définis par le label"
    else:
        threshold_slider = mo.ui.slider(
            1.0, 4.0, step=0.1, value=1.5,
            label='Seuil R/R prédit',
            show_value=True,
        )
        _hint = "R/R prédit > X → signal long · TP/SL = max_gain/max_loss prédits (médiane)"

    run_btn = mo.ui.button(
        label    = '▶ Lancer le backtest',
        on_click = lambda v: (v or 0) + 1,
        value    = 0,
    )

    mo.vstack([
        mo.md(f"### Paramètres backtest\n_{_hint}_"),
        mo.hstack([threshold_slider, run_btn]),
    ])
    return run_btn, threshold_slider


@app.cell
def _backtest(base64, capital_input, io, mo, model_type, np, os, pd, pkl_select, plt, mdates, preds, result, run_btn, threshold_slider):
    if not run_btn.value:
        mo.stop(True, mo.callout(mo.md("Clique sur **▶ Lancer le backtest** pour démarrer."), kind="neutral"))

    mo.output.replace(mo.callout(mo.md("Chargement prix + calcul backtest..."), kind="info"))

    import vectorbtpro as _vbt

    def _fig_to_html(fig):
        _buf = io.BytesIO()
        fig.savefig(_buf, format='png', dpi=130, bbox_inches='tight')
        _buf.seek(0)
        return mo.Html(f'<img src="data:image/png;base64,{base64.b64encode(_buf.read()).decode()}" style="width:100%;max-width:1400px"/>')

    def _fmt_stat(x):
        if not isinstance(x, float) and not hasattr(x, '__float__'):
            return str(x)
        try:
            v = float(x)
        except Exception:
            return str(x)
        import math as _m
        if _m.isnan(v) or _m.isinf(v):
            return str(x)
        a = abs(v)
        if a >= 10000:   return f'{v:,.0f}'
        elif a >= 100:   return f'{v:,.2f}'
        elif a >= 1:     return f'{v:.4f}'
        elif a >= 1e-4:  return f'{v:.6f}'
        else:            return f'{v:.4e}'

    try:
        _threshold = threshold_slider.value
        _init_cap  = capital_input.value
        _config    = result.get('config', {})

        # ── Prix BTC perp 1m ──────────────────────────────────────────────────
        _close_path = 'data/raw/binance/um/1m/BTCUSDT.csv'
        _pdf = pd.read_csv(_close_path, low_memory=False)
        _pdf['date'] = pd.to_datetime(_pdf['date'], unit='ms', utc=True)
        _pdf = _pdf.set_index('date').sort_index()
        _close = _pdf['close'].astype(float)

        # ── Génération des signaux ────────────────────────────────────────────
        if model_type == 'classification':
            _tp    = _config.get('tp',       0.015)
            _sl    = _config.get('sl',       0.009)
            _proba = preds.reindex(_close.index, fill_value=0.0)
            _entries = (_proba > _threshold)
            _tp_stop = _tp
            _sl_stop = _sl
            _label_bt = (f"XGB Triple Barrier · seuil={_threshold:.2f}"
                         f" · TP={_tp*100:.1f}% · SL={_sl*100:.1f}%")
        else:
            _rr_pred    = preds['rr_pred'].reindex(_close.index, fill_value=0.0)
            _entries    = (_rr_pred > _threshold)
            _gain_pred  = preds['gain_pred'].reindex(_close.index)
            _loss_pred  = preds['loss_pred'].abs().reindex(_close.index)
            _tp_stop    = max(float(_gain_pred[_entries].median()), 0.005)
            _sl_stop    = max(float(_loss_pred[_entries].median()), 0.003)
            _label_bt   = (f"R/R dynamique · seuil={_threshold:.1f}"
                           f" · TP≈{_tp_stop*100:.2f}% · SL≈{_sl_stop*100:.2f}%")

        _n = int(_entries.sum())
        if _n == 0:
            mo.stop(True, mo.callout(mo.md(f"Aucun signal au seuil {_threshold}. Baisse le seuil."), kind="danger"))

        # ── Backtest VBT ──────────────────────────────────────────────────────
        _pf = _vbt.Portfolio.from_signals(
            close       = _close,
            entries     = _entries,
            exits       = pd.Series(False, index=_close.index),
            sl_stop     = _sl_stop,
            tp_stop     = _tp_stop,
            fees        = 0.0004,
            freq        = '1T',
            init_cash   = _init_cap,
            size        = 1.0,
            size_type   = 'value',
        )

        # ── Stats ─────────────────────────────────────────────────────────────
        _stats    = _pf.stats(metrics='all', silence_warnings=True)
        _stats_df = _stats.reset_index()
        _stats_df.columns = ['Métrique', 'Valeur']
        _stats_df['Valeur'] = _stats_df['Valeur'].apply(_fmt_stat)

        # ── Equity + Drawdown ─────────────────────────────────────────────────
        _equity  = _pf.value
        _eq_s    = _equity * (_init_cap / float(_equity.iloc[0]))
        _peak    = _eq_s.cummax()
        _dd      = (_eq_s - _peak) / _peak * 100

        _f1, _a1 = plt.subplots(figsize=(15, 6))
        _a1.set_xlabel('Date')
        _a1.set_ylabel('Capital ($)', color='darkblue')
        _a1.plot(_eq_s.index, _eq_s.values, color='darkblue', linewidth=0.8)
        _a1.tick_params(axis='y', labelcolor='darkblue')
        _a1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        _a1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        _a1.xaxis.set_tick_params(rotation=45)
        _a1.grid(True, linestyle='--', linewidth=0.5)
        _a2 = _a1.twinx()
        _a2.set_ylabel('Drawdown (%)', color='darkred')
        _a2.plot(_dd.index, (-_dd).values, color='darkred', linestyle='--', alpha=0.7, linewidth=0.8)
        _a2.tick_params(axis='y', labelcolor='darkred')
        _f1.tight_layout()
        plt.title(f'Capital & Drawdown — {_label_bt}')
        _f1.legend(['Capital', 'Drawdown (%)'], loc='upper left', bbox_to_anchor=(0.08, 0.92))
        _cap_html = _fig_to_html(_f1)
        plt.close(_f1)

        # ── Monthly returns ───────────────────────────────────────────────────
        _monthly  = _eq_s.resample('ME').last().pct_change().fillna(0) * 100
        _f2, _ax  = plt.subplots(figsize=(14, 5))
        _mcolors  = ['#247B47' if x > 0 else '#C71A1A' for x in _monthly.values]
        _bars     = _ax.bar(_monthly.index.strftime('%Y-%m'), _monthly.values, color=_mcolors)
        _ax.axhline(0, color='gray', linewidth=0.8)
        _ax.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Rendements mensuels (%) — {_label_bt}')
        plt.ylabel('%')
        for _bar in _bars:
            _y = _bar.get_height()
            if abs(_y) > 0.05:
                _ax.text(_bar.get_x() + _bar.get_width() / 2,
                         _y + 0.4 if _y >= 0 else _y - 1.2,
                         f'{_y:.1f}%', va='center', ha='center', fontsize=7)
        _f2.tight_layout()
        _mret_html = _fig_to_html(_f2)
        plt.close(_f2)

        # ── Résumé rapide ─────────────────────────────────────────────────────
        _sharpe  = _stats.get('Sharpe Ratio',   float('nan'))
        _ret     = _stats.get('Total Return [%]', float('nan'))
        _mdd     = _stats.get('Max Drawdown [%]', float('nan'))
        _wr      = _stats.get('Win Rate [%]',     float('nan'))
        _ntrades = _stats.get('Total Trades',     0)

        try:
            _sharpe_v  = float(_sharpe)
            _kind_sh   = "success" if _sharpe_v >= 1.0 else ("warn" if _sharpe_v >= 0.5 else "danger")
        except Exception:
            _kind_sh   = "neutral"

        mo.output.replace(mo.vstack([
            mo.callout(
                mo.md(f"**OOS · {_label_bt}**"
                      f" · {_n:,} signaux · capital={_init_cap:,}$"),
                kind="warn",
            ),
            mo.callout(
                mo.md(f"Sharpe **{_sharpe_v:.2f}** · Return **{float(_ret):.1f}%**"
                      f" · MaxDD **{float(_mdd):.1f}%** · WinRate **{float(_wr):.1f}%**"
                      f" · Trades **{int(_ntrades):,}**"),
                kind=_kind_sh,
            ),
            mo.md("#### Stats complètes"),
            mo.ui.table(_stats_df, selection=None),
            mo.md("---\n#### Évolution du capital & Drawdown"),
            _cap_html,
            mo.md("---\n#### Rendements mensuels"),
            _mret_html,
        ]))

    except Exception as _e:
        import traceback as _tb
        mo.output.replace(mo.callout(
            mo.md(f"Erreur : `{_e}`\n```\n{_tb.format_exc()}\n```"),
            kind="danger",
        ))
    return


if __name__ == "__main__":
    app.run()
