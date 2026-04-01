import marimo

__generated_with = "0.21.1"
app = marimo.App(width="wide")


@app.cell
def _imports():
    import sys
    sys.path.insert(0, '.')
    import marimo as mo
    import glob as glob_mod
    import os
    import math
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from scipy import stats as sp_stats
    from vectorbtpro import vbt

    METRIC_COLS = [
        'Total Return [%]', 'Max Drawdown [%]', 'Win Rate [%]',
        'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
        'Profit Factor', 'Expectancy', 'Total Trades',
    ]

    # Colonnes de paramètres par stratégie
    PARAMS_BY_STRATEGY = {
        'keltner': ['ma_window', 'atr_window', 'atr_mult', 'sl_stop'],
        'ram':     ['ma_window', 'env_pct', 'sl_pct'],
    }

    _PARQUET_DIR = '.parquet_cache'

    def load_pickle(path):
        """Charge un pickle VBT → DataFrame. Cache parquet automatique pour les rechargements."""
        _pq_dir  = os.path.join(os.path.dirname(path), _PARQUET_DIR)
        _pq_path = os.path.join(_pq_dir, os.path.basename(path).replace('.pickle', '.parquet'))

        if os.path.exists(_pq_path) and os.path.getmtime(_pq_path) >= os.path.getmtime(path):
            return pd.read_parquet(_pq_path)

        raw = vbt.load(path)
        if isinstance(raw, pd.Series) and None in raw.index.names:
            df = raw.unstack(level=-1)
        elif isinstance(raw, pd.Series):
            df = raw.to_frame(name='Sharpe Ratio')
        else:
            df = raw
        for col in METRIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        try:
            os.makedirs(_pq_dir, exist_ok=True)
            df.to_parquet(_pq_path)
        except Exception:
            pass

        return df

    def info_from_filename(fname):
        """
        kc_wfsl_BTC_5m_lighter.pickle  → ('keltner', 'BTC',  '5m', 'lighter')
        ram_HYPE_5m_lighter.pickle      → ('ram',     'HYPE', '5m', 'lighter')
        """
        name  = os.path.basename(fname).replace('.pickle', '')
        parts = name.split('_')
        if parts[0] == 'ram':
            # ram_{PAIR}_{tf}_{exchange}
            return 'ram', parts[1], parts[2], parts[3] if len(parts) > 3 else 'unknown'
        else:
            # kc_wfsl_{PAIR}_{tf}_{exchange}
            return 'keltner', parts[2], parts[3], parts[4] if len(parts) > 4 else 'unknown'

    def pair_from_filename(fname):
        return info_from_filename(fname)[1]

    def get_param_cols(df):
        """Détecte les colonnes de paramètres depuis les niveaux d'index du DataFrame."""
        non_param = {'split', 'set', None}
        return [n for n in df.index.names if n not in non_param]

    def compute_dsr(sharpe_values, T):
        """
        Deflated Sharpe Ratio — Bailey & López de Prado (2014).
        sharpe_values : array des Sharpe sur toutes les combos × fenêtres
        T : nb d'observations (candles) par fenêtre de test
        Retourne DSR ∈ [0,1]. DSR < 0.05 → le SR max est probablement du bruit.
        """
        sr = np.array(sharpe_values, dtype=float)
        sr = sr[np.isfinite(sr)]
        N = len(sr)
        if N < 3 or T < 2:
            return np.nan
        sr_max = sr.max()
        sr_mean = sr.mean()
        sr_std = sr.std(ddof=1)
        if sr_std == 0:
            return np.nan
        # Espérance du max sous H0 (approximation Euler-Mascheroni)
        gamma_em = 0.5772156649
        e_max = ((1 - gamma_em) * sp_stats.norm.ppf(1 - 1 / N)
                 + gamma_em * sp_stats.norm.ppf(1 - 1 / (N * math.e)))
        sr_0 = sr_mean + e_max * sr_std
        # Variance du SR estimé (correction skewness/kurtosis)
        sk = sp_stats.skew(sr)
        ku = sp_stats.kurtosis(sr, fisher=False)   # kurtosis total (pas excess)
        variance = max((1 - sk * sr_max + (ku - 1) / 4 * sr_max ** 2) / (T - 1), 1e-12)
        sigma = math.sqrt(variance)
        z = (sr_max - sr_0) / sigma
        return float(sp_stats.norm.cdf(z))

    # Affichage lisible : pas de notation scientifique, 4 chiffres significatifs
    pd.set_option('display.float_format', '{:.4g}'.format)

    def fmt_params(params_dict):
        """Arrondit les valeurs de params pour affichage humain."""
        out = {}
        for k, v in params_dict.items():
            if k == 'ma_window':
                out[k] = int(round(v))
            else:
                out[k] = round(float(v), 4)
        return out

    return (
        PARAMS_BY_STRATEGY,
        compute_dsr,
        fmt_params,
        get_param_cols,
        glob_mod,
        go,
        info_from_filename,
        load_pickle,
        mo,
        np,
        os,
        pair_from_filename,
        pd,
        vbt,
    )


@app.cell
def _controls(glob_mod, mo, os):
    # Dossiers disponibles (cache/ + 1 niveau + 2 niveaux)
    _subdirs = set()
    for d in glob_mod.glob('cache/*'):
        if os.path.isdir(d):
            _subdirs.add(d)
    for d in glob_mod.glob('cache/*/*'):
        if os.path.isdir(d):
            _subdirs.add(d)
    _all_dirs = ['cache/'] + sorted([d + '/' for d in _subdirs])
    _dir_options = {d.removeprefix('cache/').rstrip('/') or 'cache (racine)': d for d in _all_dirs}
    _default_label = next(iter(_dir_options))
    folder_selector = mo.ui.dropdown(
        options=_dir_options,
        value=_default_label,
        label='Dossier pickles',
    )
    pattern_input = mo.ui.text(value='*lighter*', label='Filtre pickles (glob)')
    refresh_btn = mo.ui.button(label='Rafraîchir les fichiers', on_click=lambda x: x + 1, value=0)

    max_dd = mo.ui.slider(0, 100, value=35, step=1, label='Max Drawdown < X% (0=off)', show_value=True)
    min_wr = mo.ui.slider(0, 100, value=0, step=1, label='Win Rate > X% (0=off)', show_value=True)
    min_sharpe = mo.ui.slider(-5.0, 10.0, value=0.5, step=0.1, label='Sharpe > X (-5=off)', show_value=True)
    min_return = mo.ui.slider(-50.0, 100.0, value=0.0, step=0.5, label='Return > X% (-50=off)', show_value=True)
    min_pct_windows = mo.ui.slider(0, 100, value=50, step=10, label='% fenêtres min qui passent', show_value=True)
    min_trades = mo.ui.slider(0, 100, value=30, step=5, label='Trades min par fenêtre (0=off)', show_value=True)

    mo.output.replace(mo.vstack([
        mo.md("## Critères de filtrage"),
        mo.hstack([folder_selector, pattern_input, refresh_btn], justify="start", gap=2),
        mo.hstack([max_dd, min_wr, min_sharpe, min_return, min_pct_windows, min_trades], justify="start", gap=2),
    ]))
    return (
        folder_selector,
        max_dd,
        min_pct_windows,
        min_return,
        min_sharpe,
        min_trades,
        min_wr,
        pattern_input,
        refresh_btn,
    )


@app.cell
def _load_data(
    folder_selector,
    get_param_cols,
    glob_mod,
    info_from_filename,
    load_pickle,
    mo,
    os,
    pair_from_filename,
    pattern_input,
    refresh_btn,
):
    from concurrent.futures import ThreadPoolExecutor

    _pattern = os.path.join(folder_selector.value, pattern_input.value + '.pickle')
    _files = sorted(glob_mod.glob(_pattern))
    if not _files:
        mo.stop(True, mo.callout(mo.md(f"Aucun pickle trouvé : `{_pattern}`"), kind="warn"))

    def _load_one(f):
        pair = pair_from_filename(f)
        try:
            return pair, load_pickle(f), info_from_filename(f)
        except Exception:
            return pair, None, None

    import multiprocessing as _mp, subprocess as _sp
    _total_cpus = _mp.cpu_count()
    _opti_running = _sp.run(['pgrep', '-f', 'opti.py'], capture_output=True).returncode == 0
    _n_workers = max(2, int(_total_cpus * 0.2)) if _opti_running else max(4, _total_cpus - 2)
    _n_workers = min(_n_workers, len(_files))

    with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
        _results = list(_pool.map(_load_one, _files))

    all_data  = {pair: df   for pair, df, _ in _results if df is not None}
    file_info = {pair: info for pair, df, info in _results if df is not None}
    pairs = sorted(all_data.keys())

    # Détecter strategy + param_cols depuis le premier fichier chargé
    _first_df   = next(iter(all_data.values())) if all_data else None
    _first_info = next(iter(file_info.values())) if file_info else None
    detected_strategy = _first_info[0] if _first_info else 'keltner'
    detected_params   = get_param_cols(_first_df) if _first_df is not None else ['ma_window']

    mo.output.replace(mo.vstack([
        mo.md(f"**{len(pairs)} paires chargées** — stratégie : `{detected_strategy}` · params : `{detected_params}`"),
        mo.md(', '.join(f'`{p}`' for p in pairs)),
    ]))
    return all_data, detected_params, detected_strategy, file_info, pairs


@app.cell
def _filter_and_score(
    all_data,
    detected_params,
    max_dd,
    min_pct_windows,
    min_return,
    min_sharpe,
    min_trades,
    min_wr,
    mo,
    np,
    pairs,
    pd,
):
    test_data = {}     # {pair: DataFrame test (index: split×params, cols: metrics)}
    valid_combos = {}  # {pair: Index de param tuples valides}
    combo_scores = {}  # {pair: Series indexed by param tuples, values = composite score}

    for _pair in pairs:
        _df = all_data[_pair]
        if 'set' not in _df.index.names:
            continue
        _test = _df.xs('test', level='set').copy()

        # Vérifier que les colonnes nécessaires existent
        _needed = ['Max Drawdown [%]', 'Win Rate [%]', 'Sharpe Ratio', 'Total Return [%]']
        if not all(c in _test.columns for c in _needed):
            continue

        test_data[_pair] = _test

        # Dériver les params réels depuis l'index de CE DataFrame
        # (robuste aux caches parquet stale ou pickles avec params différents)
        _pair_params = [p for p in detected_params if p in _test.index.names]
        if not _pair_params:
            _pair_params = [n for n in _test.index.names if n not in {'split', 'set', None}]
        if not _pair_params:
            continue

        # Filtrage vectorisé : masque booléen sur tout le DataFrame d'un coup,
        # puis groupby.min() (C pur) pour vérifier que TOUTES les fenêtres passent
        _mask = pd.Series(True, index=_test.index)
        if max_dd.value > 0:
            _mask &= _test['Max Drawdown [%]'] < max_dd.value
        if min_wr.value > 0:
            _mask &= _test['Win Rate [%]'] > min_wr.value
        if min_sharpe.value > -5.0:
            _mask &= _test['Sharpe Ratio'] > min_sharpe.value
        if min_return.value > -50.0:
            _mask &= _test['Total Return [%]'] > min_return.value
        if min_trades.value > 0 and 'Total Trades' in _test.columns:
            _mask &= _test['Total Trades'] >= min_trades.value

        _pct_pass = _mask.groupby(level=_pair_params).mean() * 100
        valid_combos[_pair] = _pct_pass[_pct_pass >= min_pct_windows.value].index

        # Score composite sur toutes les combos (médiane cross-fenêtres)
        _agg = _test.groupby(level=_pair_params)[_needed].median()
        _agg = _agg.dropna()
        if len(_agg) == 0:
            combo_scores[_pair] = pd.Series(dtype=float)
            continue

        # Normalisation min-max
        def _norm(s):
            r = s.max() - s.min()
            return (s - s.min()) / (r if r != 0 else 1)
        _sharpe_n = _norm(_agg['Sharpe Ratio'])
        _return_n = _norm(_agg['Total Return [%]'])
        _wr_n     = _norm(_agg['Win Rate [%]'])
        _dd_n     = _norm(_agg['Max Drawdown [%]'])

        combo_scores[_pair] = (_sharpe_n + _return_n + _wr_n + (1 - _dd_n)) / 4

    # Table globale
    _rows = []
    for _pair in pairs:
        _n_valid = len(valid_combos.get(_pair, []))
        _scores = combo_scores.get(_pair, pd.Series(dtype=float))
        _n_total = len(_scores)
        _best = float(_scores.max()) if len(_scores) > 0 else np.nan
        _valid_scores = _scores.loc[_scores.index.isin(valid_combos.get(_pair, []))] if _n_valid > 0 else pd.Series(dtype=float)
        _rows.append({
            'paire': _pair,
            'combos_testées': _n_total,
            'combos_valides': _n_valid,
            'taux_validité': f"{100*_n_valid/_n_total:.1f}%" if _n_total > 0 else "0%",
            'score_max': round(_best, 3),
            'score_moy_valides': round(float(_valid_scores.mean()), 3) if _n_valid > 0 else np.nan,
        })
    global_stats_df = pd.DataFrame(_rows)

    mo.output.replace(mo.md(
        f"Filtrage terminé — {sum(len(v) for v in valid_combos.values())} combos valides au total"
    ))
    return combo_scores, global_stats_df, test_data, valid_combos


@app.cell
def _section2(combo_scores, global_stats_df, go, mo, pairs, valid_combos):
    # Bar chart : nb combos valides × score moyen
    _x = []
    _y_valid = []
    _y_best = []
    for _p in pairs:
        _x.append(_p)
        _y_valid.append(len(valid_combos.get(_p, [])))
        _sc = combo_scores.get(_p)
        _y_best.append(float(_sc.max()) if _sc is not None and len(_sc) > 0 else 0)

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        name='Combos valides', x=_x, y=_y_valid,
        marker_color=['green' if v > 0 else 'lightgray' for v in _y_valid],
    ))
    _fig.update_layout(
        title='Nombre de combos valides par paire',
        yaxis_title='Combos valides', height=320, showlegend=False,
    )

    mo.output.replace(mo.vstack([
        mo.md("## 2. Vue globale par paire"),
        mo.ui.table(global_stats_df, selection=None),
        mo.ui.plotly(_fig),
    ]))
    return


@app.cell
def _section3(
    combo_scores,
    detected_params,
    fmt_params,
    mo,
    np,
    pairs,
    pd,
    test_data,
    valid_combos,
):
    _rows = []
    for _pair in pairs:
        _vcs = valid_combos.get(_pair)
        _test = test_data.get(_pair)
        if _vcs is None or len(_vcs) == 0 or _test is None:
            continue
        _scores = combo_scores.get(_pair, pd.Series(dtype=float))
        _pair_params3 = [p for p in detected_params if p in _test.index.names] or \
                        [n for n in _test.index.names if n not in {'split', 'set', None}]

        for _combo in _vcs:
            try:
                _grp = _test.xs(_combo, level=_pair_params3)
            except KeyError:
                continue

            _sr = _grp['Sharpe Ratio'].astype(float)
            _ret = _grp['Total Return [%]'].astype(float)
            _dd = _grp['Max Drawdown [%]'].astype(float)

            _sr_mean = float(_sr.mean())
            _sr_std = float(_sr.std())
            _cv_sr = abs(_sr_std / _sr_mean) if _sr_mean != 0 else np.inf
            _pct_pos = float((_sr > 0).mean() * 100)
            _ret_mean = float(_ret.mean())
            _ret_cv = abs(float(_ret.std()) / _ret_mean) if _ret_mean != 0 else np.inf

            # Médiane de Max Drawdown Duration (en jours)
            _dd_dur_med = np.nan
            if 'Max Drawdown Duration' in _grp.columns:
                try:
                    _dd_dur_med = float(
                        pd.to_timedelta(_grp['Max Drawdown Duration'].dropna())
                        .dt.total_seconds().median() / 86400
                    )
                except Exception:
                    pass

            _score = float(_scores.loc[_combo]) if _combo in _scores.index else np.nan
            _params = fmt_params(dict(zip(_pair_params3, _combo if isinstance(_combo, tuple) else [_combo])))

            _rows.append({
                'paire': _pair,
                **_params,
                'score': round(_score, 3),
                'sharpe_med': round(_sr_mean, 3),
                'sharpe_cv': round(_cv_sr, 3),
                'return_med_%': round(_ret_mean, 2),
                'return_cv': round(_ret_cv, 3),
                'dd_med_%': round(float(_dd.mean()), 2),
                'dd_dur_med_j': round(_dd_dur_med, 1) if not np.isnan(_dd_dur_med) else np.nan,
                'pct_fenetres_positives': round(_pct_pos, 1),
            })

    mo.stop(not _rows, mo.vstack([
        mo.md("## 3. Stabilité inter-fenêtres"),
        mo.callout(mo.md("Aucune combo valide — ajuste les critères."), kind="warn"),
    ]))
    stability_df = pd.DataFrame(_rows).sort_values(['pct_fenetres_positives', 'sharpe_cv'], ascending=[False, True])
    mo.output.replace(mo.vstack([
        mo.md("## 3. Stabilité inter-fenêtres"),
        mo.md("_Triées par : % fenêtres positives ↓, CV Sharpe ↑ (plus bas = plus stable)_"),
        mo.ui.table(stability_df, selection=None),
    ]))
    return (stability_df,)


@app.cell
def _section4(combo_scores, detected_params, fmt_params, mo, np, pairs, pd, valid_combos):
    _rows = []
    for _pair in pairs:
        _vcs = valid_combos.get(_pair)
        _scores = combo_scores.get(_pair)
        if _vcs is None or len(_vcs) == 0 or _scores is None or len(_scores) == 0:
            continue

        # Params réels de cette paire (robuste aux index hétérogènes)
        if isinstance(_scores.index, pd.MultiIndex):
            _pair_params4 = list(_scores.index.names)
        else:
            _pair_params4 = [_scores.index.name] if _scores.index.name else detected_params

        # Grille de valeurs par paramètre
        _grid = {p: sorted(_scores.index.get_level_values(p).unique()) for p in _pair_params4}

        for _combo in _vcs:
            _params = fmt_params(dict(zip(_pair_params4, _combo if isinstance(_combo, tuple) else [_combo])))
            _combo_score = float(_scores.loc[_combo]) if _combo in _scores.index else np.nan

            # Trouver les voisins (±1 step sur chaque axe)
            _neighbor_scores = []
            for _p in _pair_params4:
                _vals = _grid[_p]
                _idx = _vals.index(_params[_p]) if _params[_p] in _vals else -1
                for _delta in [-1, 1]:
                    _ni = _idx + _delta
                    if 0 <= _ni < len(_vals):
                        _neighbor_params = {**_params, _p: _vals[_ni]}
                        _key = tuple(_neighbor_params[p] for p in _pair_params4)
                        if _key in _scores.index:
                            _neighbor_scores.append(float(_scores.loc[_key]))

            _n_neighbors = len(_neighbor_scores)
            _neighbor_mean = float(np.mean(_neighbor_scores)) if _neighbor_scores else np.nan
            _ratio = (_combo_score / _neighbor_mean) if (_neighbor_mean and _neighbor_mean > 0) else np.nan

            _rows.append({
                'paire':              _pair,
                **_params,
                'score':              round(_combo_score, 3),
                'score_voisins_moy':  round(_neighbor_mean, 3) if not np.isnan(_neighbor_mean) else np.nan,
                'ratio_combo/voisins':round(_ratio, 3) if not np.isnan(_ratio) else np.nan,
                'n_voisins':          _n_neighbors,
                'type':               'plateau' if (not np.isnan(_ratio) and _ratio < 1.2) else 'pic',
            })

    mo.stop(not _rows, mo.vstack([
        mo.md("## 4. Analyse de voisinage (plateaux)"),
        mo.callout(mo.md("Aucune combo valide."), kind="neutral"),
    ]))
    plateau_df = pd.DataFrame(_rows).sort_values('ratio_combo/voisins')
    mo.output.replace(mo.vstack([
        mo.md("## 4. Analyse de voisinage (plateaux)"),
        mo.md("_ratio < 1.2 → plateau (robuste) · ratio >> 1 → pic isolé (overfitting)_"),
        mo.ui.table(plateau_df, selection=None),
    ]))
    return (plateau_df,)


@app.cell
def _section5(compute_dsr, detected_params, mo, np, pairs, pd, test_data):
    _rows = []
    for _pair in pairs:
        _test = test_data.get(_pair)
        if _test is None or 'Sharpe Ratio' not in _test.columns:
            continue

        _all_sr = _test['Sharpe Ratio'].astype(float).dropna().values
        if len(_all_sr) == 0:
            continue

        # T = nb candles par fenêtre test (depuis Total Duration si dispo)
        _T = 5000  # fallback
        if 'Total Duration' in _test.columns:
            try:
                _durations = pd.to_timedelta(_test['Total Duration'].dropna().iloc[:10])
                _T = int(_durations.dt.total_seconds().median() / 300)  # 5min candles
            except Exception:
                pass

        _pair_params5 = [p for p in detected_params if p in _test.index.names] or \
                        [n for n in _test.index.names if n not in {'split', 'set', None}]
        _N = len(_test.groupby(level=_pair_params5).first())  # nb de combos
        _dsr = compute_dsr(_all_sr, _T)
        _sr_max = float(np.nanmax(_all_sr))
        _sr_med = float(np.nanmedian(_all_sr))

        _rows.append({
            'paire': _pair,
            'N_combos': _N,
            'T_candles_approx': _T,
            'SR_max': round(_sr_max, 3),
            'SR_médian': round(_sr_med, 3),
            'DSR': round(_dsr, 4) if not np.isnan(_dsr) else np.nan,
            'verdict': '✓ Signal' if (not np.isnan(_dsr) and _dsr > 0.05) else '✗ Bruit',
        })

    mo.stop(not _rows, mo.vstack([mo.md("## 5. Deflated Sharpe Ratio"), mo.callout(mo.md("Pas de données."), kind="neutral")]))

    dsr_df = pd.DataFrame(_rows).sort_values('DSR', ascending=False)

    mo.output.replace(mo.vstack([
        mo.md("## 5. Deflated Sharpe Ratio (DSR)"),
        mo.callout(mo.md("**DSR > 0.05** → le Sharpe est statistiquement significatif · **DSR < 0.05** → probablement du bruit"), kind="info"),
        mo.ui.table(dsr_df, selection=None),
    ]))
    return


@app.cell
def _section_best_params(detected_params, mo, np, pd, plateau_df, stability_df):
    best_params_table = None

    _has_data = (
        stability_df is not None and plateau_df is not None
        and len(stability_df) > 0 and len(plateau_df) > 0
    )

    if not _has_data:
        mo.output.replace(mo.callout(mo.md("Pas assez de données pour les recommandations."), kind="neutral"))
    else:
        _merge_cols = ['paire'] + detected_params

        def _round_params(df):
            df = df.copy()
            for _c in detected_params:
                if _c in df.columns and df[_c].dtype == float:
                    df[_c] = df[_c].round(6)
            return df

        _stab = _round_params(stability_df)
        _plat = _round_params(plateau_df)

        _stab_cols = [c for c in _merge_cols + ['score', 'sharpe_med', 'sharpe_cv', 'return_med_%', 'dd_med_%', 'dd_dur_med_j', 'pct_fenetres_positives'] if c in _stab.columns]
        _plat_cols = [c for c in _merge_cols + ['ratio_combo/voisins', 'type'] if c in _plat.columns]

        _merged = pd.merge(_stab[_stab_cols], _plat[_plat_cols], on=_merge_cols, how='inner')

        if 'type' in _merged.columns:
            _plateaux = _merged[_merged['type'] == 'plateau'].copy()
            if len(_plateaux) == 0:
                _plateaux = _merged.copy()
        else:
            _plateaux = _merged.copy()

        if len(_plateaux) == 0:
            mo.output.replace(mo.callout(mo.md("Aucun paramètre plateau trouvé."), kind="warn"))
        else:
            def _norm(s):
                r = s.max() - s.min()
                return (s - s.min()) / (r if r > 0 else 1.0)

            _score = pd.Series(0.0, index=_plateaux.index)
            if 'pct_fenetres_positives' in _plateaux.columns:
                _score += _norm(_plateaux['pct_fenetres_positives']) * 2
            if 'sharpe_med' in _plateaux.columns:
                _score += _norm(_plateaux['sharpe_med'].clip(upper=_plateaux['sharpe_med'].quantile(0.95))) * 2
            if 'return_med_%' in _plateaux.columns:
                _score += _norm(_plateaux['return_med_%'])
            if 'dd_med_%' in _plateaux.columns:
                _score -= _norm(_plateaux['dd_med_%'])
            if 'sharpe_cv' in _plateaux.columns:
                _score -= _norm(_plateaux['sharpe_cv'].clip(upper=_plateaux['sharpe_cv'].quantile(0.95)))
            if 'ratio_combo/voisins' in _plateaux.columns:
                _score -= _norm(_plateaux['ratio_combo/voisins'])

            _plateaux = _plateaux.copy()
            _plateaux['rec_score'] = (_score / 7).round(3)

            _display_cols = [c for c in ['paire'] + detected_params + ['rec_score', 'sharpe_med', 'return_med_%', 'dd_med_%', 'dd_dur_med_j', 'pct_fenetres_positives', 'ratio_combo/voisins'] if c in _plateaux.columns]
            _best = _plateaux.sort_values('rec_score', ascending=False).head(30)[_display_cols].reset_index(drop=True)

            best_params_table = mo.ui.table(_best, selection='single')

            mo.output.replace(mo.vstack([
                mo.md("## 6. Paramètres recommandés"),
                mo.md("_Plateau + stabilité · **rec_score** = robustesse globale · clique sur une ligne pour pré-remplir le backtest_"),
                best_params_table,
            ]))

    return (best_params_table,)


@app.cell
def _section6_controls(detected_params, mo, pairs):
    heatmap_pair = mo.ui.dropdown(
        options={p: p for p in pairs},
        value=pairs[0] if pairs else None,
        label='Paire (heatmap spécifique)',
    )
    _param_opts = {p: p for p in detected_params}
    heatmap_x = mo.ui.dropdown(
        options=_param_opts,
        value=detected_params[-1] if detected_params else 'ma_window',
        label='Axe X',
    )
    heatmap_y = mo.ui.dropdown(
        options=_param_opts,
        value=detected_params[0] if detected_params else 'ma_window',
        label='Axe Y',
    )
    mo.output.replace(mo.hstack([heatmap_pair, heatmap_x, heatmap_y], justify="start", gap=2))
    return heatmap_pair, heatmap_x, heatmap_y


@app.cell
def _section6_heatmaps(
    combo_scores,
    detected_params,
    go,
    heatmap_pair,
    heatmap_x,
    heatmap_y,
    mo,
    pairs,
    pd,
    valid_combos,
):
    _xp = heatmap_x.value
    _yp = heatmap_y.value

    if _xp == _yp:
        mo.stop(True, mo.callout(mo.md("Choisis deux axes différents."), kind="warn"))

    # ── Heatmap global : nb de paires avec combo valide à chaque (yp, xp) ──
    def _flat_idx(s):
        """Aplatit un MultiIndex en tuples pour permettre le concat cross-stratégies."""
        if isinstance(s.index, pd.MultiIndex):
            return s.set_axis(s.index.to_flat_index())
        return s

    _all_scores = pd.concat(
        [_flat_idx(combo_scores[p]).rename(p) for p in pairs if p in combo_scores and len(combo_scores[p]) > 0],
        axis=1
    ) if pairs else pd.DataFrame()

    _global_valid = {}
    for _p in pairs:
        for _c in valid_combos.get(_p, []):
            _c_dict = dict(zip(detected_params, _c if isinstance(_c, tuple) else [_c]))
            _key = (_c_dict.get(_yp), _c_dict.get(_xp))
            _global_valid[_key] = _global_valid.get(_key, 0) + 1

    _x_vals = sorted(set(k[1] for k in _global_valid)) if _global_valid else []
    _y_vals = sorted(set(k[0] for k in _global_valid)) if _global_valid else []
    _z_global = [[_global_valid.get((_y, _x), 0) for _x in _x_vals] for _y in _y_vals]

    _fig_global = go.Figure(go.Heatmap(
        z=_z_global,
        x=[str(x) for x in _x_vals],
        y=[str(y) for y in _y_vals],
        colorscale='Blues',
        colorbar=dict(title='Nb paires valides'),
        text=_z_global, texttemplate='%{text}', showscale=True,
    ))
    _fig_global.update_layout(
        title=f'Heatmap global — nb paires avec combo valide ({_yp} × {_xp})',
        xaxis_title=_xp, yaxis_title=_yp, height=380,
    )

    # ── Heatmap spécifique à la paire sélectionnée ──
    _scores = combo_scores.get(heatmap_pair.value, pd.Series(dtype=float))
    _heatmap_widget = None
    if len(_scores) > 0:
        _pivot = _scores.groupby(level=[_yp, _xp]).median().unstack(level=_xp)
        _fig_pair = go.Figure(go.Heatmap(
            z=_pivot.values,
            x=[f'{x:.2g}' for x in _pivot.columns],
            y=[f'{y:.2g}' for y in _pivot.index],
            colorscale='RdYlGn',
            colorbar=dict(title='Score médian'),
        ))
        _fig_pair.update_layout(
            title=f'Score composite — {heatmap_pair.value} ({_yp} × {_xp})',
            xaxis_title=_xp, yaxis_title=_yp, height=400,
        )
        _heatmap_widget = mo.ui.plotly(_fig_pair)

    heatmap_click = _heatmap_widget  # utilisé par la section backtest
    mo.output.replace(mo.vstack([
        mo.md("## 7. Heatmaps"),
        mo.ui.plotly(_fig_global),
        _heatmap_widget if _heatmap_widget is not None else mo.callout(mo.md("Aucun score disponible pour cette paire."), kind="neutral"),
    ]))
    return (heatmap_click,)


@app.cell
def _section7_main_controls(detected_strategy, mo, pairs):
    bt_pair    = mo.ui.dropdown(options={p: p for p in pairs}, value=pairs[0] if pairs else None, label='Paire')
    bt_capital = mo.ui.number(value=10000, start=1000, stop=1_000_000, step=1000, label='Capital ($)')
    run_btn    = mo.ui.button(label='▶ Lancer le backtest', on_click=lambda v: (v or 0) + 1, value=0)

    _strat_label = 'RAM DCA' if detected_strategy == 'ram' else 'Keltner Channel'

    mo.output.replace(mo.vstack([
        mo.md("## 8. Backtest final interactif"),
        mo.callout(mo.md(f"Stratégie détectée depuis le pickle : **{_strat_label}**"), kind="info"),
        mo.hstack([bt_pair, bt_capital, run_btn], justify="start", gap=2),
    ]))
    return bt_capital, bt_pair, run_btn


@app.cell
def _section7_param_controls(
    best_params_table,
    combo_scores,
    detected_params,
    heatmap_click,
    heatmap_pair,
    heatmap_x,
    heatmap_y,
    mo,
    np,
    pairs,
    pd,
    valid_combos,
):
    """Contrôles de paramètres — se re-run quand best_params_table change (auto-fill)."""
    # Meilleure combo valide par défaut
    _best_pair  = pairs[0] if pairs else None
    _best_combo = None
    _best_score = -np.inf
    for _p in pairs:
        _vcs = valid_combos.get(_p, [])
        _sc  = combo_scores.get(_p, pd.Series(dtype=float))
        if len(_vcs) > 0 and len(_sc) > 0:
            _valid_sc = _sc.loc[_sc.index.isin(_vcs)]
            if len(_valid_sc) > 0 and float(_valid_sc.max()) > _best_score:
                _best_score = float(_valid_sc.max())
                _best_pair  = _p
                _best_combo = _valid_sc.idxmax()

    _def = dict(zip(detected_params, _best_combo if isinstance(_best_combo, tuple) else ([_best_combo] if _best_combo else [])))

    # Override prioritaire : sélection dans le tableau best_params
    if best_params_table is not None and len(best_params_table.value) > 0:
        _row = best_params_table.value.iloc[0].to_dict()
        for _param in detected_params:
            if _param in _row:
                _def[_param] = _row[_param]
    # Sinon override depuis le clic heatmap
    elif heatmap_click is not None and heatmap_click.value:
        try:
            _pts = heatmap_click.value.get('points', [])
            if _pts:
                _def[heatmap_x.value] = float(_pts[0].get('x', 0))
                _def[heatmap_y.value] = float(_pts[0].get('y', 0))
        except Exception:
            pass

    bt_ma   = mo.ui.number(value=int(_def.get('ma_window', 20)), start=5, stop=500, step=1, label='ma_window')
    bt_env  = mo.ui.number(value=float(_def.get('env_pct', 0.03)), start=0.005, stop=0.30, step=0.005, label='env_pct (RAM)')
    bt_sl   = mo.ui.number(value=float(_def.get('sl_pct', _def.get('sl_stop', 0.05))), start=0.005, stop=0.30, step=0.005, label='sl_pct / sl_stop')
    bt_atrw = mo.ui.number(value=int(_def.get('atr_window', 10)), start=5, stop=500, step=1, label='atr_window (KC)')
    bt_atrm = mo.ui.number(value=float(_def.get('atr_mult', 2.0)), start=0.5, stop=20.0, step=0.5, label='atr_mult (KC)')

    mo.output.replace(mo.vstack([
        mo.hstack([bt_ma, bt_env, bt_sl, bt_atrw, bt_atrm], justify="start", gap=2),
        mo.callout(mo.md("**env_pct** et **sl_pct** sont pour RAM · **atr_window / atr_mult** pour Keltner · clique une ligne dans §6 pour auto-remplir"), kind="info"),
    ]))
    return bt_atrm, bt_atrw, bt_env, bt_ma, bt_sl


@app.cell
def _section7_backtest(
    bt_atrm,
    bt_atrw,
    bt_capital,
    bt_env,
    bt_ma,
    bt_pair,
    bt_sl,
    detected_strategy,
    file_info,
    mo,
    os,
    pd,
    run_btn,
    vbt,
):
    if not run_btn.value:
        mo.stop(True, mo.callout(mo.md("Clique sur **▶ Lancer le backtest** pour démarrer."), kind="neutral"))

    mo.output.replace(mo.callout(mo.md("Backtest en cours..."), kind="info"))

    import sys as _sys
    if '.' not in _sys.path:
        _sys.path.insert(0, '.')

    import matplotlib as _mpl
    _mpl.use('Agg')
    import matplotlib.pyplot as _plt
    import matplotlib.dates as _mdates
    import io as _io
    import base64 as _b64

    def _fig_to_html(fig):
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        buf.seek(0)
        b64 = _b64.b64encode(buf.read()).decode()
        return mo.Html(f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1400px"/>')

    try:
        _pair     = bt_pair.value
        # Stratégie toujours déduite du pickle chargé (detected_strategy),
        # le dropdown bt_strategy sert uniquement d'override explicite
        _strategy = detected_strategy
        _init_cap = bt_capital.value

        _info     = file_info.get(_pair)
        _tf       = _info[2] if _info else '5m'
        _exchange = _info[3] if _info else 'lighter'

        _data = None
        for _suf in ('.csv', 'USDT.csv'):
            _path = f'data/raw/{_exchange}/{_tf}/{_pair}{_suf}'
            if os.path.exists(_path):
                _data = pd.read_csv(_path)
                _data['date'] = pd.to_datetime(_data['date'], unit='ms')
                _data = _data.set_index('date').sort_index()
                break

        if _data is None:
            mo.stop(True, mo.callout(mo.md(f"Données introuvables pour `{_pair}` ({_exchange}/{_tf})"), kind="danger"))

        if _strategy == 'ram':
            from src.strategies.ram_dca import run_backtest as _rb
            _pf = _rb(_data, int(bt_ma.value), [float(bt_env.value)], [1.0], float(bt_sl.value))
            _band_label = f'RAM DCA · ma={bt_ma.value} · env={float(bt_env.value)*100:.1f}% · sl={float(bt_sl.value)*100:.0f}%'
        else:
            from src.strategies.keltner import run_backtest as _rb
            _pf = _rb(_data, int(bt_ma.value), int(bt_atrw.value), float(bt_atrm.value), float(bt_sl.value), size=1)
            _band_label = f'Keltner · ma={bt_ma.value} · atr_w={bt_atrw.value} · mult={bt_atrm.value} · sl={float(bt_sl.value)*100:.0f}%'

        # ── Stats ──────────────────────────────────────────────────────────────
        _stats = _pf.stats(metrics='all', silence_warnings=True)
        _stats_df = _stats.reset_index()
        _stats_df.columns = ['Métrique', 'Valeur']
        def _fmt_stat(x):
            if not isinstance(x, float) and not hasattr(x, '__float__'):
                return str(x)
            try:
                v = float(x)
            except Exception:
                return str(x)
            import math
            if math.isnan(v) or math.isinf(v):
                return str(x)
            a = abs(v)
            if a >= 10000:
                return f'{v:,.0f}'
            elif a >= 100:
                return f'{v:,.2f}'
            elif a >= 1:
                return f'{v:.4f}'
            elif a >= 0.0001:
                return f'{v:.6f}'
            else:
                return f'{v:.4e}'
        _stats_df['Valeur'] = _stats_df['Valeur'].apply(_fmt_stat)

        # ── Equity curve (scaled to user capital) ─────────────────────────────
        _equity = _pf.value
        _init_pf = float(_equity.iloc[0]) if len(_equity) > 0 else 1.0
        _equity_s = _equity * (_init_cap / _init_pf) if _init_pf != 0 else _equity
        _peak = _equity_s.cummax()
        _dd   = (_equity_s - _peak) / _peak * 100  # negative values

        # ── Chart 1: Capital + Drawdown ────────────────────────────────────────
        _fig1, _ax1 = _plt.subplots(figsize=(15, 6))
        _ax1.set_xlabel('Date')
        _ax1.set_ylabel('Capital ($)', color='darkblue')
        _ax1.plot(_equity_s.index, _equity_s.values, color='darkblue')
        _ax1.tick_params(axis='y', labelcolor='darkblue')
        _ax1.xaxis.set_major_locator(_mdates.MonthLocator(interval=2))
        _ax1.xaxis.set_major_formatter(_mdates.DateFormatter('%Y-%m'))
        _ax1.xaxis.set_tick_params(rotation=45)
        _ax1.grid(True, linestyle='--', linewidth=0.5)
        _ax2 = _ax1.twinx()
        _ax2.set_ylabel('Drawdown (%)', color='darkred')
        _ax2.plot(_dd.index, (-_dd).values, color='darkred', linestyle='--', alpha=0.7)
        _ax2.tick_params(axis='y', labelcolor='darkred')
        _fig1.tight_layout()
        _plt.title(f'Évolution du Capital & Drawdown — {_pair}')
        _fig1.legend(['Capital', 'Drawdown (%)'], loc='upper left', bbox_to_anchor=(0.08, 0.92))
        _cap_html = _fig_to_html(_fig1)
        _plt.close(_fig1)

        # ── Chart 2: Monthly returns ───────────────────────────────────────────
        _monthly = _equity_s.resample('ME').last().pct_change().fillna(0) * 100
        _fig2, _ax = _plt.subplots(figsize=(14, 5))
        _mcolors = ['#247B47' if x > 0 else '#C71A1A' for x in _monthly.values]
        _bars = _ax.bar(_monthly.index.strftime('%Y-%m'), _monthly.values, color=_mcolors)
        _ax.axhline(0, color='gray', linewidth=0.8)
        _ax.grid(True, linestyle='--', linewidth=0.5)
        _plt.xticks(rotation=45, ha='right')
        _plt.title(f'Rendements Mensuels (%) — {_pair}')
        _plt.ylabel('%')
        for _bar in _bars:
            _y = _bar.get_height()
            _pos = _y + 0.5 if _y >= 0 else _y - 0.5
            _ax.text(_bar.get_x() + _bar.get_width() / 2, _pos, f'{_y:.1f}%', va='center', ha='center', fontsize=8)
        _fig2.tight_layout()
        _mret_html = _fig_to_html(_fig2)
        _plt.close(_fig2)

        mo.output.replace(mo.vstack([
            mo.callout(mo.md(f"**In-sample** · {_band_label} · **{len(_data):,}** bougies · capital={_init_cap:,}$"), kind="warn"),
            mo.md("#### Stats complètes"),
            mo.ui.table(_stats_df, selection=None),
            mo.md("---\n#### Évolution du capital & Drawdown"),
            _cap_html,
            mo.md("---\n#### Rendements mensuels"),
            _mret_html,
        ]))

    except Exception as _e:
        import traceback as _tb
        mo.output.replace(mo.callout(mo.md(f"Erreur : `{_e}`\n```\n{_tb.format_exc()}\n```"), kind="danger"))
    return


if __name__ == "__main__":
    app.run()
