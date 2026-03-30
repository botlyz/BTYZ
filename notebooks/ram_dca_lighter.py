import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _imports():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import plotly.graph_objects as go
    from numba import njit
    from vectorbtpro import vbt
    import warnings
    warnings.filterwarnings('ignore')
    return go, matplotlib, mdates, mo, njit, np, pd, plt, vbt, warnings


@app.cell
def _strategy_class(go, mdates, mo, njit, np, pd, plt, vbt):

    class Ram_Strategy_SL:
        def __init__(self, pair, ma_periods, envelope_pct_config, ohlc4,
                     start_time, end_time, tf, exchange, capital, leverage, sl_pct):
            self.pair             = pair
            self.ma_periods       = ma_periods
            self.envelope_pct_config = sorted(envelope_pct_config, key=lambda x: x[0])
            self.ohlc4            = ohlc4
            self.tf               = tf
            self.exchange         = exchange
            self.start_time       = pd.to_datetime(start_time) if start_time else None
            self.end_time         = pd.to_datetime(end_time)   if end_time   else None
            self.capital          = capital
            self.leverage         = leverage
            self.sl_pct           = sl_pct
            self.results          = None

        def load_data(self):
            csv_path = f'data/raw/{self.exchange}/{self.tf}/{self.pair}.csv'
            data = pd.read_csv(csv_path, index_col=0)
            first = str(data.index[0])
            if ' ' in first or ('-' in first and ':' in first):
                data.index = pd.to_datetime(data.index)
            else:
                data.index = pd.to_datetime(data.index, unit='ms')
            data = data.sort_index()
            data = data[~data.index.duplicated(keep='first')]
            if self.start_time and self.end_time:
                data = data[(data.index >= self.start_time) & (data.index <= self.end_time)]
            return data

        @staticmethod
        def moving_average(open_p, high_p, low_p, close_p, ohlc4, periods):
            if ohlc4:
                src = (open_p + high_p + low_p + close_p) / 4
            else:
                src = close_p
            return src.rolling(window=periods).mean()

        @staticmethod
        def calculate_envelopes(ma, envelope_pct_config):
            upper, lower = {}, {}
            for i, (pct, _) in enumerate(envelope_pct_config):
                upper[f'upper_{i}'] = ma * (1 + pct)
                lower[f'lower_{i}'] = ma * (1 - pct)
            return pd.DataFrame(upper, index=ma.index), pd.DataFrame(lower, index=ma.index)

        @staticmethod
        @njit
        def ram_dca_with_sl_nb(high, low, close, ma, upper_envs, lower_envs, allocations, sl_pct):
            n        = len(close)
            n_levels = len(allocations)
            target_size = np.full(n, np.nan, dtype=np.float64)
            exec_price  = np.full(n, np.nan, dtype=np.float64)
            pos_dir = 0
            cur_lvl = 0
            avg_entry = 0.0
            cur_qty   = 0.0
            sl_triggered = False

            for i in range(1, n):
                ma_prev = ma[i - 1]
                if np.isnan(ma_prev):
                    continue

                if sl_triggered:
                    touched = low[i] <= ma_prev <= high[i]
                    crossed = min(close[i-1], close[i]) <= ma_prev <= max(close[i-1], close[i])
                    if touched or crossed:
                        sl_triggered = False
                    if sl_triggered:
                        continue

                if pos_dir != 0:
                    exit_signal = (pos_dir == 1  and high[i] >= ma_prev) or \
                                  (pos_dir == -1 and low[i]  <= ma_prev)
                    if exit_signal:
                        target_size[i] = 0.0
                        exec_price[i]  = ma_prev
                        pos_dir = 0; cur_lvl = 0; avg_entry = 0.0; cur_qty = 0.0
                        continue

                tmp_dir = pos_dir; tmp_lvl = cur_lvl
                tmp_avg = avg_entry; tmp_qty = cur_qty
                traded = False; wp = 0.0; wa = 0.0

                if tmp_dir >= 0:
                    while tmp_lvl < n_levels:
                        alloc = allocations[tmp_lvl]
                        lim   = lower_envs[tmp_lvl, i - 1]
                        if low[i] <= lim:
                            prev_val = tmp_avg * tmp_qty
                            tmp_qty += alloc
                            tmp_avg  = (prev_val + lim * alloc) / tmp_qty
                            wp += lim * alloc; wa += alloc
                            tmp_lvl += 1; tmp_dir = 1; traded = True
                        else:
                            break

                if tmp_dir <= 0:
                    while tmp_lvl < n_levels:
                        alloc = allocations[tmp_lvl]
                        lim   = upper_envs[tmp_lvl, i - 1]
                        if high[i] >= lim:
                            prev_val = tmp_avg * tmp_qty
                            tmp_qty += alloc
                            tmp_avg  = (prev_val + lim * alloc) / tmp_qty
                            wp += lim * alloc; wa += alloc
                            tmp_lvl += 1; tmp_dir = -1; traded = True
                        else:
                            break

                sl_hit = False; sl_px = 0.0
                if tmp_dir != 0:
                    if tmp_dir == 1:
                        sl_level = tmp_avg * (1 - sl_pct)
                        if low[i] <= sl_level:
                            sl_hit = True; sl_px = sl_level
                    else:
                        sl_level = tmp_avg * (1 + sl_pct)
                        if high[i] >= sl_level:
                            sl_hit = True; sl_px = sl_level

                if sl_hit:
                    target_size[i] = 0.0; exec_price[i] = sl_px
                    pos_dir = 0; cur_lvl = 0; avg_entry = 0.0; cur_qty = 0.0
                    sl_triggered = True
                elif traded:
                    pos_dir = tmp_dir; cur_lvl = tmp_lvl
                    avg_entry = tmp_avg; cur_qty = tmp_qty
                    target_size[i] = cur_qty if pos_dir == 1 else -cur_qty
                    exec_price[i]  = wp / wa if wa > 0 else np.nan

            return target_size, exec_price

        def run_strategy(self):
            data = self.load_data()
            if data.empty:
                return None
            o, h, l, c = data['open'], data['high'], data['low'], data['close']
            ma = self.moving_average(o, h, l, c, self.ohlc4, self.ma_periods)
            upper_df, lower_df = self.calculate_envelopes(ma, self.envelope_pct_config)

            n_levels = len(self.envelope_pct_config)
            n_bars   = len(data)
            upper_np = np.zeros((n_levels, n_bars))
            lower_np = np.zeros((n_levels, n_bars))
            allocs   = np.zeros(n_levels)
            for i in range(n_levels):
                upper_np[i] = upper_df[f'upper_{i}'].values
                lower_np[i] = lower_df[f'lower_{i}'].values
                allocs[i]   = self.envelope_pct_config[i][1]

            target_size, exec_price = self.ram_dca_with_sl_nb(
                h.values, l.values, c.values, ma.values,
                upper_np, lower_np, allocs, self.sl_pct,
            )

            freq_map = {'1m':'1min','3m':'3min','5m':'5min','15m':'15min',
                        '30m':'30min','1h':'1h','2h':'2h','4h':'4h','1d':'1D'}
            freq = freq_map.get(self.tf, self.tf)

            pf = vbt.Portfolio.from_orders(
                close=c,
                size=pd.Series(target_size, index=data.index),
                price=pd.Series(exec_price,  index=data.index),
                size_type='TargetPercent',
                init_cash=self.capital,
                leverage=self.leverage,
                fees=0.00075,
                freq=freq,
            )
            self.results = pf
            return pf

        def get_trade_details(self):
            pf     = self.run_strategy()
            trades = pf.trades.records_readable.copy()
            trades['Entry Index'] = trades['Entry Index'].dt.strftime('%Y-%m-%d %H:%M:%S')
            capital = self.capital
            capitals, peak, max_dds = [capital], capital, [0]
            for _, row in trades.iterrows():
                capital += row['PnL']
                capitals.append(capital)
                if capital > peak:
                    peak = capital
                    max_dds.append(0)
                else:
                    max_dds.append((peak - capital) / peak * 100)
            trades['Capital']      = capitals[1:]
            trades['Max DD (%)']   = max_dds[1:]
            return trades

        def data_plot(self):
            trades = self.get_trade_details()
            trades['Entry Index'] = pd.to_datetime(trades['Entry Index'])
            fig, ax1 = plt.subplots(figsize=(15, 6))
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Capital', color='darkblue')
            ax1.plot(trades['Entry Index'], trades['Capital'], color='darkblue')
            ax1.tick_params(axis='y', labelcolor='darkblue')
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_tick_params(rotation=45)
            ax1.grid(True, linestyle='--', linewidth=0.5)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Max Drawdown (%)', color='darkred')
            ax2.plot(trades['Entry Index'], trades['Max DD (%)'], color='darkred', linestyle='--')
            ax2.tick_params(axis='y', labelcolor='darkred')
            ax2.invert_yaxis()
            fig.tight_layout()
            plt.title('Évolution du Capital & Max Drawdown')
            fig.legend(['Capital', 'Max DD (%)'], loc='upper left',
                       bbox_to_anchor=(0.08, 0.92))
            return fig

        def plot_monthly_returns(self):
            trades = self.get_trade_details()
            trades['Entry Index'] = pd.to_datetime(trades['Entry Index'])
            trades = trades.set_index('Entry Index')
            monthly = trades['Capital'].resample('ME').last().pct_change().fillna(0) * 100
            fig, ax = plt.subplots(figsize=(14, 5))
            colors  = ['#247B47' if x > 0 else '#C71A1A' for x in monthly.values]
            bars    = ax.bar(monthly.index.strftime('%Y-%m'), monthly.values, color=colors)
            ax.axhline(0, color='gray', linewidth=0.8)
            ax.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.title('Rendements Mensuels (%)')
            plt.ylabel('%')
            for bar in bars:
                y   = bar.get_height()
                pos = y + 0.5 if y >= 0 else y - 0.5
                ax.text(bar.get_x() + bar.get_width() / 2, pos,
                        f'{y:.1f}%', va='center', ha='center', fontsize=8)
            fig.tight_layout()
            return fig

    return Ram_Strategy_SL,


@app.cell
def _controls(mo):
    capital_sl  = mo.ui.number(start=1000, stop=1_000_000, step=1000,  value=10000, label='Capital ($)')
    leverage_sl = mo.ui.number(start=1,    stop=10,        step=1,     value=1,     label='Levier')
    run_btn     = mo.ui.button(label='▶ Lancer le backtest', value=0, on_click=lambda v: v + 1)

    mo.output.replace(mo.vstack([
        mo.md('## RAM DCA — HYPE / Lighter 5m'),
        mo.md('**Params :** `ma=230` · `enveloppes=[(3%, 50%), (6%, 50%)]` · `SL=10%` · `OHLC4=True`'),
        mo.hstack([capital_sl, leverage_sl, run_btn], justify='start', gap=3),
    ]))
    return capital_sl, leverage_sl, run_btn


@app.cell
def _run(Ram_Strategy_SL, capital_sl, leverage_sl, mo, run_btn):
    if not run_btn.value:
        mo.stop(True, mo.callout(mo.md('Clique **▶ Lancer le backtest** pour démarrer.'), kind='neutral'))

    mo.output.replace(mo.callout(mo.md('Calcul en cours…'), kind='info'))

    _capital  = capital_sl.value
    _leverage = leverage_sl.value

    _strat = Ram_Strategy_SL(
        'HYPE',
        230,
        [(0.03, 0.5), (0.06, 0.5)],
        True,
        None, None,
        '5m', 'lighter',
        _capital, _leverage, 0.1,
    )

    _pf = _strat.run_strategy()

    import io as _io, base64 as _b64, plotly.graph_objects as _go

    def _fig_to_html(fig):
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        buf.seek(0)
        b64 = _b64.b64encode(buf.read()).decode()
        return mo.Html(f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1200px"/>')

    _stats = _pf.stats(metrics='all', silence_warnings=True)
    _stats_df = _stats.reset_index()
    _stats_df.columns = ['Métrique', 'Valeur']
    _stats_df['Valeur'] = _stats_df['Valeur'].apply(
        lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)
    )

    _fig_cap  = _strat.data_plot()
    _fig_mret = _strat.plot_monthly_returns()

    import matplotlib.pyplot as _plt
    _plt.close('all')

    mo.output.replace(mo.vstack([
        mo.md(f'### Résultats — HYPE Lighter 5m · capital={_capital:,}$ · levier={_leverage}×'),
        mo.md('#### Stats complètes'),
        mo.ui.table(_stats_df, selection=None),
        mo.md('---\n#### Évolution du capital & Drawdown'),
        _fig_to_html(_fig_cap),
        mo.md('---\n#### Rendements mensuels'),
        _fig_to_html(_fig_mret),
    ]))
    return


if __name__ == '__main__':
    app.run()
