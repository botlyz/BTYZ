"""LIQ_FADE_v1 — kernel tick event-driven : fade des cascades de liquidation.

Thèse : une cascade de liquidations = flux FORCÉ (le liquidé vend/achète au marché,
price-insensitive) qui pousse le prix en overshoot. Après épuisement du flux forcé,
le prix reverte. On fournit la liquidité APRÈS l'épuisement et on encaisse le rebond.

Pourquoi tick et pas barre : le signal (intensité de liq) et l'exécution sont
sub-seconde. Resampler en bougies détruirait le timing. On streame les trades bruts.

Données = ticks Lighter (data/lighter_ticks/<SYM>.parquet) : timestamp(ms), px, sz,
is_maker_ask (True=trade à l'ask/achat taker, False=au bid/vente taker), trade_type.

EXÉCUTION RÉALISTE (sans L2) :
  - On ne se remplit PAS au creux instantané (c'est le flux forcé qui l'a pris).
  - Entrée = APRÈS épuisement, en taker → fill au prix du prochain trade du côté qu'on
    prend (on fade : liq de longs = vente forcée → on ACHÈTE → fill ~ ask) + buffer.
  - Le buffer de slippage est calibré sur la DISPERSION de la cascade (proxy de finesse
    du carnet, gratuit sans L2).

Signal (params optimisables) :
  win_s        : fenêtre d'accumulation de l'intensité de liq (secondes)
  z_enter      : seuil Z-score d'intensité pour armer (cascade anormale)
  exhaust_frac : on entre quand l'intensité retombe sous z_enter*exhaust_frac
  hold_s       : durée de détention avant sortie (secondes)  [exit temporel]
  buffer_mult  : multiplicateur du buffer de slippage (× dispersion cascade)
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _liq_fade_nb(ts, px, sz, ask, is_liq,
                 gap_ms, delay_ms, hold_ms, buffer_mult,
                 base_cost_frac, min_notional, stop_frac):
    """Streame les ticks. Détection BURST (réplique l'event study validé) :
    cluster de liquidations avec gaps < gap_ms = 1 burst ; on entre delay_ms après
    la FIN du burst (le rebond est rapide -> il faut entrer vite), fade, sortie +hold_ms.

    Retourne (entry_ts, entry_px, exit_ts, exit_px, dir, ret_frac) par trade.
    dir = +1 LONG (fade d'une vente forcée), -1 SHORT. ret = PnL net (coûts AR inclus).
    """
    n = ts.shape[0]
    MAXT = 200000
    e_ts = np.zeros(MAXT); e_px = np.zeros(MAXT); x_ts = np.zeros(MAXT)
    x_px = np.zeros(MAXT); d_ar = np.zeros(MAXT); r_ar = np.zeros(MAXT)
    nt = 0

    burst_active = False; burst_signed = 0.0; last_liq_ts = 0
    casc_hi = 0.0; casc_lo = 1e18
    pending = False; pend_ts = 0; pend_dir = 0.0; pend_buf = 0.0
    in_pos = False; pos_dir = 0.0; pos_entry_px = 0.0; pos_exit_ts = 0; pos_buf = 0.0

    for i in range(n):
        t = ts[i]

        # 1) sortie de position : stop-loss (cascade qui ne reverte pas) OU horizon
        if in_pos:
            # excursion adverse courante (avant buffer) : pour un long, perte si px < entrée
            adverse = pos_dir * (px[i] / pos_entry_px - 1.0)
            stop_hit = stop_frac > 0.0 and adverse <= -stop_frac
            if stop_hit or t >= pos_exit_ts:
                xp = px[i] * (1.0 - pos_dir * pos_buf)
                gross = pos_dir * (xp / pos_entry_px - 1.0)
                x_ts[nt] = t; x_px[nt] = xp
                r_ar[nt] = gross - base_cost_frac
                nt += 1; in_pos = False
                if nt >= MAXT:
                    break

        # 2) entrée en attente -> exécutée au 1er tick après pend_ts
        if pending and (not in_pos) and t >= pend_ts:
            ep = px[i] * (1.0 + pend_dir * pend_buf)
            e_ts[nt] = t; e_px[nt] = ep; d_ar[nt] = pend_dir
            in_pos = True; pos_dir = pend_dir; pos_entry_px = ep
            pos_exit_ts = t + hold_ms; pos_buf = pend_buf
            pending = False

        # 3) fin de burst détectée (un tick arrive > gap_ms après la dernière liq)
        if burst_active and (t - last_liq_ts) > gap_ms:
            if (abs(burst_signed) >= min_notional) and (not in_pos) and (not pending):
                disp = (casc_hi - casc_lo) / px[i] if px[i] > 0 else 0.0
                pend_dir = -np.sign(burst_signed)      # FADE
                pend_buf = buffer_mult * disp
                pend_ts = last_liq_ts + delay_ms
                pending = True
            burst_active = False; burst_signed = 0.0

        # 4) tick de liquidation -> accumule le burst courant
        if is_liq[i]:
            if not burst_active:
                burst_active = True; burst_signed = 0.0
                casc_hi = px[i]; casc_lo = px[i]
            s = 1.0 if ask[i] else -1.0
            burst_signed += s * sz[i] * px[i]
            if px[i] > casc_hi: casc_hi = px[i]
            if px[i] < casc_lo: casc_lo = px[i]
            last_liq_ts = t

    return (e_ts[:nt], e_px[:nt], x_ts[:nt], x_px[:nt], d_ar[:nt], r_ar[:nt])


def run_liq_fade_arr(ts, px, sz, ask, is_liq, gap_s=3, delay_s=2, hold_s=300,
                     stop_frac=0.0, buffer_mult=0.1, base_cost_bps=8.0,
                     min_notional=10_000.0):
    """Variante arrays (légère mémoire) : pas de pandas/colonne string."""
    e_ts, e_px, x_ts, x_px, d_ar, r_ar = _liq_fade_nb(
        ts, px, sz, ask, is_liq,
        gap_s * 1000, delay_s * 1000, hold_s * 1000, buffer_mult,
        base_cost_bps * 1e-4, min_notional, stop_frac,
    )
    return dict(entry_ts=e_ts, entry_px=e_px, exit_ts=x_ts, exit_px=x_px,
                dir=d_ar, ret=r_ar)


def run_liq_fade(df, gap_s=3, delay_s=2, hold_s=300, stop_frac=0.0,
                 buffer_mult=0.1, base_cost_bps=8.0, min_notional=10_000.0):
    """Wrapper df (pratique pour tests). Réplique l'event study validé.
    gap_s : gap max entre 2 liq d'un même burst. delay_s : on entre delay_s après la
    fin du burst. stop_frac : stop-loss (fraction, 0=off). base_cost_bps = coût AR."""
    return run_liq_fade_arr(
        df["timestamp"].to_numpy(np.int64), df["px"].to_numpy(np.float64),
        df["sz"].to_numpy(np.float64), df["is_maker_ask"].to_numpy(np.bool_),
        (df["trade_type"].to_numpy().astype("U12") != "trade"),
        gap_s, delay_s, hold_s, stop_frac, buffer_mult, base_cost_bps, min_notional)
