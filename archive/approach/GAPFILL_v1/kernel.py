"""GapfillKernel — machine à états pure, SOURCE DE VÉRITÉ unique du gap-fill.

Reproduit EXACTEMENT la boucle de `Strategy.compute_target_arrays` (strategy.py),
mais en *streaming* (une barre à la fois) pour le live. Le backtest peut le valider
via `run_vectorized` (doit être bit-identique à compute_target_arrays).

Convention 1:1 backtest (lag 1 barre) : à la barre i, TOUTES les décisions
(entrée / sortie / stop-loss / retournement) utilisent les données de la barre i-1 ;
l'exécution (et `entry_px`) se fait au prix de la barre i. Le stop-loss compare
`px[i-1]` à `entry_px×(1∓sl)`. Identique à strategy.py.

Le live appelle `step(px, fair_value, real_age_min)` à chaque nouvelle barre 5m
complétée → renvoie la position cible `s ∈ {-1, 0, +1}` à détenir.
"""
from __future__ import annotations

import numpy as np


class GapfillKernel:
    def __init__(self, entry_thresh: float, exit_thresh: float,
                 recent_min: float, sl_pct: float):
        self.entry = float(entry_thresh)
        self.exit = float(exit_thresh)
        self.recent_min = float(recent_min)
        self.sl = float(sl_pct)
        # État persistant
        self.s = 0                # position courante {-1,0,1}
        self.entry_px = 0.0
        self.cooldown = False
        # Données de la barre précédente (None tant qu'aucune barre vue)
        self.prev_px = None
        self.prev_dev = None
        self.prev_closed = None

    def step(self, px: float, fair_value: float, real_age_min: float) -> int:
        """Traite une barre. Renvoie la position cible APRÈS cette barre."""
        dev = (px - fair_value) / fair_value
        closed = bool(real_age_min > self.recent_min)
        s = self.s
        new_s = s

        # Décisions seulement si une barre précédente existe (équiv. i>=1 + _lag)
        if self.prev_px is not None:
            long_e = self.prev_closed and (self.prev_dev < -self.entry)
            short_e = self.prev_closed and (self.prev_dev > self.entry)
            gap_filled = abs(self.prev_dev) < self.exit
            exit_sig = gap_filled or (not self.prev_closed)

            # 1) Stop-loss (sur px[i-1] vs entry_px)
            if s != 0 and self.sl > 0:
                if (s == 1 and self.prev_px <= self.entry_px * (1.0 - self.sl)) or \
                   (s == -1 and self.prev_px >= self.entry_px * (1.0 + self.sl)):
                    new_s = 0
                    self.cooldown = True

            # 2) Sortie standard (gap rempli / réouverture)
            if new_s != 0 and exit_sig:
                new_s = 0

            # Release cooldown
            if self.cooldown and exit_sig:
                self.cooldown = False

            # 3) Entrée / retournement (bloqué pendant cooldown)
            if not self.cooldown:
                if long_e:
                    new_s = 1
                elif short_e:
                    new_s = -1

            if new_s != s:
                if new_s != 0:
                    self.entry_px = px
                self.s = new_s

        # Mémoriser la barre courante pour la prochaine décision
        self.prev_px = float(px)
        self.prev_dev = float(dev)
        self.prev_closed = closed
        return self.s

    def run_vectorized(self, px, fair, age) -> np.ndarray:
        """Rejoue tout l'historique → cible sparse (NaN sauf aux changements),
        format identique à compute_target_arrays pour le test de parité."""
        n = len(px)
        target = np.full(n, np.nan)
        s_prev = 0
        for i in range(n):
            s_new = self.step(float(px[i]), float(fair[i]), float(age[i]))
            if s_new != s_prev:
                target[i] = float(s_new)
            s_prev = s_new
        return target

    # --- Persistance JSON (état entre cycles cron) ---
    def to_dict(self) -> dict:
        return {
            "s": self.s, "entry_px": self.entry_px, "cooldown": self.cooldown,
            "prev_px": self.prev_px, "prev_dev": self.prev_dev,
            "prev_closed": self.prev_closed,
        }

    def from_dict(self, d: dict) -> "GapfillKernel":
        self.s = int(d.get("s", 0))
        self.entry_px = float(d.get("entry_px", 0.0))
        self.cooldown = bool(d.get("cooldown", False))
        self.prev_px = d.get("prev_px")
        self.prev_dev = d.get("prev_dev")
        self.prev_closed = d.get("prev_closed")
        return self
