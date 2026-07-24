"""Terminal d'avancement du pipeline (rich).

API stable (SPEC) :
    with PipelineProgress(strategy_id, stage, ledger=None) as prog:
        t = prog.task("screening BTC", total=120)
        t.advance(1, sharpe=1.2, pair="BTC")
        t.done()

TTY : rich.Live avec header Panel (stratégie, étape, research_debt N, seuil
Sharpe crédible √(2 ln N)) + barres (compteur, vitesse, ETA, postfix libre).
Non-TTY : pas de Live, une ligne de log tous les 10 %.
"""
from __future__ import annotations

import math
import sys
import time


def _fmt_postfix(postfix: dict) -> str:
    parts = []
    for k, v in postfix.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3g}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)


class TaskHandle:
    """Poignée d'une barre de progression (ou de son fallback log)."""

    def __init__(self, parent: "PipelineProgress", name: str, total: int,
                 rich_task_id=None):
        self._parent = parent
        self.name = name
        self.total = max(int(total), 1)
        self.completed = 0
        self._rich_id = rich_task_id
        self._last_logged_decile = -1
        self._t0 = time.monotonic()

    def advance(self, n: int = 1, **postfix):
        self.completed += n
        if self._rich_id is not None:
            self._parent._progress.update(
                self._rich_id, advance=n, postfix=_fmt_postfix(postfix))
        else:
            decile = min(self.completed * 10 // self.total, 10)
            if decile > self._last_logged_decile:
                self._last_logged_decile = decile
                elapsed = time.monotonic() - self._t0
                rate = self.completed / elapsed if elapsed > 0 else 0.0
                eta = ((self.total - self.completed) / rate) if rate > 0 else float("inf")
                print(f"[{self._parent.strategy_id}:{self._parent.stage}] "
                      f"{self.name} {decile * 10}% ({self.completed}/{self.total}) "
                      f"{rate:.1f}/s eta {eta:.0f}s  {_fmt_postfix(postfix)}",
                      file=sys.stderr, flush=True)

    def done(self):
        if self._rich_id is not None:
            self._parent._progress.update(self._rich_id, completed=self.total)
        elif self._last_logged_decile < 10:
            print(f"[{self._parent.strategy_id}:{self._parent.stage}] "
                  f"{self.name} terminé ({self.completed}/{self.total})",
                  file=sys.stderr, flush=True)


class PipelineProgress:
    """Affichage d'une étape du pipeline. `progress=None` accepté partout :
    les modules créent le leur si besoin."""

    def __init__(self, strategy_id: str, stage: str, ledger=None):
        self.strategy_id = strategy_id
        self.stage = stage
        self._ledger = ledger
        self._live = None
        self._progress = None
        from rich.console import Console
        self._console = Console(stderr=True)
        # décision Live sur le VRAI tty (FORCE_COLOR force is_terminal chez rich)
        try:
            self._is_tty = sys.stderr.isatty()
        except Exception:
            self._is_tty = False

    # ------------------------------------------------------------- header
    def _header(self):
        from rich.panel import Panel
        from rich.text import Text
        debt = None
        if self._ledger is not None:
            try:
                debt = self._ledger.research_debt(self.strategy_id)
            except Exception:
                debt = None
        txt = Text()
        txt.append(f"{self.strategy_id}", style="bold cyan")
        txt.append(f"  étape: {self.stage}", style="bold")
        if debt is not None:
            thresh = math.sqrt(2 * math.log(max(debt, 2)))
            txt.append(f"  research_debt: {debt}", style="yellow")
            txt.append(f"  seuil Sharpe crédible √(2lnN): {thresh:.2f}",
                       style="magenta")
        return Panel(txt, border_style="dim")

    # ------------------------------------------------------------- lifecycle
    def __enter__(self):
        if self._is_tty:
            from rich.console import Group
            from rich.live import Live
            from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                       ProgressColumn, TaskProgressColumn,
                                       TextColumn, TimeRemainingColumn)
            from rich.text import Text as _Text

            class _RateColumn(ProgressColumn):
                def render(self, task):
                    speed = task.finished_speed or task.speed
                    if speed is None:
                        return _Text("-/s", style="dim")
                    return _Text(f"{speed:.1f}/s", style="dim")

            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                _RateColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[postfix]}", style="green"),
                console=self._console,
            )
            self._live = Live(Group(self._header(), self._progress),
                              console=self._console, refresh_per_second=5)
            self._live.__enter__()
        else:
            print(f"[{self.strategy_id}:{self.stage}] démarrage",
                  file=sys.stderr, flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None
        return False

    # ------------------------------------------------------------- tasks
    def task(self, name: str, total: int) -> TaskHandle:
        if self._progress is not None:
            tid = self._progress.add_task(name, total=int(total), postfix="")
            return TaskHandle(self, name, total, rich_task_id=tid)
        return TaskHandle(self, name, total)
