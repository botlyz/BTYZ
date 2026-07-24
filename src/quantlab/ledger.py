"""Ledger SQLite du pipeline quantlab.

Compte chaque backtest (research_debt par FAMILLE, versions confondues),
journalise runs/verdicts, et matérialise le gate one-shot du holdout.
Thread-safe : une connexion par appel + lock process-local, WAL activé.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from quantlab import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS families (
    family TEXT PRIMARY KEY,
    created_at TEXT,
    research_debt INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT,
    strategy_id TEXT,
    stage TEXT,
    code_hash TEXT,
    space_hash TEXT,
    seed INTEGER,
    n_tests INTEGER,
    meta_json TEXT,
    verdict TEXT,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT,
    kind TEXT,
    payload_json TEXT,
    at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_sid ON runs(strategy_id, stage);
CREATE INDEX IF NOT EXISTS idx_events_family ON events(family, kind);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def family_of(strategy_id: str) -> str:
    """'OI_FADE_v2' -> 'OI_FADE' (le research_debt est par famille)."""
    return re.sub(r"_v\d+$", "", str(strategy_id))


def _sid(strategy) -> str:
    if isinstance(strategy, str):
        return strategy
    return strategy.strategy_id()


class Ledger:
    """Journal des tests. Utilisable en context-manager ou instance simple."""

    def __init__(self, db_path: str | Path = config.LEDGER_DB):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_done = False

    # ------------------------------------------------------------- connexion
    @contextmanager
    def _conn(self):
        with self._lock:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(self.db_path, timeout=30)
            try:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute("PRAGMA busy_timeout=30000")
                if not self._init_done:
                    con.executescript(_SCHEMA)
                    self._init_done = True
                yield con
                con.commit()
            except Exception:
                con.rollback()
                raise
            finally:
                con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _ensure_family(self, con, family: str):
        con.execute(
            "INSERT OR IGNORE INTO families(family, created_at, research_debt) "
            "VALUES (?, ?, 0)", (family, _now()))

    # ------------------------------------------------------------- comptage
    def record_tests(self, family: str, n: int, stage: str, meta: dict | None = None):
        """Incrémente le research_debt de la famille de `n` backtests."""
        family = family_of(family)
        with self._conn() as con:
            self._ensure_family(con, family)
            con.execute(
                "UPDATE families SET research_debt = research_debt + ? WHERE family = ?",
                (int(n), family))
            con.execute(
                "INSERT INTO events(family, kind, payload_json, at) VALUES (?,?,?,?)",
                (family, "tests",
                 json.dumps({"n": int(n), "stage": stage, "meta": meta or {}}),
                 _now()))

    def research_debt(self, family: str) -> int:
        family = family_of(family)
        with self._conn() as con:
            row = con.execute(
                "SELECT research_debt FROM families WHERE family = ?",
                (family,)).fetchone()
        return int(row[0]) if row else 0

    def credible_sharpe(self, family: str) -> float:
        """Seuil de Sharpe crédible (en σ) : sqrt(2 ln N), N = research_debt."""
        n = max(self.research_debt(family), 2)
        return math.sqrt(2.0 * math.log(n))

    # ------------------------------------------------------------- runs
    def start_run(self, strategy, stage: str, meta: dict | None = None) -> int:
        meta = dict(meta or {})
        sid = _sid(strategy)
        family = family_of(sid)
        code_hash = ""
        if not isinstance(strategy, str):
            try:
                code_hash = strategy.code_hash()
            except Exception:
                pass
        with self._conn() as con:
            self._ensure_family(con, family)
            cur = con.execute(
                "INSERT INTO runs(family, strategy_id, stage, code_hash, space_hash,"
                " seed, n_tests, meta_json, verdict, started_at, finished_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (family, sid, stage, code_hash, meta.pop("space_hash", ""),
                 int(meta.pop("seed", 42)), 0, json.dumps(meta), None, _now(), None))
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, verdict: str, n_tests: int, meta: dict | None = None):
        with self._conn() as con:
            if meta:
                row = con.execute("SELECT meta_json FROM runs WHERE id = ?",
                                  (run_id,)).fetchone()
                merged = {**(json.loads(row[0]) if row and row[0] else {}), **meta}
                con.execute("UPDATE runs SET meta_json = ? WHERE id = ?",
                            (json.dumps(merged), run_id))
            con.execute(
                "UPDATE runs SET verdict = ?, n_tests = ?, finished_at = ? WHERE id = ?",
                (verdict, int(n_tests), _now(), run_id))

    def record_verdict(self, strategy, stage: str, verdict_dict: dict):
        """Enregistre le verdict d'étape (source de vérité de can_run)."""
        sid = _sid(strategy)
        family = family_of(sid)
        with self._conn() as con:
            self._ensure_family(con, family)
            con.execute(
                "INSERT INTO runs(family, strategy_id, stage, code_hash, space_hash,"
                " seed, n_tests, meta_json, verdict, started_at, finished_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (family, sid, stage, "", "", 0,
                 int(verdict_dict.get("n_tests", 0)),
                 json.dumps(verdict_dict),
                 str(verdict_dict.get("verdict", "")),
                 verdict_dict.get("at", _now()), _now()))

    # ------------------------------------------------------------- état pipeline
    def pipeline_state(self, strategy_id: str) -> dict:
        """stage -> dernier verdict enregistré ('PASS'/'REJECT'/...) ou None."""
        sid = _sid(strategy_id)
        state = {stage: None for stage in config.STAGES}
        with self._conn() as con:
            rows = con.execute(
                "SELECT stage, verdict FROM runs WHERE strategy_id = ?"
                " AND verdict IS NOT NULL AND verdict != '' ORDER BY id",
                (sid,)).fetchall()
        for stage, verdict in rows:
            if stage in state:
                state[stage] = verdict
        return state

    def can_run(self, strategy_id: str, stage: str) -> tuple[bool, str]:
        """Toutes les étapes précédentes (jusqu'à deploy_rule) doivent être PASS."""
        if stage not in config.STAGES:
            return False, f"étape inconnue: {stage}"
        idx = config.STAGES.index(stage)
        if idx == 0:
            return True, "ok ('contract' n'a pas de prérequis)"
        last_auto = config.STAGES.index("deploy_rule")
        prereqs = [s for s in config.STAGES[:idx]
                   if config.STAGES.index(s) <= last_auto]
        state = self.pipeline_state(strategy_id)
        for s in prereqs:
            if state.get(s) != "PASS":
                return False, f"prérequis '{s}' non PASS (verdict: {state.get(s)})"
        return True, "ok"

    # ------------------------------------------------------------- holdout
    def holdout_available(self, family: str) -> bool:
        family = family_of(family)
        with self._conn() as con:
            row = con.execute(
                "SELECT 1 FROM events WHERE family = ? AND kind = 'holdout_consumed'"
                " LIMIT 1", (family,)).fetchone()
        return row is None

    def consume_holdout(self, family: str) -> str | None:
        """Brûle le holdout de la famille (one-shot). None si déjà consommé."""
        family = family_of(family)
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT 1 FROM events WHERE family = ? AND kind = 'holdout_consumed'"
                " LIMIT 1", (family,)).fetchone()
            if row is not None:
                return None
            self._ensure_family(con, family)
            token = str(uuid.uuid4())
            con.execute(
                "INSERT INTO events(family, kind, payload_json, at) VALUES (?,?,?,?)",
                (family, "holdout_consumed", json.dumps({"token": token}), _now()))
            return token

    # ------------------------------------------------------------- funnel
    def status_table(self):
        """DataFrame stratégies × étapes + research_debt + seuil Sharpe crédible."""
        import pandas as pd

        with self._conn() as con:
            fams = con.execute(
                "SELECT family, research_debt FROM families").fetchall()
            sids = [r[0] for r in con.execute(
                "SELECT DISTINCT strategy_id FROM runs WHERE strategy_id != ''"
            ).fetchall()]
        debt = {f: int(d) for f, d in fams}
        # familles sans run mais avec debt -> ligne quand même
        all_sids = sorted(set(sids) | {f for f in debt if f not in
                                       {family_of(s) for s in sids}})
        rows = []
        for sid in all_sids:
            fam = family_of(sid)
            state = self.pipeline_state(sid)
            n = max(debt.get(fam, 0), 2)
            rows.append({"strategy_id": sid, "family": fam, **state,
                         "research_debt": debt.get(fam, 0),
                         "credible_sharpe": round(math.sqrt(2 * math.log(n)), 3)})
        cols = ["strategy_id", "family", *config.STAGES,
                "research_debt", "credible_sharpe"]
        return pd.DataFrame(rows, columns=cols).set_index("strategy_id")


# instance module-level partagée (chemin par défaut config.LEDGER_DB)
ledger = Ledger()
