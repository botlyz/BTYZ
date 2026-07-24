"""Façade CLI du pipeline quantlab.

    python -m quantlab.cli <commande> [STRATEGY...] [options]

Commandes : list, status, check, screen, optimize, plateau, validate,
deploy-rule, run, holdout, incubation, report, data sync, data status.
Codes retour : 0 PASS, 1 REJECT, 2 erreur — scriptable.

Imports lourds différés dans les commandes (--help instantané) ; le fan-out
process (spawn) tourne dans les modules du pipeline, jamais dans __main__.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

EXIT_PASS, EXIT_REJECT, EXIT_ERROR = 0, 1, 2
_PASS_VERDICTS = {"PASS", "FIXED", "ADAPTIVE"}
# commande CLI -> étape ledger (gating can_run)
_STAGE_OF = {"check": "contract", "screen": "screening", "optimize": "optimize",
             "plateau": "plateau", "validate": "stats",
             "deploy-rule": "deploy_rule", "holdout": "holdout"}

_CONSOLE = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _console():
    global _CONSOLE
    if _CONSOLE is None:
        from rich.console import Console
        _CONSOLE = Console()
        if not _CONSOLE.is_terminal:   # sortie pipée: ne pas écraser les tables
            _CONSOLE = Console(width=200)
    return _CONSOLE


# ------------------------------------------------------------------ affichage
def _style_of(verdict) -> str:
    v = str(verdict)
    if v in _PASS_VERDICTS or v == "COHERENT":
        return "bold green"
    if v in ("REJECT", "ERROR", "EFFONDREMENT"):
        return "bold red"
    if v in ("MARGINAL", "WARNING", "WAITING_LIVE_DATA"):
        return "yellow"
    return "dim"


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "oui" if v else "non"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _show_verdict(v: dict, title: str = ""):
    """Panel rich vert/rouge : verdict + reason + métriques scalaires."""
    from rich.panel import Panel
    from rich.table import Table

    verdict = str(v.get("verdict", v.get("status", "?")))
    style = _style_of(verdict)
    border = ("green" if verdict in _PASS_VERDICTS
              else "red" if verdict in ("REJECT", "ERROR") else "yellow")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("verdict", f"[{style}]{verdict}[/]")
    if v.get("mode") and v["mode"] != verdict:
        grid.add_row("mode", str(v["mode"]))
    if v.get("reason"):
        grid.add_row("raison", str(v["reason"]))
    if v.get("instructions"):
        grid.add_row("instructions", str(v["instructions"]))
    n_rows = 0
    for k, val in (v.get("metrics") or {}).items():
        if isinstance(val, (int, float, str, bool)) or val is None:
            grid.add_row(k, _fmt(val))
            n_rows += 1
        if n_rows >= 14:
            break
    if v.get("n_tests") is not None:
        grid.add_row("n_tests", str(v["n_tests"]))
    _console().print(Panel(grid, title=title or v.get("stage", ""),
                           border_style=border))


def _exit_of(v: dict) -> int:
    return EXIT_PASS if str(v.get("verdict")) in _PASS_VERDICTS else EXIT_REJECT


def _write_verdict(sid: str, stage: str, verdict: dict):
    from quantlab import config
    out_dir = config.RESULTS_ROOT / sid / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))


# ------------------------------------------------------------------ sélection
def _resolve_sids(arg_list, *, multi: bool = False) -> list[str] | None:
    """Stratégies demandées, ou sélection interactive questionary si absentes."""
    from quantlab import registry
    known = sorted(registry.discover())
    if isinstance(arg_list, str):   # nargs="?" -> str, nargs="*" -> list
        arg_list = [arg_list]
    if arg_list:
        sids = list(arg_list)
        unknown = [s for s in sids if s not in known]
        if unknown:
            _console().print(f"[red]Stratégie(s) inconnue(s) : {unknown}[/] "
                             f"(découvertes : {known})")
            return None
        return sids
    if not known:
        _console().print("[red]Aucune stratégie découverte dans strategies/[/]")
        return None
    if not sys.stdin.isatty():
        _console().print("[red]Aucune stratégie fournie et pas de TTY pour la "
                         f"sélection interactive.[/] Choix : {known}")
        return None
    import questionary
    if multi:
        sel = questionary.checkbox("Stratégies ?", choices=known).ask()
    else:
        one = questionary.select("Stratégie ?", choices=known).ask()
        sel = [one] if one else []
    if not sel:
        _console().print("[yellow]Sélection vide — abandon.[/]")
        return None
    return sel


def _gate(led, sid: str, stage: str, force: bool) -> bool:
    """ledger.can_run avant chaque étape ; --force outrepasse (tests comptés)."""
    ok, why = led.can_run(sid, stage)
    if ok:
        return True
    if force:
        _console().print(f"[yellow]{sid}: gating outrepassé (--force) — {why}[/]")
        return True
    _console().print(f"[red]{sid}: étape '{stage}' bloquée — {why}.[/] "
                     "(--force pour re-run explicite)")
    return False


# ================================================================== list/status
def cmd_list(args) -> int:
    from rich.table import Table
    from quantlab import config, registry
    from quantlab.ledger import family_of, ledger as led

    strats = registry.discover()
    if not strats:
        _console().print("[yellow]Aucune stratégie dans strategies/[/]")
        return EXIT_PASS

    table = Table(title="Stratégies découvertes", show_lines=False)
    for col in ("stratégie", "famille", "source", "tf"):
        table.add_column(col, style="bold cyan" if col == "stratégie" else None)
    for stage in config.STAGES:
        table.add_column(stage, justify="center")
    table.add_column("debt", justify="right")

    for sid in sorted(strats):
        cls = strats[sid]
        state = led.pipeline_state(sid)
        cells = []
        for stage in config.STAGES:
            v = state.get(stage)
            cells.append(f"[{_style_of(v)}]{v or '—'}[/]")
        table.add_row(sid, family_of(sid), cls.DATA_SOURCE, cls.TF, *cells,
                      str(led.research_debt(family_of(sid))))
    _console().print(table)
    return EXIT_PASS


def cmd_status(args) -> int:
    from rich.table import Table
    from quantlab import config
    from quantlab.ledger import ledger as led

    df = led.status_table()
    if df.empty:
        _console().print("[yellow]Ledger vide — aucun run enregistré.[/]")
        return EXIT_PASS

    table = Table(title="Funnel quantlab (ledger)")
    table.add_column("stratégie", style="bold cyan")
    table.add_column("famille")
    for stage in config.STAGES:
        table.add_column(stage, justify="center")
    table.add_column("research_debt N", justify="right", style="yellow")
    table.add_column("seuil √(2lnN)", justify="right", style="magenta")

    for sid, row in df.iterrows():
        cells = [f"[{_style_of(row[s])}]{row[s] or '—'}[/]"
                 for s in config.STAGES]
        table.add_row(sid, row["family"], *cells,
                      str(row["research_debt"]), f"{row['credible_sharpe']:.3f}")
    _console().print(table)
    return EXIT_PASS


# ================================================================== check
def _dev_data_for_check(strategy):
    """(pair, DataFrame dev) d'une paire liquide de la source de la stratégie."""
    from quantlab import screening

    source, tf = strategy.DATA_SOURCE, strategy.TF
    if screening._is_store_source(source):
        from quantlab.data import store
        pairs = store.universe(source, tf)
    elif source == "oi":
        pairs = screening._oi_pairs()
    else:
        pairs = ["BTC"]
    if not pairs:
        raise RuntimeError(f"aucune paire disponible pour la source '{source}'")
    pair = "BTC" if "BTC" in pairs else pairs[0]
    data = screening._load_cell_dev(source, pair, tf)
    if data is None or not len(data):
        raise RuntimeError(f"données dev introuvables ({source}/{pair}/{tf})")
    return pair, data


def _do_check(sid: str) -> dict:
    """Contrat + anti look-ahead sur données réelles dev -> verdict 'contract'."""
    from quantlab import lookahead, registry
    from quantlab.ledger import ledger as led

    try:
        strategy = registry.load(sid)
    except Exception as exc:
        verdict = {"stage": "contract", "verdict": "REJECT",
                   "reason": f"contrat invalide : {exc}",
                   "metrics": {}, "n_tests": 0, "at": _now()}
        _write_verdict(sid, "contract", verdict)
        led.record_verdict(sid, "contract", verdict)
        return verdict

    pair, data = _dev_data_for_check(strategy)
    res = lookahead.check(strategy, data)
    detail = res["detail"]
    fd = res["first_divergence"]
    if res["passed"]:
        verdict_str = "PASS"
        reason = (f"contrat OK + look-ahead OK : {detail['n_prefixes_checked']} "
                  f"préfixes identiques sur {strategy.DATA_SOURCE}/{pair}/"
                  f"{strategy.TF} ({detail['n_bars']} barres dev)")
    else:
        verdict_str = "REJECT"
        reason = (f"look-ahead DÉTECTÉ : divergence à {fd['timestamp']} "
                  f"(série '{fd['series']}', préfixe {fd['prefix']} barres) — "
                  "signaux non causaux")
    verdict = {
        "stage": "contract", "verdict": verdict_str, "reason": reason,
        "metrics": {
            "source": strategy.DATA_SOURCE, "pair": pair, "tf": strategy.TF,
            "n_bars": int(detail["n_bars"]),
            "n_prefixes": int(detail["n_prefixes_checked"]),
            "first_divergence": None if fd is None else {
                "timestamp": str(fd["timestamp"]), "series": fd["series"],
                "prefix": int(fd["prefix"])},
        },
        "n_tests": 0, "at": _now(),
    }
    _write_verdict(sid, "contract", verdict)
    led.record_verdict(strategy, "contract", verdict)
    return verdict


def cmd_check(args) -> int:
    sids = _resolve_sids(args.strategy, multi=False)
    if sids is None:
        return EXIT_ERROR
    v = _do_check(sids[0])
    _show_verdict(v, title=f"{sids[0]} — check (contract)")
    return _exit_of(v)


# ================================================================== étapes 1-5
def _kw(args, *names) -> dict:
    """Options CLI non-nulles -> kwargs du module."""
    out = {}
    for name in names:
        val = getattr(args, name.replace("-", "_"), None)
        if val is not None:
            out[name.replace("-", "_")] = val
    return out


def _do_screen(sid: str, args) -> dict:
    from quantlab import registry, screening
    strategy = registry.load(sid)
    kw = _kw(args, "pairs", "tfs", "sources", "fees_bps", "bootstrap", "n_null")
    if getattr(args, "workers", None):
        kw["max_workers"] = args.workers
    return screening.run(strategy, seed=args.seed, **kw)


def _do_optimize(sid: str, args) -> dict:
    from quantlab import optimize, registry
    strategy = registry.load(sid)
    kw = _kw(args, "pairs", "tf", "source", "fees_bps", "trials", "k_folds",
             "embargo", "workers")
    return optimize.run(strategy, seed=args.seed, **kw)


def _do_plateau(sid: str, args) -> dict:
    from quantlab import plateau, registry
    strategy = registry.load(sid)
    kw = _kw(args, "top_frac", "workers")
    return plateau.run(strategy, **kw)


def _do_validate(sid: str, args) -> dict:
    """Batterie PBO+DSR ; --permutation ajoute la permutation MC (long)."""
    from quantlab import registry, stats
    from quantlab.ledger import ledger as led
    strategy = registry.load(sid)
    v = stats.run_battery(strategy)
    if v["verdict"] == "REJECT" or not getattr(args, "permutation", False):
        return v
    _show_verdict(v, title=f"{sid} — validate (PBO+DSR)")
    kw = _kw(args, "budget_frac")
    if getattr(args, "n_runs", None):
        kw["n_runs"] = args.n_runs
    perm = stats.run_permutation(strategy, seed=args.seed, **kw)
    if perm["verdict"] == "REJECT":
        # la permutation invalide l'étape stats au ledger (état pipeline)
        led.record_verdict(strategy, "stats", perm)
        return perm
    perm["reason"] = f"PBO+DSR PASS ; permutation {perm['reason']}"
    return perm


def _do_deploy_rule(sid: str, args) -> dict:
    from quantlab import deploy_rule, registry
    strategy = registry.load(sid)
    return deploy_rule.run(strategy,
                           adaptive=not getattr(args, "no_adaptive", False))


_STEP_FN = {"check": lambda sid, args: _do_check(sid),
            "screen": _do_screen, "optimize": _do_optimize,
            "plateau": _do_plateau, "validate": _do_validate,
            "deploy-rule": _do_deploy_rule}


def _run_step(cmd: str, sid: str, args) -> tuple[dict | None, int]:
    """Gating + exécution d'une étape. (verdict|None, code retour)."""
    from quantlab.ledger import ledger as led
    stage = _STAGE_OF[cmd]
    if stage != "contract" and not _gate(led, sid, stage, args.force):
        return None, EXIT_ERROR
    v = _STEP_FN[cmd](sid, args)
    _show_verdict(v, title=f"{sid} — {cmd}")
    return v, _exit_of(v)


def _make_stage_cmd(cmd: str, multi: bool = False):
    def _fn(args) -> int:
        sids = _resolve_sids(args.strategy, multi=multi)
        if sids is None:
            return EXIT_ERROR
        worst = EXIT_PASS
        for sid in sids:
            _, code = _run_step(cmd, sid, args)
            worst = max(worst, code)
        return worst
    return _fn


cmd_screen = _make_stage_cmd("screen", multi=True)
cmd_optimize = _make_stage_cmd("optimize")
cmd_plateau = _make_stage_cmd("plateau")
cmd_validate = _make_stage_cmd("validate")
cmd_deploy_rule = _make_stage_cmd("deploy-rule")


# ================================================================== run (chaîne)
_RUN_SEQUENCE = ["check", "screen", "optimize", "plateau", "validate",
                 "deploy-rule"]


def _funnel_summary(sid: str, results: dict[str, dict | None]):
    from rich.table import Table
    table = Table(title=f"Funnel {sid}")
    table.add_column("étape")
    table.add_column("verdict", justify="center")
    table.add_column("raison", overflow="fold")
    for cmd in _RUN_SEQUENCE:
        v = results.get(cmd)
        if v is None:
            table.add_row(cmd, "[dim]—[/]", "")
        else:
            verdict = str(v.get("verdict", "?"))
            table.add_row(cmd, f"[{_style_of(verdict)}]{verdict}[/]",
                          str(v.get("reason", ""))[:200])
    _console().print(table)


def cmd_run(args) -> int:
    sids = _resolve_sids(args.strategy, multi=True)
    if sids is None:
        return EXIT_ERROR
    recap: dict[str, str] = {}
    worst = EXIT_PASS
    for sid in sids:
        _console().rule(f"[bold cyan]run {sid}[/]")
        results: dict[str, dict | None] = {}
        code = EXIT_PASS
        for cmd in _RUN_SEQUENCE:
            v, code = _run_step(cmd, sid, args)
            results[cmd] = v
            if code != EXIT_PASS:
                stopped = ("REJECT" if code == EXIT_REJECT else "ERREUR")
                recap[sid] = f"{stopped} à l'étape {cmd}"
                break
        else:
            recap[sid] = "PASS complet (jusqu'à deploy-rule)"
        _funnel_summary(sid, results)
        worst = max(worst, code)
    if len(sids) > 1:
        _console().rule("[bold]récapitulatif[/]")
        for sid, msg in recap.items():
            style = "green" if msg.startswith("PASS") else "red"
            _console().print(f"  [{style}]{sid}[/] : {msg}")
    return worst


# ================================================================== holdout
def cmd_holdout(args) -> int:
    from rich.panel import Panel
    from quantlab import gates
    from quantlab.ledger import family_of, ledger as led

    sids = _resolve_sids(args.strategy, multi=False)
    if sids is None:
        return EXIT_ERROR
    sid = sids[0]
    family = family_of(sid)

    if not led.holdout_available(family):
        _console().print(f"[bold red]REFUS : le holdout de la famille "
                         f"'{family}' a déjà été consommé — famille brûlée, "
                         "aucun retry possible.[/]")
        return EXIT_ERROR
    if not _gate(led, sid, "holdout", args.force):
        return EXIT_ERROR

    _console().print(Panel(
        f"[bold red]GATE HUMAIN — HOLDOUT one-shot[/]\n\n"
        f"Tu t'apprêtes à consommer le holdout de la famille "
        f"[bold]{family}[/] (derniers min(6 mois, 20 %) de chaque série).\n"
        f"• UN SEUL passage, succès OU échec : la famille sera BRÛLÉE.\n"
        f"• Le résultat sera écrit quel qu'il soit. AUCUN retry.\n"
        f"• Méfie-toi de tout résultat qui te fait plaisir.",
        border_style="red", title="⚠ avertissement"))
    try:
        answer = input(f'Tape exactement "BURN {family}" pour confirmer '
                       "(toute autre saisie = abandon) : ")
    except EOFError:
        answer = ""
    if answer.strip() != f"BURN {family}":
        _console().print("[yellow]Abandon — holdout NON consommé "
                         "(famille intacte).[/]")
        return EXIT_ERROR

    v = gates.run_holdout(sid)   # consomme lui-même le token one-shot
    _show_verdict(v, title=f"{sid} — holdout ({v.get('result', '')})")
    return _exit_of(v)


# ================================================================== incubation
def cmd_incubation(args) -> int:
    from quantlab import gates
    sids = _resolve_sids(args.strategy, multi=False)
    if sids is None:
        return EXIT_ERROR
    out = gates.incubation_report(sids[0], fills_csv=args.fills)
    _show_verdict(out, title=f"{sids[0]} — incubation")
    if out.get("status") == "WAITING_LIVE_DATA":
        return EXIT_PASS
    return _exit_of(out)


# ================================================================== report
def cmd_report(args) -> int:
    from quantlab import config, report
    sids = _resolve_sids(args.strategy, multi=False)
    if sids is None:
        return EXIT_ERROR
    sid = sids[0]
    report.render_markdown(sid)
    path = config.RESULTS_ROOT / sid / "REPORT.md"
    _console().print(f"[green]Fiche écrite :[/] {path}")
    return EXIT_PASS


# ================================================================== data
def cmd_data_sync(args) -> int:
    from rich.table import Table
    from quantlab import data

    stats = data.sync(sources=args.sources, tfs=args.tfs, pairs=args.pairs)
    table = Table(title="data sync")
    for col in ("source", "paires", "cellules ok", "cellules vides",
                "téléchargées", "manquantes"):
        table.add_column(col)
    for sname, st in stats.items():
        table.add_row(sname, str(st.get("pairs")), str(st.get("cells_ok")),
                      str(st.get("cells_empty")),
                      str(len(st.get("downloaded", []))),
                      str(len(st.get("missing", []))))
    _console().print(table)
    return EXIT_PASS


def cmd_data_status(args) -> int:
    from rich.table import Table
    from quantlab import data

    df = data.coverage()
    if args.sources:
        df = df[df["source"].isin(args.sources)]
    if args.tfs:
        df = df[df["tf"].isin(args.tfs)]
    if args.pairs:
        df = df[df["pair"].isin(args.pairs)]
    if df.empty:
        _console().print("[yellow]Store vide (lancer `data sync`).[/]")
        return EXIT_PASS

    if len(df) > 200:   # résumé par source × tf
        agg = df.groupby(["source", "tf"]).agg(
            n_pairs=("pair", "nunique"), n_bars=("n_bars", "sum"),
            start=("start", "min"), end=("end", "max")).reset_index()
        table = Table(title=f"Couverture du store (résumé, {len(df)} cellules)")
        for col in ("source", "tf", "paires", "barres", "début", "fin"):
            table.add_column(col)
        for _, r in agg.iterrows():
            table.add_row(r["source"], r["tf"], str(r["n_pairs"]),
                          f"{int(r['n_bars']):,}", str(r["start"])[:16],
                          str(r["end"])[:16])
    else:
        table = Table(title=f"Couverture du store ({len(df)} cellules)")
        for col in ("source", "paire", "tf", "barres", "début", "fin",
                    "cutoff holdout"):
            table.add_column(col)
        for _, r in df.iterrows():
            table.add_row(r["source"], r["pair"], r["tf"], str(r["n_bars"]),
                          str(r["start"])[:16], str(r["end"])[:16],
                          str(r["holdout_cutoff"])[:16])
    _console().print(table)
    return EXIT_PASS


# ================================================================== parser
def _add_strategy(sp, multi: bool = False):
    sp.add_argument("strategy", nargs="*" if multi else "?", metavar="STRATEGY",
                    help="id de stratégie (sélection interactive si absent)")
    sp.add_argument("--force", action="store_true",
                    help="outrepasse le gating ledger.can_run (re-run explicite "
                         "— les tests restent comptés)")


def _add_seed(sp):
    sp.add_argument("--seed", type=int, default=42, help="seed (défaut 42)")


def _add_screen_opts(sp):
    sp.add_argument("--pairs", nargs="+", help="paires imposées (ex. BTC ETH)")
    sp.add_argument("--tfs", nargs="+", help="timeframes screenées")
    sp.add_argument("--sources", nargs="+", help="sources (lighter binance)")
    sp.add_argument("--fees-bps", type=float, help="frais en bps")
    sp.add_argument("--bootstrap", type=int,
                    help="resamples bootstrap p-value (défaut config)")
    sp.add_argument("--n-null", type=int,
                    help="stratégies nulles par cellule (défaut config)")
    sp.add_argument("--workers", type=int, help="processus parallèles")


def _add_optimize_opts(sp):
    sp.add_argument("--pairs", nargs="+", help="paires (défaut: survivantes du "
                                               "screening)")
    sp.add_argument("--tf", help="timeframe (défaut: TF native)")
    sp.add_argument("--source", help="source de données")
    sp.add_argument("--fees-bps", type=float, help="frais en bps")
    sp.add_argument("--trials", type=int, help="budget Optuna (défaut config)")
    sp.add_argument("--k-folds", type=int, help="folds walk-forward")
    sp.add_argument("--embargo", type=int, help="embargo en barres (défaut auto)")
    sp.add_argument("--workers", type=int, help="processus parallèles")


def _add_validate_opts(sp):
    sp.add_argument("--permutation", action="store_true",
                    help="ajoute la permutation Monte-Carlo (long)")
    sp.add_argument("--n-runs", type=int,
                    help="runs de permutation (défaut config)")
    sp.add_argument("--budget-frac", type=float,
                    help="fraction du budget trials par permutation")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quantlab",
        description="Pipeline de validation de stratégies quant "
                    "(codes retour : 0 PASS, 1 REJECT, 2 erreur)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list", help="stratégies découvertes + état pipeline")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("status", help="funnel global (verdicts, N, seuil)")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("check", help="contrat + anti look-ahead (étape 0)")
    _add_strategy(sp)
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("screen", help="screening multi-paires (étape 1)")
    _add_strategy(sp, multi=True)
    _add_screen_opts(sp)
    _add_seed(sp)
    sp.set_defaults(func=cmd_screen)

    sp = sub.add_parser("optimize", help="Optuna walk-forward poolé (étape 2)")
    _add_strategy(sp)
    _add_optimize_opts(sp)
    _add_seed(sp)
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("plateau", help="centre du plateau + perturbations "
                                        "(étape 3)")
    _add_strategy(sp)
    sp.add_argument("--top-frac", type=float, help="fraction des meilleurs "
                                                   "trials (défaut 0.2)")
    sp.add_argument("--workers", type=int, help="processus parallèles")
    sp.set_defaults(func=cmd_plateau)

    sp = sub.add_parser("validate", help="batterie stats PBO+DSR (étape 4)")
    _add_strategy(sp)
    _add_validate_opts(sp)
    _add_seed(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("deploy-rule", help="WFE + stabilité + méta-backtest "
                                            "(étape 5)")
    _add_strategy(sp)
    sp.add_argument("--no-adaptive", action="store_true",
                    help="saute le méta-backtest adaptatif")
    sp.set_defaults(func=cmd_deploy_rule)

    sp = sub.add_parser("run", help="chaîne check→screen→optimize→plateau→"
                                    "validate→deploy-rule (stop au 1er REJECT)")
    _add_strategy(sp, multi=True)
    _add_screen_opts(sp)
    sp.add_argument("--tf", help="timeframe optimize")
    sp.add_argument("--source", help="source optimize")
    sp.add_argument("--trials", type=int, help="budget Optuna")
    sp.add_argument("--k-folds", type=int, help="folds walk-forward")
    sp.add_argument("--embargo", type=int, help="embargo en barres")
    sp.add_argument("--top-frac", type=float, help="fraction plateau")
    _add_validate_opts(sp)
    sp.add_argument("--no-adaptive", action="store_true",
                    help="deploy-rule sans méta-backtest")
    _add_seed(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("holdout", help="gate humain one-shot — famille brûlée "
                                        "(étape 6)")
    _add_strategy(sp)
    sp.set_defaults(func=cmd_holdout)

    sp = sub.add_parser("incubation", help="rapport incubation live (étape 7)")
    _add_strategy(sp)
    sp.add_argument("--fills", help="CSV des fills live (ts,pair,side,qty,px,"
                                    "fee,funding)")
    sp.set_defaults(func=cmd_incubation)

    sp = sub.add_parser("report", help="fiche REPORT.md de la stratégie")
    _add_strategy(sp)
    sp.set_defaults(func=cmd_report)

    dp = sub.add_parser("data", help="store parquet : sync / status")
    dsub = dp.add_subparsers(dest="data_command", required=True)
    sp = dsub.add_parser("sync", help="télécharge le manquant + matérialise le "
                                      "store")
    sp.add_argument("--sources", nargs="+", help="sources (défaut: toutes)")
    sp.add_argument("--tfs", nargs="+", help="timeframes (défaut: toutes)")
    sp.add_argument("--pairs", nargs="+", help="paires (défaut: univers)")
    sp.set_defaults(func=cmd_data_sync)
    sp = dsub.add_parser("status", help="couverture paires × TF × source")
    sp.add_argument("--sources", nargs="+", help="filtre sources")
    sp.add_argument("--tfs", nargs="+", help="filtre timeframes")
    sp.add_argument("--pairs", nargs="+", help="filtre paires")
    sp.set_defaults(func=cmd_data_status)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _console().print("\n[yellow]Interrompu.[/]")
        return EXIT_ERROR
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        _console().print(f"[bold red]Erreur : {type(exc).__name__}: {exc}[/]")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
