# SPEC quantlab — contrat inter-modules (document de travail des agents)

Pipeline de validation de stratégies quant (voir docs/QUANTLAB.md pour le workflow
métier). Ce fichier fixe les APIs que chaque module DOIT respecter pour que
l'intégration se fasse sans friction. Lis aussi `src/quantlab/config.py` et
`src/quantlab/contract.py` (déjà écrits, ne pas modifier sans raison).

## Conventions globales

- Python 3, venv: `/home/botlyz-gpu/BTYZ/.venv` (vectorbtpro, optuna 4.7, pandas 2.3,
  numba, scipy, rich, marimo, plotly, questionary installés).
- Exécution: `cd BTYZ/src && python -m quantlab.cli ...`. Imports absolus:
  `from quantlab.x import y`, `from engine.data_loader import load_ohlcv`.
- Métriques de portefeuille: EXCLUSIVEMENT via `engine.metrics.extract_from_portfolio(pf)`
  (règle: pas de math manuelle Sharpe/DD). Trades via `engine.trades.extract_trades_df`.
- Toutes les sorties fichiers sous `config.RESULTS_ROOT/<STRATEGY_ID>/<stage>/...`
  (JSON pour les verdicts, parquet pour les matrices).
- Chaque module qui consomme des backtests DOIT les compter au ledger
  (`ledger.record_tests(family, n, stage, meta)`).
- Verdict d'étape: dict `{"stage": str, "verdict": "PASS"|"REJECT", "reason": str,
  "metrics": {...}, "n_tests": int, "at": iso8601}` écrit en
  `RESULTS_ROOT/<sid>/<stage>/verdict.json` ET enregistré au ledger.
- Affichage progression: utiliser `quantlab.progress` (voir plus bas). Aucun print brut
  dans les boucles chaudes.
- Pas de `random` sans seed. Seed par défaut 42, loggée.

## Modules et signatures

### quantlab/backtest.py (cœur partagé — écrit par l'agent "core")
```python
def run_signals_backtest(strategy, data: pd.DataFrame, params: dict, *,
                         fees: float, slippage: float = config.DEFAULT_SLIPPAGE,
                         init_cash: float = config.DEFAULT_INIT_CASH,
                         tf: str) -> "vbt.Portfolio":
    # signals = strategy.signals(data, strategy.full_params(params)) ; .align(data.index)
    # vbt.Portfolio.from_signals(close=..., entries=..., exits=..., short_entries=...,
    #   short_exits=..., sl_stop=..., tp_stop=..., td_stop=pd.Timedelta si td_stop,
    #   fees=fees, slippage=slippage, init_cash=init_cash, freq=config.FREQ_MAP[tf])

def pooled_backtest(strategy, datas: dict[str, pd.DataFrame], params, *, fees, tf,
                    weights: dict[str, float] | None = None) -> "vbt.Portfolio":
    # backtest multi-paires mêmes params: colonnes = paires (close en DataFrame large,
    #   signaux par paire), group_by=True cash partagé -> un seul portefeuille poolé.

def portfolio_returns(pf) -> pd.Series   # pf.returns() (par barre)
```

### quantlab/data/ (agent "data")
```python
# sources/base.py
class Source(ABC):
    name: str
    def pairs(self) -> list[str]
    def tfs_native(self) -> list[str]                      # tfs dispo sans resample
    def load_raw(self, pair, tf) -> pd.DataFrame | None    # OHLCV index UTC
    def download(self, pair, tf, progress=None) -> bool    # récupère le brut manquant
SOURCES: dict[str, Source]  # {"lighter": ..., "binance": ...} dans sources/__init__.py

# sources/lighter.py : wrappe engine.data_loader (CSV data/raw/lighter/<tf>/<BASE>.csv),
#   resample depuis 1m pour tfs manquantes (3m/4h). download() = no-op (données locales,
#   log un warning si absent).
# sources/binance.py : CSV data/raw/binance/<tf>/<PAIR>USDT.csv + download klines
#   mensuelles data.binance.vision (um futures) -> 1m, resample vers le reste.
#   Réutiliser la mécanique zip->csv de scripts/download_binance_alt.py.

# store.py — cache canonique parquet, invalidé si le brut est plus récent
def load(source: str, pair: str, tf: str) -> pd.DataFrame | None
    # lit STORE_ROOT/<source>/<tf>/<pair>.parquet ; sinon construit depuis Source.load_raw
    # + resample OHLCV correct (o=first,h=max,l=min,c=last,v=sum), écrit le parquet.
def universe(source: str, tf: str, min_bars=config.MIN_BARS) -> list[str]

# splits.py — holdout lock (étape 0.2)
def dev_holdout_cut(index: pd.DatetimeIndex) -> pd.Timestamp
    # cutoff = fin - min(HOLDOUT_MAX_DAYS, HOLDOUT_FRACTION * span)
def load_dev(source, pair, tf) -> pd.DataFrame | None      # store.load tronqué AVANT cutoff
def load_holdout(source, pair, tf, *, _token) -> pd.DataFrame
    # REFUSE sans token émis par ledger.consume_holdout(family) — le pipeline ne peut
    # physiquement pas lire le holdout hors gate.
def holdout_hash(source, pair, tf) -> str                  # sha256 du segment holdout

# sync.py
def sync(sources=None, tfs=None, pairs=None, progress=None) -> dict   # stats couverture
def coverage() -> pd.DataFrame   # source×pair×tf: n_bars, start, end, cutoff holdout
```

### quantlab/ledger.py (agent "core")
SQLite `config.LEDGER_DB`, WAL. Tables:
- `families(family TEXT PK, created_at TEXT, research_debt INTEGER DEFAULT 0)`
- `runs(id INTEGER PK, family, strategy_id, stage, code_hash, space_hash, seed,
   n_tests, meta_json, verdict, started_at, finished_at)`
- `events(id INTEGER PK, family, kind, payload_json, at)`  # holdout_consumed, gate_*
```python
class Ledger:  # context-manager ou instance module-level `ledger = Ledger()`
    def record_tests(self, family, n, stage, meta=None)          # incrémente research_debt
    def start_run(self, strategy, stage, meta) -> int            # run_id
    def finish_run(self, run_id, verdict, n_tests, meta=None)
    def research_debt(self, family) -> int
    def credible_sharpe(self, family) -> float                   # sqrt(2*ln(N)) (en σ)
    def pipeline_state(self, strategy_id) -> dict                # stage -> verdict|None
    def can_run(self, strategy_id, stage) -> tuple[bool, str]    # étapes précédentes PASS ?
    def consume_holdout(self, family) -> str | None              # token one-shot; None si brûlé
    def holdout_available(self, family) -> bool
    def record_verdict(self, strategy, stage, verdict_dict)
    def status_table(self) -> pd.DataFrame                       # funnel global
```

### quantlab/registry.py (agent "core")
```python
def discover() -> dict[str, type]      # STRATEGY_ID -> classe (import strategies/<dir>/strategy.py)
def load(strategy_id) -> BaseStrategy  # instancie, validate_strategy, RATIONALE.md présent sinon erreur
def read_manifest(strategy_id) -> dict # manifest.yaml optionnel
```
Import picklable pour ProcessPool: module enregistré `sys.modules[f"strategies.{dir}.strategy"]`
(même technique que archive/engine/approach_loader.py).

### quantlab/lookahead.py (agent "core")
```python
def check(strategy, data, params=None, n_prefixes=config.LOOKAHEAD_N_PREFIXES) -> dict
    # signals(full) vs signals(data.iloc[:i]) pour n_prefixes valeurs de i
    # (espacées log entre LOOKAHEAD_MIN_PREFIX et len). Compare les 4 séries bool
    # sur [0, i). dict: {"passed": bool, "first_divergence": ts|None, "detail": ...}
```

### quantlab/screening.py (agent "screen") — étape 1
```python
def run(strategy, *, sources=None, tfs=None, fees_bps=None, seed=42,
        progress=None) -> dict   # verdict complet
```
1. Cellules = source × tf (SCREEN_TFS ou [TF]) × paires de `data.universe` (dev set via
   `splits.load_dev` UNIQUEMENT). Params = DEFAULT_PARAMS partout.
2. Par cellule: backtest -> Sharpe + p-value par bootstrap stationnaire des returns
   (scipy/numpy, SCREEN_BOOTSTRAP resamples, H0 Sharpe<=0).
3. BH-FDR sur TOUTES les p-values ensemble -> q-values.
4. Nulle empirique: par cellule survivante, NULL_STRATEGIES_PER_CELL strats aléatoires
   appariées (mêmes nombre d'entrées, permutation circulaire des blocs d'entrée,
   même td/sl) -> le Sharpe candidat doit battre NULL_QUANTILE.
5. PASS si q<FDR_Q sur assez de cellules ET fraction de paires positives (dans la tf
   native) >= MIN_POSITIVE_PAIR_FRAC ET nulle battue.
Sorties: `screening/cells.parquet` (1 ligne/cellule: pair, tf, source, sharpe, pval,
qval, null_q95, n_trades, fees), `verdict.json`. Compte n_tests = cellules + nulles.

### quantlab/optimize.py (agent "optuna") — étape 2
```python
def run(strategy, *, pairs=None, tf=None, source="lighter", fees_bps=None,
        k_folds=config.DEFAULT_K_FOLDS, trials=config.DEFAULT_TRIALS,
        embargo=None, seed=42, multi_objective=False, progress=None) -> dict
```
- Paires par défaut: survivantes du screening (lues dans screening/cells.parquet).
- Folds temporels K sur le dev set, purge+embargo (embargo auto = WARMUP_BARS +
  td_stop max estimé). Objectif par trial: backtest POOLÉ multi-paires sur chaque
  fold de validation, returns de validation CONCATÉNÉS -> Sharpe annualisé de la
  série concaténée. Hard-reject (-10) si trades/param < MIN_TRADES_PER_PARAM dans
  un fold. Refus AVANT run si n_free_params > MAX_FREE_PARAMS.
- Storage Optuna: `RESULTS_ROOT/<sid>/optimize/optuna.db`, study name = run_id ledger.
- Sauver par trial: params, sharpe_concat, sharpe par fold, n_trades par fold, ET la
  série de returns de validation par périodes (matrice trials × périodes ->
  `optimize/trial_returns.parquet`, indispensable au PBO). Diagnostics train aussi
  (folds IS) pour le WFE de l'étape 5 -> `optimize/folds_is_oos.parquet`.
- n_tests = trials × k_folds.

### quantlab/plateau.py (agent "optuna") — étape 3
```python
def run(strategy, *, top_frac=0.2, progress=None) -> dict
```
- Depuis l'étude Optuna: régions (binning par param), médiane par région, choisir le
  centre du plateau (médiane haute + faible dispersion), PAS le best trial.
- Perturbation: chaque param ±PERTURBATION_PCT (params numériques; catégoriels: voisins),
  re-backtest WF complet -> chute Sharpe > MAX_SHARPE_DROP -> REJECT.
- Sorties: `plateau/selected_params.json` (+ ensemble 3-5 jeux optionnel),
  `plateau/perturbations.parquet`, verdict.

### quantlab/stats/ (agent "stats") — étape 4
```python
# pbo.py
def pbo_cscv(trial_returns: pd.DataFrame, s_blocks=config.PBO_BLOCKS) -> dict
    # matrice (périodes × trials). Retourne {"pbo": float, "logits": [...], ...}
# dsr.py  (reprendre archive/notebooks/analyse_full.py compute_dsr, adapter)
def deflated_sharpe(candidate_returns: pd.Series, *, n_effective: int,
                    sharpe_variance: float) -> dict   # {"dsr": p, "pass": bool, ...}
def effective_n(trial_returns: pd.DataFrame, research_debt: int) -> int
    # corrélation moyenne inter-trials -> N_eff = debt * (1-rho) + rho (clip >=1)
# permutation.py
def run(strategy, *, n_runs=config.PERMUTATION_RUNS, budget_frac=0.1, seed=42,
        progress=None) -> dict
    # block-permute les returns dev (PERMUTATION_BLOCK_BARS), reconstruit un OHLC
    # synthétique, relance screening(micro)+optimize(trials*budget_frac) -> best sharpe.
    # p-value = frac(perm_best >= real). ProcessPoolExecutor.
# __init__.py : run_battery(strategy, progress=None) -> verdict combiné PBO puis DSR
#   (permutation séparée, opt-in CLI --permutation)
```

### quantlab/deploy_rule.py (agent "stats") — étape 5
```python
def run(strategy, *, reopt_months=(1,2,3), progress=None) -> dict
```
- WFE = sharpe_OOS_annualisé / sharpe_IS_annualisé depuis optimize/folds_is_oos.parquet.
- Stabilité: distance L2 normalisée des optima entre folds + Spearman des surfaces.
- Méta-backtest: "tous les M mois ré-opti sur T mois puis applique" (budget réduit)
  vs jeu fixe du plateau -> equity OOS concaténée, net de frais. verdict FIXED|ADAPTIVE|REJECT.

### quantlab/gates.py (agent "stats") — étapes 6-7
```python
def run_holdout(strategy, *, confirm_token: str, progress=None) -> dict
    # exige token = ledger.consume_holdout(family) (one-shot). Backtest params retenus
    # sur load_holdout de toutes les paires retenues. Résultat écrit QUEL QU'IL SOIT.
def incubation_report(strategy, fills_csv=None) -> dict   # squelette comparaison
    # slippage/funding réels vs modèle (données à brancher plus tard sur le serveur live)
```

### quantlab/report.py (agent "cli")
```python
def build(strategy) -> dict    # fiche complète: verdicts par étape + params + debt
def render_markdown(strategy) -> str   # RESULTS_ROOT/<sid>/REPORT.md
```

### quantlab/progress.py (agent "cli", ÉCRIT EN PREMIER — les autres l'utilisent)
rich.progress. API minimale STABLE :
```python
class PipelineProgress:
    def __init__(self, strategy_id: str, stage: str, ledger=None): ...
    def __enter__ / __exit__          # Live display: header = stratégie, étape,
                                      # research_debt N, seuil sharpe crédible
    def task(self, name: str, total: int) -> TaskHandle
class TaskHandle:
    def advance(self, n=1, **postfix)   # postfix affiché (ex. sharpe=1.2, pair=BTC)
    def done(self)
```
Barres rich avec ETA + vitesse + compteur (le "terminal d'avancement" demandé:
étape, ETA, nombre de tests). Fallback silencieux si non-TTY (log 1 ligne/10%).
`progress=None` accepté partout (les modules créent le leur si None).

### quantlab/cli.py (agent "cli")
argparse + questionary (sélection interactive si args manquants):
```
python -m quantlab.cli list | check | screen | optimize | plateau | validate
                      | deploy-rule | holdout | status | data sync | data status | report
```
- `run <STRAT>` = enchaîne check->screen->optimize->plateau->validate->deploy-rule
  en s'arrêtant au premier REJECT (le flow du mermaid, tout l'automatisable).
- Chaque commande vérifie `ledger.can_run` (pas de saut d'étape), `--force` pour re-runs
  explicites (re-run = nouveaux tests comptés, jamais décomptés).
- `holdout` demande une confirmation interactive explicite ("BURN <family>" à taper).
- `status`: table rich du funnel (toutes stratégies × étapes, verdicts, debt, seuil).

## Style / qualité
- Docstrings courtes en français (comme l'existant), pas de sur-commentaire.
- Chaque agent livre un smoke-test exécutable `tests/smoke_<module>.py`
  (script simple, pas pytest requis) qui tourne en < 2 min sur données réelles locales.
- Ne JAMAIS toucher au holdout hors gates.py. Ne jamais lire data/raw directement
  hors quantlab/data/.
