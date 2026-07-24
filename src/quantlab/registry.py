"""Découverte + import des stratégies BTYZ/strategies/<DIR>/strategy.py.

Les modules sont enregistrés dans sys.modules sous `strategies.<dir>.strategy`
(même technique que archive/engine/approach_loader.py) pour rester picklables
dans les workers ProcessPool.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from quantlab import config
from quantlab.contract import BaseStrategy, validate_strategy

_PKG_ROOT = "strategies"


def _ensure_package(name: str, path: Path):
    """Crée (si absent) un module-package minimal dans sys.modules."""
    if name in sys.modules:
        return
    init = path / "__init__.py"
    if init.exists():
        spec = importlib.util.spec_from_file_location(name, init)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
    else:  # package-namespace synthétique
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod


def list_strategy_dirs() -> list[str]:
    """Dossiers de STRATEGIES_ROOT contenant un strategy.py."""
    root = config.STRATEGIES_ROOT
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "strategy.py").exists()
                  and not p.name.startswith((".", "__")))


def _import_strategy_module(dir_name: str):
    mod_name = f"{_PKG_ROOT}.{dir_name}.strategy"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    root = config.STRATEGIES_ROOT
    mod_path = root / dir_name / "strategy.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Stratégie introuvable: {mod_path}")
    _ensure_package(_PKG_ROOT, root)
    _ensure_package(f"{_PKG_ROOT}.{dir_name}", root / dir_name)
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Spec illisible: {mod_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


def discover() -> dict[str, type]:
    """STRATEGY_ID -> classe Strategy (les dossiers non importables sont ignorés)."""
    out: dict[str, type] = {}
    for d in list_strategy_dirs():
        try:
            mod = _import_strategy_module(d)
            cls = getattr(mod, "Strategy", None)
            if cls is not None and issubclass(cls, BaseStrategy):
                out[cls.strategy_id()] = cls
        except Exception as e:  # dossier cassé -> visible mais non bloquant
            print(f"[registry] {d}: import impossible ({e})", file=sys.stderr)
    return out


def _dir_for(strategy_id: str) -> str | None:
    dirs = list_strategy_dirs()
    if strategy_id in dirs:
        return strategy_id
    for d in dirs:  # dossier dont la classe porte cet id
        try:
            mod = _import_strategy_module(d)
            cls = getattr(mod, "Strategy", None)
            if cls is not None and cls.strategy_id() == strategy_id:
                return d
        except Exception:
            continue
    return None


def load(strategy_id: str) -> BaseStrategy:
    """Importe, valide le contrat + RATIONALE.md, instancie."""
    d = _dir_for(strategy_id)
    if d is None:
        raise KeyError(f"Stratégie inconnue: {strategy_id} "
                       f"(dossiers: {list_strategy_dirs()})")
    mod = _import_strategy_module(d)
    cls = getattr(mod, "Strategy", None)
    if cls is None:
        raise AttributeError(f"{d}/strategy.py doit exporter une classe `Strategy`")
    errors = validate_strategy(cls)
    if not (config.STRATEGIES_ROOT / d / "RATIONALE.md").exists():
        errors.append("RATIONALE.md manquant (thèse économique obligatoire)")
    if errors:
        raise ValueError(f"{strategy_id}: contrat invalide: {errors}")
    return cls()


# --------------------------------------------------------------- manifest
def _parse_simple_yaml(text: str) -> dict:
    """Parser minimal clé: valeur (fallback sans pyyaml). Listes inline [a, b] ok."""
    out: dict = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or ":" not in line or line.startswith((" ", "\t", "-")):
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if not val:
            continue
        if val.startswith("[") and val.endswith("]"):
            out[key] = [_coerce(v.strip()) for v in val[1:-1].split(",") if v.strip()]
        else:
            out[key] = _coerce(val)
    return out


def _coerce(v: str):
    v = v.strip().strip("'\"")
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def read_manifest(strategy_id: str) -> dict:
    """manifest.yaml optionnel du dossier stratégie ({} si absent)."""
    d = _dir_for(strategy_id)
    if d is None:
        return {}
    path = config.STRATEGIES_ROOT / d / "manifest.yaml"
    if not path.exists():
        return {}
    text = path.read_text()
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        return _parse_simple_yaml(text)
