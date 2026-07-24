"""Dynamic discovery + import of strategies in src/approach/<APPROACH_ID>/strategy.py."""
import importlib.util
import sys
from pathlib import Path

from .config import APPROACH_ROOT


def list_approaches() -> list[str]:
    """Return all approach IDs present in src/approach/."""
    if not APPROACH_ROOT.exists():
        return []
    return sorted([
        p.name for p in APPROACH_ROOT.iterdir()
        if p.is_dir() and (p / "strategy.py").exists() and not p.name.startswith(".") and not p.name.startswith("__")
    ])


def load_strategy_module(approach_id: str):
    """Import src/approach/<approach_id>/strategy.py and return the module.

    Module is registered under sys.modules as `approach.<approach_id>.strategy`
    so subprocess workers can pickle / re-import it.
    """
    mod_path = APPROACH_ROOT / approach_id / "strategy.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Strategy not found: {mod_path}")

    mod_name = f"approach.{approach_id}.strategy"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    # Ensure parent packages exist in sys.modules for pickle support
    if "approach" not in sys.modules:
        spec = importlib.util.spec_from_file_location("approach", APPROACH_ROOT / "__init__.py")
        if spec and spec.loader:
            parent = importlib.util.module_from_spec(spec)
            sys.modules["approach"] = parent
            try:
                spec.loader.exec_module(parent)
            except Exception:
                pass

    pkg_name = f"approach.{approach_id}"
    if pkg_name not in sys.modules:
        pkg_init = APPROACH_ROOT / approach_id / "__init__.py"
        if pkg_init.exists():
            spec = importlib.util.spec_from_file_location(pkg_name, pkg_init)
            if spec and spec.loader:
                pkg = importlib.util.module_from_spec(spec)
                sys.modules[pkg_name] = pkg
                try:
                    spec.loader.exec_module(pkg)
                except Exception:
                    pass

    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {mod_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


def instantiate_strategy(approach_id: str):
    """Instantiate the Strategy class exported from approach/<id>/strategy.py."""
    mod = load_strategy_module(approach_id)
    if not hasattr(mod, "Strategy"):
        raise AttributeError(f"{approach_id}/strategy.py must export a class named `Strategy`")
    return mod.Strategy()
