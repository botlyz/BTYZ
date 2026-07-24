"""
Worker persistant pour l'optimisation.
Importé une seule fois, reçoit des tâches via stdin, libère la RAM avec malloc_trim entre chaque.
"""
import sys
import os
import gc
import ctypes
import json
import warnings
import io

# Limiter les threads internes (numpy/numba/openblas) à 1
# Car on a déjà 24 workers en parallèle — pas besoin de threading interne
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Sauver le vrai stdout pour les RESULT
_real_stdout = sys.stdout

# Rediriger stdout pendant l'import VBT (il peut print des trucs)
sys.stdout = io.StringIO()
from vectorbtpro import vbt
from src.opti import process_chunk
sys.stdout = _real_stdout

# malloc_trim pour rendre la mémoire à l'OS
_libc = ctypes.CDLL("libc.so.6")

def _cleanup():
    vbt.flush()
    gc.collect()
    _libc.malloc_trim(0)

# Signaler qu'on est prêt
_real_stdout.write("READY\n")
_real_stdout.flush()

# Boucle : lire des tâches depuis stdin, exécuter, répondre sur stdout
for line in sys.stdin:
    line = line.strip()
    if not line or line == "EXIT":
        break

    try:
        task = json.loads(line)
        pair = task['pair']
        chunk_idx = task['chunk_idx']
        sub_grid = task['sub_grid']
        strategy = task['strategy']
        exchange = task['exchange']
        tf = task['tf']
        cache_dir = task['cache_dir']

        # Capturer stdout pendant le calcul (VBT peut print des trucs)
        sys.stdout = io.StringIO()
        try:
            p, c, ok, msg = process_chunk(pair, chunk_idx, sub_grid, strategy, exchange, tf, cache_dir)
        finally:
            sys.stdout = _real_stdout

        _real_stdout.write(f"RESULT:{p}:{c}:{ok}:{msg}\n")
        _real_stdout.flush()

        _cleanup()

    except Exception as e:
        sys.stdout = _real_stdout
        _real_stdout.write(f"RESULT:{pair}:{chunk_idx}:False:{e}\n")
        _real_stdout.flush()
        _cleanup()
