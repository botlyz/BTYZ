"""vérifie rapidement les résultats d'opti depuis le terminal
usage : python tools/check_cache.py cache/full_lighter_5m/
        python tools/check_cache.py cache/full_lighter_5m/kc_wfsl_BTC_5m_lighter.pickle"""
import sys
import os
import pandas as pd
from vectorbtpro import vbt


def check_one(path):
    """affiche un résumé rapide d'un pickle d'opti"""
    name = os.path.basename(path).replace('.pickle', '')
    results = vbt.load(path)

    #separer train/test
    train = results.xs('train', level='set')
    test = results.xs('test', level='set')

    #sharpe median par combo
    train_median = train.groupby(['ma_window', 'atr_window', 'atr_mult', 'sl_stop']).median()
    test_median = test.groupby(['ma_window', 'atr_window', 'atr_mult', 'sl_stop']).median()

    #combos avec sharpe > 0 sur 50%+ des fenetres test
    test_grouped = test.groupby(['ma_window', 'atr_window', 'atr_mult', 'sl_stop'])
    pct_positive = test_grouped.apply(lambda x: (x > 0).sum() / len(x))
    robust = pct_positive[pct_positive >= 0.5]

    best_train = train_median.sort_values(ascending=False).head(5)
    best_test = test_median.sort_values(ascending=False).head(5)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Combos: {len(train_median)}")
    print(f"  Robustes (>0 sur 50%+ fenetres): {len(robust)}")
    print(f"  Sharpe median train: min={train_median.min():.2f}, max={train_median.max():.2f}")
    print(f"  Sharpe median test:  min={test_median.min():.2f}, max={test_median.max():.2f}")
    print(f"\n  Top 5 test:")
    for idx, val in best_test.items():
        print(f"    ma={idx[0]:>3}, atr_w={idx[1]:>2}, atr_m={idx[2]:.1f}, sl={idx[3]:.2f} -> sharpe {val:+.3f}")
    print()


def check_dir(dirpath):
    """résumé de tous les pickles dans un dossier + ranking"""
    pickles = sorted([f for f in os.listdir(dirpath) if f.endswith('.pickle')])
    print(f"{len(pickles)} fichiers dans {dirpath}\n")

    ranking = []
    for f in pickles:
        path = os.path.join(dirpath, f)
        name = f.replace('.pickle', '')
        try:
            results = vbt.load(path)
            test = results.xs('test', level='set')
            test_median = test.groupby(['ma_window', 'atr_window', 'atr_mult', 'sl_stop']).median()
            best = test_median.max()

            test_grouped = test.groupby(['ma_window', 'atr_window', 'atr_mult', 'sl_stop'])
            pct_pos = test_grouped.apply(lambda x: (x > 0).sum() / len(x))
            n_robust = (pct_pos >= 0.5).sum()

            ranking.append({'name': name, 'best_sharpe': best, 'robust': n_robust})
        except Exception as e:
            print(f"  {name}: erreur {e}")

    if not ranking:
        return

    ranking.sort(key=lambda x: x['best_sharpe'], reverse=True)

    print(f"{'='*60}")
    print(f"  RANKING (par meilleur sharpe test median)")
    print(f"{'='*60}")
    print(f"  {'Rank':<5} {'Paire':<30} {'Sharpe':>8} {'Robust':>8}")
    print(f"  {'-'*55}")
    for i, r in enumerate(ranking):
        marker = ' ***' if r['best_sharpe'] > 0 else ''
        print(f"  {i+1:<5} {r['name']:<30} {r['best_sharpe']:>+7.3f} {r['robust']:>8}{marker}")
    print(f"{'='*60}")

    positives = [r for r in ranking if r['best_sharpe'] > 0]
    print(f"\n  {len(positives)}/{len(ranking)} paires avec sharpe test > 0")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tools/check_cache.py cache/full_lighter_5m/")
        print("  python tools/check_cache.py cache/full_lighter_5m/kc_wfsl_BTC_5m_lighter.pickle")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        check_dir(target)
    elif os.path.isfile(target):
        check_one(target)
    else:
        print(f"'{target}' introuvable")
