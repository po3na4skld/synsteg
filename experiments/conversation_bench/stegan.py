"""Steganalysis of Conversation-Steganography: 400 stegotexts vs 400 length-matched GPT-2 covers per configuration,
whole texts, random 80/20 split stratified by label, 15 % of train for validation, seeds 0/1/2.
Detectors and training loop are imported from experiments/steganalysis/steganalysis.py (not copied).

Usage: $PY experiments/conversation_bench/stegan.py <cfg>   -> results/steganalysis_<cfg>.json (one process per
configuration, so the three can run in parallel)
"""
import json, os, sys, time
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg'))
sys.path.insert(0, f'{ROOT}/experiments/steganalysis')
sys.path.insert(0, f'{ROOT}/experiments/conversation_bench')
import steganalysis as sa  # noqa: E402
from covers import RES, load_stego  # noqa: E402


def main(cfg):
    res = {'config': {'maxlen': sa.MAXLEN, 'seeds': sa.SEEDS, 'epochs_max': sa.EPOCHS, 'patience': sa.PATIENCE,
                      'batch': sa.BS, 'val_frac_of_train': sa.VAL_FRAC, 'test_frac': 0.2}, 'runs': []}
    t0 = time.time()
    stego, _, _ = load_stego(cfg)
    covers = [json.loads(l)['text'] for l in open(f'{RES}/covers_{cfg}.jsonl')]
    X = np.array(covers + [r['stegotext'] for r in stego], dtype=object)
    y = np.array([0] * len(covers) + [1] * len(stego))
    for seed in sa.SEEDS:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        tr, va = train_test_split(tr, test_size=sa.VAL_FRAC, stratify=y[tr], random_state=seed)
        for name in sa.DETECTORS:
            t = time.time()
            m, fz, info = sa.fit(name, list(X[tr]), y[tr], list(X[va]), y[va], seed)
            r = sa.evaluate(name, m, fz, list(X[te]), y[te], None, per_L=False)
            res['runs'].append({'config': cfg, 'detector': name, 'seed': seed, 'n_train': len(tr), 'n_val': len(va),
                                'n_test': len(te), 'train': info, 'test': r, 'sec': round(time.time() - t, 1)})
            print(f'{cfg} s{seed} {name:14s} acc={r["acc"]:.3f} ep={info["epochs_run"]} {time.time() - t:.0f}s', flush=True)
    res['config']['runtime_sec'] = round(time.time() - t0)
    json.dump(res, open(f'{RES}/steganalysis_{cfg}.json', 'w'), indent=1, default=float)
    print('wrote', f'{RES}/steganalysis_{cfg}.json', res['config']['runtime_sec'], 's')


if __name__ == '__main__':
    main(sys.argv[1])
