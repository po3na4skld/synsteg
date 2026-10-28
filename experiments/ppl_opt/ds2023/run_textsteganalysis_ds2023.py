"""Protocol A' (yjs1224/TextSteganalysis, notebook cell 8 config, 60/20/20, use_processor, BERT for lstmatt) on the
DeepStego corpus: ds2023_base (= artifacts/synsteg_*) vs ds2023_pplopt, seeds 42/43/44 (config seed).
Usage: experiments/steganalysis/.venv_ts/bin/python run_textsteganalysis_ds2023.py -> textsteganalysis_ds2023_summary.json"""
import json, os, subprocess, sys, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import pairs

BASE = os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg')); REPO = f'{BASE}/experiments/steganalysis/TextSteganalysis'
DATA = f'{BASE}/datasets/textsteganalysis_runs'; OUT = f'{BASE}/experiments/ppl_opt/ds2023/textsteganalysis_ds2023_summary.json'
CORPORA = {'ds2023_base': pairs('base')[0], 'ds2023_pplopt': pairs('opt')[0]}
MODELS = ["rbilstmc", "lstmatt", "cnn", "bilstmdense", "birnn"]  # notebook order
SEEDS = [42, 43, 44]

nb = json.load(open(f'{BASE}/notebooks/steganalysis_training/training_steganalysis_models.ipynb'))
src = ''.join(nb['cells'][8]['source']).replace('config = ', '', 1)
summary = json.load(open(OUT)) if os.path.exists(OUT) else {}
for seed in SEEDS:
    for corpus, prs in CORPORA.items():
        d = f'{DATA}/{corpus}'; os.makedirs(d, exist_ok=True)
        open(f'{d}/clean.txt', 'w').write('\n'.join(a.replace('\n', ' ') for a, _ in prs))
        open(f'{d}/stego.txt', 'w').write('\n'.join(b.replace('\n', ' ') for _, b in prs))
        for m in MODELS:
            k = f'{corpus}/{m}/s{seed}'
            if k in summary: continue
            cfg = eval(src, {'DATA_PATH': d, 'True': True, 'False': False})
            cfg['model'] = m; cfg['use_plm'] = m in ('sesy', 'lstmatt'); cfg['seed'] = seed
            cfg['out_dir'] = f'{BASE}/reports/textsteganalysis/{corpus}_{m}_s{seed}'
            cfg['Dataset'].update(stego_file=f'{d}/stego.txt', cover_file=f'{d}/clean.txt', csv_dir=f'{d}/csv_s{seed}')
            cp = f'{d}/config_{m}_s{seed}.json'; json.dump(cfg, open(cp, 'w'), indent=1)
            t = time.time()
            out = subprocess.run([sys.executable, 'main.py', '--config_path', cp], cwd=REPO, capture_output=True, text=True)
            log = out.stdout + out.stderr
            g = lambda n: (lambda v: float(v[-1]) if v else None)(re.findall(rf'- +{n} = ([0-9.]+)', log))
            summary[k] = {'acc': g('accuracy'), 'precision': g('precision'), 'recall': g('recall'), 'rc': out.returncode, 'sec': round(time.time() - t)}
            open(f'{d}/log_{m}_s{seed}.txt', 'w').write(log)
            print(k, summary[k], flush=True)
            json.dump(summary, open(OUT, 'w'), indent=1)
print('done')
