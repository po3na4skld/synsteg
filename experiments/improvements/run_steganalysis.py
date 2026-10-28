"""Protocol A' (yjs1224/TextSteganalysis, notebook cell 8 config, 60/20/20 random sentence split, BERT for lstmatt),
exactly as ppl_opt/ds2023/run_textsteganalysis_ds2023.py, on the DeepStego corpus (reports/json_reports) re-encoded
with each profile. The sentence pairs are written by profiles_ds2023.py to datasets/textsteganalysis_runs/imp_<arm>/.
Arms: imp_base (published keys = control, byte-identical to artifacts/synsteg_*) and imp_p1..p3. Seeds 42/43/44.
Usage: experiments/steganalysis/.venv_ts/bin/python run_steganalysis.py -> steganalysis_summary.json (resumable)"""
import json, os, subprocess, sys, re, time

BASE = os.path.expanduser('~/PycharmProjects/Personal/SynSteg'); REPO = f'{BASE}/experiments/steganalysis/TextSteganalysis'
DATA = f'{BASE}/datasets/textsteganalysis_runs'; OUT = f'{BASE}/experiments/improvements/steganalysis_summary.json'
ARMS = sys.argv[1:] or ['imp_base', 'imp_p1', 'imp_p2', 'imp_p3']
MODELS = ["rbilstmc", "lstmatt", "cnn", "bilstmdense", "birnn"]  # notebook order
SEEDS = [42, 43, 44]

nb = json.load(open(f'{BASE}/notebooks/steganalysis_training/training_steganalysis_models.ipynb'))
src = ''.join(nb['cells'][8]['source']).replace('config = ', '', 1)
summary = json.load(open(OUT)) if os.path.exists(OUT) else {}
for seed in SEEDS:
    for corpus in ARMS:
        d = f'{DATA}/{corpus}'
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
