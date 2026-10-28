"""Run yjs1224/TextSteganalysis with the exact config of notebooks/steganalysis_training/training_steganalysis_models.ipynb
(Colab, 11.2024) on sentence corpora: old DeepStego corpus (control), gpt-6-luna and Qwen series.
Only local change: device MPS instead of CUDA. Output: ../../reports/textsteganalysis_<corpus>_<model>/ and textsteganalysis_summary.json."""
import json, os, subprocess, sys, re, time
sys.path.insert(0, os.path.join(os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg')), 'experiments/steganalysis'))
from steganalysis_sentences import series, lines

HERE = os.path.join(os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg')), 'experiments/steganalysis'); REPO = f'{HERE}/TextSteganalysis'
BASE = os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg')); DATA = f'{BASE}/datasets/textsteganalysis_runs'
CORPORA = {'gpt4o_base': series('gpt_omni_reports')[0], 'gpt4o_pplopt32': series('../experiments/ppl_opt/series/opt_k32')[0]}
MODELS = ["rbilstmc", "lstmatt", "cnn", "bilstmdense", "birnn"]  # notebook order

nb = json.load(open(f'{BASE}/notebooks/steganalysis_training/training_steganalysis_models.ipynb'))
src = ''.join(nb['cells'][8]['source']).replace('config = ', '', 1)
summary = {}
for corpus, pairs in CORPORA.items():
    d = f'{DATA}/{corpus}'; os.makedirs(d, exist_ok=True)
    open(f'{d}/clean.txt', 'w').write('\n'.join(a.replace('\n', ' ') for a, _ in pairs))
    open(f'{d}/stego.txt', 'w').write('\n'.join(b.replace('\n', ' ') for _, b in pairs))
    for m in MODELS:
        cfg = eval(src, {'DATA_PATH': d, 'True': True, 'False': False})
        cfg['model'] = m; cfg['use_plm'] = m in ('sesy', 'lstmatt')
        cfg['out_dir'] = f'{BASE}/reports/textsteganalysis/{corpus}_{m}'
        cfg['Dataset'].update(stego_file=f'{d}/stego.txt', cover_file=f'{d}/clean.txt', csv_dir=f'{d}/csv')
        cp = f'{d}/config_{m}.json'; json.dump(cfg, open(cp, 'w'), indent=1)
        t = time.time()
        out = subprocess.run([sys.executable, 'main.py', '--config_path', cp], cwd=REPO, capture_output=True, text=True)
        log = out.stdout + out.stderr
        acc = re.findall(r'accuracy = ([0-9.]+)', log)
        summary[f'{corpus}/{m}'] = {'acc': float(acc[-1]) if acc else None, 'rc': out.returncode, 'sec': round(time.time() - t)}
        open(f'{d}/log_{m}.txt', 'w').write(log)
        print(corpus, m, summary[f'{corpus}/{m}'], flush=True)
        json.dump(summary, open(f'{BASE}/experiments/ppl_opt/textsteganalysis_opt_summary.json', 'w'), indent=1)
print('done')
