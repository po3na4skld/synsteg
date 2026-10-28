"""Markdown tables from reports/steganalysis_missing.json (mean ± SD over seeds)."""
import json, os, numpy as np
R = json.load(open(os.path.join(os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg')), 'reports/steganalysis_missing.json')))
DET = ['TS-CNN', 'TS-RNN', 'Bi-LSTM-Dense', 'BERT-LSTM-Att', 'length-baseline']
ms = lambda v: f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}'

def table(setting, series):
    print(f'\n### {series} ({setting})\n\n| Detector | Accuracy | Precision | Recall | F1 | Acc range | Pred. stego share | One-class runs | Epochs |\n|---|---|---|---|---|---|---|---|---|')
    for d in DET:
        rs = [r for r in R['runs'] if r['setting'] == setting and r['series'] == series and r['detector'] == d]
        t = [r['test'] for r in rs]; a = [x['acc'] for x in t]
        ep = '-'.join(map(str, sorted(r['train']['epochs_run'] for r in rs))) if 'train' in rs[0] else '-'
        print(f"| {d} | {ms(a)} | {ms([x['prec'] for x in t])} | {ms([x['rec'] for x in t])} | {ms([x['f1'] for x in t])} "
              f"| {min(a):.4f}-{max(a):.4f} | {', '.join(f'{x['frac_pred_stego']:.2f}' for x in t)} | {sum(x['degenerate'] for x in t)}/3 | {ep} |")

def per_L(setting, series):
    print(f'\n### {series} ({setting}) per L: accuracy / F1\n\n| Detector | L = 128 | L = 256 | L = 512 |\n|---|---|---|---|')
    for d in DET[:4]:
        t = [r['test']['per_L'] for r in R['runs'] if r['setting'] == setting and r['series'] == series and r['detector'] == d]
        print(f'| {d} | ' + ' | '.join(f"{ms([x[l]['acc'] for x in t])} / {ms([x[l]['f1'] for x in t])}" for l in ('128', '256', '512')) + ' |')

for s in ('gpt6_luna_b5', 'qwen3_30b_b3'): table('in-series', s)
per_L('in-series', 'gpt6_luna_b5'); table('cross', 'gpt6->qwen'); per_L('cross', 'gpt6->qwen')
print('\nper-run train info:'); [print(r['setting'], r['series'], r['detector'], r['seed'], r.get('train'), r.get('sec')) for r in R['runs'] if 'train' in r]
print(R['config'])
