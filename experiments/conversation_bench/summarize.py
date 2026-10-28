"""Table 8 values per configuration -> results/summary.json (also prints a table).
Perplexity: GPT-2 124M, function ppl() imported from experiments/model_series/metrics.py (same as Tables 2-4).

Usage: $PY experiments/conversation_bench/summarize.py
"""
import json, os, sys, collections
import numpy as np

ROOT = os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg'))
sys.path.insert(0, f'{ROOT}/experiments/model_series')
sys.path.insert(0, f'{ROOT}/experiments/conversation_bench')
from covers import RES, load_stego  # noqa: E402

CFGS = {'default': 'default arithmetic top-n 8', 'u32': 'uniform top-n 32', 'u1024': 'uniform top-n 1024'}
rows = lambda p: [json.loads(l) for l in open(p)] if os.path.exists(p) else []
mean = lambda v: float(np.mean(v)) if len(v) else None


def capacity(cfg):
    by_L = collections.defaultdict(list)
    for r in rows(f'{RES}/capacity_{cfg}.jsonl'):
        by_L[r['L']].append(r)
    cells = {}
    for L, rs in sorted(by_L.items()):
        ok = [r for r in rs if not r['encode_error']]
        cells[L] = {'runs': len(rs), 'encode_errors': len(rs) - len(ok),
                    'encode_error_msgs': sorted({r['encode_error'] for r in rs if r['encode_error']}),
                    'decode_errors': sum(bool(r.get('decode_error')) for r in ok),
                    'decode_error_msgs': sorted({r['decode_error'] for r in ok if r.get('decode_error')}),
                    'exact': sum(bool(r.get('exact')) for r in rs),
                    **{k: mean([r[k] for r in ok]) for k in ('bpw', 'words', 'gpt2_tokens', 'data_tokens', 'finish_tokens',
                                                              'encode_s', 'decode_s', 'mean_logit_regret')}}
    return cells


def chain(cfg):
    rs = rows(f'{RES}/chain_{cfg}.jsonl')
    return {'sends': len(rs), 'completed': sum(r['completed'] for r in rs), 'decoded': sum(bool(r.get('decoded')) for r in rs),
            'failures': [{'send': r['send'], 'stage': 'send' if not r['completed'] else 'receive',
                          'error': r['send_error'] or r.get('receive_error')} for r in rs if not r.get('decoded')],
            'hidden_bytes': [r['hidden_bytes'] for r in rs], 'encode_s': [round(r['encode_s'], 1) for r in rs]}


def steganalysis(cfg):
    p = f'{RES}/steganalysis_{cfg}.json'
    if not os.path.exists(p):
        return None
    acc = collections.defaultdict(list)
    for r in json.load(open(p))['runs']:
        acc[r['detector']].append(r['test']['acc'])
    det = {d: {'mean': mean(a), 'sd': float(np.std(a, ddof=1)), 'seeds': a} for d, a in acc.items()}
    means = [v['mean'] for v in det.values()]
    return {'detectors': det, 'range': [min(means), max(means)] if means else None}


def perplexity(cfgs):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import metrics  # experiments/model_series/metrics.py
    tok = AutoTokenizer.from_pretrained('openai-community/gpt2'); tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained('openai-community/gpt2').to(metrics.DEV).eval()
    out = {}
    for cfg in cfgs:
        cp = f'{RES}/covers_{cfg}.jsonl'
        if not os.path.exists(cp):
            continue
        stego = [r['stegotext'] for r in load_stego(cfg)[0]]
        covers = [json.loads(l)['text'] for l in open(cp)]
        ps, pc = metrics.ppl(stego, tok, m), metrics.ppl(covers, tok, m)
        out[cfg] = {'median_ppl_stego': float(np.median(ps)), 'median_ppl_cover': float(np.median(pc)),
                    'ratio': float(np.median(ps) / np.median(pc)), 'n_stego': len(ps), 'n_cover': len(pc)}
    return out


def rng(cells, k):
    v = [c[k] for c in cells.values() if c[k] is not None]
    return [min(v), max(v)] if v else None


def main():
    ppl = perplexity(CFGS)
    summary = {}
    for cfg, label in CFGS.items():
        cells = capacity(cfg)
        c = os.path.exists(f'{RES}/corpus_{cfg}.jsonl') and load_stego(cfg)
        summary[cfg] = {'label': label, 'capacity': cells, 'bpw_range': rng(cells, 'bpw'), 'encode_s_range': rng(cells, 'encode_s'),
                        'decode_s_range': rng(cells, 'decode_s'), 'logit_regret_range': rng(cells, 'mean_logit_regret'),
                        'corpus': {'L': 128, 'used': len(c[0]), 'attempted': c[1], 'encoded': c[2]} if c else None,
                        'steganalysis': steganalysis(cfg), 'perplexity': ppl.get(cfg), 'full_protocol': chain(cfg)}
    json.dump(summary, open(f'{RES}/summary.json', 'w'), indent=1)
    f = lambda r, d=2: '-' if not r else f'{r[0]:.{d}f}-{r[1]:.{d}f}'
    print(f'{"config":28s} {"bits/word":11s} {"encode s":12s} {"regret":10s} {"steg acc":12s} {"ppl":7s} protocol')
    for cfg, s in summary.items():
        st, p, fp = s['steganalysis'], s['perplexity'], s['full_protocol']
        print(f'{s["label"]:28s} {f(s["bpw_range"]):11s} {f(s["encode_s_range"]):12s} {f(s["logit_regret_range"]):10s} '
              f'{f(st and st["range"], 3):12s} {(p and format(p["ratio"], ".2f") + "x") or "-":7s} '
              f'{fp["decoded"]}/{fp["sends"]} (completed {fp["completed"]})')


if __name__ == '__main__':
    main()
