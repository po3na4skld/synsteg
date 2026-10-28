"""Deep-learning steganalysis of two SynSteg series (cover=0 vs stego=1).

Detectors re-implemented from the paper descriptions:
  TS-CNN        Yang Z. et al. 2019  - multi-size 1-D convs (h=1,2,3) over word embeddings + average pool
  TS-RNN        Yang Z. et al. 2019  - TS-BiRNN: 2-layer BiLSTM, last fwd/bwd states -> fusion -> FC
  Bi-LSTM-Dense Yang H. et al. 2020 (IH&MMSec) - densely connected Bi-LSTM, max-pooled pyramid of all layer outputs
  BERT-LSTM-Att Zou J. et al. 2020   - frozen bert-base-uncased -> Bi-LSTM -> additive attention -> FC
Split by pair (cover and its stego together), 80/20 stratified by L, seeds 0/1/2.
Usage: python steganalysis.py   -> ../../reports/steganalysis_missing.json
"""
import glob, json, re, time, collections, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

ROOT = os.path.join(os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg')), 'reports')
OUT = f'{ROOT}/steganalysis_missing.json'
MAXLEN = 256            # word tokens (word models) / WordPiece tokens (BERT), head-truncated, right-padded
SEEDS = (0, 1, 2)
EPOCHS, PATIENCE, BS, LR, SGD_LR, VAL_FRAC = 60, 8, 32, 1e-3, 0.05, 0.15
SERIES = {'gpt6_luna_b5': 'gpt6_luna_missing', 'qwen3_30b_b3': 'qwen_missing_b3'}


def load(folder):
    pairs = []
    for f in sorted(glob.glob(f'{ROOT}/{folder}/*.json')):
        r = json.load(open(f))
        pairs.append((r['container'], r['encoded_message'], len(r['message'])))
    return pairs


tok = lambda s: re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[^\sa-z0-9]", s.lower())


def flatten(pairs, idx):
    X, y, L = [], [], []
    for i in idx:
        c, s, l = pairs[i]
        X += [c, s]; y += [0, 1]; L += [l, l]
    return X, np.array(y), np.array(L)


# ---------------- models ----------------
def run_lstm(lstm, x, lens):
    out, (h, _) = lstm(pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False))
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
    return out, h


def masked_max(h, mask):
    return h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values


class TSCNN(nn.Module):  # paper: random-init emb 400, kernel heights 1/2/3 x 128, average pooling, FC+softmax, SGD m=0.9
    def __init__(s, V, E=400, K=(1, 2, 3), C=128):
        super().__init__(); s.emb = nn.Embedding(V, E, padding_idx=0); s.K = K
        s.convs = nn.ModuleList(nn.Conv1d(E, C, k) for k in K); s.drop = nn.Dropout(0.5); s.fc = nn.Linear(C * len(K), 2)
    def forward(s, x, lens):
        e = s.drop(s.emb(x)).transpose(1, 2); feats = []
        for k, c in zip(s.K, s.convs):  # mean over the valid (non-pad) windows only
            h = F.relu(c(e)); n = (lens - k + 1).clamp(min=1).to(h.device)
            m = torch.arange(h.size(-1), device=h.device)[None] < n[:, None]
            feats.append((h * m[:, None]).sum(-1) / n[:, None])
        return s.fc(s.drop(torch.cat(feats, 1)))


class TSRNN(nn.Module):  # paper (TS-BiRNN variant): emb 256, 2-layer BiLSTM 100 units, last fwd+bwd outputs
    def __init__(s, V, E=256, H=100):     # concatenated -> fusion layer (tanh) -> softmax, dropout 0.5, Adam 1e-3
        super().__init__(); s.emb = nn.Embedding(V, E, padding_idx=0)
        s.rnn = nn.LSTM(E, H, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        s.drop = nn.Dropout(0.5); s.fuse = nn.Linear(2 * H, H); s.fc = nn.Linear(H, 2)
    def forward(s, x, lens):
        _, h = run_lstm(s.rnn, s.drop(s.emb(x)), lens)
        return s.fc(s.drop(torch.tanh(s.fuse(torch.cat([h[-2], h[-1]], 1)))))


class BiLSTMDense(nn.Module):  # emb 300, 3 densely connected BiLSTM layers (input = emb ++ all previous outputs),
    def __init__(s, V, E=300, H=100, N=3):  # max-pool every level (emb + each layer) -> concat pyramid -> FC
        super().__init__(); s.emb = nn.Embedding(V, E, padding_idx=0)
        s.layers = nn.ModuleList(nn.LSTM(E + i * 2 * H, H, bidirectional=True, batch_first=True) for i in range(N))
        s.drop = nn.Dropout(0.5); s.fc = nn.Linear(E + N * 2 * H, 2)
    def forward(s, x, lens):
        mask = x != 0; feats = [s.drop(s.emb(x))]
        for l in s.layers:
            feats.append(run_lstm(l, torch.cat(feats, -1), lens)[0])
        return s.fc(s.drop(torch.cat([masked_max(f, mask) for f in feats], 1)))


class BERTLSTMAtt(nn.Module):  # input: frozen BERT last-layer states (768), BiLSTM 128/dir, additive attention
    def __init__(s, D=768, H=128):
        super().__init__(); s.rnn = nn.LSTM(D, H, bidirectional=True, batch_first=True)
        s.att = nn.Sequential(nn.Linear(2 * H, 2 * H), nn.Tanh(), nn.Linear(2 * H, 1, bias=False))
        s.drop = nn.Dropout(0.5); s.fc = nn.Linear(2 * H, 2)
    def forward(s, x, lens):
        h, _ = run_lstm(s.rnn, s.drop(x), lens)
        mask = torch.arange(x.size(1), device=x.device)[None] < lens[:, None].to(x.device)
        a = s.att(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(1)
        return s.fc(s.drop((a.unsqueeze(-1) * h).sum(1)))


# ---------------- features ----------------
class WordFeats:
    def __init__(s, train_texts):
        c = collections.Counter(t for x in train_texts for t in tok(x))
        s.vocab = {w: i + 2 for i, w in enumerate(sorted(c))}  # 0 pad, 1 unk
    def __call__(s, texts):
        ids = [[s.vocab.get(t, 1) for t in tok(x)][:MAXLEN] for x in texts]
        X = torch.zeros(len(ids), MAXLEN, dtype=torch.long)
        for i, r in enumerate(ids): X[i, :len(r)] = torch.tensor(r)
        return X, torch.tensor([len(r) for r in ids])


BERT_CACHE = {}
def bert_embed(texts):  # frozen bert-base-uncased, last hidden layer, cached per text
    from transformers import AutoTokenizer, AutoModel
    if 'm' not in BERT_CACHE:
        BERT_CACHE['t'] = AutoTokenizer.from_pretrained('bert-base-uncased')
        BERT_CACHE['m'] = AutoModel.from_pretrained('bert-base-uncased').eval().to(DEV_BERT)
    todo = [t for t in dict.fromkeys(texts) if t not in BERT_CACHE]
    with torch.no_grad():
        for i in range(0, len(todo), 16):
            b = todo[i:i + 16]
            enc = BERT_CACHE['t'](b, truncation=True, max_length=MAXLEN, padding='max_length', return_tensors='pt').to(DEV_BERT)
            hs = BERT_CACHE['m'](**enc).last_hidden_state.half().cpu(); n = enc['attention_mask'].sum(1).cpu()
            for t, h, l in zip(b, hs, n): BERT_CACHE[t] = (h, int(l))
    return torch.stack([BERT_CACHE[t][0] for t in texts]).float(), torch.tensor([BERT_CACHE[t][1] for t in texts])


# ---------------- training ----------------
DEV = torch.device('cpu')           # small recurrent models: CPU was more reliable than MPS for packed LSTMs
DEV_BERT = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
DETECTORS = {'TS-CNN': TSCNN, 'TS-RNN': TSRNN, 'Bi-LSTM-Dense': BiLSTMDense, 'BERT-LSTM-Att': BERTLSTMAtt}


def predict(m, X, lens):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 64): out.append(m(X[i:i + 64].to(DEV), lens[i:i + 64]).softmax(-1)[:, 1].cpu())
    return torch.cat(out).numpy()


def fit(name, tr_texts, ytr, va_texts, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if name == 'BERT-LSTM-Att':
        featurize, m = bert_embed, BERTLSTMAtt()
    else:
        featurize = WordFeats(tr_texts); m = DETECTORS[name](len(featurize.vocab) + 2)
    m.to(DEV); Xtr, ltr = featurize(tr_texts); Xva, lva = featurize(va_texts)
    opt = (torch.optim.SGD(m.parameters(), lr=SGD_LR, momentum=0.9) if name == 'TS-CNN'
           else torch.optim.Adam(m.parameters(), lr=LR)); ytr_t = torch.tensor(ytr)
    best, best_state, bad, ep_done = (-1, 1e9), None, 0, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        m.train()
        for b in torch.randperm(len(Xtr), generator=g).split(BS):
            opt.zero_grad(); F.cross_entropy(m(Xtr[b].to(DEV), ltr[b]), ytr_t[b].to(DEV)).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        p = predict(m, Xva, lva)
        acc = ((p > .5) == yva).mean(); loss = F.binary_cross_entropy(torch.tensor(p).clamp(1e-6, 1 - 1e-6), torch.tensor(yva, dtype=torch.float)).item()
        ep_done = ep + 1
        if (acc, -loss) > (best[0], -best[1]):
            best, bad = (acc, loss), 0; best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    m.load_state_dict(best_state)
    return m, featurize, {'epochs_run': ep_done, 'val_acc': float(best[0]), 'params': sum(p.numel() for p in m.parameters())}


def metrics(y, pred):
    return {'acc': accuracy_score(y, pred), 'prec': precision_score(y, pred, zero_division=0),
            'rec': recall_score(y, pred, zero_division=0), 'f1': f1_score(y, pred, zero_division=0),
            'n': int(len(y)), 'frac_pred_stego': float(pred.mean()), 'degenerate': bool(len(set(pred)) == 1)}


def split_pairs(pairs, seed):
    idx = np.arange(len(pairs)); L = [p[2] for p in pairs]
    tr, te = train_test_split(idx, test_size=0.2, stratify=L, random_state=seed)
    tr, va = train_test_split(tr, test_size=VAL_FRAC, stratify=[L[i] for i in tr], random_state=seed)
    return tr, va, te


def length_baseline(pairs, tr, te):  # sanity check: logistic regression on word count alone
    Xtr, ytr, _ = flatten(pairs, tr); Xte, yte, _ = flatten(pairs, te)
    f = lambda X: np.log([[len(x.split())] for x in X])
    return metrics(yte, LogisticRegression().fit(f(Xtr), ytr).predict(f(Xte)))


def evaluate(name, m, featurize, X, y, L, per_L):
    pred = (predict(m, *featurize(X)) > .5).astype(int)
    r = metrics(y, pred)
    if per_L: r['per_L'] = {int(l): metrics(y[L == l], pred[L == l]) for l in sorted(set(L))}
    return r


def main():
    data = {k: load(v) for k, v in SERIES.items()}
    res = {'config': {'maxlen': MAXLEN, 'seeds': SEEDS, 'epochs_max': EPOCHS, 'patience': PATIENCE, 'batch': BS, 'lr_adam': LR, 'lr_sgd_tscnn': SGD_LR,
                      'val_frac_of_train_pairs': VAL_FRAC, 'n_pairs': {k: len(v) for k, v in data.items()}},
           'runs': []}
    t0 = time.time()
    for series, pairs in data.items():
        for seed in SEEDS:
            tr, va, te = split_pairs(pairs, seed)
            res['runs'].append({'setting': 'in-series', 'series': series, 'detector': 'length-baseline', 'seed': seed,
                                'test': length_baseline(pairs, tr, te)})
            Xtr, ytr, _ = flatten(pairs, tr); Xva, yva, _ = flatten(pairs, va); Xte, yte, Lte = flatten(pairs, te)
            for name in DETECTORS:
                t = time.time(); m, fz, info = fit(name, Xtr, ytr, Xva, yva, seed)
                r = evaluate(name, m, fz, Xte, yte, Lte, per_L=True)
                res['runs'].append({'setting': 'in-series', 'series': series, 'detector': name, 'seed': seed, 'train': info,
                                    'test': r, 'sec': round(time.time() - t, 1)})
                print(f'{series} s{seed} {name:14s} acc={r["acc"]:.3f} f1={r["f1"]:.3f} stego%={r["frac_pred_stego"]:.2f} ep={info["epochs_run"]} {time.time()-t:.0f}s', flush=True)
    # pooled / cross-generator: train on all gpt-6-luna pairs (val slice held out), test on all Qwen pairs
    g, q = data['gpt6_luna_b5'], data['qwen3_30b_b3']
    Xq, yq, Lq = flatten(q, range(len(q)))
    for seed in SEEDS:
        idx = np.arange(len(g))
        tr, va = train_test_split(idx, test_size=VAL_FRAC, stratify=[p[2] for p in g], random_state=seed)
        Xtr, ytr, _ = flatten(g, tr); Xva, yva, _ = flatten(g, va)
        res['runs'].append({'setting': 'cross', 'series': 'gpt6->qwen', 'detector': 'length-baseline', 'seed': seed,
                            'test': length_baseline(g + q, tr, range(len(g), len(g) + len(q)))})
        for name in DETECTORS:
            t = time.time(); m, fz, info = fit(name, Xtr, ytr, Xva, yva, seed)
            r = evaluate(name, m, fz, Xq, yq, Lq, per_L=True)
            res['runs'].append({'setting': 'cross', 'series': 'gpt6->qwen', 'detector': name, 'seed': seed, 'train': info,
                                'test': r, 'sec': round(time.time() - t, 1)})
            print(f'cross s{seed} {name:14s} acc={r["acc"]:.3f} f1={r["f1"]:.3f} stego%={r["frac_pred_stego"]:.2f} ep={info["epochs_run"]}', flush=True)
    res['config']['runtime_sec'] = round(time.time() - t0)
    res['config']['torch'] = torch.__version__; res['config']['devices'] = {'models': str(DEV), 'bert': str(DEV_BERT)}
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    print('wrote', OUT, res['config']['runtime_sec'], 's')


if __name__ == '__main__':
    main()
