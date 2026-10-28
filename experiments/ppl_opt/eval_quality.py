"""Recovery, BPW, replaced-word share, R_PPL (GPT-2 = selector, Qwen2.5-1.5B = held-out judge), Flesch, optimisation time.
Arms: base = published keys (reports/gpt_omni_reports, stored stegotexts), opt_k{K} = series/opt_k{K} from optimize.py.
Usage: <detector venv python> eval_quality.py -> quality.json + markdown tables on stdout
"""
import glob, json, os, pathlib, sys, gc
import numpy as np, torch, textstat
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "model_series"))
from metrics import ppl, edit_rate, DEV  # noqa: E402

ARMS = ["base"] + [p.name for p in sorted((HERE / "series").glob("opt_k*"), key=lambda p: -int(p.name[5:]))]


def load():
    base = {d["uuid"]: d for d in (json.load(open(f)) for f in glob.glob(str(HERE.parents[1] / "reports/gpt_omni_reports/*.json")))}
    arms = {a: {d["uuid"]: d for d in (json.load(open(f)) for f in glob.glob(str(HERE / "series" / a / "*.json")))} for a in ARMS[1:]}
    uu = sorted(set(base).intersection(*arms.values()))
    return uu, base, arms


def main():
    uu, base, arms = load()
    opt = arms[ARMS[1]]
    rows = {u: dict(L=len(base[u]["message"]), container=base[u]["container"], used=opt[u]["base_used_bits"], words=len(base[u]["secret_key"]),
                    t_key=base[u]["encoding_time_report"].get("secret_key_generation")) for u in uu}
    for u in uu:
        rows[u]["base_rec"] = base[u]["decoded_message"].startswith(base[u]["message"][:rows[u]["used"]])
        rows[u]["base_exact"] = base[u]["decoded_message"] == base[u]["message"]
    texts = {"container": [base[u]["container"] for u in uu], "base": [base[u]["encoded_message"] for u in uu]}
    for a in ARMS[1:]:
        texts[a] = [arms[a][u]["encoded_message"] for u in uu]
        for u in uu:
            d = arms[a][u]
            rows[u][f"{a}_rec"] = d["recovered"] and d["reencode_ok"] and d["used_bits"] == d["base_used_bits"]
            rows[u][f"{a}_exact"] = d["decoded_message"] == d["message"]
            rows[u].update({f"{a}_time": d["opt_time"], f"{a}_retries": d["retries"], f"{a}_kept": d["kept_original"] / max(d["positions"], 1)})
    for k in ARMS:
        for u, t in zip(uu, texts[k]):
            rows[u][f"{k}_flesch"] = textstat.flesch_reading_ease(t)
            if k != "container":
                rows[u][f"{k}_edit"] = edit_rate(base[u]["container"], t)
    for u, t in zip(uu, texts["container"]):
        rows[u]["container_flesch"] = textstat.flesch_reading_ease(t)
    for judge, name, bs in (("gpt2", "openai-community/gpt2", 16), ("qwen", "Qwen/Qwen2.5-1.5B", 4)):
        tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "right"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(DEV).eval()
        for k, ts in texts.items():
            for u, p in zip(uu, ppl(ts, tok, m, bs=bs)):
                rows[u][f"{k}_ppl_{judge}"] = p
        del m; gc.collect(); torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        print("judge done", judge, flush=True)
    json.dump(dict(uuids=uu, rows=rows), open(HERE / "quality.json", "w"))
    report(rows)


def report(rows):
    R = list(rows.values())
    groups = [("all", R)] + [(str(L), [r for r in R if r["L"] == L]) for L in (128, 256, 512)]
    f = lambda x: f"{np.median(x):.3f} / {np.mean(x):.3f}"
    print("\n| L | n | arm | recovered (embedded prefix) | exact full message | BPW | edit rate | kept original at active pos. | R_PPL GPT-2 med/mean | R_PPL Qwen med/mean | dFlesch med/mean | opt time s med/mean | retries |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g, rs in groups:
        bpw = np.mean([r["used"] / r["words"] for r in rs])
        for a in ARMS:
            ratio = lambda j: [r[f"{a}_ppl_{j}"] / r[f"container_ppl_{j}"] for r in rs]
            dfl = [r[f"{a}_flesch"] - r["container_flesch"] for r in rs]
            opt = a != "base"
            print(f"| {g} | {len(rs)} | {a} | {sum(r[f'{a}_rec'] for r in rs)}/{len(rs)} | {sum(r[f'{a}_exact'] for r in rs)}/{len(rs)} | {bpw:.3f} | "
                  f"{np.mean([r[f'{a}_edit'] for r in rs]):.3f} | {np.mean([r[f'{a}_kept'] for r in rs]) if opt else float('nan'):.3f} | {f(ratio('gpt2'))} | {f(ratio('qwen'))} | "
                  f"{f(dfl)} | {f([r[f'{a}_time'] for r in rs]) if opt else '-'} | {sum(r[f'{a}_retries'] > 0 for r in rs) if opt else '-'} |")
    print("\nabsolute PPL (median): ", {g: {f"{k}_{j}": round(float(np.median([r[f"{k}_ppl_{j}"] for r in rs])), 1) for k in ["container"] + ARMS for j in ("gpt2", "qwen")} for g, rs in groups[:1]})
    from scipy.stats import wilcoxon
    for a in ARMS[1:]:
        for j in ("gpt2", "qwen"):
            x = [r[f"{a}_ppl_{j}"] - r[f"base_ppl_{j}"] for r in R]
            print(f"Wilcoxon {a} vs base PPL {j}: p = {wilcoxon(x).pvalue:.2e}; opt lower in {sum(v < 0 for v in x)}/{len(x)}")
    print("key generation time (API, published) s med/mean:", f([r["t_key"] for r in R if r["t_key"]]))


if __name__ == "__main__":
    main() if "--report" not in sys.argv else report(json.load(open(HERE / "quality.json"))["rows"])
