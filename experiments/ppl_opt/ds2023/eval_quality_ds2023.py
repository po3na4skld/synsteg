"""eval_quality.py for the DeepStego corpus: recovery (2023 decoder), BPW, edit rate, R_PPL (GPT-2 selector, Qwen2.5-1.5B
held-out judge), dFlesch, optimisation time. Arms: base = stored reports/json_reports stegotexts, opt = series/ds2023_opt.
Usage: <ai_detectors venv python> eval_quality_ds2023.py -> quality_ds2023.json + markdown on stdout"""
import copy, gc, json, os, pathlib, sys
import numpy as np, torch, textstat
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "model_series"))
import codec2023 as C  # noqa: E402
from corpus import REPORTS, OPT  # noqa: E402
from metrics import ppl, edit_rate, DEV  # noqa: E402

ARMS = ("base", "opt")


def main():
    fs = sorted(f for f in os.listdir(REPORTS) if f.endswith(".json"))
    rows, texts = {}, {"container": [], "base": [], "opt": []}
    for f in fs:
        b, o = json.load(open(REPORTS / f)), json.load(open(OPT / f))
        st, used = C.encode(b["container"], copy.deepcopy(b["secret_key"]), b["message"])
        r = dict(L=len(b["message"]), used=used, words=len(b["secret_key"]), base_reenc=st == b["encoded_message"],
                 t_key=b["encoding_time_report"].get("secret_key_generation"), opt_time=o["opt_time"], retries=o["retries"],
                 kept=o["kept_original"] / max(o["positions"], 1), opt_reenc=o["reencode_ok"], opt_used=o["used_bits"])
        for a, s, k in (("base", b["encoded_message"], b["secret_key"]), ("opt", o["encoded_message"], o["secret_key"])):
            dec = C.decode(s, k)
            r[f"{a}_rec_exact"] = dec == b["message"][:used]       # exactly the embedded bits
            r[f"{a}_rec_prefix"] = dec.startswith(b["message"][:used])
            r[f"{a}_full"] = dec == b["message"]
            r[f"{a}_edit"] = edit_rate(b["container"], s)
            texts[a].append(s)
        texts["container"].append(b["container"])
        rows[f] = r
    for k, ts in texts.items():
        for f, t in zip(fs, ts):
            rows[f][f"{k}_flesch"] = textstat.flesch_reading_ease(t)
    for judge, name, bs in (("gpt2", "openai-community/gpt2", 16), ("qwen", "Qwen/Qwen2.5-1.5B", 4)):
        tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "right"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(DEV).eval()
        with torch.no_grad():
            for k, ts in texts.items():
                for f, p in zip(fs, ppl(ts, tok, m, bs=bs)):
                    rows[f][f"{k}_ppl_{judge}"] = p
        del m; gc.collect(); torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        print("judge done", judge, flush=True)
    json.dump(rows, open(HERE / "quality_ds2023.json", "w"))
    report(rows)


def report(rows):
    R = list(rows.values()); f = lambda x: f"{np.median(x):.2f} / {np.mean(x):.2f}"
    print("base re-encoding identical to stored stego:", sum(r["base_reenc"] for r in R), "/", len(R),
          "| opt re-encode ok:", sum(r["opt_reenc"] for r in R), "| opt used == base used:", sum(r["opt_used"] == r["used"] for r in R))
    print("\n| L | n | arm | rec exact embedded bits | rec prefix | full message | BPW | edit rate | R_PPL GPT-2 med/mean | R_PPL Qwen med/mean | dFlesch med/mean |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for g, rs in [("all", R)] + [(str(L), [r for r in R if r["L"] == L]) for L in (64, 128, 256)]:
        for a in ARMS:
            ratio = lambda j: [r[f"{a}_ppl_{j}"] / r[f"container_ppl_{j}"] for r in rs]
            print(f"| {g} | {len(rs)} | {a} | {sum(r[f'{a}_rec_exact'] for r in rs)} | {sum(r[f'{a}_rec_prefix'] for r in rs)} | {sum(r[f'{a}_full'] for r in rs)} | "
                  f"{np.mean([r['used'] / r['words'] for r in rs]):.3f} | {np.mean([r[f'{a}_edit'] for r in rs]):.3f} | {f(ratio('gpt2'))} | {f(ratio('qwen'))} | "
                  f"{f([r[f'{a}_flesch'] - r['container_flesch'] for r in rs])} |")
    print("abs PPL median:", {f"{k}_{j}": round(float(np.median([r[f'{k}_ppl_{j}'] for r in R])), 1) for k in ("container", "base", "opt") for j in ("gpt2", "qwen")})
    from scipy.stats import wilcoxon
    for j in ("gpt2", "qwen"):
        x = [r[f"opt_ppl_{j}"] - r[f"base_ppl_{j}"] for r in R]
        print(f"Wilcoxon opt vs base PPL {j}: p = {wilcoxon(x).pvalue:.2e}; opt lower in {sum(v < 0 for v in x)}/{len(x)}")
    print("opt time s med/mean:", f([r["opt_time"] for r in R]), "per L:", {L: f([r["opt_time"] for r in R if r["L"] == L]) for L in (64, 128, 256)})
    print("retried runs:", sum(r["retries"] > 0 for r in R), "retries hist:", np.bincount([r["retries"] for r in R]).tolist(),
          "| kept original at active pos.:", round(float(np.mean([r["kept"] for r in R])), 3))
    print("key generation time (API, stored) s med/mean:", f([r["t_key"] for r in R if r["t_key"]]))


if __name__ == "__main__":
    main() if "--report" not in sys.argv else report(list(json.load(open(HERE / "quality_ds2023.json")).values()))
