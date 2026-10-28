"""Markdown tables from results/luna_rows.json and results/luna_paired.json (no ML)."""
import json
import pathlib
HERE = pathlib.Path(__file__).resolve().parent

rows = {r["attack"]: r for r in json.load(open(HERE / "results/luna_rows.json"))}
pairs = json.load(open(HERE / "results/luna_paired.json"))
base_ppl = rows["no attack"]["ppl"]


def line(name, label=None):
    r = rows[name]
    dec = "" if r.get("decoded") is None or name.startswith("control") else f"{r['decoded']}/{r['n_decoded']}"
    return (f"| {label or name} | {r['auc']:.3f} | {r['tpr1']:.3f} | {r['mean_g']:.4f} | {100 * r['residual']:.1f} % | "
            f"{100 * (1 - r['residual']):.1f} % | {r['share']:.3f} | {r['bits']:.0f} | {r['ppl']:.1f} ({r['ppl'] / base_ppl:.1f}×) | "
            f"{r.get('ppl_ratio_median', 1.0):.2f}× | {dec} |")


HEAD = ("| Attack | AUC | TPR @ 1 % FPR | Mean g | Residual | Removed | Replaced share | Bits/text | PPL (ratio of means) | "
        "Median per-text PPL ratio | Exact decode (both classes) |\n|" + "---|" * 11)
order = ["no attack"]
for b in range(1, 6):
    order += [f"synsteg b={b}", f"control for synsteg b={b}", f"pplopt b={b}"]
main = [HEAD] + [line(n) for n in order]
low = [HEAD] + [line("no attack")]
for v in ("window", "window+opt"):
    for b in (1, 2):
        for d in (0.0, 0.25, 0.5, 1.0, 2.0):
            n = f"{v} b={b} d={d}"
            low += [line(n), line(f"control for {n}", "  control, same count")]
ph = "| Variant vs its control | n | Mean diff g | 95 % CI | Significant | Variant lower g, texts | Share of signal |\n|" + "---|" * 7
pr = [ph] + [f"| {k} | {p['n']} | {p['diff']:+.5f} | [{p['lo']:+.5f}; {p['hi']:+.5f}] | {'yes' if p['significant'] else 'no'} | "
             f"{p['a_better']} of {p['n']} | {100 * p['share']:+.1f} % |" for k, p in pairs.items()]
print("\n".join(main), "\n\n", "\n".join(low), "\n\n", "\n".join(pr))
