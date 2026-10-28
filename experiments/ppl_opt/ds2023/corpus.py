"""Sentence pairs of the DeepStego corpus exactly as artifacts/synsteg_{clean,stego}.txt were built: reports/json_reports in
os.listdir order, each text split on '.', lines stripped, empty lines dropped (byte-identical to the artifacts for the base arm)."""
import json, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPORTS, OPT = ROOT / "reports/json_reports", ROOT / "experiments/ppl_opt/series/ds2023_opt"


def pairs(arm):
    out, mismatch = [], 0
    for f in (f for f in os.listdir(REPORTS) if f.endswith(".json")):
        d = json.load(open(REPORTS / f))
        s = d["encoded_message"] if arm == "base" else json.load(open(OPT / f))["encoded_message"]
        c, s = d["container"].split("."), s.split(".")
        mismatch += len(c) != len(s)
        out += [(a.strip(), b.strip()) for a, b in zip(c, s) if a.strip() and b.strip()]
    return out, mismatch


if __name__ == "__main__":
    cl = [l.strip() for l in open(ROOT / "artifacts/synsteg_clean.txt")]; sl = [l.strip() for l in open(ROOT / "artifacts/synsteg_stego.txt")]
    art = [(a, b) for a, b in zip(cl, sl) if a and b]
    for arm in ("base", "opt"):
        p, mm = pairs(arm)
        print(arm, len(p), "texts with '.'-count mismatch", mm, "identical pairs", sum(a == b for a, b in p),
              "equal to artifacts" if p == art else f"clean lines equal to artifacts: {[a for a, _ in p] == [a for a, _ in art]}")
