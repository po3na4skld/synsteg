"""Build the evaluation corpus for the AI-generated-text-detector experiment.

Arms (all derived offline from reports/gpt_omni_reports, no API calls):
  human      IMDB reviews, truncated to the cover length distribution
  cover      GPT-4o container as generated (no payload)
  stego_b{b} SynSteg re-encoding with the same message and the same synonym slots,
             restricted to the first 2**b synonyms of every slot (b = 1..5)
  stego_sel  detector-aware key selection: K random bit->synonym permutations per
             text, guide detector picks the least "AI-looking" one (b = 5)

Output: corpus.jsonl with one record per (uuid, arm).
"""
import json, glob, random, re, sys, pathlib, csv
from itertools import product

random.seed(0)
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from steganography import helper  # noqa: E402

OUT = pathlib.Path(__file__).parent / "corpus.jsonl"
K_PERM = 16  # permutations tried for stego_sel (encoded here, scored in select_key.py)


def encode_with_key(container: str, key: list[dict], bits: str) -> tuple[str, int]:
    """Same walk as steganography.core.encode_message, for a fixed key. Returns (stego, bits_used)."""
    out, idx = [], 0
    for token, slot in zip(container.split(), key):
        special, token = helper.check_endswith_special(token)
        if helper.is_token_replacable(token, slot) and idx < len(bits) and not helper.has_duplicates(slot):
            token, was_cap = helper.check_capitalization(token, slot)
            size = len(list(slot[token].keys())[0])
            part = bits[idx: idx + size]
            mapping = slot[token]
            if len(part) != size:
                mapping = {"".join(i): t for i, t in zip(product("01", repeat=len(part)), list(mapping.values())[: 2 ** len(part)])}
            new = mapping[part]
            if was_cap:
                new = new.capitalize()
            out.append(new + special)
            idx += size
        else:
            out.append(token + special)
    return " ".join(out), idx


def restrict_key(key: list[dict], b: int, permute: bool = False) -> list[dict]:
    """Keep the first 2**b synonyms of every 32-way slot; optionally shuffle bit assignment."""
    new = []
    for slot in key:
        base, mapping = next(iter(slot.items()))
        syns = list(mapping.values())
        if len(set(syns)) < 2 ** b or len(syns) < 2 ** b:  # untouched slot ({"0": w, "1": w}) or too few
            new.append({base: {"0": base, "1": base}})
            continue
        syns = syns[: 2 ** b]
        if permute:
            random.shuffle(syns)
        new.append({base: {"".join(i): s for i, s in zip(product("01", repeat=b), syns)}})
    return new


def main():
    reports = []
    for f in sorted(glob.glob(str(ROOT / "reports/gpt_omni_reports/*.json"))):
        d = json.load(open(f))
        if d["message"].startswith(d["decoded_message"]):
            reports.append(d)
    print("reports", len(reports))

    # human arm: IMDB reviews truncated to a sampled cover length (words)
    cover_lengths = [len(d["container"].split()) for d in reports]
    with open(ROOT / "datasets/IMDB_sample.csv", newline="") as fh:
        reviews = [r["review"] for r in csv.DictReader(fh)]
    reviews = [re.sub(r"<br\s*/?>", " ", r) for r in reviews]
    reviews = [re.sub(r"\s+", " ", r).strip() for r in reviews]

    recs = []
    for d, rev in zip(reports, reviews):
        L = len(d["container"].split())
        words = rev.split()
        recs.append(dict(uuid=d["uuid"], arm="human", bits=0, text=" ".join(words[:L]), n_words=min(L, len(words))))
        recs.append(dict(uuid=d["uuid"], arm="cover", bits=0, text=d["container"], n_words=L))
        msg = d["message"]
        for b in range(1, 6):
            key = restrict_key(d["secret_key"], b)
            stego, used = encode_with_key(d["container"], key, msg)
            recs.append(dict(uuid=d["uuid"], arm=f"stego_b{b}", bits=used, text=stego, n_words=L))
        # sanity: b=5 with the original key reproduces the stored stego text
        orig, _ = encode_with_key(d["container"], d["secret_key"], msg)
        assert orig == d["encoded_message"], d["uuid"]
        # candidate permutations for detector-aware selection
        for k in range(K_PERM):
            key = restrict_key(d["secret_key"], 5, permute=True)
            stego, used = encode_with_key(d["container"], key, msg)
            recs.append(dict(uuid=d["uuid"], arm="cand_b5", cand=k, bits=used, text=stego, n_words=L))

    with open(OUT, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r["arm"] for r in recs))
    print("mean bits per arm", {a: round(sum(r["bits"] for r in recs if r["arm"] == a) / max(1, sum(1 for r in recs if r["arm"] == a)), 1) for a in sorted(set(r["arm"] for r in recs))})


if __name__ == "__main__":
    main()
