"""Score every corpus record with every detector. Resumable: scores/<detector>.jsonl, one line per record id."""
import json, sys, pathlib, time
from detectors import DETECTORS

HERE = pathlib.Path(__file__).parent
CORPUS = HERE / "corpus.jsonl"
SCORES = HERE / "scores"
SCORES.mkdir(exist_ok=True)


def rid(r):
    return f'{r["uuid"]}|{r["arm"]}|{r.get("cand", -1)}'


def main(names):
    recs = [json.loads(l) for l in open(CORPUS)]
    for name in names:
        out = SCORES / f"{name}.jsonl"
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
        todo = [r for r in recs if rid(r) not in done]
        print(f"{name}: {len(todo)} to score", flush=True)
        if not todo:
            continue
        det = DETECTORS[name]()
        t0 = time.time()
        with open(out, "a") as fh:
            for i in range(0, len(todo), 64):
                chunk = todo[i:i + 64]
                sc = det.score([r["text"] for r in chunk])
                for r, s in zip(chunk, sc):
                    fh.write(json.dumps({"id": rid(r), "uuid": r["uuid"], "arm": r["arm"], "cand": r.get("cand", -1), "score": s}) + "\n")
                fh.flush()
                if (i // 64) % 10 == 0:
                    print(f"  {name} {i + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        del det
        import torch, gc; gc.collect(); torch.mps.empty_cache() if torch.backends.mps.is_available() else None


if __name__ == "__main__":
    main(sys.argv[1:] or list(DETECTORS))
