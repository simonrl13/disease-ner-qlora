"""Pre-run checks (pre-registration §2 and Amendment 1). Writes docs/prerun_checks.json.

Stops with an error, before anything else, if a PMID appears in more than one split.
Run: uv run python -m scripts.prerun_checks
"""

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from ner.data import bio_spans, gold_strings, pmids, read_conll
from ner.score import counts, map_strings

RAW = Path("data/raw")
SPLITS = {"train": "train", "dev": "devel", "test": "test"}
ORIG = {"train": "train/NCBItrainset_corpus.txt", "dev": "devel/NCBIdevelopset_corpus.txt",
        "test": "test/NCBItestset_corpus.txt"}
EXPECTED = {"train": 5433, "dev": 924, "test": 941}


def main() -> None:
    out: dict = {}

    # 1. PMID leakage: checked first, stops everything if it fails.
    ids = {s: pmids(RAW / "original-data" / f) for s, f in ORIG.items()}
    out["pmids_per_split"] = {s: len(v) for s, v in ids.items()}
    overlap = {f"{a}&{b}": sorted(ids[a] & ids[b]) for a, b in combinations(ids, 2)}
    out["pmid_overlap"] = overlap
    if any(overlap.values()):
        print(json.dumps(out, indent=2))
        sys.exit("STOP: a PMID appears in more than one split. Owner decision needed.")

    # 2. Split sizes.
    data = {s: read_conll(RAW / "conll" / f"{f}.tsv") for s, f in SPLITS.items()}
    out["sentences"] = {s: len(v) for s, v in data.items()}
    if out["sentences"] != EXPECTED:
        sys.exit(f"STOP: sentence counts {out['sentences']} != published {EXPECTED}")

    # 3. Mentions, empty sentences, oracle ceiling (§3) per split.
    out["mentions"], out["no_mention_sentences"], out["oracle"] = {}, {}, {}
    for s, sents in data.items():
        tp = fp = fn = affected = 0
        n_ment = n_empty = 0
        for toks, tags in sents:
            gold = bio_spans(tags)
            n_ment += len(gold)
            n_empty += not gold
            spans, ph = map_strings(toks, gold_strings(toks, gold))
            t, f, n = counts(gold, spans, ph)
            tp, fp, fn = tp + t, fp + f, fn + n
            affected += bool(f or n)
        out["mentions"][s], out["no_mention_sentences"][s] = n_ment, n_empty
        out["oracle"][s] = {"tp": tp, "fp": fp, "fn": fn, "sentences_affected": affected,
                            "max_f1": round(2 * tp / (2 * tp + fp + fn), 4)}

    # 4. Identical sentences across splits (same token sequence), and within-split duplicates.
    seqs = {s: Counter(tuple(t) for t, _ in v) for s, v in data.items()}
    out["identical_across_splits"] = {}
    for a, b in combinations(seqs, 2):
        common = sorted(set(seqs[a]) & set(seqs[b]))
        out["identical_across_splits"][f"{a}&{b}"] = {
            "distinct": len(common),
            "occurrences": {a: sum(seqs[a][c] for c in common), b: sum(seqs[b][c] for c in common)},
            "texts": [" ".join(c) for c in common],
        }
    out["duplicates_within_split"] = {s: sum(n - 1 for n in c.values() if n > 1) for s, c in seqs.items()}

    # 5. Amendment 1: test sentences that also appear in train.
    in_train = sum(1 for t, _ in data["test"] if tuple(t) in seqs["train"])
    out["amendment1"] = {"test_sentences_in_train": in_train, "test_subset_size": len(data["test"]) - in_train,
                         "check_runs": in_train > 0}

    Path("docs/prerun_checks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
