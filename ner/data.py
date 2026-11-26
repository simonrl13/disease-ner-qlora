"""CoNLL (token<TAB>tag) reading and BIO span extraction, per pre-registration §2."""

import re
from pathlib import Path

Span = tuple[int, int]  # token indices, end exclusive


def read_conll(path: Path) -> list[tuple[list[str], list[str]]]:
    """Sentences as (tokens, tags). Blank lines separate sentences."""
    sents, toks, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if toks:
                sents.append((toks, tags))
                toks, tags = [], []
            continue
        tok, tag = line.split("\t")
        toks.append(tok)
        tags.append(tag)
    if toks:
        sents.append((toks, tags))
    return sents


def bio_spans(tags: list[str]) -> list[Span]:
    """Contiguous B-/I- runs. A stray I- (after O or at the start) starts a new span."""
    spans, start = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-") or (t.startswith("I-") and start is None):
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == "O":
            if start is not None:
                spans.append((start, i))
            start = None
        elif not t.startswith("I-"):
            raise ValueError(f"unknown tag {t!r}")
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def gold_strings(tokens: list[str], spans: list[Span]) -> list[str]:
    """Training target / oracle answer: distinct gold strings in order of first occurrence (§3)."""
    return list(dict.fromkeys(" ".join(tokens[a:b]) for a, b in spans))


def pmids(path: Path) -> set[str]:
    """PMIDs in an original NCBI corpus file (title lines `PMID|t|...`)."""
    return set(re.findall(r"^(\d+)\|t\|", path.read_text(encoding="utf-8"), flags=re.M))
