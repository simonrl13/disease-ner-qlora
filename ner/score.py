"""Output parsing, string -> span mapping and entity-level counts, per pre-registration §3 and §7."""

import json
import re

from ner.data import Span

_FENCE = re.compile(r"\A```[A-Za-z]*[ \t]*\n(.*)\n```\Z", re.S)


def parse_output(raw: str, hit_limit: bool) -> list[str] | None:
    """The predicted strings, or None for an invalid output (§3). No retry, no repair."""
    if hit_limit:
        return None
    s = raw.strip()
    m = _FENCE.match(s)
    if m:  # at most one enclosing fence
        s = m.group(1)
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(out, list) or not all(isinstance(x, str) for x in out):
        return None
    return out


def map_strings(tokens: list[str], strings: list[str]) -> tuple[list[Span], int]:
    """Map predicted strings to spans. Returns (spans, phantoms).

    Whole tokens only, case-sensitive. Each distinct string maps to every non-overlapping occurrence, left to
    right; longer strings (more tokens) are placed first and later strings can't reuse claimed tokens.
    A string with no remaining occurrence, or an empty one, is a phantom (one false positive).
    """
    distinct = list(dict.fromkeys(strings))
    distinct.sort(key=lambda s: -len(s.split()))  # stable: ties keep list order
    used = [False] * len(tokens)
    spans, phantoms = [], 0
    for s in distinct:
        q = s.split()
        found = False
        i = 0
        while q and i + len(q) <= len(tokens):
            if tokens[i : i + len(q)] == q and not any(used[i : i + len(q)]):
                spans.append((i, i + len(q)))
                for j in range(i, i + len(q)):
                    used[j] = True
                found = True
                i += len(q)
            else:
                i += 1
        if not found:
            phantoms += 1
    return sorted(spans), phantoms


def counts(gold: list[Span], pred: list[Span], phantoms: int = 0) -> tuple[int, int, int]:
    """(TP, FP, FN) for one sentence, exact span match."""
    g, p = set(gold), set(pred)
    return len(g & p), len(p - g) + phantoms, len(g - p)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
