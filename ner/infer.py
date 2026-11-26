"""Base-model and QLoRA inference, per pre-registration §4. Greedy, one sentence at a time.

  uv run python -m ner.infer shots                      # draw the 3 shots once -> docs/shots.json
  uv run python -m ner.infer run B0 smoke               # 20 train sentences: format, VRAM and timing only
  uv run python -m ner.infer run B0 dev                 # full dev, scored
  uv run python -m ner.infer run B0 test --test-once    # full test: once per final system

Per-sentence outputs go to runs/<system>_<split>.jsonl; an existing file is never overwritten. One summary record
per run is appended to runs/runs.jsonl. torch/transformers are imported lazily so CI can test without them.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ner.data import bio_spans, gold_strings, read_conll
from ner.score import counts, map_strings, parse_output, prf

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MAX_NEW_TOKENS = 256
SEED = 20261126
SYSTEM_PROMPT = (  # pre-registration §4, verbatim
    "You extract disease mentions from biomedical text. Read the sentence and return a JSON array of strings. "
    "Each string must be a disease mention copied exactly from the sentence, with the same spelling, capitalisation "
    "and spacing. Include diseases, disorders, syndromes and their abbreviations. Return [] if the sentence contains "
    "no disease mention. Output only the JSON array."
)
CONLL = Path("data/raw/conll")
SPLIT_FILES = {"smoke": "train", "dev": "devel", "test": "test"}
SHOTS = Path("docs/shots.json")
RUNS = Path("runs")


def answer(strings: list[str]) -> str:
    return json.dumps(strings, ensure_ascii=False)


def draw_shots() -> dict:
    """§4: seed 20261126 draws 2 train sentences with >=1 mention, then 1 with none. Order: as drawn."""
    train = read_conll(CONLL / "train.tsv")
    with_m = [i for i, (_, tags) in enumerate(train) if bio_spans(tags)]
    without = [i for i, (_, tags) in enumerate(train) if not bio_spans(tags)]
    rng = random.Random(SEED)
    idx = rng.sample(with_m, 2) + rng.sample(without, 1)
    shots = [{"train_index": i, "sentence": " ".join(train[i][0]),
              "answer": answer(gold_strings(train[i][0], bio_spans(train[i][1])))} for i in idx]
    return {"seed": SEED, "rule": "random.Random(seed): sample(with_mention, 2) + sample(no_mention, 1); order as drawn",
            "shots": shots}


def messages(sentence: str, system: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if system == "B3":
        for s in json.loads(SHOTS.read_text(encoding="utf-8"))["shots"]:
            msgs += [{"role": "user", "content": s["sentence"]}, {"role": "assistant", "content": s["answer"]}]
    return msgs + [{"role": "user", "content": sentence}]


def load(adapter: str | None = None):
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"  # weights come from the local cache only
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=False)
    kw = {"revision": REVISION, "trust_remote_code": False}
    tok = AutoTokenizer.from_pretrained(MODEL, **kw)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, device_map={"": 0},
                                                 use_safetensors=True, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.eval()


def generate(tok, model, msgs: list[dict]) -> dict:
    import torch

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    new = out[0, ids["input_ids"].shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    return {"raw": tok.decode(new, skip_special_tokens=True), "seconds": round(time.perf_counter() - t0, 3),
            "prompt_tokens": int(ids["input_ids"].shape[1]), "output_tokens": len(new),
            "hit_limit": len(new) >= MAX_NEW_TOKENS and new[-1] not in eos,
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)}


def environment() -> dict:
    import bitsandbytes
    import torch
    import transformers

    free, total = torch.cuda.mem_get_info()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True).stdout.strip())
    return {"commit": commit, "dirty": dirty, "model": MODEL, "revision": REVISION, "torch": torch.__version__,
            "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__,
            "BNB_CUDA_VERSION": os.environ.get("BNB_CUDA_VERSION"), "gpu": torch.cuda.get_device_name(0),
            "free_vram_at_start_gib": round(free / 2**30, 3), "total_vram_gib": round(total / 2**30, 3)}


def run(system: str, split: str, adapter: str | None, test_once: bool) -> None:
    if split == "test" and not test_once:
        sys.exit("test is scored once per final system: pass --test-once deliberately")
    if system == "Q" and not adapter:
        sys.exit("Q needs --adapter")
    name = f"{system}{'-' + Path(adapter).name if adapter else ''}_{split}"
    out_path = RUNS / f"{name}.jsonl"
    if out_path.exists():
        sys.exit(f"{out_path} exists: runs are never overwritten (log a deviation and ask the owner)")
    sents = read_conll(CONLL / f"{SPLIT_FILES[split]}.tsv")
    if split == "smoke":
        sents = sents[:20]

    env = environment()
    if env["dirty"] and split != "smoke":
        sys.exit("uncommitted changes to tracked files: commit before a scored run")
    tok, model = load(adapter)
    env["free_vram_after_load_gib"] = round(__import__("torch").cuda.mem_get_info()[0] / 2**30, 3)
    print(json.dumps(env), flush=True)
    generate(tok, model, messages("Warm-up sentence .", system))  # warm-up, excluded

    RUNS.mkdir(exist_ok=True)
    tp = fp = fn = invalid = 0
    recs = []
    t_start = time.perf_counter()
    with out_path.open("x", encoding="utf-8") as f:
        for i, (toks, tags) in enumerate(sents):
            r = generate(tok, model, messages(" ".join(toks), system))
            pred = parse_output(r["raw"], r["hit_limit"])
            gold = bio_spans(tags)
            spans, ph = map_strings(toks, pred) if pred is not None else ([], 0)
            t, p, n = counts(gold, spans, ph)
            tp, fp, fn, invalid = tp + t, fp + p, fn + n, invalid + (pred is None)
            rec = {"i": i, **r, "valid": pred is not None, "pred": pred, "pred_spans": spans, "phantoms": ph,
                   "gold_spans": gold, "tp": t, "fp": p, "fn": n}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            recs.append(rec)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(sents)}  {time.perf_counter() - t_start:.0f}s", flush=True)

    P, R, F = prf(tp, fp, fn)
    secs = sorted(x["seconds"] for x in recs)
    summary = {"time": datetime.now(timezone.utc).isoformat(), "system": system, "split": split, "adapter": adapter,
               "n": len(recs), **env, "tp": tp, "fp": fp, "fn": fn, "precision": round(P, 4),
               "recall": round(R, 4), "f1": round(F, 4), "invalid": invalid,
               "invalid_rate": round(invalid / len(recs), 4), "hit_limit": sum(x["hit_limit"] for x in recs),
               "sec_mean": round(sum(secs) / len(secs), 3), "sec_p50": secs[len(secs) // 2],
               "sec_p95": secs[int(0.95 * (len(secs) - 1))], "wall_seconds": round(time.perf_counter() - t_start, 1),
               "peak_alloc_gib": max(x["peak_alloc_gib"] for x in recs),
               "peak_reserved_gib": max(x["peak_reserved_gib"] for x in recs),
               "prompt_tokens_mean": round(sum(x["prompt_tokens"] for x in recs) / len(recs), 1),
               "output_tokens_mean": round(sum(x["output_tokens"] for x in recs) / len(recs), 1)}
    if split == "smoke":  # §6 step 6: not a score
        for k in ("tp", "fp", "fn", "precision", "recall", "f1"):
            summary.pop(k)
    with (RUNS / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shots")
    r = sub.add_parser("run")
    r.add_argument("system", choices=["B0", "B3", "Q"])
    r.add_argument("split", choices=list(SPLIT_FILES))
    r.add_argument("--adapter")
    r.add_argument("--test-once", action="store_true")
    a = ap.parse_args()
    if a.cmd == "shots":
        if SHOTS.exists():
            sys.exit(f"{SHOTS} exists: shots are drawn once, no redraws")
        SHOTS.write_text(json.dumps(draw_shots(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(SHOTS.read_text(encoding="utf-8"))
    else:
        run(a.system, a.split, a.adapter, a.test_once)


if __name__ == "__main__":
    main()
