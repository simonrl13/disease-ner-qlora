# Plan

Status: **approved by the owner on 2026-11-26**, with the changes below. Nothing was installed or downloaded before
the pre-registration commit.

## Goal

Measure whether QLoRA fine-tuning of Qwen3-4B-Instruct-2507 improves disease-mention extraction on the NCBI Disease
corpus, compared with the same base model zero-shot and 3-shot, on one consumer GPU. This is a base-vs-fine-tuned
comparison, not a state-of-the-art attempt: BERT-family taggers score much higher on this corpus.

## Steps

Each step ends with a commit. Nothing is scored before step 3 is committed.

1. **Repo skeleton.** `git init`, README, SECURITY.md, this plan, `.gitignore` (data/, runs/, adapters/, .env),
   pre-commit with gitleaks. CI like Scout's, with every action pinned to a commit SHA:
   - `ci.yml`: tests, pip-audit (including a separate audit of the Windows-only pins) and gitleaks;
   - `codeql.yml`: CodeQL for Python and Actions.

   The tests and audit jobs need `pyproject.toml` and `uv.lock` from step 2, so they can't pass before then.
2. **Dependencies.** peft is added only before training (step 7), not now. `pyproject.toml` with the Scout pins, the same
   explicit PyTorch cu134 index for torch only, win32 markers on torch/accelerate/bitsandbytes, Python 3.12.
   `uv.lock` committed.
3. **Pre-registration.** `docs/PREREGISTRATION.md` finalised and committed. Its commit hash goes in the README.
4. **Data.** `scripts/get_data.py` downloads, from `spyysalo/ncbi-disease` at commit
   `5a266c6fc76fb2b5184cfa397ff0a5ecdb133aec` (raw.githubusercontent.com, plain HTTPS, no loading script):
   - `conll/{train,devel,test}.tsv`;
   - `original-data/*/NCBI*set_corpus.txt`, for PMIDs;
   - `original-data/LICENSE`.

   It writes SHA-256 hashes to `data/SHA256SUMS` (committed; the data itself is gitignored) and fails unless the
   sentence counts are exactly 5,433 / 924 / 941.

   **Pre-run checks,** reported in the README before any model run (pre-registration §2):
   - PMID leakage across splits;
   - identical sentences across splits;
   - the §3 ambiguity count;
   - mentions per split and sentences with no mention.
5. **Code, with tests before any model run.** One package, about four modules:
   `data.py` (CoNLL → sentences + BIO spans), `score.py` (parse, string → span mapping, micro-P/R/F1, invalid rate,
   paired bootstrap; stdlib only), `infer.py` (prompting + greedy generation + run log), `train.py`
   (transformers `Trainer` + peft LoRA, loss on the answer tokens only). `tests/test_score.py` covers the parser,
   span mapping and F1 on hand-made cases.
6. **Smoke test** on 20 *train* sentences: loads the model, logs free/peak VRAM, checks the output format.
   Not a score; no dev or test sentence is touched.
7. **Dev phase.** Base zero-shot and 3-shot on full dev. QLoRA grid (pre-registration §5) scored on full dev.
   Pick the QLoRA config and the stronger base variant by dev F1.
8. **Test phase.** Each of the three final systems scored on test exactly once. Bootstrap CIs.
9. **Write-up.** README results table with exactly what was measured, deviations log, limitations.

## Hardware and run logging

RTX 3050, 6 GB (Scout measured 5.04 GiB free with the display attached). 4-bit NF4, bf16 compute, no double
quantisation. Every run appends one JSON line to `runs/runs.jsonl` (gitignored; summary tables committed): commit,
model revision, package versions, config, free VRAM at start, peak allocated/reserved VRAM, wall time, token counts,
number of outputs that hit `max_new_tokens`.

**Out of memory:** the run stops, the error is logged in `docs/DEVIATIONS.md`, and I ask the owner before any
change (batch size, sequence length, optimiser, hardware). No silent fallback.

## Dependencies

**Owner decision, 2026-11-26:**
- **peft 0.21.3** is approved for use from 2026-11-27, with the OSV check repeated that day. The lock must add no new
  transitive packages, or they are shown to the owner first.
- **trl** is dropped.
- **pytest, pre-commit and pip-audit** are approved as dev/CI tools.
- **Scout's vetted pins** are reused unchanged.

**Lock (step 1, 2026-11-26):** 54 packages.
- **Two-week rule:** `exclude-newer = 2026-11-12` applies it to every package, transitive ones included.
- **Same as Scout:** 20 transitive packages would have resolved to newer patch releases that Scout never vetted.
  `constraint-dependencies` holds them at Scout's vetted versions, so the only packages not in Scout's lock are
  pytest 9.2.0 and pre-commit 4.7.0, both approved.
- **OSV:** the batch query on 2026-11-26 returned no advisories for any locked version.
- **bitsandbytes:** it loads its CUDA 13.2 build under torch's cu134 build, as it did in Scout.

The table below is the original vetting, checked on PyPI and OSV on 2026-11-26 with a cutoff of 2026-11-12.

| Package | Proposed pin | Released | Newer (too new) | Licence | OSV (pinned version) | Needed? |
|---|---|---|---|---|---|---|
| **peft** | 0.21.2 | 2026-10-01 | 0.21.3 (2026-11-13, eligible 2026-11-27); 0.22.0 (2026-11-19) | Apache-2.0 | no advisories (none ever for peft) | **Yes**: LoRA adapters on the 4-bit model |
| **trl** | 1.17.1 | 2026-10-30 | 1.18.0–1.19.0 (2026-11-13 → 11-25) | Apache-2.0 | no advisories (none ever for trl) | **Recommend no** |
| pytest | 9.2.0 | 2026-10-27 | — | MIT | no advisories | dev: scoring tests |
| pre-commit | 4.7.0 | 2026-10-12 | — | MIT | no advisories | dev: runs gitleaks |
| pip-audit | 2.10.1 | 2026-06-10 | — | Apache-2.0 (PyPI classifier) | no advisories | CI only |
| gitleaks (pre-commit hook, not PyPI) | v8.30.0 | 2025-11-26 | not checked | MIT | n/a | same rev as Scout |

- **peft:** 0.21.3 is a patch release one day short of the rule. Option: pin 0.21.2 now, or wait until 2026-11-27 and
  pin 0.21.3. Its direct requirements (numpy, packaging, psutil, pyyaml, tqdm, torch, transformers, accelerate,
  safetensors, huggingface_hub) are all already in the Scout set or its transitive closure, so I expect **no new
  transitive packages**. Confirmed at lock time and shown to you before install.
- **Why not trl:** 1.17.1 requires `datasets`, `pandas`, `pyarrow` and `jinja2` on top. Its `SFTTrainer` saves
  about 20 lines (completion-only loss masking) over plain `transformers.Trainer`. Not worth four extra packages.
- **No** seqeval, numpy-based bootstrap, outlines/xgrammar or vLLM. Span scoring and the bootstrap are stdlib Python
  (10,000 × 941 sentences is a few seconds). JSON validity is measured, so decoding is not constrained.

## Time and GPU estimate

These are estimates only, not measurements. The smoke test (step 6) measures real throughput, and I re-estimate
before the dev phase.

- **Assumptions:** about 140 tokens per training example (prompt + sentence + answer), so about 0.76M tokens per
  epoch. Training at 300–600 tokens/s. Greedy decoding, one sentence at a time, 1.5–3 s per sentence
  (bitsandbytes 4-bit decoding is slow).

| Item | GPU time |
|---|---|
| Smoke test | ~10 min |
| Base dev: zero-shot + 3-shot (2 × 924) | 0.8–1.5 h |
| QLoRA training: 2 learning rates × 2 epochs | 1.4–2.8 h |
| QLoRA dev: 4 checkpoints × 924 | 1.5–3 h |
| Test: 3 systems × 941 | 1.2–2.4 h |
| **Total** | **≈ 5–10 GPU-hours** on the RTX 3050 |

- **Spend: US$0.** Everything is local. Electricity isn't costed. Kaggle or any cloud is not planned. Using one
  would be a hardware deviation, and I would ask first.
- **Wall-clock:** about 3–4 working sessions. Most of it is unattended GPU runs; the coding plus tests take about
  one session.

## Decisions (owner, 2026-11-26)

1. **Dependencies:** as above.
2. **String → span rule:** approved, with whole-token, case-sensitive matches only, the same rule for every system,
   and the ambiguity count reported before any run.
3. **Base comparator:** the stronger of zero-shot and 3-shot, chosen on dev only.
4. **Leakage check added:** no PMID or identical sentence may appear in more than one split, and the result is
   reported before any run.
5. **Licence wording:** "public domain, United States Government Work, per NCBI's notice". Not CC0.
6. **Citation:** checked by the owner against PubMed. J Biomed Inform 2014;47:1–10, doi 10.1016/j.jbi.2013.12.006.
