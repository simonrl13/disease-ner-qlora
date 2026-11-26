# Pre-registration

Status: **binding from the commit that adds this file** (2026-11-26), made before any data download, model output or
score. Any later change goes in a dated amendment in §9, never as a silent edit.

## 1. Question

On the NCBI Disease corpus test split, does QLoRA fine-tuning of Qwen3-4B-Instruct-2507 raise entity-level
micro-F1 over the same base model prompted zero-shot or 3-shot?

## 2. Data

- **Source:** `spyysalo/ncbi-disease` at commit `5a266c6fc76fb2b5184cfa397ff0a5ecdb133aec`.
  - `conll/train.tsv`, `conll/devel.tsv` and `conll/test.tsv` supply the sentences and gold spans.
  - `original-data/{train,devel,test}/NCBI*set_corpus.txt` are used only to read PMIDs for the leakage check.
- **Hashes:** the SHA-256 of every file is recorded on download.
- **Counts:** expected sentence counts are 5,433 / 924 / 941. Any mismatch stops the project.
- **Known quirk:** document PMID 8528200 appears twice in the original *train* set, and this conversion keeps the
  duplicate (repo README, issue #1). It is kept, so that the published counts match. It touches only train.
- **Gold:** contiguous `B-Disease`/`I-Disease` spans as given in the files, with no re-annotation. A stray `I-` with no
  `B-` before it starts a new span.
- **Model input:** the sentence's tokens joined by single spaces, exactly as in the file.

### Pre-run checks, each reported in the README before any model run

1. **Leakage, PMIDs:** no PMID appears in more than one split. If one does, the project stops and the owner decides.
   The within-train duplicate above isn't cross-split.
2. **Leakage, sentences:** no identical sentence (same token sequence) appears in more than one split. Any are
   reported with their count and text. The published splits are **not** modified, so results stay comparable with
   other work on the same splits.
3. **Ambiguity:** the number of gold spans affected by the ceiling in §3, per split.

## 3. Output format, parsing and span mapping

The same rules apply to every system.

- **Expected output:** a JSON array of strings, each an exact copy of a disease mention from the sentence. `[]` if
  there is none.
- **Parsing:** strip surrounding whitespace and at most one enclosing Markdown code fence, then `json.loads`.
- **Invalid output:** anything that doesn't parse, isn't a list, has a non-string element, or stops at
  `max_new_tokens` (256). It counts as zero predictions for that sentence, so every gold span there is a false
  negative, and it counts toward the invalid-output rate. There is no retry and no repair.
- **String → span mapping:**
  - each distinct predicted string is split on whitespace into tokens;
  - it matches only whole sentence tokens: case-sensitive, exact, never part of a token. `"cancer"` does not match
    the token `cancers`, and `"Cancer"` does not match `cancer`;
  - it maps to every non-overlapping occurrence of that token sequence, scanning left to right;
  - longer strings (more tokens) are placed first, and a later string can't claim tokens already used;
  - a string with no remaining occurrence is one false positive;
  - an empty string, or one that is only whitespace, is one false positive.
- **Known ceiling:** if gold labels one occurrence of a string and not another in the same sentence, a perfect
  string list still scores a false positive. That count is a pre-run check (§2).
- **Training targets:** the distinct gold strings, in order of first occurrence.

## 4. Systems

All systems use the same model and loading:
- revision `cdbee75f17c01a7cc42f958dc650907174af0554`, safetensors only, `trust_remote_code=False`;
- `HF_HUB_DISABLE_TELEMETRY=1`;
- 4-bit NF4 with bf16 compute and no double quantisation;
- the model's own chat template;
- greedy decoding, one sentence at a time, `max_new_tokens=256`.

**System prompt**, fixed for every system and never tuned:

```text
You extract disease mentions from biomedical text. Read the sentence and return a JSON array of strings. Each string must be a disease mention copied exactly from the sentence, with the same spelling, capitalisation and spacing. Include diseases, disorders, syndromes and their abbreviations. Return [] if the sentence contains no disease mention. Output only the JSON array.
```

The user turn is the sentence text alone. The assistant answer is the compact JSON array: `json.dumps` with default
separators and `ensure_ascii=False`.

- **B0 (zero-shot):** the system prompt, then the sentence.
- **B3 (3-shot):** the same, plus three earlier user/assistant turns taken from train. Seed 20261126 draws 2
  sentences with at least one mention and 1 sentence with none, from train only. They are drawn once, with no
  redraws, and committed before any model run.
- **Q (QLoRA):** B0's prompt format, with a LoRA adapter trained on train only.

## 5. QLoRA hyperparameters (dev only)

- **Fixed:** LoRA r=16, alpha=32, dropout 0.05, on all attention and MLP projections. Loss on the answer tokens only.
  Effective batch 16, cosine schedule, 3% warm-up, `paged_adamw_8bit`, gradient checkpointing, seed 20261126, max
  sequence length 512.
- **Searched:** learning rate ∈ {1e-4, 2e-4} × checkpoint after epoch {1, 2}. That gives 4 candidates, each scored on
  the full dev split.
- **Selection:** highest dev micro-F1. Ties go to the lower invalid rate, then to the earlier checkpoint.
- **Base comparator:** whichever of B0 and B3 has the higher dev micro-F1, with ties going to B0. It is chosen on dev
  only and committed before any test run.

## 6. Test protocol

- **Scored once:** B0, B3 and the selected Q are each run on test once, after every dev decision is committed.
- **No re-runs:** a crash or out-of-memory error is logged as a deviation in `docs/DEVIATIONS.md`. Nothing is re-run
  without a dated amendment that explains why.

## 7. Metrics

- **Primary:** entity-level micro-F1 on test, with exact span match.
- **Also reported:** precision, recall and invalid-output rate for every system, plus free and peak VRAM, time per
  sentence and training time.
- **Confidence intervals:** paired bootstrap over test sentences, with 10,000 resamples and seed 20261126. Each
  resample recomputes micro-F1 from summed TP/FP/FN, and the interval is the 2.5th to 97.5th percentile of
  F1(Q) − F1(base).
- **Limitation:** sentences from the same abstract are correlated, so a sentence-level bootstrap may make the
  intervals too narrow.

## 8. Decision rule, stated in advance

Primary comparison: Q against the dev-selected base comparator, on test.

- **"Fine-tuning helped":** the lower bound of the 95% CI on F1(Q) − F1(base) is above 0.
- **"Fine-tuning hurt":** the upper bound is below 0.
- **Otherwise "inconclusive",** reported as such.

Secondary comparisons are descriptive only, with no claim attached: Q vs the other base variant, and B3 vs B0.

Every number is reported whatever the outcome, including invalid rates and any deviations.

## 9. Amendments

### Amendment 1 (2026-11-26, before any data download)

**Train/test overlap sensitivity check.** If any test sentence also appears verbatim in train (same token sequence,
per the §2 check), test F1 is also reported, for every system, on the test subset without those sentences. The
bootstrap for that subset uses the same settings as §7.

This is a sensitivity check only. The main result, and the §8 decision rule, stay on the full published test split.
If no test sentence appears in train, this check isn't run, and the README says so.

### Amendment 2 (2026-11-26, owner-approved, before the remaining pre-run checks and any model output)

**Expected sentence counts.** The §2 counts (5,433 / 924 / 941) came from the Hugging Face `ncbi_disease` dataset card.
They include one empty end-of-file example per split: the HF loading script, at line 140, emits a final example after
its read loop without checking that it has any tokens, and each file ends with a blank line.

The downloaded files match the pinned commit byte for byte. They contain **5,432 / 923 / 940** non-empty sentences,
and those are now the expected counts.

**Document counts, added as a check.** The authors' published counts are 593 / 100 / 100 abstracts. Train must have
593 title lines, made of 592 unique PMIDs plus the documented duplicate, PMID 8528200. Any other result stops the
project.

The data and all other rules are unchanged.
