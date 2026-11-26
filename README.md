# disease-ner-qlora

Does QLoRA fine-tuning help a small instruction-tuned LLM extract disease mentions? This repo measures it for
Qwen3-4B-Instruct-2507 on the NCBI Disease corpus, on a single 6 GB consumer GPU (RTX 3050).

**This is not a state-of-the-art attempt.** BERT-family token classifiers reach much higher F1 on this corpus. The
question here is narrower: base model (zero-shot and 3-shot) against the same model after QLoRA, under one
pre-registered protocol.

**Status:** pre-registered, no results yet. See [docs/PLAN.md](docs/PLAN.md) and the
[pre-registration](docs/PREREGISTRATION.md), which was committed before any data download, model output or score, in
commit `6909410ee1031ddedbffa448194ca6515b3a0c70` (2026-11-26).

## Task

The model reads one sentence and returns the disease mentions as a JSON list of exact strings. Scoring is
entity-level micro-F1 on exact span match against the gold BIO spans, with precision, recall and the invalid-output
rate. Unparseable output counts as a failed prediction.

## Pre-run checks (2026-11-26, before any model output)

These come from `scripts/prerun_checks.py`, and the full output is in
[docs/prerun_checks.json](docs/prerun_checks.json). The rules are in pre-registration §2 and Amendments 1–2.

- **Files:** all 7 downloaded files match the git blob hashes at commit `5a266c6`. Their SHA-256 hashes are in
  [data/SHA256SUMS](data/SHA256SUMS).
- **Sentences:** 5,432 / 923 / 940, as expected under Amendment 2. The HF dataset card's 5,433 / 924 / 941 include one
  empty end-of-file example per split.
- **Documents:** 593 / 100 / 100 abstracts, matching the authors' counts. Train has 592 unique PMIDs plus the
  documented duplicate, PMID 8528200.
- **PMID leakage:** none. No PMID appears in more than one split.
- **Gold mentions:** 5,145 / 787 / 960 (6,892 in all), matching the mention counts in Doğan et al. 2014. Sentences
  with no mention: 2,502 / 434 / 401.
- **Identical sentences across splits:** they are reported here, and the splits are not modified.

  | Splits | Distinct | Sentences |
  |---|---|---|
  | train & dev | 5 | `1 1 .`, `2 2 .`, `3 .`, `METHODS .`, `RESULTS .` (section labels and fragments; no disease mention) |
  | train & test | 1 | `Hereditary deficiency of the fifth component of complement in man .` |
  | dev & test | 1 | `5 of embryogenesis .` (fragment; no disease mention) |

  The train and test sentence is the start of the titles of two different papers, part I (PMID 932197, test) and
  part II (PMID 777027, train), cut at "man." by sentence splitting. Its gold label is the same in both splits:
  "Hereditary deficiency of the fifth component of complement".
- **Duplicates within a split:** train has 32 repeated sentences, including the duplicated document, and test has 3.
  This is reported, not changed.
- **Amendment 1:** 1 test sentence also appears in train, so test F1 will also be reported on the remaining 939 test
  sentences as a sensitivity check. The main result stays on the full 940.
- **String-output ceiling (§3):** this is what a perfect answer, the gold strings themselves, scores under the
  mapping rule.

  | Split | Extra FPs | Sentences affected | Max micro-F1 |
  |---|---|---|---|
  | train | 22 | 20 | 0.9979 |
  | dev | 1 | 1 | 0.9994 |
  | test | 9 | 9 | 0.9953 |

  The FPs come from a gold string repeated unlabelled elsewhere in the same sentence. The oracle has no false
  negatives.

## Results

_None yet. Results will be reported exactly as measured, including failures and deviations._

## Data

NCBI Disease corpus: R. Islamaj Doğan, R. Leaman, Z. Lu. "NCBI disease corpus: A resource for disease name
recognition and concept normalization." *Journal of Biomedical Informatics* 47 (2014): 1–10.
doi:10.1016/j.jbi.2013.12.006.

- **Files used:** the CoNLL conversion in [spyysalo/ncbi-disease](https://github.com/spyysalo/ncbi-disease), pinned to
  commit `5a266c6fc76fb2b5184cfa397ff0a5ecdb133aec`. SHA-256 hashes are in `data/SHA256SUMS` once downloaded.
- **Licence:** public domain, United States Government Work, per NCBI's
  [Public Domain Notice](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/disclaimer.html). The same notice
  ships as `original-data/LICENSE` at the pinned commit.
- **Other citations:** the notice also asks for two more works to be cited, which it lists by link:
  - [An improved corpus of disease mentions in PubMed citations](https://aclanthology.org/W12-2411/) (BioNLP 2012);
  - Disease Name Normalization with Pairwise Learning to Rank (PubMed 23969135).
- **Conversion repo:** it has no licence file. Only its data files are downloaded, and none of its code is used.

## Model

[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) at revision
`cdbee75f17c01a7cc42f958dc650907174af0554` (Apache-2.0). It is loaded from safetensors only, with
`trust_remote_code=False`, in 4-bit NF4 with bf16 compute.
