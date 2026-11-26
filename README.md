# disease-ner-qlora

Does QLoRA fine-tuning help a small instruction-tuned LLM extract disease mentions? This repo measures it for
Qwen3-4B-Instruct-2507 on the NCBI Disease corpus, on a single 6 GB consumer GPU (RTX 3050).

**This is not a state-of-the-art attempt.** BERT-family token classifiers reach much higher F1 on this corpus. The
question here is narrower: base model (zero-shot and 3-shot) against the same model after QLoRA, under one
pre-registered protocol.

**Status:** pre-registered, no results yet. See [docs/PLAN.md](docs/PLAN.md) and the
[pre-registration](docs/PREREGISTRATION.md), which was committed before any data download, model output or score.
The commit hash will be recorded here.

## Task

The model reads one sentence and returns the disease mentions as a JSON list of exact strings. Scoring is
entity-level micro-F1 on exact span match against the gold BIO spans, with precision, recall and the invalid-output
rate. Unparseable output counts as a failed prediction.

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
