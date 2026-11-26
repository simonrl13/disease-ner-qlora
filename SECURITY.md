# Security

## Secrets

- **No secrets needed:** the model isn't gated and the data is public, so no Hugging Face token is used.
- **If one is ever needed:** it goes in an environment variable or a gitignored `.env`, never in code, logs or
  commits.
- **Secret scanning:** gitleaks runs as a pre-commit hook and in CI. Hooks and CI are never bypassed.

## Auth model

There is none. This is a local, offline experiment with no server, API or user accounts.

## Dependencies

- **Exact pins** in `pyproject.toml`, with `uv.lock` committed.
- **Age:** each release is at least 14 days old at install, a patched release, and checked on OSV.
- **Approval:** the owner approves every new package by version, release date, licence and OSV status. The list is
  in [docs/PLAN.md](docs/PLAN.md).
- **torch index:** torch comes only from the explicit PyTorch cu134 index, Windows only. Nothing else resolves
  from that index.
- **CI audit:** pip-audit runs in CI.

## Model and data provenance

- **Model:** pinned revision, safetensors only, `trust_remote_code=False`, and `HF_HUB_DISABLE_TELEMETRY=1`.
- **Data:** raw CoNLL files fetched over HTTPS from a pinned Git commit, with their SHA-256 hashes recorded and
  checked. No dataset loading scripts.
- **Downloads are untrusted:** they are parsed as plain text and never executed.

## AI features

- **What reaches the model:** only public NCBI corpus sentences, and inference runs locally. Nothing is sent to an
  external API.
- **Prompt injection:** model output is data. It is parsed with `json.loads` and never executed or evaluated.
  Malformed output is counted as invalid, not repaired.
- **Rate limits:** not applicable, since there are no external calls.
- **AI-generated code:** reviewed before merge. Scoring code has unit tests written before any model run.
