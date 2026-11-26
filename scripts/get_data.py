"""Download the NCBI Disease files from the pinned commit, verify them, write data/SHA256SUMS.

Each file must match the git blob hash GitHub records for it at the pinned commit, so a byte-exact copy is
guaranteed. Files are treated as untrusted text: hashed and parsed, never executed.
Run: uv run python -I scripts/get_data.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "5a266c6fc76fb2b5184cfa397ff0a5ecdb133aec"
BASE = f"https://raw.githubusercontent.com/spyysalo/ncbi-disease/{COMMIT}/"
# path -> git blob SHA-1 at COMMIT (GitHub tree API, 2026-11-26)
FILES = {
    "conll/train.tsv": "8e48dda77208747197e17346b1a8e455177634a6",
    "conll/devel.tsv": "152e9b9d3b4275a1bb9a435b0783946ef40208fe",
    "conll/test.tsv": "6217f611e1517a70e5bd645b9dea34f7f9bf594f",
    "original-data/train/NCBItrainset_corpus.txt": "df5b923f1e10dd4a36d84429606ae13de0325b26",
    "original-data/devel/NCBIdevelopset_corpus.txt": "11397ce039ef99dbbbbadde81a2ce7c849c47618",
    "original-data/test/NCBItestset_corpus.txt": "c098c9115a9b858dbc3795c730d86bf85556ecf2",
    "original-data/LICENSE": "498b69f93dd27d7cf1547004fe7ea0e5601dbd6f",
}
RAW = Path("data/raw")


def git_blob_sha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main() -> None:
    sums = []
    for path, blob in FILES.items():
        dest = RAW / path
        if not dest.exists():
            with urllib.request.urlopen(BASE + path, timeout=60) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        data = dest.read_bytes()
        if git_blob_sha1(data) != blob:
            sys.exit(f"{path}: does not match the git blob at {COMMIT}")
        sums.append(f"{hashlib.sha256(data).hexdigest()}  raw/{path}")
    Path("data/SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(sums))


if __name__ == "__main__":
    main()
