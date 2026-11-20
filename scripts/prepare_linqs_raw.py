"""Unpack a LINQS archive (cora.tgz or citeseer.tgz) into data/raw/<name>/.

Usage:  python scripts/prepare_linqs_raw.py cora path/to/cora.tgz

Only what the model needs is kept: the citation file as it is, the README, and a
small tab file with paper id and class label. The 1433 / 3703 word columns of the
.content file are not used (same as for PubMed-Diabetes) and would add 8-25 MB.
"""
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(name, archive):
    out = ROOT / "data" / "raw" / name
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for wanted in (f"{name}.cites", "README"):
            (out / wanted).write_bytes(tar.extractfile(members[wanted]).read())
        content = tar.extractfile(members[f"{name}.content"])
        with open(out / f"{name}.nodes.tab", "w", encoding="utf-8", newline="\n") as nodes:
            nodes.write("paper_id\tlabel\n")
            for line in content:
                parts = line.decode("utf-8", errors="replace").split()
                nodes.write(f"{parts[0]}\t{parts[-1]}\n")
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
