#!/usr/bin/env python3
"""Join the parts of the data release, check them and extract the archive.

    python release/join_release_data.py                       # parts in data/release/, extracts ./release_data/
    python release/join_release_data.py --parts DIR --out DIR

The archive release_data.tar.xz is cut into parts of at most 9.5 MB (release_data.tar.xz.001, .002, ...), so that it
fits GitHub's limits. release_data.SHA256SUMS holds the SHA-256 of the archive and of each part, and MANIFEST.json,
inside the archive, that of every file. This script checks the three and extracts the archive. It needs only the
Python standard library.

By hand: ``cat release_data.tar.xz.0* > release_data.tar.xz && tar -xJf release_data.tar.xz`` (Linux, macOS), or open
release_data.tar.xz.001 with 7-Zip (Windows).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default=os.path.join(HERE, "..", "data", "release"), help="folder with the parts")
    ap.add_argument("--out", default=".", help="folder where release_data/ is extracted")
    ap.add_argument("--name", default="release_data", help="name of the archive, without .tar.xz")
    a = ap.parse_args(argv)

    parts = sorted(glob.glob(os.path.join(a.parts, f"{a.name}.tar.xz.[0-9][0-9][0-9]")))
    if not parts:
        raise SystemExit(f"no parts {a.name}.tar.xz.001, .002, ... in {a.parts}")
    sums = {}
    sums_path = os.path.join(a.parts, f"{a.name}.SHA256SUMS")
    if os.path.exists(sums_path):
        for line in open(sums_path, encoding="utf-8"):
            digest, _, fname = line.strip().partition("  ")
            if fname:
                sums[fname] = digest
    else:
        print(f"warning: {sums_path} not found; the parts are not checked")
    for p in parts:
        name = os.path.basename(p)
        if name in sums and sha256(p) != sums[name]:
            raise SystemExit(f"{name}: SHA-256 does not match {os.path.basename(sums_path)}")
    missing = [n for n in sums if n.startswith(f"{a.name}.tar.xz.") and not os.path.exists(os.path.join(a.parts, n))]
    if missing:
        raise SystemExit(f"missing parts: {', '.join(missing)}")

    os.makedirs(a.out, exist_ok=True)
    archive = os.path.join(a.out, f"{a.name}.tar.xz")
    with open(archive, "wb") as g:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, g)
    if f"{a.name}.tar.xz" in sums and sha256(archive) != sums[f"{a.name}.tar.xz"]:
        raise SystemExit(f"{a.name}.tar.xz: SHA-256 does not match {os.path.basename(sums_path)}")

    with tarfile.open(archive, "r:xz") as tf:
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in m.name.split("/") or not (m.isfile() or m.isdir()):
                raise SystemExit(f"unexpected member in the archive: {m.name}")
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(a.out, **kwargs)

    top = os.path.join(a.out, a.name)
    man = json.load(open(os.path.join(top, "MANIFEST.json"), encoding="utf-8"))
    bad = [f["path"] for f in man["files"] if sha256(os.path.join(top, f["path"])) != f["sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} files do not match MANIFEST.json, e.g. {bad[:3]}")
    print(f"{len(parts)} parts joined and checked; {len(man['files'])} files extracted to {top}")
    return top


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
