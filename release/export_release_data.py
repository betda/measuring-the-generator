#!/usr/bin/env python3
"""Export the text-free data release of the study from its Drive folder.

Run it in Colab, with Google Drive mounted, on the folder that the notebooks write to:

    !python release/export_release_data.py --root /content/drive/MyDrive/CLM_losses/colab --out /content/release_data

It writes ``<out>/``, the archive ``<out>.tar.xz``, the same archive cut into parts of at most 9.5 MB
(``<out>.tar.xz.001``, ``.002``, ...; they fit the upload limits of GitHub's web interface and of the Drive connector)
and ``<out>.SHA256SUMS``. ``release/join_release_data.py`` joins the parts, checks them and extracts the archive.
The folder mirrors the layout under ``--root`` and keeps only files without text:

- ``config/``   study configuration, protocol freeze and addendum, as the notebook generated them
- ``data/``     document identifiers of each held-out split; item metadata (doc_id, pair_idx, row) of each evaluation
                set; subsample indices; candidate indices and cosines of the random and mined sets; item indices of the
                T-gen sets. Embeddings, token IDs and texts are skipped.
- ``runs/``     per run: config.json, DONE.json, log.jsonl, eval/metrics.json and the per-item arrays
                (correct_*, argmax_*, items_*, scores_*). Checkpoints are skipped.
- ``results/``  JSON and CSV result files of every phase and of the post hoc analyses
- ``logs/``     pip freeze of the sessions
- ``labels/``   the judge's false-negative labels by item and candidate, without the judge's raw answers

Every file is checked before it is copied, and the run ends with ``REPORT.md`` (what was exported and what was
skipped, with the reason), ``EXCLUDED.csv`` and ``MANIFEST.json`` (size and SHA-256 of every exported file).
Review ``REPORT.md`` before publishing.

Rules (the release must not redistribute the texts of Nemotron-CC, of CLM's negatives or of the generated negatives):

- ``.npy``: only names on an allow list, with a numeric dtype, and without an embedding-sized last axis;
- ``.json`` / ``.csv`` / the ``.jsonl`` logs of ``runs/`` and ``config/``: skipped if a field or column that holds data
  text (question, answer, candidates, prompt, raw output, ...; also as part of a longer name, such as
  ``pregunta_entrenamiento``) has a value of three or more words; if a string is a sentence (15 or more words) that is
  not Spanish prose; or if a string of five or more words reads as English (two or more English function words and no
  Spanish ones). The data texts are in English and the study's notes and decision rules are in Spanish, so the notes
  are kept and the texts are not. Configuration, freeze and rule files may hold English prompt templates (Appendix B
  of the article); for them only the overlap check below applies;
- every exported text-like file is also compared with the study's texts (the questions, answers and candidates in
  ``results/PRE/textos/*.jsonl`` and the panel's ``kept.jsonl`` and ``items_text.jsonl``, when they exist): a file
  that shares a run of six words with them is skipped;
- files that name the annotators' labeling sheets are skipped, so that the annotators stay anonymous;
- everything else (checkpoints, embeddings, token IDs, generations, prompts, spreadsheets, reports in Markdown, the
  BM25 vocabulary of the training answers, figures and server logs) is skipped.

Licenses: the exported data are released under CC BY 4.0 (see LICENSE-DATA.md in the repository).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile

import numpy as np

SECTIONS = ("config", "data", "runs", "results", "logs")
SKIP_DIRS = {"textos", "eval_local", "cache", "__pycache__", ".ipynb_checkpoints", "tmp", "hub", "models", "hf_cache"}

# Fields that carry data text anywhere in the study's files.
TEXT_KEYS = {
    "question", "questions", "answer", "answers", "q", "a", "gold", "oro", "pregunta", "preguntas", "candidato",
    "candidatos", "cands", "candidates", "candidate_text", "positive", "positives", "negative", "negatives", "negs",
    "prompt", "prompts", "raw", "text", "texts", "texto", "textos", "response", "responses", "output", "outputs",
    "completion", "completions", "generation", "generations", "state", "action", "doc", "document", "passage",
    "context", "example", "examples", "ejemplo", "ejemplos", "muestra", "muestras", "neg_text", "pos_text",
    "corpus", "notas", "notes",
}
TEXT_KEY_PARTS = {k for k in TEXT_KEYS if len(k) > 2}   # 'pregunta_entrenamiento' and 'oro_test' are text fields too
LONG_WORDS = 15          # a string with this many words is a sentence
SHORT_WORDS = 5          # a string with this many words is checked for English
PROSE_OK = [re.compile(p) for p in (r"^config/", r"/REGLA\.json$", r"^logs/pip_freeze")]
JSONL_OK = [re.compile(p) for p in (r"^runs/", r"^config/")]   # training logs and the F0 decision log
LABEL_SHEET = re.compile(r"etiquetar_", re.I)                    # the sheets are named after the annotators
SKIP_FILES = {"bm25_corpus.json": "BM25 vocabulary and document frequencies of the training answers"}

# Function words, to tell the study's notes (Spanish) from the data texts (English).
SPANISH = set("""de la el que y en los se del las por un para con una es al lo como más mas pero sus su le ya sin sobre
este esta estos estas entre cuando sólo solo también hay si porque cada donde son fue ser está están todos todas otro
otros otra otras mismo misma antes después aunque así ni ese esa eso dos tres según desde hasta bajo contra durante
mientras menos muy nos qué cuál sí cuyo cuya sigue siguen queda quedan""".split())
ENGLISH = set("""the of and to in is that it for on with as was by are this be or an which from at were has have not
but they their his her its can will would there been these those what when who how than then into about after before
also more most such only other some any each we you he she them our your do does did should could may might must""".split())
LETTERS = re.compile(r"[^\W\d_]{2,}")

NPY_ALLOW = [re.compile(p) for p in (
    r"^correct_.*\.npy$", r"^argmax_.*\.npy$", r"^items(_.*)?\.npy$", r"^scores_.*\.npy$", r"^.*_idx\.npy$",
    r"^.*sims\.npy$", r"^subsample\.npy$", r"^mask_.*\.npy$", r"^neg_cos\.npy$", r"^.*_correct\.npy$",
    r"^perm.*\.npy$", r"^blind_.*\.npy$", r"^panel_order\.npy$",
)]
NPY_DENY = re.compile(r"(token|emb|latent|^w_|^chunk_|^c\d{4}_[qan]\.npy$|^negs\.npy$|^pos\.npy$|_text)", re.I)
MAX_LAST_AXIS = 256      # embeddings have 512-4096 columns; candidate and score arrays have at most ~50

SHINGLE = 6
PART_BYTES = 9_500_000   # size of each part of the archive


def words(s: str) -> list:
    return re.findall(r"\w+", s.lower())


def shingles(s: str):
    w = words(s)
    for i in range(len(w) - SHINGLE + 1):
        yield hash(" ".join(w[i:i + SHINGLE]))


def tokens(s: str) -> list:
    """Words of a string as a reader counts them: pieces between spaces with at least two letters."""
    return [t for t in s.split() if LETTERS.search(t)]


def function_words(toks: list) -> tuple:
    """(Spanish, English) function words among the tokens."""
    w = [t.lower().strip(".,;:()[]{}'\"¿?¡!«»") for t in toks]
    return sum(x in SPANISH for x in w), sum(x in ENGLISH for x in w)


def public_path(rel: str) -> str:
    """The path as listed in REPORT.md and EXCLUDED.csv: a labeling sheet's name, which may carry the annotator's
    name, is replaced by a wildcard."""
    head, _, name = rel.rpartition("/")
    if LABEL_SHEET.match(name):
        stem, dot, ext = name.partition(".")
        name = stem[:len("etiquetar_")] + "*" + dot + ext
    return f"{head}/{name}" if head else name


def is_text_key(key) -> bool:
    if key is None:
        return False
    return key in TEXT_KEYS or any(p in TEXT_KEY_PARTS for p in re.split(r"[_\-\s.]+", key))


def strings_in(obj, key=None):
    """(key, string) for every string in a JSON value, with the key it hangs from."""
    if isinstance(obj, str):
        yield key, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings_in(v, str(k).lower())
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from strings_in(v, key)


class Exporter:
    def __init__(self, root: str, out: str, with_scores: bool = True):
        self.root, self.out, self.with_scores = root, out, with_scores
        self.included, self.excluded = [], []      # (relpath, reason)
        self.text_shingles: set = set()

    # ---------------------------------------------------------------- texts to compare against
    def text_sources(self) -> list:
        """The files that hold the study's texts: questions, answers and candidates of the evaluation sets
        (results/PRE/textos) and the questions, answers and kept negatives of the T-gen panel."""
        out = []
        tdir = os.path.join(self.root, "results", "PRE", "textos")
        if os.path.isdir(tdir):
            out += [os.path.join(tdir, fn) for fn in sorted(os.listdir(tdir)) if fn.endswith(".jsonl")]
        for dirpath, dirs, files in os.walk(os.path.join(self.root, "data", "eval", "tgen")):
            dirs.sort()
            out += [os.path.join(dirpath, fn) for fn in sorted(files) if fn in ("kept.jsonl", "items_text.jsonl")]
        return out

    def load_study_texts(self):
        n = 0
        for path in self.text_sources():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for _, s in strings_in(rec):
                        if len(words(s)) >= SHINGLE:
                            self.text_shingles.update(shingles(s))
                            n += 1
        return n

    def shares_text(self, s: str) -> bool:
        if not self.text_shingles or len(words(s)) < SHINGLE:
            return False
        return any(h in self.text_shingles for h in shingles(s))

    # ---------------------------------------------------------------- checks per file type
    def check_strings(self, rel: str, pairs) -> str | None:
        # Configuration, freeze and rule files hold the study's own notes and prompt templates (Appendix B of the
        # article); for them only the overlap with the study's texts is checked.
        prose_ok = any(p.search(rel) for p in PROSE_OK)
        for key, s in pairs:
            if LABEL_SHEET.search(s):
                return "names the labeling sheets (they identify the annotators)"
            if not prose_ok:
                toks = tokens(s)
                if is_text_key(key) and len(toks) >= 3:
                    return f"text field '{key}'"
                if len(toks) >= SHORT_WORDS:
                    es, en = function_words(toks)
                    if len(toks) >= LONG_WORDS and not (es >= 3 and es > en):
                        return "contains sentences that are not the study's notes in Spanish"
                    if en >= 2 and es == 0:
                        return "contains English text"
            if self.shares_text(s):
                return "shares text with the study's texts"
        return None

    def check_npy(self, path: str, name: str) -> str | None:
        if NPY_DENY.search(name):
            return "embeddings or token IDs"
        if not any(p.match(name) for p in NPY_ALLOW):
            return "array not on the allow list"
        if name.startswith("scores_") and not self.with_scores:
            return "scores left out (--no-scores)"
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as e:  # object arrays, damaged files
            return f"unreadable array ({type(e).__name__})"
        if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
            return f"non-numeric array ({arr.dtype})"
        if arr.ndim >= 2 and arr.shape[-1] > MAX_LAST_AXIS:
            return f"embedding-sized array {tuple(arr.shape)}"
        return None

    def check_json(self, path: str, rel: str) -> str | None:
        try:
            obj = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            return f"unreadable JSON ({type(e).__name__})"
        return self.check_strings(rel, strings_in(obj))

    def check_jsonl(self, path: str, rel: str) -> str | None:
        if not any(p.search(rel) for p in JSONL_OK):
            return "JSON Lines outside runs/ and config/ (prompts, generations or texts)"
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return "unreadable JSON Lines"
                why = self.check_strings(rel, strings_in(rec))
                if why:
                    return why
        return None

    def check_csv(self, path: str, rel: str) -> str | None:
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel
                rows = csv.reader(f, dialect)
                # A column is judged by its values, as a JSON field is: a column named 'A' that holds arm names
                # (the pairwise tables) is kept; one named 'pregunta' that holds questions is not.
                header = [str(h).strip().lower().split("(")[0].strip() for h in next(rows, [])]
                pairs = ((header[i] if i < len(header) else None, c) for row in rows for i, c in enumerate(row))
                why = self.check_strings(rel, pairs)
                return why.replace("text field", "text column") if why else None
        except UnicodeDecodeError:
            return "not UTF-8 text"

    def check_txt(self, path: str, rel: str) -> str | None:
        name = os.path.basename(rel)
        if name.endswith("_docs.txt"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t and (" " in t or len(t) > 128):
                        return "not a list of identifiers"
            return None
        if name.startswith("pip_freeze"):
            return None
        return "text file"

    def check_md(self, path: str, rel: str) -> str | None:
        if re.search(r"^config/PROTOCOL_FREEZE(_ADDENDUM)?\.md$", rel) or rel.endswith("/REGLA.md"):
            text = open(path, encoding="utf-8").read()
            if LABEL_SHEET.search(text):
                return "names the labeling sheets (they identify the annotators)"
            if any(self.shares_text(line) for line in text.splitlines()):
                return "shares text with the study's questions or answers"
            return None
        return "report in Markdown (its numbers are in the JSON and CSV files)"

    # ---------------------------------------------------------------- special files
    def export_judge(self, path: str):
        """labels/judge_labels.csv from the judge's raw output: first valid labels of each item, without the raw answer."""
        judge, attempt = {}, {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("labels") is not None and r["i"] not in judge:
                    judge[r["i"]] = [int(x) for x in r["labels"]]
                    attempt[r["i"]] = r.get("attempt")
        dst = os.path.join(self.out, "labels", "judge_labels.csv")
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        def order(k):
            s, _, i = str(k).rpartition(":")
            return (s, int(i) if i.isdigit() else i)

        with open(dst, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "candidate", "label", "attempt"])
            for k in sorted(judge, key=order):
                for n, lab in enumerate(judge[k], 1):
                    w.writerow([k, n, lab, attempt[k]])
        self.included.append(("labels/judge_labels.csv", f"from {os.path.relpath(path, self.root)} ({len(judge)} items)"))

    # ---------------------------------------------------------------- walk
    def classify(self, path: str, rel: str) -> str | None:
        """None if the file can be exported as it is; otherwise the reason to skip it."""
        name = os.path.basename(rel)
        low = name.lower()
        if any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
            return "text or cache folder"
        if name in SKIP_FILES:
            return SKIP_FILES[name]
        if re.match(r"^(etiquetar|etiquetadas|adjudicar)", low):
            return "labeling sheet with texts (the labels are released in data/labels of the repository)"
        if low.endswith(".npy"):
            return self.check_npy(path, name)
        if low.endswith(".json"):
            return self.check_json(path, rel)
        if low.endswith(".jsonl"):
            return self.check_jsonl(path, rel)
        if low.endswith(".csv"):
            return self.check_csv(path, rel)
        if low.endswith(".txt"):
            return self.check_txt(path, rel)
        if low.endswith(".md"):
            return self.check_md(path, rel)
        if low.endswith((".pt", ".pth", ".bin", ".safetensors", ".ckpt")):
            return "model checkpoint"
        if low.endswith((".xlsx", ".xls", ".xlsm")):
            return "spreadsheet"
        if low.endswith((".npz", ".parquet", ".pkl", ".pickle", ".arrow", ".feather")):
            return "binary data container"
        if low in ("done",) or low.endswith((".done", ".lock", ".tmp")):
            return "marker file"
        return "file type not on the allow list"

    def run(self):
        if os.path.exists(self.out):
            shutil.rmtree(self.out)
        os.makedirs(self.out)
        n_texts = self.load_study_texts()
        present = [s for s in SECTIONS if os.path.isdir(os.path.join(self.root, s))]
        if not present:
            raise SystemExit(f"{self.root}: none of {SECTIONS} found. Point --root to the folder that holds them "
                             "(for the study, /content/drive/MyDrive/CLM_losses/colab).")
        for sec in present:
            for dirpath, dirs, files in os.walk(os.path.join(self.root, sec)):
                dirs.sort()
                for fn in sorted(files):
                    p = os.path.join(dirpath, fn)
                    rel = os.path.relpath(p, self.root).replace(os.sep, "/")
                    if fn.startswith("juez_raw") and fn.endswith(".jsonl"):
                        if fn == "juez_raw.jsonl":
                            self.export_judge(p)
                        self.excluded.append((rel, "judge's raw output (exported as labels/judge_labels.csv)"))
                        continue
                    why = self.classify(p, rel)
                    if why:
                        self.excluded.append((rel, why))
                        continue
                    dst = os.path.join(self.out, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(p, dst)
                    self.included.append((rel, "copied"))
        self.write_readme()
        self.write_manifest()
        self.write_report(n_texts)
        return self.write_archive()

    def write_archive(self) -> list:
        """<out>.tar.xz (files in name order, without owner names), its parts and <out>.SHA256SUMS. Returns the paths
        of the parts."""
        base = self.out.rstrip("/")
        top = os.path.basename(base)
        tpath = base + ".tar.xz"

        def anonymous(ti):
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            return ti

        with tarfile.open(tpath, "w:xz", preset=9) as tf:
            for dirpath, dirs, files in os.walk(self.out):
                dirs.sort()
                for fn in sorted(files):
                    p = os.path.join(dirpath, fn)
                    tf.add(p, arcname=f"{top}/{os.path.relpath(p, self.out).replace(os.sep, '/')}", filter=anonymous)
        for old in [f for f in os.listdir(os.path.dirname(tpath)) if f.startswith(os.path.basename(tpath) + ".")]:
            os.remove(os.path.join(os.path.dirname(tpath), old))
        parts = []
        with open(tpath, "rb") as f:
            for k, chunk in enumerate(iter(lambda: f.read(PART_BYTES), b""), 1):
                parts.append(f"{tpath}.{k:03d}")
                with open(parts[-1], "wb") as g:
                    g.write(chunk)

        def sha(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            return h.hexdigest()

        with open(base + ".SHA256SUMS", "w", encoding="utf-8") as f:
            for path in [tpath] + parts:
                f.write(f"{sha(path)}  {os.path.basename(path)}\n")
        return parts

    # ---------------------------------------------------------------- outputs
    def write_manifest(self):
        files = []
        for dirpath, dirs, fns in os.walk(self.out):
            dirs.sort()
            for fn in sorted(fns):
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, self.out).replace(os.sep, "/")
                if rel == "MANIFEST.json":
                    continue
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                files.append({"path": rel, "bytes": os.path.getsize(p), "sha256": h.hexdigest()})
        json.dump({"created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                   "root": self.root, "files": files},
                  open(os.path.join(self.out, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)

    def write_report(self, n_texts: int):
        by_reason: dict = {}
        excluded = [(public_path(rel), why) for rel, why in self.excluded]
        for rel, why in excluded:
            by_reason.setdefault(why, []).append(rel)
        size = sum(os.path.getsize(os.path.join(self.out, r)) for r, _ in self.included
                   if os.path.exists(os.path.join(self.out, r)))
        lines = ["# Export report", "",
                 f"- Root: `{self.root}`",
                 f"- Exported: {len(self.included)} files ({size / 1e6:.1f} MB before compression)",
                 f"- Skipped: {len(self.excluded)} files",
                 f"- Study texts used for the overlap check: {n_texts} strings from "
                 f"{len(self.text_sources())} files (results/PRE/textos and the T-gen panel)" + (
                     "" if n_texts else " (no text files found: the overlap check did not run)"),
                 "", "## Skipped, by reason", ""]
        for why, rels in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"- **{why}**: {len(rels)} files, e.g. " + ", ".join(f"`{r}`" for r in rels[:3]))
        lines += ["", "The full list is in `EXCLUDED.csv`. Review the skipped JSON and CSV files: a result file that "
                  "should be public may have been skipped only because it holds a long note.", ""]
        open(os.path.join(self.out, "REPORT.md"), "w", encoding="utf-8").write("\n".join(lines))
        with open(os.path.join(self.out, "EXCLUDED.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "reason"])
            w.writerows(excluded)

    def write_readme(self):
        text = """# Measuring the Generator, Not the State: data release (text-free)

Exported from the study's working folder with `release/export_release_data.py` of the code repository.
The layout mirrors that folder: `config/` (study configuration, protocol freeze and addendum), `data/` (document
identifiers of each split, item metadata and candidate indices of the evaluation sets), `runs/` (per run:
configuration, summary, training log, metrics and per-item results), `results/` (result files of every phase and of
the post hoc analyses) and `labels/` (the judge's false-negative labels).

No text is included: neither the question-answer pairs of Nemotron-CC v2.1, whose agreement forbids redistribution,
nor CLM's Gemini negatives, nor the negatives written by the panel generators. Items are identified by their
position in each evaluation set and, through `data/eval/*/c*_meta.json`, by the `doc_id` and `pair_idx` of
Nemotron-CC v2.1 High-Quality-DQA as distributed by CLM.

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). See `REPORT.md` for what was exported and skipped.
"""
        open(os.path.join(self.out, "README.md"), "w", encoding="utf-8").write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="study folder with config/, data/, runs/, results/ and logs/")
    ap.add_argument("--out", required=True,
                    help="output folder (replaced if it exists); the archive and its parts are written next to it")
    ap.add_argument("--no-scores", action="store_true", help="leave out the per-item score arrays (scores_*.npy)")
    a = ap.parse_args(argv)
    ex = Exporter(os.path.abspath(a.root), os.path.abspath(a.out), with_scores=not a.no_scores)
    parts = ex.run()
    print(f"exported {len(ex.included)} files, skipped {len(ex.excluded)}; see {os.path.join(a.out, 'REPORT.md')}")
    print(f"archive: {a.out.rstrip('/')}.tar.xz in {len(parts)} parts: " + ", ".join(os.path.basename(p) for p in parts))


if __name__ == "__main__":
    sys.exit(main())
