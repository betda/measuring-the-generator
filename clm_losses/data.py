"""Datos: stores de embeddings con el layout de Hugging Face, split por documento,
sets de evaluación en Drive y streams de entrenamiento reanudables.

Layout verificado (2026-09-30):
    Contrastive-LM/CLM-v0.1-Pretrain-Nemotron/shard_00/chunk_NNNN_{q,a}.npy   fp16 [100000, 4096]
                                                   /chunk_NNNN_meta.parquet   doc_id, pair_idx
    jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings/shard_00/chunk_NNNN_n.npy    fp16 [100000, 10, 4096]
                                                  /chunk_NNNN_ncnt.npy         uint8 [100000]
                                                  /chunk_NNNN_nmeta.parquet    doc_id, pair_idx
    La fila i de chunk_NNNN_n.npy corresponde a la fila i de chunk_NNNN_{q,a}.npy (NNNN = 0000..0029).
    Todas las filas están ordenadas por (doc_id, pair_idx); los embeddings están normalizados L2.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass, field

import numpy as np

from .utils import atomic_np_save, load_json, makedirs, rng, save_json


def slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def pre_name(shard: str, c: int, kind: str) -> str:
    return f"{shard}/chunk_{c:04d}_{kind}" + (".parquet" if kind == "meta" else ".npy")


def hn_name(shard: str, c: int, kind: str) -> str:
    suffix = {"n": "n.npy", "ncnt": "ncnt.npy", "nmeta": "nmeta.parquet"}[kind]
    return f"{shard}/chunk_{c:04d}_{suffix}"


def read_meta(path: str) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["doc_id", "pair_idx"])
    return (np.asarray(t.column("doc_id").to_pylist(), dtype=object),
            t.column("pair_idx").to_numpy().astype(np.int32))


def write_meta(path: str, doc_ids, pair_idx) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    makedirs(os.path.dirname(path))
    tmp = path + ".tmp"
    pq.write_table(pa.table({"doc_id": pa.array(list(doc_ids), pa.string()),
                             "pair_idx": pa.array(np.asarray(pair_idx, np.uint16), pa.uint16())}), tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
class HFDownloader:
    """Descarga archivos sueltos de datasets de HF a la caché local (reanudable)."""

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("HF_TOKEN")

    def fetch(self, repo_id: str, filename: str, local_dir: str, repo_type: str = "dataset") -> str:
        target = os.path.join(local_dir, filename)
        if os.path.exists(target):
            return target
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type,
                               local_dir=local_dir, token=self.token)

    def remote_info(self, repo_id: str, filenames: list[str], repo_type: str = "dataset") -> dict:
        """Tamaño y sha256 (LFS) según la API de HF, para el manifiesto de datos."""
        from huggingface_hub import HfApi
        out = {}
        api = HfApi(token=self.token)
        for i in range(0, len(filenames), 50):
            for info in api.get_paths_info(repo_id, filenames[i:i + 50], repo_type=repo_type, expand=True):
                lfs = getattr(info, "lfs", None)
                out[info.path] = {"size": getattr(info, "size", None),
                                  "sha256": getattr(lfs, "sha256", None) if lfs else None}
        return out


class Store:
    """Acceso perezoso a los chunks. Si falta un archivo y hay `downloader`, lo descarga."""

    def __init__(self, pre_root: str, hn_root: str, shard: str = "shard_00",
                 downloader: HFDownloader | None = None, pre_repo: str | None = None,
                 hn_repo: str | None = None):
        self.pre_root, self.hn_root, self.shard = pre_root, hn_root, shard
        self.downloader, self.pre_repo, self.hn_repo = downloader, pre_repo, hn_repo
        self._cache: dict = {}

    # rutas --------------------------------------------------------------------------
    def path(self, which: str, c: int, kind: str) -> str:
        if which == "pre":
            root, name, repo = self.pre_root, pre_name(self.shard, c, kind), self.pre_repo
        else:
            root, name, repo = self.hn_root, hn_name(self.shard, c, kind), self.hn_repo
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
        if self.downloader is None or repo is None:
            raise FileNotFoundError(p)
        return self.downloader.fetch(repo, name, root)

    def has(self, which: str, c: int, kind: str) -> bool:
        name = pre_name(self.shard, c, kind) if which == "pre" else hn_name(self.shard, c, kind)
        return os.path.exists(os.path.join(self.pre_root if which == "pre" else self.hn_root, name))

    def ensure(self, pre_chunks, hn_chunks, log=print) -> None:
        for c in pre_chunks:
            for kind in ("meta", "q", "a"):
                if not self.has("pre", c, kind):
                    log(f"[data] descargando pre chunk {c:04d} {kind}")
                self.path("pre", c, kind)
        for c in hn_chunks:
            for kind in ("nmeta", "ncnt", "n"):
                if not self.has("hn", c, kind):
                    log(f"[data] descargando HN chunk {c:04d} {kind}")
                self.path("hn", c, kind)

    # arrays ---------------------------------------------------------------------------
    def _mm(self, which: str, c: int, kind: str):
        key = (which, c, kind)
        if key not in self._cache:
            self._cache[key] = np.load(self.path(which, c, kind), mmap_mode="r")
        return self._cache[key]

    def q(self, c: int):
        return self._mm("pre", c, "q")

    def a(self, c: int):
        return self._mm("pre", c, "a")

    def n(self, c: int):
        return self._mm("hn", c, "n")

    def ncnt(self, c: int) -> np.ndarray:
        key = ("hn", c, "ncnt_full")
        if key not in self._cache:
            self._cache[key] = np.asarray(np.load(self.path("hn", c, "ncnt")))
        return self._cache[key]

    def meta(self, c: int) -> tuple[np.ndarray, np.ndarray]:
        key = ("pre", c, "meta_full")
        if key not in self._cache:
            self._cache[key] = read_meta(self.path("pre", c, "meta"))
        return self._cache[key]

    def nmeta(self, c: int) -> tuple[np.ndarray, np.ndarray]:
        key = ("hn", c, "nmeta_full")
        if key not in self._cache:
            self._cache[key] = read_meta(self.path("hn", c, "nmeta"))
        return self._cache[key]

    def nrows(self, c: int) -> int:
        return int(self.q(c).shape[0])

    @property
    def hidden(self) -> int:
        for key, v in self._cache.items():
            if key[2] in ("q", "a"):
                return int(v.shape[1])
        raise RuntimeError("abre algún chunk antes de consultar hidden")

    def release(self) -> None:
        self._cache.clear()


def check_alignment(store: Store, c: int) -> dict:
    """Comprueba que meta (q/a) y nmeta (HN) coinciden fila a fila."""
    d1, p1 = store.meta(c)
    d2, p2 = store.nmeta(c)
    ok = len(d1) == len(d2) and bool(np.all(d1 == d2)) and bool(np.all(p1 == p2))
    return {"chunk": c, "rows": int(len(d1)), "aligned": ok,
            "zero_neg_rows": int((store.ncnt(c) == 0).sum()),
            "full_neg_rows": int((store.ncnt(c) >= 10).sum())}


# --------------------------------------------------------------------------------------
# Split por documento
def doc_bucket(doc_id: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{doc_id}".encode()).hexdigest()
    return int(h[:13], 16) / float(16 ** 13)


@dataclass
class Split:
    salt: str
    test_frac: float
    val_frac: float
    hn_chunks: list
    test_docs: set
    val_docs: set
    confirm_frac: float = 0.0
    confirm_docs: set = field(default_factory=set)

    def role(self, doc_id: str) -> str:
        if doc_id in self.test_docs:
            return "test"
        if doc_id in self.val_docs:
            return "val"
        if doc_id in self.confirm_docs:
            return "confirm"
        return "train"

    def docs_of(self, which: str) -> set:
        return {"test": self.test_docs, "val": self.val_docs, "confirm": self.confirm_docs}[which]

    def holdout(self) -> set:
        return self.test_docs | self.val_docs | self.confirm_docs

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for name, docs in (("test", self.test_docs), ("val", self.val_docs), ("confirm", self.confirm_docs)):
            h.update(name.encode())
            for d in sorted(docs):
                h.update(d.encode())
        return h.hexdigest()[:10]

    def save(self, d: str) -> None:
        makedirs(d)
        for name, docs in (("test_docs.txt", self.test_docs), ("val_docs.txt", self.val_docs),
                           ("confirm_docs.txt", self.confirm_docs)):
            tmp = os.path.join(d, name + ".tmp")
            with open(tmp, "w") as f:
                f.write("\n".join(sorted(docs)))
            os.replace(tmp, os.path.join(d, name))
        save_json({"salt": self.salt, "test_frac": self.test_frac, "val_frac": self.val_frac,
                   "confirm_frac": self.confirm_frac, "hn_chunks": list(self.hn_chunks),
                   "n_test_docs": len(self.test_docs), "n_val_docs": len(self.val_docs),
                   "n_confirm_docs": len(self.confirm_docs)}, os.path.join(d, "split.json"))

    @classmethod
    def load(cls, d: str) -> "Split":
        meta = load_json(os.path.join(d, "split.json"))
        if meta is None:
            raise FileNotFoundError(os.path.join(d, "split.json"))

        def rd(name):
            p = os.path.join(d, name)
            if not os.path.exists(p):
                return set()
            with open(p) as f:
                return {x for x in f.read().split("\n") if x}
        return cls(meta["salt"], meta["test_frac"], meta["val_frac"], meta["hn_chunks"],
                   rd("test_docs.txt"), rd("val_docs.txt"), meta.get("confirm_frac", 0.0),
                   rd("confirm_docs.txt"))


def build_split(store: Store, dcfg, out_dir: str, log=print) -> Split:
    """Sólo los documentos de hn_chunks son elegibles para test/val/confirm (necesitan HN).
    Asignación determinista por hash del doc_id: [0, test) test, [test, test+val) val,
    [test+val, test+val+confirm) confirm; el resto, entrenamiento."""
    if os.path.exists(os.path.join(out_dir, "split.json")):
        sp = Split.load(out_dir)
        same = (sp.salt == dcfg.split_salt and abs(sp.test_frac - dcfg.test_frac) < 1e-12
                and abs(sp.val_frac - dcfg.val_frac) < 1e-12
                and abs(sp.confirm_frac - dcfg.confirm_frac) < 1e-12
                and list(sp.hn_chunks) == list(dcfg.hn_chunks))
        if not same:
            raise RuntimeError(f"{out_dir} tiene un split con otros parámetros; bórralo a mano si quieres rehacerlo")
        log(f"[split] reutilizado: {len(sp.test_docs)} docs test, {len(sp.val_docs)} val, "
            f"{len(sp.confirm_docs)} confirm")
        return sp
    eligible = set()
    for c in dcfg.hn_chunks:
        ids, _ = store.meta(c)
        eligible.update(ids.tolist())
    test, val, conf = set(), set(), set()
    t1 = dcfg.test_frac
    t2 = t1 + dcfg.val_frac
    t3 = t2 + dcfg.confirm_frac
    for d in eligible:
        b = doc_bucket(d, dcfg.split_salt)
        if b < t1:
            test.add(d)
        elif b < t2:
            val.add(d)
        elif b < t3:
            conf.add(d)
    sp = Split(dcfg.split_salt, dcfg.test_frac, dcfg.val_frac, list(dcfg.hn_chunks), test, val,
               dcfg.confirm_frac, conf)
    sp.save(out_dir)
    log(f"[split] {len(eligible)} docs elegibles -> {len(test)} test, {len(val)} val, {len(conf)} confirm")
    return sp


def train_masks(store: Store, split: Split, chunks, out_dir: str, stage: str) -> dict:
    """Máscara de filas de entrenamiento por chunk (se guarda en Drive).
    stage='pre': excluye docs de test/val.  stage='mid': además exige >=1 HN."""
    makedirs(out_dir)
    hold = split.holdout()
    sid = split.fingerprint()
    out = {}
    for c in chunks:
        p = os.path.join(out_dir, f"mask_{stage}_c{c:04d}_{sid}.npy")   # atada al split exacto
        if os.path.exists(p):
            out[c] = np.load(p)
            continue
        ids, _ = store.meta(c)
        m = np.fromiter((d not in hold for d in ids), dtype=bool, count=len(ids))
        if stage == "mid":
            m &= store.ncnt(c) >= 1
        atomic_np_save(m, p)
        out[c] = m
    return out


# --------------------------------------------------------------------------------------
# Sets de evaluación (1 oro + K HN), guardados en Drive por chunk
def _cap_per_doc(docs: np.ndarray, max_per_doc: int | None, seed: int) -> np.ndarray:
    """Índices (ordenados) con a lo sumo max_per_doc ítems por documento, elegidos al azar."""
    n = len(docs)
    if max_per_doc is None or n == 0:
        return np.arange(n)
    rs = rng(seed, 515151)
    perm = rs.permutation(n)
    seen: dict = {}
    keep = []
    for i in perm:
        d = docs[i]
        c = seen.get(d, 0)
        if c < max_per_doc:
            seen[d] = c + 1
            keep.append(i)
    return np.sort(np.asarray(keep, np.int64))


def extract_eval_set(store: Store, split: Split, which: str, chunks, out_dir: str, k: int = 10,
                     max_items: int | None = None, seed: int = 0, delete_hn_after: bool = False,
                     keep_hn_chunks=(), max_per_doc: int | None = None, log=print) -> dict:
    makedirs(out_dir)
    docs = split.docs_of(which)
    for c in chunks:
        done = os.path.join(out_dir, f"c{c:04d}.done")
        if os.path.exists(done):
            continue
        ids, pidx = store.meta(c)
        ncnt = store.ncnt(c)
        sel = np.fromiter((d in docs for d in ids), dtype=bool, count=len(ids)) & (ncnt >= k)
        rows = np.nonzero(sel)[0]
        log(f"[eval:{which}] chunk {c:04d}: {len(rows)} ítems")
        atomic_np_save(np.asarray(store.q(c)[rows]), os.path.join(out_dir, f"c{c:04d}_q.npy"))
        atomic_np_save(np.asarray(store.a(c)[rows]), os.path.join(out_dir, f"c{c:04d}_a.npy"))
        atomic_np_save(np.asarray(store.n(c)[rows, :k]), os.path.join(out_dir, f"c{c:04d}_n.npy"))
        save_json({"doc_id": ids[rows].tolist(), "pair_idx": pidx[rows].tolist(), "row": rows.tolist()},
                  os.path.join(out_dir, f"c{c:04d}_meta.json"), indent=None)
        with open(done, "w") as f:
            f.write("ok")
        if delete_hn_after and c not in keep_hn_chunks:
            p = os.path.join(store.hn_root, hn_name(store.shard, c, "n"))
            store.release()
            if os.path.exists(p):
                os.remove(p)
    counts, all_docs = {}, []
    for c in chunks:
        ids_c = load_json(os.path.join(out_dir, f"c{c:04d}_meta.json"))["doc_id"]
        counts[int(c)] = len(ids_c)
        all_docs.extend(ids_c)
    total = sum(counts.values())
    all_docs = np.asarray(all_docs, dtype=object)
    sub_path = os.path.join(out_dir, "subsample.npy")
    if not os.path.exists(sub_path) and (max_per_doc is not None or (max_items is not None and total > max_items)):
        idx = _cap_per_doc(all_docs, max_per_doc, seed)
        if max_items is not None and len(idx) > max_items:
            idx = np.sort(rng(seed, 424242).choice(idx, size=max_items, replace=False))
        atomic_np_save(idx, sub_path)
    n_used = len(np.load(sub_path)) if os.path.exists(sub_path) else total
    man = {"which": which, "chunks": [int(c) for c in chunks], "counts": counts, "total": total,
           "k": k, "max_items": max_items, "max_per_doc": max_per_doc, "n_used": int(n_used),
           "n_docs": int(len(set(all_docs.tolist())))}
    save_json(man, os.path.join(out_dir, "manifest.json"))
    return man


class EvalSet:
    """Set de evaluación: q y a en RAM; negativos por lotes desde memmap."""

    def __init__(self, d: str, use_subsample: bool = True):
        self.dir = d
        man = load_json(os.path.join(d, "manifest.json"))
        if man is None:
            raise FileNotFoundError(f"{d}/manifest.json: construye el set primero (F0.5)")
        self.k = man["k"]
        qs, as_, docs, pidx, part, local = [], [], [], [], [], []
        self._n = []
        for j, c in enumerate(man["chunks"]):
            meta = load_json(os.path.join(d, f"c{c:04d}_meta.json"))
            m = len(meta["row"])
            qs.append(np.load(os.path.join(d, f"c{c:04d}_q.npy"), mmap_mode="r"))
            as_.append(np.load(os.path.join(d, f"c{c:04d}_a.npy"), mmap_mode="r"))
            self._n.append(np.load(os.path.join(d, f"c{c:04d}_n.npy"), mmap_mode="r"))
            docs.extend(meta["doc_id"])
            pidx.extend(meta["pair_idx"])
            part.append(np.full(m, j, np.int32))
            local.append(np.arange(m, dtype=np.int64))
        part = np.concatenate(part) if part else np.zeros(0, np.int32)
        local = np.concatenate(local) if local else np.zeros(0, np.int64)
        sub = os.path.join(d, "subsample.npy")
        idx = np.load(sub) if (use_subsample and os.path.exists(sub)) else np.arange(len(part))
        self.part, self.local = part[idx], local[idx]
        self.doc_ids = np.asarray(docs, dtype=object)[idx]
        self.pair_idx = np.asarray(pidx, dtype=np.int32)[idx]
        self.n = len(idx)
        self._q_parts, self._a_parts = qs, as_
        self._q = self._a = None

    def _gather(self, arrs, sel: np.ndarray) -> np.ndarray:
        parts, locs = self.part[sel], self.local[sel]
        first = arrs[0]
        out = np.empty((len(sel),) + first.shape[1:], dtype=first.dtype)
        for j in np.unique(parts):
            w = np.nonzero(parts == j)[0]
            order = np.argsort(locs[w])
            rows = locs[w][order]
            out[w[order]] = np.asarray(arrs[j][rows])
        return out

    def q_all(self) -> np.ndarray:
        if self._q is None:
            self._q = self._gather(self._q_parts, np.arange(self.n))
        return self._q

    def a_all(self) -> np.ndarray:
        if self._a is None:
            self._a = self._gather(self._a_parts, np.arange(self.n))
        return self._a

    def negatives(self, sel: np.ndarray) -> np.ndarray:
        return self._gather(self._n, np.asarray(sel))

    def iter_batches(self, bs: int):
        q, a = self.q_all(), self.a_all()
        for s in range(0, self.n, bs):
            sel = np.arange(s, min(self.n, s + bs))
            yield sel, q[sel], a[sel], self.negatives(sel)


# --------------------------------------------------------------------------------------
# Streams de entrenamiento: bloques contiguos, barajados por época, reanudables
class Stream:
    """Plan determinista por (seed, epoch): se permutan bloques de filas contiguas, se agrupan,
    cada grupo se carga a RAM (con prefetch del siguiente), se baraja y se corta en batches
    (drop_last por grupo). `iterate(epoch, start_batch)` salta sin leer datos hasta el batch pedido."""

    def __init__(self, store: Store, chunks, masks: dict, batch: int, block_rows: int,
                 group_blocks: int, seed: int, with_hn: bool = False, k: int = 10,
                 row_fraction: float = 1.0):
        self.store, self.batch, self.seed, self.with_hn, self.k = store, batch, seed, with_hn, k
        self.group_blocks = group_blocks
        self.masks = {}
        for c in chunks:
            m = masks[c].copy()
            if row_fraction < 1.0:  # subconjunto fijo (HPO): mismas filas para todos los brazos
                keep = rng(4040, c).random(len(m)) < row_fraction
                m &= keep
            self.masks[c] = m
        self.blocks = []
        for c in chunks:
            nr = len(self.masks[c])
            for s in range(0, nr, block_rows):
                e = min(nr, s + block_rows)
                cnt = int(self.masks[c][s:e].sum())
                if cnt:
                    self.blocks.append((int(c), s, e, cnt))
        if not self.blocks:
            raise ValueError("el stream no tiene filas de entrenamiento")
        self.n_rows = sum(b[3] for b in self.blocks)
        # Batches grandes (receta de CLM, 16,384): con los grupos configurados un grupo podría no llenar
        # ni un batch (drop_last por grupo => épocas vacías). Si el promedio de batches por grupo es < 4,
        # se agrandan los grupos para tener ~8 batches cada uno. Con los batches del F1 piloto (2,048 / 512)
        # nunca se activa, así que el orden de los datos de las corridas previas no cambia.
        n_groups = -(-len(self.blocks) // self.group_blocks)
        if self.n_rows / max(1, n_groups) < 4 * self.batch:
            per_block = self.n_rows / len(self.blocks)
            self.group_blocks = int(min(len(self.blocks), max(self.group_blocks, -(-8 * self.batch // max(1, int(per_block))))))
        if self.n_rows < self.batch:
            raise ValueError(f"el stream tiene {self.n_rows} filas, menos que un batch ({self.batch})")
        self._pool = cf.ThreadPoolExecutor(max_workers=1)

    def groups(self, epoch: int) -> list:
        order = rng(self.seed, 7919, epoch).permutation(len(self.blocks))
        blocks = [self.blocks[i] for i in order]
        return [blocks[i:i + self.group_blocks] for i in range(0, len(blocks), self.group_blocks)]

    def _nb(self, group) -> int:
        return sum(b[3] for b in group) // self.batch

    def epoch_batches(self, epoch: int = 0) -> int:
        return sum(self._nb(g) for g in self.groups(epoch))

    def _load(self, group) -> dict:
        out = {"es": [], "ea": []}
        if self.with_hn:
            out["en"], out["ncnt"] = [], []
        for c, s, e, _ in group:
            rows = s + np.nonzero(self.masks[c][s:e])[0]
            out["es"].append(np.asarray(self.store.q(c)[rows]))
            out["ea"].append(np.asarray(self.store.a(c)[rows]))
            if self.with_hn:
                out["en"].append(np.asarray(self.store.n(c)[rows, :self.k]))
                out["ncnt"].append(np.asarray(self.store.ncnt(c)[rows]))
        return {key: np.concatenate(v) for key, v in out.items()}

    def iterate(self, epoch: int, start_batch: int = 0):
        groups = self.groups(epoch)
        todo = []
        skip = start_batch
        for gi, g in enumerate(groups):
            nb = self._nb(g)
            if nb == 0:
                continue
            if skip >= nb:
                skip -= nb
                continue
            todo.append((gi, g, skip, nb))
            skip = 0
        fut = self._pool.submit(self._load, todo[0][1]) if todo else None
        for j, (gi, g, first, nb) in enumerate(todo):
            data = fut.result()
            fut = self._pool.submit(self._load, todo[j + 1][1]) if j + 1 < len(todo) else None
            total = len(data["es"])
            perm = rng(self.seed, 104729, epoch, gi).permutation(total)[: nb * self.batch]
            for b in range(first, nb):
                sl = np.sort(perm[b * self.batch:(b + 1) * self.batch])
                yield {key: v[sl] for key, v in data.items()}


# --------------------------------------------------------------------------------------
# Datos sintéticos con el mismo layout (perfil "smoke" y tests)
def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def make_synthetic_store(pre_root: str, hn_root: str, shard: str = "shard_00", n_chunks: int = 4,
                         hn_chunks=(0, 1), rows_per_chunk: int = 3000, hidden: int = 128,
                         pairs_per_doc: int = 4, latent: int = 24, seed: int = 0, k: int = 10,
                         noise: float = 0.35, hardness: float = 0.55, artifact: float = 0.12,
                         zero_neg_frac: float = 0.045) -> dict:
    """Pares (estado, acción) con latente compartido; HN = latente perturbado + un sesgo fijo del
    "generador" (artifact) que un modelo ciego puede aprender. Se guardan los latentes para
    poder generar negativos de un "segundo generador" sin artefacto (T-gen2 sintético)."""
    done = os.path.join(pre_root, shard, "SYNTHETIC.json")
    want = {"n_chunks": n_chunks, "hn_chunks": list(hn_chunks), "rows_per_chunk": rows_per_chunk,
            "hidden": hidden, "latent": latent, "seed": seed, "k": k, "noise": noise,
            "hardness": hardness, "artifact": artifact, "pairs_per_doc": pairs_per_doc,
            "zero_neg_frac": zero_neg_frac}
    if os.path.exists(done):
        have = load_json(done)
        if all(have.get(key) == v for key, v in want.items()):
            return have
        # parámetros distintos: se regeneran (los datos sintéticos son baratos)
        for root in (pre_root, hn_root):
            shutil.rmtree(os.path.join(root, shard), ignore_errors=True)
    rs = np.random.default_rng(seed)
    Ws = rs.standard_normal((latent, hidden)) / np.sqrt(latent)
    Wa = rs.standard_normal((latent, hidden)) / np.sqrt(latent)
    v_art = _l2(rs.standard_normal(hidden))
    total = n_chunks * rows_per_chunk
    n_docs = int(np.ceil(total / pairs_per_doc))
    docs = sorted(str(uuid.UUID(bytes=rs.bytes(16), version=4)) for _ in range(n_docs))
    doc_of_row = np.repeat(np.arange(n_docs), pairs_per_doc)[:total]
    pair_of_row = np.tile(np.arange(pairs_per_doc), n_docs)[:total]
    for c in range(n_chunks):
        r0, r1 = c * rows_per_chunk, (c + 1) * rows_per_chunk
        u = rs.standard_normal((rows_per_chunk, latent)).astype(np.float32)
        q = _l2(u @ Ws + noise * rs.standard_normal((rows_per_chunk, hidden)))
        a = _l2(u @ Wa + noise * rs.standard_normal((rows_per_chunk, hidden)))
        d = os.path.join(pre_root, shard)
        makedirs(d)
        np.save(os.path.join(d, f"chunk_{c:04d}_q.npy"), q.astype(np.float16))
        np.save(os.path.join(d, f"chunk_{c:04d}_a.npy"), a.astype(np.float16))
        np.save(os.path.join(d, f"chunk_{c:04d}_latent.npy"), u)
        ids = [docs[i] for i in doc_of_row[r0:r1]]
        write_meta(os.path.join(d, f"chunk_{c:04d}_meta.parquet"), ids, pair_of_row[r0:r1])
        if c in hn_chunks:
            dn = os.path.join(hn_root, shard)
            makedirs(dn)
            delta = rs.standard_normal((rows_per_chunk, k, latent)).astype(np.float32)
            un = u[:, None, :] + hardness * delta
            n = _l2(un @ Wa + noise * rs.standard_normal((rows_per_chunk, k, hidden)) + artifact * v_art)
            ncnt = np.full(rows_per_chunk, k, np.uint8)
            z = rs.random(rows_per_chunk) < zero_neg_frac
            ncnt[z] = 0
            partial = (~z) & (rs.random(rows_per_chunk) < 0.03)
            ncnt[partial] = rs.integers(3, k, partial.sum())
            for i in np.nonzero(ncnt < k)[0]:
                n[i, ncnt[i]:] = 0.0
            np.save(os.path.join(dn, f"chunk_{c:04d}_n.npy"), n.astype(np.float16))
            np.save(os.path.join(dn, f"chunk_{c:04d}_ncnt.npy"), ncnt)
            write_meta(os.path.join(dn, f"chunk_{c:04d}_nmeta.parquet"), ids, pair_of_row[r0:r1])
    info = dict(want)
    np.save(os.path.join(pre_root, shard, "W_a.npy"), Wa)
    save_json(info, done)
    return info


def synthetic_gen2_negatives(pre_root: str, shard: str, test: EvalSet, k: int = 10, seed: int = 1,
                             noise: float | None = None, hardness: float | None = None) -> np.ndarray:
    """Negativos de un 'segundo generador' (sin artefacto) para los ítems de un EvalSet sintético.
    Por defecto usa la misma dificultad (hardness/noise) que el generador principal."""
    info = load_json(os.path.join(pre_root, shard, "SYNTHETIC.json"))
    noise = info["noise"] if noise is None else noise
    hardness = info["hardness"] if hardness is None else hardness
    Wa = np.load(os.path.join(pre_root, shard, "W_a.npy"))
    rs = np.random.default_rng(seed)
    man = load_json(os.path.join(test.dir, "manifest.json"))
    lat_rows = []
    for c in man["chunks"]:
        rows = load_json(os.path.join(test.dir, f"c{c:04d}_meta.json"))["row"]
        lat = np.load(os.path.join(pre_root, shard, f"chunk_{c:04d}_latent.npy"))
        lat_rows.append(lat[rows])
    lat = np.concatenate(lat_rows)
    sub = os.path.join(test.dir, "subsample.npy")
    if os.path.exists(sub):
        lat = lat[np.load(sub)]
    delta = rs.standard_normal((len(lat), k, info["latent"])).astype(np.float32)
    n = _l2((lat[:, None, :] + hardness * delta) @ Wa + noise * rs.standard_normal((len(lat), k, Wa.shape[1])))
    return n.astype(np.float16)
