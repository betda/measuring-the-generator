"""F0: compuerta antes de cualquier comparación. Cada paso guarda su resultado en
Drive (<perfil>/f0/F0.x_*/result.json) y, si ya existe, no se repite."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from .config import (CLM_HEAD_REPO, CLM_STAGE_HEADS_REPO, DEEPSWE_HEADS_REPO, HN_REPO, PRETRAIN_REPO,
                     TOKENS_REPO)
from .data import (EvalSet, HFDownloader, Split, build_split, check_alignment, extract_eval_set, hn_name,
                   pre_name, slug, train_masks)
from .config import recipe_overrides
from .evaluate import build_mine_pool, build_pool_indices, cos_matrix, geometry, quick_val, rank_metrics
from .heads import heads_from_clm_checkpoint, load_torch
from .losses import BlindScorer
from .pipeline import base_cfg, extra_sets, load_bundle, make_store, run_one
from .utils import atomic_torch_save, load_json, makedirs, now_iso, rng, save_json

# Umbrales de las compuertas (P4): fijarlos ANTES de correr F0.
GATES = {
    "G1_cos_median": 0.999, "G1_cos_p01": 0.995,
    "G2_tol_pp": 3.0, "G2_ref": {"pre": 0.521, "mid": 0.692, "start_peak": 0.624},
    "G3_bands": (0.15, 0.30),
    "G4_rankme_frac": 0.05, "G4_rand_margin": 0.02,
    "G5_ref_agreement": 0.658, "G5_tol": 0.05,
}


def _out(study, name: str) -> str:
    return makedirs(study.paths.d("f0", name))


def _cached(path: str):
    return load_json(path) if os.path.exists(path) else None


# --------------------------------------------------------------------------------------
def f0_1_checkpoints(study, log=print) -> dict:
    """Carga los cabezales públicos: cfg, parámetros, logit_scale, sha256."""
    out = _out(study, "F0.1_checkpoints")
    res_p = os.path.join(out, "result.json")
    if (r := _cached(res_p)) is not None:
        return r
    from huggingface_hub import hf_hub_download
    targets = [(CLM_HEAD_REPO, "CLM_v0.1-8B.pt"), (CLM_STAGE_HEADS_REPO, "qwen3_8b_pretrained_head.pt"),
               (CLM_STAGE_HEADS_REPO, "qwen3_8b_midtrained_head.pt"),
               (CLM_STAGE_HEADS_REPO, "qwen3_8b_posttrained_head.pt"), (DEEPSWE_HEADS_REPO, "best_head.pt")]
    res = {"heads": {}}
    for repo, fn in targets:
        path = hf_hub_download(repo, fn, local_dir=study.paths.c("hf", slug(repo)))
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        entry = {"path": path, "bytes": os.path.getsize(path), "sha256": h.hexdigest()}
        try:
            ck = load_torch(path)
            heads = heads_from_clm_checkpoint(ck)
            entry.update(cfg_in_file=ck.get("cfg"), cfg_effective=heads.cfg, n_params=heads.n_params(),
                         logit_scale=float(heads.logit_scale), scale=float(heads.scale()))
        except Exception as e:
            entry["error"] = str(e)
        res["heads"][f"{repo}/{fn}"] = entry
        log(f"[F0.1] {repo}/{fn}: {entry.get('cfg_effective')} params={entry.get('n_params')}")
    ours = study.train
    ref = res["heads"].get(f"{CLM_HEAD_REPO}/CLM_v0.1-8B.pt", {}).get("cfg_effective") or {}
    res["matches_our_arch"] = bool(ref) and all(ref.get(k) == v for k, v in (
        ("width", ours.width), ("depth", ours.depth), ("projection_dim", ours.proj), ("layernorm", ours.layernorm)))
    save_json(res, res_p)
    return res


def f0_2_manifest(study, log=print) -> dict:
    """Tamaños y sha256 (según HF) de cada archivo que usará el perfil + licencias conocidas."""
    out = _out(study, "F0.2_manifest")
    res_p = os.path.join(out, "DATA_MANIFEST.json")
    if (r := _cached(res_p)) is not None:
        return r
    d = study.data
    dl = HFDownloader()
    pre = [pre_name(d.shard, c, k) for c in sorted(set(d.pre_chunks) | set(d.hn_chunks)) for k in ("q", "a", "meta")]
    hn = [hn_name(d.shard, c, k) for c in d.hn_chunks for k in ("n", "ncnt", "nmeta")]
    from .texts import NEG_PARTS, QA_PARTS
    man = {"created": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
           "repos": {PRETRAIN_REPO: dl.remote_info(PRETRAIN_REPO, pre), HN_REPO: dl.remote_info(HN_REPO, hn),
                     TOKENS_REPO: dl.remote_info(TOKENS_REPO, QA_PARTS + NEG_PARTS)},
           "licenses_known": {
               PRETRAIN_REPO: "sin licencia declarada (2026-09-30)",
               HN_REPO: "sin verificar", TOKENS_REPO: "nvidia-open-data-license; dice regirse por Nemotron-CC-v2.1 (gated)",
               "nvidia/Nemotron-CC-v2.1": "acuerdo NVIDIA para entrenamiento de modelos; no redistribuir"},
           "note": "No redistribuir datos derivados de Nemotron sin revisar la licencia [AUTHOR DECISION NEEDED]."}
    total = sum((v.get("size") or 0) for repo in man["repos"].values() for v in repo.values())
    man["total_gb_listed"] = round(total / 1e9, 1)
    save_json(man, res_p)
    log(f"[F0.2] manifiesto: {man['total_gb_listed']} GB listados")
    return man


def f0_download(study, log=print) -> dict:
    from .pipeline import resources
    r = resources(study)
    log(f"[descarga] {r}")
    if not r["alcanza"] and not study.data.synthetic:
        raise RuntimeError(f"no hay disco suficiente: {r}")
    store = make_store(study)
    store.ensure(sorted(set(study.data.pre_chunks) | set(study.data.hn_chunks)), study.data.hn_chunks, log=log)
    return r


def f0_3_embedding_equivalence(study, n: int = 2000, n_neg_items: int = 100, backend: str = "hf", log=print) -> dict:
    """G1: nuestro pipeline de embeddings reproduce los embeddings públicos (q, a y el orden de los HN)."""
    out = _out(study, "F0.3_embeddings")
    res_p = os.path.join(out, "result.json")
    if (r := _cached(res_p)) is not None:
        return r
    from .encoder import make_encoder
    from .texts import lookup
    store = make_store(study)
    c = study.data.hn_chunks[0]
    ids, pidx = store.meta(c)
    rows = np.sort(rng(7, c).choice(len(ids), size=min(n, len(ids)), replace=False))
    docs = set(ids[rows].tolist())
    qa = {(r["doc_id"], r["pair_idx"]): r for r in lookup(study.paths, docs, "qa",
                                                          ["doc_id", "pair_idx", "question_ids", "answer_ids"])}
    keep = [r for r in rows if (ids[r], int(pidx[r])) in qa]
    enc = make_encoder(backend)
    Eq = enc.embed_ids([qa[(ids[r], int(pidx[r]))]["question_ids"] for r in keep])
    Ea = enc.embed_ids([qa[(ids[r], int(pidx[r]))]["answer_ids"] for r in keep])

    def cmp(E, ref):
        ref = np.asarray(ref, np.float32)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12
        cs = (E * ref).sum(1)
        return {"median": float(np.median(cs)), "p01": float(np.quantile(cs, 0.01)), "min": float(cs.min()),
                "mean": float(cs.mean())}
    res = {"backend": backend, "n": len(keep), "chunk": int(c),
           "q": cmp(Eq, store.q(c)[keep]), "a": cmp(Ea, store.a(c)[keep])}
    # orden de los negativos: ¿el k-ésimo embedding corresponde a neg_idx = k?
    sub = [r for r in keep[:n_neg_items] if store.ncnt(c)[r] >= 10]
    negs = {}
    for r in lookup(study.paths, {ids[x] for x in sub}, "neg", ["doc_id", "pair_idx", "neg_idx", "neg_ids"]):
        negs.setdefault((r["doc_id"], r["pair_idx"]), {})[r["neg_idx"]] = r["neg_ids"]
    same, best = [], []
    for r in sub:
        m = negs.get((ids[r], int(pidx[r])), {})
        if len(m) < 10:
            continue
        E = enc.embed_ids([m[j] for j in sorted(m)][:10])
        ref = np.asarray(store.n(c)[r, :10], np.float32)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12
        S = E @ ref.T
        same.append(float(np.diag(S).mean()))
        best.append(float(S.max(1).mean()))
    res["neg_order"] = {"items": len(same), "diag_cos_mean": float(np.mean(same)) if same else None,
                        "best_match_cos_mean": float(np.mean(best)) if best else None}
    ok = all(res[k]["median"] >= GATES["G1_cos_median"] and res[k]["p01"] >= GATES["G1_cos_p01"] for k in ("q", "a"))
    res["G1_pass"] = bool(ok)
    save_json(res, res_p)
    log(f"[F0.3] G1={'PASA' if ok else 'NO PASA'} q={res['q']} a={res['a']} negs={res['neg_order']}")
    return res


def g1_recheck(study, enc, n: int = 256, n_neg_items: int = 20, lookup_fn=None, log=print) -> dict:
    """Re-verifica G1 con el encoder y el entorno ACTUALES antes de embeber negativos nuevos (celda 6.2).
    F0.3 se corrió una vez; si después cambió el entorno (p. ej. transformers ≥ 5.5 para Gemma 4), los negativos
    del panel podrían quedar en un espacio levemente distinto al de los embeddings públicos del oro, y eso sería
    un atajo. Mismo criterio que la 6.2 aplica a F0.3: respuestas con coseno mediano ≥ 0.999 y p01 ≥ 0.995, y
    orden de los HN con coseno de la diagonal ≥ 0.999. Informa además cuántas respuestas públicas sobreviven
    exactas a decodificar y volver a tokenizar (así entran los negativos del panel). Guarda el resultado."""
    import importlib.metadata as md
    if lookup_fn is None:
        from .texts import lookup as lookup_fn
    store = make_store(study)
    c = study.data.hn_chunks[0]
    ids, pidx = store.meta(c)
    rows = np.sort(rng(8, c).choice(len(ids), size=min(n, len(ids)), replace=False))
    qa = {(r["doc_id"], int(r["pair_idx"])): r
          for r in lookup_fn(study.paths, set(ids[rows].tolist()), "qa", ["doc_id", "pair_idx", "answer_ids"])}
    keep = [r for r in rows if (ids[r], int(pidx[r])) in qa]
    a_ids = [list(qa[(ids[r], int(pidx[r]))]["answer_ids"]) for r in keep]

    def unit(E):
        E = np.asarray(E, np.float32)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    cs = (unit(enc.embed_ids(a_ids)) * unit(store.a(c)[keep])).sum(1)
    a = {"median": float(np.median(cs)), "p01": float(np.quantile(cs, 0.01)), "min": float(cs.min()), "n": len(keep)}
    sub = [r for r in keep[:n_neg_items] if store.ncnt(c)[r] >= 10]
    negs = {}
    for r in lookup_fn(study.paths, {ids[x] for x in sub}, "neg", ["doc_id", "pair_idx", "neg_idx", "neg_ids"]):
        negs.setdefault((r["doc_id"], int(r["pair_idx"])), {})[int(r["neg_idx"])] = r["neg_ids"]
    diag = []
    for r in sub:
        m = negs.get((ids[r], int(pidx[r])), {})
        if len(m) >= 10:
            S = unit(enc.embed_ids([list(m[j]) for j in sorted(m)][:10])) @ unit(store.n(c)[r, :10]).T
            diag.append(float(np.diag(S).mean()))
    rt = None
    tok = getattr(enc, "tok", None)
    if tok is not None and hasattr(enc, "tokenize"):
        cap = getattr(enc, "max_len", None) or 10 ** 9
        back = enc.tokenize([tok.decode(x) for x in a_ids], keep="head")
        rt = float(np.mean([list(b) == list(x)[:cap] for b, x in zip(back, a_ids)]))
    env = {}
    for pkg in ("transformers", "torch", "tokenizers"):
        try:
            env[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            env[pkg] = None
    res = {"created": now_iso(), "env": env, "a": a, "neg_items": len(diag),
           "neg_diag_cos_mean": float(np.mean(diag)) if diag else None, "roundtrip_exact": rt}
    res["ok"] = bool(a["median"] >= GATES["G1_cos_median"] and a["p01"] >= GATES["G1_cos_p01"]
                     and res["neg_diag_cos_mean"] is not None and res["neg_diag_cos_mean"] >= 0.999)
    save_json(res, os.path.join(_out(study, "F0.3_embeddings"), f"recheck_transformers_{env.get('transformers')}.json"))
    log(f"[G1 de nuevo] transformers {env.get('transformers')}: respuestas mediana {a['median']:.5f} p01 {a['p01']:.5f}; "
        f"orden de HN {res['neg_diag_cos_mean']}; tokenización ida y vuelta exacta {rt} -> "
        f"{'OK' if res['ok'] else 'NO PASA'}")
    return res


def f0_4_splits(study, log=print) -> dict:
    out = _out(study, "F0.4_splits")
    res_p = os.path.join(out, "result.json")
    if (r := _cached(res_p)) is not None:
        return r
    store = make_store(study)
    d = study.data
    sdir = study.paths.d("data", "splits")
    split = build_split(store, d, sdir, log=log)
    align = [check_alignment(store, c) for c in d.hn_chunks]
    mp = train_masks(store, split, d.pre_chunks, sdir, "pre")
    mm = train_masks(store, split, d.hn_chunks, sdir, "mid")
    # fuga: ningún documento de test/val puede quedar en filas de entrenamiento
    hold = split.holdout()
    leaks = 0
    for c in d.pre_chunks:
        ids, _ = store.meta(c)
        leaks += int(sum(1 for i in np.nonzero(mp[c])[0] if ids[i] in hold))
    res = {"test_docs": len(split.test_docs), "val_docs": len(split.val_docs),
           "train_rows_pre": int(sum(m.sum() for m in mp.values())),
           "train_rows_mid": int(sum(m.sum() for m in mm.values())), "alignment": align,
           "all_aligned": all(a["aligned"] for a in align), "leaked_rows": leaks}
    save_json(res, res_p)
    log(f"[F0.4] {res}")
    return res


def f0_5_eval_sets(study, lengths: bool = True, qc: bool = True, log=print) -> dict:
    out = _out(study, "F0.5_eval_sets")
    res_p = os.path.join(out, "result.json")
    if (r := _cached(res_p)) is not None:
        return r
    store = make_store(study)
    d, p = study.data, study.paths
    split = Split.load(p.d("data", "splits"))
    tdir, vdir, cdir = p.d("data", "eval", "test"), p.d("data", "eval", "val"), p.d("data", "eval", "confirm")
    mt = extract_eval_set(store, split, "test", d.hn_chunks, tdir, k=d.k_neg, max_items=d.n_test_max,
                          max_per_doc=d.max_per_doc, log=log)
    mv = extract_eval_set(store, split, "val", d.hn_chunks, vdir, k=d.k_neg, max_items=d.n_val,
                          max_per_doc=d.max_per_doc, log=log)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test = EvalSet(tdir)
    build_pool_indices(test, tdir, study.eval.n_rand, study.eval.n_mine, study.eval.mine_fn_threshold, device=dev)
    res = {"test": {"items": test.n, "docs": int(len(set(test.doc_ids.tolist()))), **mt},
           "val": {"items": EvalSet(vdir).n, **mv}}
    if d.confirm_frac > 0:
        # Test confirmatorio: se extrae ahora (los chunks de HN están en disco) pero queda SELLADO:
        # load_bundle(test_split="confirm") exige PROTOCOL_FREEZE.json. Aquí sólo se cuentan ítems.
        mc = extract_eval_set(store, split, "confirm", d.hn_chunks, cdir, k=d.k_neg, max_items=d.n_confirm_max,
                              max_per_doc=d.max_per_doc, log=log)
        conf = EvalSet(cdir)
        build_pool_indices(conf, cdir, study.eval.n_rand, study.eval.n_mine, study.eval.mine_fn_threshold,
                           device=dev)
        res["confirm_sealed"] = {"items": conf.n, **mc}
        del conf
    sims = np.load(os.path.join(tdir, "mine_sims.npy"))
    res["tmine_cos_to_gold"] = {"mean": float(sims.mean()), "p90": float(np.quantile(sims, 0.9))}
    if lengths and not d.synthetic:
        from .texts import eval_lengths, export_qc_sample
        eval_lengths(p, test, os.path.join(tdir, "lengths.npz"), log=log)
        eval_lengths(p, EvalSet(vdir), os.path.join(vdir, "lengths.npz"), log=log)
        res["lengths"] = True
        if qc:
            res["qc_csv"] = export_qc_sample(p, test, os.path.join(out, "QC_negativos_gemini.csv"))
    save_json(res, res_p)
    log(f"[F0.5] test={res['test']['items']} val={res['val']['items']}")
    return res


def f0_5b_pools(study, log=print) -> dict:
    """F0.5b (enmienda A1): pool de val-mine (selección, A1.2) y T-mine-50 (secundaria) en test y confirm.
    Mismo procedimiento que T-mine: espacio crudo del encoder, otros documentos, coseno ≤ 0.95. No evalúa
    ningún modelo (no abre el test confirmatorio)."""
    from .evaluate import raw_cosine_scores
    out = _out(study, "F0.5b_pools")
    res_p = os.path.join(out, "result.json")
    if (r := _cached(res_p)) is not None:
        return r
    p, e = study.paths, study.eval
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"n_mine": e.n_mine, "n_mine_hard": e.n_mine_hard, "fn_threshold": e.mine_fn_threshold}
    vdir = p.d("data", "eval", "val")
    val = EvalSet(vdir)
    I = build_mine_pool(val, os.path.join(vdir, "mine_idx.npy"), e.n_mine, e.mine_fn_threshold, dev,
                        os.path.join(vdir, "mine_sims.npy"))
    ps = os.path.join(vdir, "mine_sims.npy")
    sims = np.load(ps) if os.path.exists(ps) else None
    res["val_mine"] = {"items": int(val.n), "cos_mean": None if sims is None else float(sims.mean()),
                       "raw_cosine_top1": rank_metrics(raw_cosine_scores(val, "pool", pool_idx=I))[0]["top1"]}
    for split in ("test", "confirm"):
        d = p.d("data", "eval", split)
        if not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        ev = EvalSet(d)
        I = build_mine_pool(ev, os.path.join(d, "mine50_idx.npy"), e.n_mine_hard, e.mine_fn_threshold, dev,
                            os.path.join(d, "mine50_sims.npy"))
        ps = os.path.join(d, "mine50_sims.npy")
        sims = np.load(ps) if os.path.exists(ps) else None
        res[f"{split}_mine50"] = {"items": int(ev.n), "cos_mean": None if sims is None else float(sims.mean())}
        if split == "test":    # línea base sin entrenamiento (no se calcula en el confirmatorio sellado)
            res["test_mine50"]["raw_cosine_top1"] = rank_metrics(raw_cosine_scores(ev, "pool", pool_idx=I))[0]["top1"]
        del ev
    save_json(res, res_p)
    log(f"[F0.5b] val-mine: {res['val_mine']} · T-mine-50: {res.get('test_mine50')}")
    return res


# --------------------------------------------------------------------------------------
def _cands(q, a, n):
    return np.concatenate([a[:, None, :], n], 1)


def train_blind(bundle, out_dir: str, epochs: float = 1.0, batch: int = 512, lr: float = 1e-3,
                seed: int = 0, min_steps: int = 300, log=print) -> BlindScorer:
    """Modelo ciego: sólo ve las acciones (oro + HN), nunca el estado."""
    from .data import Stream
    p = os.path.join(out_dir, "blind.pt")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = BlindScorer(bundle.hidden).to(dev)
    if os.path.exists(p):
        model.load_state_dict(torch.load(p, map_location=dev))
        return model.eval()
    d = bundle.dcfg
    n_rows = int(sum(m.sum() for m in bundle.masks_mid.values()))
    batch = int(min(batch, max(32, n_rows // 40)))
    st = Stream(bundle.store, d.hn_chunks, bundle.masks_mid, batch, d.mid_block_rows, d.mid_group_blocks,
                seed=seed + 11, with_hn=True, k=d.k_neg)
    total = max(min_steps, int(epochs * st.epoch_batches(0)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    step, ep = 0, 0
    while step < total:
        for b in st.iterate(ep):
            c = torch.from_numpy(_cands(None, b["ea"], b["en"])).float().to(dev)
            K = c.shape[1] - 1
            valid = torch.cat([torch.ones(len(c), 1, dtype=torch.bool),
                               torch.arange(K)[None] < torch.from_numpy(b["ncnt"]).long()[:, None]], 1).to(dev)
            s = model(c).masked_fill(~valid, -1e9)
            loss = torch.nn.functional.cross_entropy(s, torch.zeros(len(c), dtype=torch.long, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                log(f"[blind] {step}/{total} loss={float(loss.detach()):.4f}")
            if step >= total:
                break
        ep += 1
    atomic_torch_save(model.state_dict(), p)
    return model.eval()


@torch.no_grad()
def blind_top1(model, evalset, cand_fn=None, bs: int = 1024) -> np.ndarray:
    dev = next(model.parameters()).device
    out = []
    for sel, q, a, n in evalset.iter_batches(bs):
        c = _cands(q, a, n) if cand_fn is None else cand_fn(sel, a)
        s = model(torch.from_numpy(c).float().to(dev)).cpu().numpy()
        out.append(s)
    S = np.concatenate(out)
    return rank_metrics(S)[1]


def f0_6_shortcuts(study, bundle=None, log=print) -> dict:
    """G3: ¿se puede acertar sin mirar el estado? Modelo ciego + heurística de longitud."""
    out = _out(study, "F0.6_shortcuts")
    res_p = os.path.join(out, "result.json")
    if (r := _cached(res_p)) is not None:
        return r
    bundle = bundle or load_bundle(study)
    model = train_blind(bundle, out, log=log)
    test = bundle.test
    res = {"chance": 1 / (1 + study.data.k_neg), "blind": {}}
    res["blind"]["tgem"] = float(blind_top1(model, test).mean())
    A = test.a_all()
    for name, fn in (("trand", "rand_idx.npy"), ("tmine", "mine_idx.npy")):
        pth = os.path.join(bundle.pools_dir, fn)
        if os.path.exists(pth):
            I = np.load(pth)
            res["blind"][name] = float(blind_top1(model, test, cand_fn=lambda sel, a: np.concatenate(
                [a[:, None, :], A[I[sel]]], 1)).mean())
    ex = extra_sets(bundle)
    if "tgen2" in ex:
        items, negs = ex["tgen2"]
        with torch.no_grad():
            dev = next(model.parameters()).device
            c = np.concatenate([A[items][:, None, :], np.asarray(negs)], 1)
            S = np.concatenate([model(torch.from_numpy(c[s:s + 1024]).float().to(dev)).cpu().numpy()
                                for s in range(0, len(c), 1024)])
        res["blind"]["tgen2"] = float(rank_metrics(S)[1].mean())
    # línea base sin entrenamiento: coseno crudo del encoder (cuánto aportan los cabezales)
    from .evaluate import raw_cosine_scores
    res["raw_cosine"] = {"tgem": rank_metrics(raw_cosine_scores(test, "tgem"))[0]["top1"]}
    for name, fn in (("trand", "rand_idx.npy"), ("tmine", "mine_idx.npy")):
        pth = os.path.join(bundle.pools_dir, fn)
        if os.path.exists(pth):
            res["raw_cosine"][name] = rank_metrics(raw_cosine_scores(test, "pool", pool_idx=np.load(pth)))[0]["top1"]
    if "tgen2" in ex:
        items, negs = ex["tgen2"]
        res["raw_cosine"]["tgen2"] = rank_metrics(raw_cosine_scores(test, "external", items=items,
                                                                    negs=np.asarray(negs)))[0]["top1"]
    lp_t = os.path.join(study.paths.d("data", "eval", "test"), "lengths.npz")
    lp_v = os.path.join(study.paths.d("data", "eval", "val"), "lengths.npz")
    if os.path.exists(lp_t) and os.path.exists(lp_v):
        from .texts import length_heuristic
        zt, zv = np.load(lp_t), np.load(lp_v)
        val_acc = {r: float(np.nanmean(length_heuristic(zv["gold"], zv["neg"], r))) for r in ("longest", "shortest")}
        rule = max(val_acc, key=val_acc.get)
        res["length_heuristic"] = {"val": val_acc, "rule": rule,
                                   "test": float(np.nanmean(length_heuristic(zt["gold"], zt["neg"], rule)))}
    lo, hi = GATES["G3_bands"]
    b = res["blind"]["tgem"]
    res["G3"] = "despreciable" if b < lo else ("moderado" if b < hi else "ALTO: cambiar métrica primaria")
    res["G3_pass"] = b < hi
    save_json(res, res_p)
    log(f"[F0.6] ciego={res['blind']} coseno_crudo={res['raw_cosine']} longitud={res.get('length_heuristic')} "
        f"-> G3 {res['G3']}")
    return res


# --------------------------------------------------------------------------------------
REPRO_VARIANTS = {
    "infonce_pre": dict(arm="infonce", stages=["pre"]),
    "infonce_hn_hnonly": dict(arm="infonce_hn", loss_params={"hn_denominator": "hn_only"}),
    "infonce_hn_inbatch": dict(arm="infonce_hn", loss_params={"hn_denominator": "hn_plus_inbatch"}),
    "infonce_hn_start": dict(arm="infonce_hn", hn_from_start=True, loss_params={"hn_denominator": "hn_only"}),
}


def f0_7_reproduction(study, bundle=None, seeds=(0, 1), variants=None, log=print) -> dict:
    """G2: reproducir la receta de CLM (pre / pre→mid / HN desde el inicio)."""
    out = _out(study, "F0.7_reproduction")
    bundle = bundle or load_bundle(study)
    res = load_json(os.path.join(out, "result.json"), {}) or {}
    res.setdefault("runs", {})
    for name, kw in (variants or REPRO_VARIANTS).items():
        for s in seeds:
            key = f"{name}/seed_{s}"
            if key in res["runs"]:
                continue
            kw2 = dict(kw)
            cfg = base_cfg(study, kw2.pop("arm"), seed=s, **kw2)
            r = run_one(study, bundle, cfg, "F0.7", f"{name}/seed_{s}")
            hist = []
            for stg in r.get("stages", {}):
                sj = load_json(study.paths.d("runs", "F0.7", name, f"seed_{s}", f"stage_{stg}.json"), {}) or {}
                hist += [h.get("val_top1") for h in sj.get("history", []) if not h.get("init")]
            res["runs"][key] = {"tgem_top1": r.get("test", {}).get("tgem", {}).get("top1"),
                                "val_peak": max([h for h in hist if h is not None], default=None),
                                "val_final": hist[-1] if hist else None}
            save_json(res, os.path.join(out, "result.json"))
    # cabezales públicos de CLM en tgem (referencia CONTAMINADA: vieron estos pares)
    res.setdefault("clm_public_contaminated", {})
    ck1 = load_json(study.paths.d("f0", "F0.1_checkpoints", "result.json"), {}) or {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for k, v in (ck1.get("heads") or {}).items():
        if k in res["clm_public_contaminated"] or "deepswe" in k or "path" not in v:
            continue
        try:
            h = heads_from_clm_checkpoint(load_torch(v["path"]), dev)
            C, _ = cos_matrix(h, bundle.test, dev)
            res["clm_public_contaminated"][k] = rank_metrics(C)[0]["top1"]
        except Exception as e:
            res["clm_public_contaminated"][k] = f"error: {e}"
    # G2
    def mean_of(prefix, field):
        xs = [v[field] for kk, v in res["runs"].items() if kk.startswith(prefix + "/") and v.get(field) is not None]
        return float(np.mean(xs)) if xs else None
    pre, mid_a, mid_b = mean_of("infonce_pre", "tgem_top1"), mean_of("infonce_hn_hnonly", "tgem_top1"), \
        mean_of("infonce_hn_inbatch", "tgem_top1")
    start_peak = mean_of("infonce_hn_start", "val_peak")
    mid = max(x for x in (mid_a, mid_b) if x is not None) if (mid_a or mid_b) else None
    tol = GATES["G2_tol_pp"] / 100
    ref = GATES["G2_ref"]
    full_scale = study.data.profile in ("full",)
    res["G2"] = {"pre": pre, "mid_hnonly": mid_a, "mid_inbatch": mid_b, "start_peak_val": start_peak,
                 "ordering_ok": None if None in (pre, mid, start_peak) else bool(mid > start_peak > pre)}
    if full_scale and None not in (pre, mid):
        res["G2"]["abs_ok"] = bool(abs(pre - ref["pre"]) <= tol and abs(mid - ref["mid"]) <= tol)
    else:
        res["G2"]["abs_ok"] = None
        res["G2"]["nota"] = "perfil reducido: G2 sólo cualitativo (orden mid > pico de HN desde el inicio > pre)"
    res["G2_pass"] = bool(res["G2"]["ordering_ok"]) if res["G2"]["abs_ok"] is None else bool(
        res["G2"]["ordering_ok"] and res["G2"]["abs_ok"])
    save_json(res, os.path.join(out, "result.json"))
    log(f"[F0.7] G2={res['G2']}")
    return res


# --------------------------------------------------------------------------------------
# F0.7b (enmienda A1.5): receta actual vs receta real de CLM, con selección por val-mine y "HN + in-batch"
F07B_CONDITIONS = {
    "pre": dict(arm="infonce", stages=["pre"]),
    "mid_inbatch": dict(arm="infonce_hn", stages=["mid"], init="pre", loss_params={"hn_denominator": "hn_plus_inbatch"}),
    "start_inbatch": dict(arm="infonce_hn", hn_from_start=True, loss_params={"hn_denominator": "hn_plus_inbatch"}),
    "mid_hnonly": dict(arm="infonce_hn", stages=["mid"], init="pre", loss_params={"hn_denominator": "hn_only"}),
}
G2B_CONDITIONS = ("pre", "mid_inbatch", "start_inbatch")     # las que entran en la regla (mid_hnonly no)
F07B_RECIPES = ("clm", "actual")                              # primero la de CLM: es la que decide la regla


def _f07b_record(study, rd: str, r: dict) -> dict:
    t = r.get("test", {}) or {}

    def g(k, f="top1"):
        return (t.get(k) or {}).get(f)
    st = r.get("stages", {}) or {}
    return {"status": r.get("status"), "tmine": g("tmine"), "tmine_shuf": g("tmine_shuf"), "tgem": g("tgem"),
            "tgem_shuf": g("tgem_shuf"), "trand": g("trand"), "tmine50": g("tmine50"),
            "abst_tmine": (t.get("tabst_mine") or {}).get("abst_auroc_maxcos"), "val_best": r.get("val_best"),
            "steps": {k: v.get("steps") for k, v in st.items()}, "best_step": {k: v.get("best_step") for k, v in st.items()},
            "train_seconds": r.get("train_seconds"), "run_dir": rd}


def g2b(study, res: dict, recipe: str, seeds, bundle=None) -> dict:
    """G2b: orden de CLM (mid_inbatch > start_inbatch > pre) en top-1 de T-mine (test), media de semillas.
    Requiere todas las semillas de las tres condiciones; si no, ordering_ok = None (sin decisión)."""
    from .stats import clusters_from_docs, contrast
    means, complete = {}, True
    for c in G2B_CONDITIONS:
        xs = [(res["runs"].get(f"{recipe}/{c}/seed_{s}") or {}).get("tmine") for s in seeds]
        if any(x is None for x in xs):
            complete = False
            means[c] = float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
        else:
            means[c] = float(np.mean(xs))
    out = {"recipe": recipe, "tmine_mean": means, "complete": complete,
           "ordering_ok": (bool(means["mid_inbatch"] > means["start_inbatch"] > means["pre"]) if complete else None)}
    if complete and bundle is not None:
        cl = clusters_from_docs(bundle.test.doc_ids)

        def items_of(c):
            return np.vstack([np.load(os.path.join(res["runs"][f"{recipe}/{c}/seed_{s}"]["run_dir"], "eval",
                                                   "correct_tmine.npy")) for s in seeds]).astype(np.float32)
        try:
            M = {c: items_of(c) for c in G2B_CONDITIONS}
            out["contrasts_tmine"] = {}
            for a, b in (("mid_inbatch", "start_inbatch"), ("start_inbatch", "pre"), ("mid_inbatch", "pre")):
                cc = contrast(M[a], M[b], cl, study.rope_pp / 100, B=2000)
                out["contrasts_tmine"][f"{a} − {b}"] = {k: cc[k] for k in ("diff", "ci95_lo", "ci95_hi", "p",
                                                                         "tost_equivalent")}
        except FileNotFoundError:
            pass
    return out


def f0_7b_recipes(study, bundle=None, recipes=F07B_RECIPES, seeds=(0, 1), conditions=None, log=print) -> dict:
    """A1.5: la receta real de CLM (batch 16,384, LR constante) frente a la actual, con selección por val-mine
    y "HN + in-batch". Los mid parten del checkpoint de pre de la misma semilla y receta. La regla
    (pre-especificada, ENMIENDA_03 §4): F1b usa la receta de CLM si y sólo si G2b se cumple con ella."""
    out = _out(study, "F0.7b_recipes")
    res_p = os.path.join(out, "result.json")
    bundle = bundle or load_bundle(study)
    if getattr(bundle, "val_mine_idx", None) is None:
        raise RuntimeError("Falta el pool de val-mine: corre F0.5b (celda 3.6b) antes de F0.7b.")
    res = load_json(res_p, {}) or {}
    res.setdefault("runs", {})
    conds = list(conditions or F07B_CONDITIONS)
    order = [c for c in G2B_CONDITIONS if c in conds] + [c for c in conds if c not in G2B_CONDITIONS]
    for recipe in recipes:
        rov = recipe_overrides(study, recipe)
        for cond in order:
            for s in seeds:
                key = f"{recipe}/{cond}/seed_{s}"
                if (res["runs"].get(key) or {}).get("status") == "ok":
                    continue
                kw = dict(F07B_CONDITIONS[cond])
                arm, init = kw.pop("arm"), kw.pop("init", None)
                cfg = base_cfg(study, arm, seed=s, select_metric="valmine_top1", **rov, **kw)
                if init:
                    pre_dir = study.paths.d("runs", "F0.7b", recipe, init, f"seed_{s}")
                    if (load_json(os.path.join(pre_dir, "DONE.json")) or {}).get("status") != "ok":
                        log(f"[F0.7b] {key}: falta la condición '{init}' de esta semilla; se omite por ahora")
                        continue
                    cfg.init_from = pre_dir
                log(f"[F0.7b] {key}")
                r = run_one(study, bundle, cfg, "F0.7b", f"{recipe}/{cond}/seed_{s}")
                res["runs"][key] = _f07b_record(study, study.paths.d("runs", "F0.7b", recipe, cond, f"seed_{s}"), r)
                save_json(res, res_p)
    res["G2b"] = {r: g2b(study, res, r, seeds, bundle) for r in recipes}
    res["seeds"] = list(seeds)
    save_json(res, res_p)
    decide_recipe(study, res, log=log)
    return res


def decide_recipe(study, res: dict | None = None, log=print) -> dict | None:
    """Aplica la regla pre-especificada y la registra en config/RECETA_F1b.json (una sola vez)."""
    from .utils import code_hash
    res = res or load_json(study.paths.d("f0", "F0.7b_recipes", "result.json"))
    if not res or "G2b" not in res:
        return None
    gc = res["G2b"].get("clm") or {}
    if gc.get("ordering_ok") is None:
        log("[F0.7b] sin decisión de receta: faltan corridas de la receta de CLM (pre, mid_inbatch, start_inbatch × semillas)")
        return None
    receta = "clm" if gc["ordering_ok"] else "actual"
    p = study.paths.d("config", "RECETA_F1b.json")
    old = load_json(p)
    dec = {"receta": receta, "fecha": now_iso(), "code_hash": code_hash(),
           "regla": "F1b usa la receta de CLM si y sólo si G2b (mid_inbatch > start_inbatch > pre en top-1 de "
                    "T-mine, media de semillas) se cumple con ella (ENMIENDA_03 §4)",
           "G2b_clm": gc, "G2b_actual": res["G2b"].get("actual")}
    if old is not None and old.get("receta") != receta:
        log(f"[F0.7b] ATENCIÓN: la decisión registrada ({old.get('receta')}) no coincide con la que sale ahora "
            f"({receta}). No se sobrescribe; revisa qué corridas cambiaron.")
        return old
    if old is None:
        save_json(dec, p)
        log(f"[F0.7b] receta para F1b: {receta} (G2b con la receta de CLM: {gc.get('tmine_mean')})")
        return dec
    return old


COLLAPSE_ARMS = ["simsiam_x", "byol_x", "barlow_x", "vicreg_x", "mse_sigreg", "align_unif", "dino_x", "swav_x"]


def f0_8_collapse(study, bundle=None, arms=None, aug_levels=(0.0, 0.5), row_fraction: float = 0.2,
                  max_steps: int | None = 300, log=print) -> dict:
    """G4: distinguir colapso por bug de colapso del método (corridas cortas, con/sin aumentación)."""
    out = _out(study, "F0.8_collapse")
    res_p = os.path.join(out, "result.json")
    res = load_json(res_p, {}) or {}
    bundle = bundle or load_bundle(study)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chance = 1 / (1 + study.data.k_neg)
    for arm in (arms or COLLAPSE_ARMS):
        for aug in aug_levels:
            key = f"{arm}|aug={aug}"
            if key in res:
                continue
            cfg = base_cfg(study, arm, stages=["pre"], max_steps_pre=max_steps, aug_noise_std=aug, save_final=True)
            name = f"{arm}_aug{aug}".replace(".", "p")
            r = run_one(study, bundle, cfg, "F0.8", name, evaluate=False, row_fraction=row_fraction)
            if r.get("status") != "ok":
                res[key] = {"status": r.get("status"), "error": r.get("error"), "collapsed": True}
                save_json(res, res_p)
                continue
            from .evaluate import cos_with_pool
            from .train import Trainer
            rdir = study.paths.d("runs", "F0.8", name)
            tr = Trainer(cfg, bundle, rdir, row_fraction=row_fraction)
            # el colapso se diagnostica en el estado FINAL (no en el mejor checkpoint de validación)
            st = torch.load(os.path.join(rdir, "final_pre.pt"), map_location=tr.device, weights_only=False)
            tr.heads.load_state_dict(st["heads"])
            g = geometry(tr.heads, bundle.val, dev, n=min(2000, bundle.val.n))
            v = quick_val(tr.heads, bundle.val, dev)
            nv = bundle.val.n
            I = rng(3, nv).integers(0, nv - 1, size=(nv, 10))
            I = I + (I >= np.arange(nv)[:, None])          # nunca el propio ítem
            Crand = cos_with_pool(tr.heads, bundle.val, I, dev)
            rand_top1 = float(rank_metrics(Crand)[1].mean())
            rk_min = GATES["G4_rankme_frac"] * study.train.proj
            collapsed = g["rankme_zs"] < rk_min or rand_top1 <= chance + GATES["G4_rand_margin"]
            sj = load_json(os.path.join(rdir, "stage_pre.json"), {}) or {}
            res[key] = {"status": "ok", "steps": sj.get("steps"), "rankme_zs": g["rankme_zs"],
                        "rankme_za": g["rankme_za"], "rankme_min": rk_min, "val_top1_final": v["val_top1"],
                        "val_top1_best": sj.get("best"), "val_top1_init": sj.get("init_val"),
                        "rand_top1_final": rand_top1, "collapsed": bool(collapsed)}
            save_json(res, res_p)
            log(f"[F0.8] {key}: {res[key]}")
            for fn in os.listdir(study.paths.d("runs", "F0.8", name)):
                if fn.endswith(".pt"):
                    os.remove(study.paths.d("runs", "F0.8", name, fn))
    return res


def f0_9_trex(study, log=print) -> dict:
    out = _out(study, "F0.9_trex")
    res_p = os.path.join(out, "result.json")
    if (r := _cached(res_p)) is not None:
        return r
    from huggingface_hub import hf_hub_download
    from .trex import run_trex, setup_clm_repo
    repo = setup_clm_repo()
    head = hf_hub_download(CLM_HEAD_REPO, "CLM_v0.1-8B.pt", local_dir=study.paths.c("hf", slug(CLM_HEAD_REPO)))
    r = run_trex(head, out, repo=repo, log=log)
    ag = (r.get("shield") or {}).get("mean_agreement_with_planner")
    ok = ag is not None and abs(ag - GATES["G5_ref_agreement"]) <= GATES["G5_tol"] and \
        (r["shield"].get("survived") == 5) and (r["shield"].get("host_stall_seconds", 0) < 1.0)
    res = {**r, "G5_pass": bool(ok), "referencia_repo": {"shield_agreement": GATES["G5_ref_agreement"]}}
    save_json(res, res_p)
    return res


def f0_10_deepswe(study, log=print) -> dict:
    from .deepswe import f0_deepswe
    from .trex import setup_clm_repo
    return f0_deepswe(study.paths, repo=setup_clm_repo(), log=log)


def f0_report(study, log=print) -> dict:
    """Tabla de compuertas G1–G6 a partir de los resultados guardados."""
    p = study.paths.d("f0")
    g = {}
    r3 = load_json(os.path.join(p, "F0.3_embeddings", "result.json"))
    g["G1 pipeline de embeddings"] = None if r3 is None else r3.get("G1_pass")
    r7 = load_json(os.path.join(p, "F0.7_reproduction", "result.json"))
    g["G2 reproducción InfoNCE"] = None if r7 is None else r7.get("G2_pass")
    r6 = load_json(os.path.join(p, "F0.6_shortcuts", "result.json"))
    g["G3 atajos (modelo ciego)"] = None if r6 is None else r6.get("G3_pass")
    r8 = load_json(os.path.join(p, "F0.8_collapse", "result.json"))
    g["G4 colapso explicado"] = None if not r8 else not all(v.get("collapsed") for v in r8.values())
    r9 = load_json(os.path.join(p, "F0.9_trex", "result.json"))
    g["G5 arnés T-Rex"] = None if r9 is None else r9.get("G5_pass")
    r10 = load_json(os.path.join(p, "F0.10_deepswe", "result.json"))
    g["G6 identidad DeepSWE"] = None if r10 is None else r10.get("G6_pass")
    r7b = load_json(os.path.join(p, "F0.7b_recipes", "result.json"))
    g["G2b orden de CLM en T-mine, receta CLM (A1.5; no condiciona F1b, decide su receta)"] = \
        None if r7b is None else ((r7b.get("G2b") or {}).get("clm") or {}).get("ordering_ok")
    r4 = load_json(os.path.join(p, "F0.4_splits", "result.json"))
    extra = {"splits_sin_fuga": None if r4 is None else (r4.get("leaked_rows") == 0 and r4.get("all_aligned")),
             "pools de val-mine y T-mine-50 (F0.5b)": None if load_json(os.path.join(p, "F0.5b_pools", "result.json")) is None else True}
    dec = load_json(study.paths.d("config", "RECETA_F1b.json"))
    lines = ["# F0 — reporte de compuertas", "", "| Compuerta | Estado |", "|---|---|"]
    for k, v in {**g, **extra}.items():
        lines.append(f"| {k} | {'PENDIENTE' if v is None else ('PASA' if v else 'NO PASA')} |")
    lines.append(f"| Receta de F1b (regla de A1.5) | {dec['receta'] if dec else 'PENDIENTE'} |")
    lines += ["", "G4 'PASA' = al menos una configuración no colapsa; revisar el detalle por brazo en F0.8.",
              "G2b es la G2 medida en T-mine (enmienda A1.5): no detiene nada; decide la receta de F1b.",
              "Una compuerta que no pasa se registra y se diagnostica; no se cambia el umbral después de ver el resultado."]
    md = "\n".join(lines)
    with open(os.path.join(p, "F0_REPORT.md"), "w") as f:
        f.write(md)
    out = {"gates": g, **extra, "receta_F1b": dec["receta"] if dec else None}
    save_json(out, os.path.join(p, "F0_REPORT.json"))
    log(md)
    return out
