"""Evaluación de un par de cabezales entrenado.

Conjuntos (todos con el oro en la columna 0):
    tgem        1 oro + 10 HN del generador original (Gemini)            -> top-1, MRR, calibración
    tabst       derivado de tgem: con oro (oro + 9 HN) vs sin oro (10 HN) -> AUROC/AUPRC de abstención
    trand       1 oro + 10 respuestas aleatorias de otros documentos
    tmine       1 oro + 10 respuestas minadas (más parecidas al oro, de otros documentos)  [PRIMARIA, A1.1]
    tabst_mine  abstención sobre T-mine: oro + 9 minadas vs 10 minadas      [abstención primaria, A1.3]
    tmine50     1 oro + 50 minadas (secundaria; techo de T-mine)
    tgen_<g>    1 oro + 10 HN del generador g del panel (con su propia abstención)
    tgen2       1 oro + 10 HN de un segundo generador (formato de v0.2; si existe)
Para cada conjunto también se mide el top-1 con ESTADOS BARAJADOS (<conjunto>_shuf o "shuf_top1"): cada
conjunto de candidatos se puntúa con el estado de otro ítem (permutación fija, A1.4).
Cada evaluación guarda el acierto por ítem para rehacer la estadística sin re-proyectar.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .heads import l2n
from .utils import atomic_np_save, load_json, save_json


# --------------------------------------------------------------------------------------
@torch.no_grad()
def project(module, E, device, bs: int = 8192) -> torch.Tensor:
    out = []
    for i in range(0, len(E), bs):
        x = torch.from_numpy(np.asarray(E[i:i + bs], dtype=np.float32)).to(device)
        out.append(module(x).float())
    return torch.cat(out) if out else torch.zeros(0, device=device)


@torch.no_grad()
def cos_matrix(heads, evalset, device, bs: int = 2048, alt=None,
               states: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Cosenos (N x (1+K)) oro + HN. `alt(zs, zc)` -> puntuación alternativa (predictor/prototipos).
    `states` (N x H) reemplaza a los estados de cada ítem (p. ej. barajados)."""
    heads.eval()
    rows, alt_rows = [], []
    for sel, q, a, n in evalset.iter_batches(bs):
        if states is not None:
            q = states[sel]
        b, K, H = n.shape
        zs = project(heads.state_head, q, device)
        cands = np.concatenate([a[:, None, :], n], 1).reshape(b * (K + 1), H)
        zc = project(heads.action_head, cands, device).view(b, K + 1, -1)
        rows.append(torch.einsum("bd,bkd->bk", l2n(zs), l2n(zc)).cpu().numpy())
        if alt is not None:
            alt_rows.append(alt(zs, zc).cpu().numpy())
    C = np.concatenate(rows).astype(np.float32)
    A = np.concatenate(alt_rows).astype(np.float32) if alt_rows else None
    return C, A


@torch.no_grad()
def cos_with_pool(heads, evalset, idx: np.ndarray, device, states: np.ndarray | None = None) -> np.ndarray:
    """Oro + candidatos tomados de las respuestas del propio set (índices N x k)."""
    heads.eval()
    zs = l2n(project(heads.state_head, evalset.q_all() if states is None else states, device))
    za = l2n(project(heads.action_head, evalset.a_all(), device))
    gold = (zs * za).sum(-1, keepdim=True)
    I = np.asarray(idx, np.int64)
    out = []
    for s in range(0, len(I), 4096):         # por bloques: con 50 candidatos la matriz entera no hace falta
        e = slice(s, s + 4096)
        neg = torch.einsum("nd,nkd->nk", zs[e], za[torch.from_numpy(I[e]).to(device)])
        out.append(torch.cat([gold[e], neg], 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def cos_with_external(heads, evalset, item_idx: np.ndarray, negs: np.ndarray, device, bs: int = 1024,
                      states: np.ndarray | None = None, pos: np.ndarray | None = None) -> np.ndarray:
    """Oro + negativos externos (panel de generadores) para un subconjunto de ítems. `states` (n x H),
    alineado con item_idx, reemplaza a los estados (p. ej. barajados). `pos` (n x H), alineado con item_idx,
    reemplaza al oro como candidato correcto (construcción P: el oro reescrito por el generador)."""
    heads.eval()
    item_idx = np.asarray(item_idx)
    q = evalset.q_all()[item_idx] if states is None else states
    a = evalset.a_all()[item_idx] if pos is None else np.asarray(pos, np.float32)
    out = []
    for s in range(0, len(item_idx), bs):
        e = slice(s, s + bs)
        zs = l2n(project(heads.state_head, q[e], device))
        nb = np.asarray(negs[e])
        b, K, H = nb.shape
        cands = np.concatenate([a[e][:, None, :], nb], 1).reshape(b * (K + 1), H)
        zc = l2n(project(heads.action_head, cands, device)).view(b, K + 1, -1)
        out.append(torch.einsum("bd,bkd->bk", zs, zc).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def shuffle_perm(n: int, seed: int = 0) -> np.ndarray:
    """Permutación de estados barajados: la misma de las celdas 5.4 y 5.5 (np.random.default_rng(0))."""
    return np.random.default_rng(seed).permutation(n)


def _l2np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def raw_cosine_scores(evalset, which: str = "tgem", pool_idx: np.ndarray | None = None,
                      items: np.ndarray | None = None, negs: np.ndarray | None = None, bs: int = 4096,
                      pos: np.ndarray | None = None) -> np.ndarray:
    """Línea base sin entrenamiento (clm-raw): coseno entre los embeddings crudos del encoder. En "external",
    `pos` (alineado con items) reemplaza al oro (panel T-gen con el oro reescrito, construcción P)."""
    q, a = _l2np(evalset.q_all()), _l2np(evalset.a_all())
    if which == "tgem":
        rows = []
        for s0 in range(0, evalset.n, bs):
            sel = np.arange(s0, min(evalset.n, s0 + bs))
            n = _l2np(evalset.negatives(sel))
            rows.append(np.concatenate([(q[sel] * a[sel]).sum(1, keepdims=True),
                                        np.einsum("bh,bkh->bk", q[sel], n)], 1))
        return np.concatenate(rows)
    if which == "pool":
        rows = []
        for s0 in range(0, evalset.n, 1024):
            sel = slice(s0, min(evalset.n, s0 + 1024))
            rows.append(np.concatenate([(q[sel] * a[sel]).sum(1, keepdims=True),
                                        np.einsum("nh,nkh->nk", q[sel], a[pool_idx[sel]])], 1))
        return np.concatenate(rows)
    if which == "external":
        qq, aa, nn = q[items], (a[items] if pos is None else _l2np(pos)), _l2np(negs)
        return np.concatenate([(qq * aa).sum(1, keepdims=True), np.einsum("nh,nkh->nk", qq, nn)], 1)
    raise ValueError(which)


# --------------------------------------------------------------------------------------
# Métricas a partir de la matriz de puntuaciones (oro en columna 0)
def rank_metrics(C: np.ndarray) -> dict:
    """top-1 y MRR ESPERADOS con desempate al azar: si el oro empata con t negativos (típico de un
    modelo colapsado), cuenta 1/(t+1). `correct` es por ítem (float en [0, 1])."""
    bad = ~np.isfinite(C).all(1)          # puntuaciones no finitas: el ítem cuenta como error
    gold, neg = C[:, 0], C[:, 1:]
    g = (neg > gold[:, None]).sum(1)
    t = (neg == gold[:, None]).sum(1)
    correct = np.where(g == 0, 1.0 / (1 + t), 0.0).astype(np.float32)
    rr = np.zeros(len(C))
    for tv in np.unique(t):
        m = t == tv
        j = np.arange(tv + 1)[None, :]
        rr[m] = (1.0 / (1 + g[m, None] + j)).mean(1)
    correct[bad], rr[bad] = 0.0, 0.0
    g, t = np.where(bad, C.shape[1] - 1, g), np.where(bad, 0, t)
    rank = g + t / 2.0
    with np.errstate(invalid="ignore"):
        margin = np.where(bad, np.nan, gold - neg.max(1))
    margin = np.nan_to_num(margin, nan=-2.0)
    return {"top1": float(correct.mean()), "mrr": float(rr.mean()), "tie_frac": float((t > 0).mean()),
            "nonfinite_frac": float(bad.mean()),
            "margin_mean": float(margin.mean()), "margin_median": float(np.median(margin)),
            "margin_p10": float(np.quantile(margin, 0.1)), "n": int(len(C))}, correct, rank


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC por rangos (Mann–Whitney), con empates promediados."""
    from scipy.stats import rankdata
    labels = labels.astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision con umbrales en los valores distintos (los empates no favorecen a nadie)."""
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order].astype(float)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]     # último índice de cada grupo de empate
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    P = max(1.0, y.sum())
    prec = tp / (tp + fp)
    rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def abstention_metrics(C: np.ndarray, abst_k: int = 9, T: float | None = None,
                       null_cos: np.ndarray | None = None) -> dict:
    """Con oro = columnas [0, 1..abst_k]; sin oro = columnas [1..abst_k+1]. Mismo tamaño."""
    K = C.shape[1] - 1
    if abst_k + 1 > K:
        raise ValueError("se necesitan al menos abst_k+1 negativos")
    with_g = C[:, : abst_k + 1]
    without = C[:, 1: abst_k + 2]
    labels = np.r_[np.ones(len(C)), np.zeros(len(C))]
    out = {}
    s = np.r_[with_g.max(1), without.max(1)]
    out["abst_auroc_maxcos"] = auroc(s, labels)
    out["abst_auprc_maxcos"] = auprc(s, labels)
    srt_w, srt_o = np.sort(with_g, 1), np.sort(without, 1)
    s = np.r_[srt_w[:, -1] - srt_w[:, -2], srt_o[:, -1] - srt_o[:, -2]]
    out["abst_auroc_margin"] = auroc(s, labels)
    if T is not None:
        def maxp(M):
            z = T * M
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            return (p / p.sum(1, keepdims=True)).max(1)
        s = np.r_[maxp(with_g), maxp(without)]
        out["abst_auroc_maxprob"] = auroc(s, labels)
    if null_cos is not None:
        s = np.r_[with_g.max(1) - null_cos, without.max(1) - null_cos]
        out["abst_auroc_null"] = auroc(s, labels)
    # riesgo selectivo sobre los conjuntos con oro
    conf = with_g.max(1)
    err = 1.0 - rank_metrics(with_g)[1]            # error esperado con desempate al azar
    order = np.argsort(-conf, kind="stable")
    risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    out["aurc"] = float(risk.mean())
    return out


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(C_val: np.ndarray, lo: float = 1.0, hi: float = 500.0) -> float:
    """Temperature scaling: escala que minimiza la NLL del oro en validación."""
    from scipy.optimize import minimize_scalar

    def nll(logT):
        p = softmax_np(math.exp(logT) * C_val)
        return -np.log(np.clip(p[:, 0], 1e-12, 1)).mean()
    r = minimize_scalar(nll, bounds=(math.log(lo), math.log(hi)), method="bounded")
    return float(math.exp(r.x))


def listwise_calibration(C: np.ndarray, T: float, bins: int = 15) -> dict:
    p = softmax_np(T * C)
    conf = p.max(1)
    correct = rank_metrics(C)[1].astype(float)     # acierto esperado (empates sin favorecer al oro)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    y = np.zeros_like(p)
    y[:, 0] = 1
    return {"ece": float(ece), "brier": float(((p - y) ** 2).sum(1).mean()),
            "nll": float(-np.log(np.clip(p[:, 0], 1e-12, 1)).mean()), "T": float(T)}


def fit_platt(C_val: np.ndarray) -> tuple[float, float]:
    """Probabilidad por par σ(a·cos + b) ajustada en validación (oro=1, HN=0), SIN pesos: queda
    calibrada para la proporción real (1 oro : K HN), que es la misma del test."""
    from scipy.optimize import minimize
    x = np.r_[C_val[:, 0], C_val[:, 1:].ravel()]
    y = np.r_[np.ones(len(C_val)), np.zeros(C_val[:, 1:].size)]
    w = np.full(len(y), 1.0 / len(y))

    def f(ab):
        z = ab[0] * x + ab[1]
        return float((w * (np.logaddexp(0, z) - y * z)).sum())
    r = minimize(f, x0=np.array([10.0, -5.0]), method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def pointwise_calibration(C: np.ndarray, a: float, b: float, bins: int = 15) -> dict:
    x = np.r_[C[:, 0], C[:, 1:].ravel()]
    y = np.r_[np.ones(len(C)), np.zeros(C[:, 1:].size)]
    p = 1 / (1 + np.exp(-(a * x + b)))
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return {"pw_ece": float(ece), "pw_brier": float(((p - y) ** 2).mean()), "pw_a": a, "pw_b": b}


# --------------------------------------------------------------------------------------
@torch.no_grad()
def geometry(heads, evalset, device, n: int = 5000, hub_k: int = 10, seed: int = 0) -> dict:
    heads.eval()
    rs = np.random.default_rng(seed)
    sel = np.sort(rs.choice(evalset.n, size=min(n, evalset.n), replace=False))
    zs = project(heads.state_head, evalset.q_all()[sel], device)
    za = project(heads.action_head, evalset.a_all()[sel], device)
    s, a = l2n(zs), l2n(za)

    def rankme(z):
        sv = torch.linalg.svdvals(z.double())
        p = sv / sv.sum() + 1e-7
        return float(torch.exp(-(p * p.log()).sum()))

    def unif(x):
        return float(torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log())
    sim = s @ a.t()
    top = sim.topk(min(hub_k, sim.shape[1]), dim=1).indices.flatten()
    counts = torch.bincount(top, minlength=sim.shape[1]).double()
    sk = float(((counts - counts.mean()) ** 3).mean() / (counts.std() ** 3 + 1e-12))
    return {"alignment": float((s - a).pow(2).sum(-1).mean()), "uniformity_s": unif(s), "uniformity_a": unif(a),
            "rankme_zs": rankme(zs), "rankme_za": rankme(za), "rankme_s": rankme(s), "rankme_a": rankme(a),
            "modality_gap": float((s.mean(0) - a.mean(0)).norm()), "hubness_skew": sk,
            "retrieval_r1_pool": float((sim.argmax(1) == torch.arange(len(s), device=device)).float().mean()),
            "n_geo": int(len(sel))}


# --------------------------------------------------------------------------------------
def build_pool_indices(evalset, out_dir: str, n_rand: int = 10, n_mine: int = 10,
                       fn_threshold: float = 0.95, seed: int = 0, device="cpu") -> dict:
    """Índices fijos para T-rand y T-mine (sobre el espacio crudo del encoder, sin cabezales).
    T-mine excluye el mismo documento y candidatos con coseno > fn_threshold al oro
    (posibles falsos negativos / duplicados)."""
    p_rand, p_mine = os.path.join(out_dir, "rand_idx.npy"), os.path.join(out_dir, "mine_idx.npy")
    docs = evalset.doc_ids
    N = evalset.n
    if not os.path.exists(p_rand):
        rs = np.random.default_rng(seed)
        idx = np.zeros((N, n_rand), np.int64)
        for i in range(N):
            c = rs.choice(N, size=n_rand * 3, replace=False)
            c = c[docs[c] != docs[i]][:n_rand]
            while len(c) < n_rand:
                extra = rs.choice(N, size=n_rand, replace=False)
                c = np.r_[c, extra[docs[extra] != docs[i]]][:n_rand]
            idx[i] = c
        atomic_np_save(idx, p_rand)
    if not os.path.exists(p_mine):
        build_mine_pool(evalset, p_mine, n_mine, fn_threshold, device, os.path.join(out_dir, "mine_sims.npy"))
    return {"rand": p_rand, "mine": p_mine}


def build_mine_pool(evalset, path: str, n_mine: int = 10, fn_threshold: float = 0.95, device="cpu",
                    sims_path: str | None = None) -> np.ndarray:
    """Índices de las n_mine respuestas más parecidas al oro (coseno crudo del encoder), de OTROS documentos
    y con coseno ≤ fn_threshold (posibles falsos negativos / duplicados fuera). Mismo procedimiento para
    T-mine, val-mine (A1.2) y T-mine-50."""
    if os.path.exists(path):
        return np.load(path)
    docs = evalset.doc_ids
    N = evalset.n
    if N - 1 < n_mine:
        raise ValueError(f"el set tiene {N} ítems; no alcanza para {n_mine} negativos minados")
    A = torch.from_numpy(np.asarray(evalset.a_all(), np.float32)).to(device)
    A = F.normalize(A, dim=-1)
    _, inv = np.unique(np.asarray(docs, dtype=object).astype(str), return_inverse=True)
    doc_t = torch.from_numpy(inv).to(device)
    idx = np.zeros((N, n_mine), np.int64)
    sims = np.zeros((N, n_mine), np.float32)
    for s in range(0, N, 1024):
        e = min(N, s + 1024)
        sim = A[s:e] @ A.t()
        same = doc_t[s:e, None] == doc_t[None, :]
        sim = sim.masked_fill(same | (sim > fn_threshold), -2.0)
        v, j = sim.topk(n_mine, dim=1)
        idx[s:e], sims[s:e] = j.cpu().numpy(), v.cpu().numpy()
    if sims_path:                      # primero los cosenos: si existe el pool, existen también sus cosenos
        atomic_np_save(sims, sims_path)
    atomic_np_save(idx, path)
    return idx


POOL_FILES = {"trand": "rand_idx.npy", "tmine": "mine_idx.npy", "tmine50": "mine50_idx.npy"}


def expected_eval_keys(pools_dir: str | None, extra_names=(), shuffle: bool = True) -> list:
    """Bloques que una evaluación completa de v0.3 debe tener en metrics.json (para add_missing_evals)."""
    keys = ["tgem"] + (["tgem_shuf"] if shuffle else [])
    if pools_dir is not None:
        for name, fn in POOL_FILES.items():
            if os.path.exists(os.path.join(pools_dir, fn)):
                keys.append(name)
                if shuffle:
                    keys.append(f"{name}_shuf")
                if name == "tmine":
                    keys.append("tabst_mine")
    return keys + list(extra_names)


def _small(m: dict) -> dict:
    return {k: m[k] for k in ("top1", "mrr", "n") if k in m}


def evaluate_run(heads, loss_mod, test, val, out_dir: str, device, ecfg, pools_dir: str | None = None,
                 extra_sets: dict | None = None, log=print, shuffle: bool = True) -> dict:
    """Evaluación completa de un checkpoint. Reanudable por bloque (salta los ya hechos). Los bloques de
    v0.3 (estados barajados, T-abst-mine, T-mine-50, panel T-gen) se agregan a corridas viejas sin rehacer
    los que ya estaban."""
    os.makedirs(out_dir, exist_ok=True)
    summ_path = os.path.join(out_dir, "metrics.json")
    metrics = load_json(summ_path, {}) or {}
    if loss_mod is not None:
        loss_mod.eval()
    alt = getattr(loss_mod, "alt_score", None)
    # validación (para temperature scaling / Platt)
    Cv, _ = cos_matrix(heads, val, device, ecfg.eval_batch)
    T_ts = fit_temperature(Cv)
    platt = fit_platt(Cv)
    native = loss_mod.native_scale(heads) if loss_mod is not None else None
    null_cos_fn = None
    if loss_mod is not None and getattr(loss_mod, "null", False):
        def null_cos_fn(es):
            zs = l2n(project(heads.state_head, es, device))
            return (zs @ l2n(loss_mod.null_vec.detach())).cpu().numpy()
    Q = test.q_all()
    perm = shuffle_perm(test.n, getattr(ecfg, "shuffle_seed", 0)) if shuffle else None

    def save():
        save_json(metrics, summ_path)

    def npy(name, arr):
        atomic_np_save(arr, os.path.join(out_dir, name))

    # T-gem + abstención + calibración
    if "tgem" not in metrics:
        C, A = cos_matrix(heads, test, device, ecfg.eval_batch, alt=alt)
        npy("scores_tgem.npy", C.astype(np.float32))
        m, correct, rank = rank_metrics(C)
        npy("correct_tgem.npy", correct)
        npy("argmax_tgem.npy", C.argmax(1).astype(np.int8))
        null_cos = null_cos_fn(Q) if null_cos_fn else None
        m.update(abstention_metrics(C, ecfg.abst_k, T=T_ts, null_cos=null_cos))
        m.update({f"ts_{k}": v for k, v in listwise_calibration(C, T_ts, ecfg.calib_bins).items()})
        if native is not None:
            m.update({f"native_{k}": v for k, v in listwise_calibration(C, native, ecfg.calib_bins).items()})
        m.update(pointwise_calibration(C, *platt, bins=ecfg.calib_bins))
        if loss_mod is not None and getattr(loss_mod, "t_prime", None) is not None:   # SigLIP
            tt = float(loss_mod.t().detach())
            m.update({f"native_{k}": v for k, v in
                      pointwise_calibration(C, tt, float(loss_mod.b.detach()), ecfg.calib_bins).items()})
        if A is not None:
            ma, ca, _ = rank_metrics(A)
            m.update({f"alt_{k}": v for k, v in ma.items()})
            npy("correct_tgem_alt.npy", ca)
        metrics["tgem"] = m
        save()
        log(f"[eval] tgem top1={m['top1']:.4f} abst_auroc={m['abst_auroc_maxcos']:.4f} ece_ts={m['ts_ece']:.4f}")
    # T-gem con estados barajados (A1.4)
    if perm is not None and "tgem_shuf" not in metrics:
        Cs, _ = cos_matrix(heads, test, device, ecfg.eval_batch, states=Q[perm])
        m, correct, _ = rank_metrics(Cs)
        npy("correct_tgem_shuf.npy", correct)
        npy("argmax_tgem_shuf.npy", Cs.argmax(1).astype(np.int8))
        metrics["tgem_shuf"] = _small(m)
        save()
        log(f"[eval] tgem barajado top1={m['top1']:.4f}")
    # T-rand / T-mine / T-mine-50 (+ abstención con negativos minados y estados barajados)
    if pools_dir is not None:
        for name, fn in POOL_FILES.items():
            p = os.path.join(pools_dir, fn)
            if not os.path.exists(p):
                continue
            I, C = None, None
            if name not in metrics:
                I = np.load(p)
                C = cos_with_pool(heads, test, I, device)
                m, correct, _ = rank_metrics(C)
                npy(f"correct_{name}.npy", correct)
                metrics[name] = m
                save()
                log(f"[eval] {name} top1={m['top1']:.4f}")
            if name == "tmine" and "tabst_mine" not in metrics:
                if C is None:
                    I = np.load(p)
                    C = cos_with_pool(heads, test, I, device)
                npy("scores_tmine.npy", C.astype(np.float32))
                if C.shape[1] - 1 >= ecfg.abst_k + 1:
                    metrics["tabst_mine"] = abstention_metrics(C, ecfg.abst_k, T=T_ts,
                                                               null_cos=null_cos_fn(Q) if null_cos_fn else None)
                    save()
                    log(f"[eval] T-abst-mine auroc={metrics['tabst_mine']['abst_auroc_maxcos']:.4f}")
            if perm is not None and f"{name}_shuf" not in metrics:
                if I is None:
                    I = np.load(p)
                m, correct, _ = rank_metrics(cos_with_pool(heads, test, I, device, states=Q[perm]))
                npy(f"correct_{name}_shuf.npy", correct)
                metrics[f"{name}_shuf"] = _small(m)
                save()
    # conjuntos externos (panel T-gen, T-gen2 de v0.2): dict nombre -> (item_idx, negs[, huella del contenido])
    sig_path = os.path.join(out_dir, "extra_sigs.json")
    sigs = load_json(sig_path, {}) or {}
    for name, val in (extra_sets or {}).items():
        item_idx, negs = np.asarray(val[0]), val[1]
        sig = val[2] if len(val) > 2 else None
        pos = val[3] if len(val) > 3 else None             # positivo propio del set (construcción P)
        p_items = os.path.join(out_dir, f"items_{name}.npy")
        if name in metrics and os.path.exists(p_items) and not np.array_equal(np.load(p_items), item_idx):
            log(f"[eval] {name}: sus ítems cambiaron desde la última evaluación; se re-evalúa")
            metrics.pop(name)
        if name in metrics and sig is not None and sigs.get(name) != sig:
            log(f"[eval] {name}: sus negativos cambiaron desde la última evaluación; se re-evalúa")
            metrics.pop(name)
        if name in metrics and (perm is None or "shuf_top1" in metrics[name]):
            continue
        C = cos_with_external(heads, test, item_idx, negs, device, pos=pos)
        m, correct, _ = rank_metrics(C)
        if C.shape[1] - 1 >= ecfg.abst_k + 1:
            m.update(abstention_metrics(C, ecfg.abst_k, T=T_ts,
                                        null_cos=null_cos_fn(Q[item_idx]) if null_cos_fn else None))
        npy(f"correct_{name}.npy", correct)
        npy(f"items_{name}.npy", item_idx)
        npy(f"scores_{name}.npy", C.astype(np.float32))
        if perm is not None:
            ms, cs, _ = rank_metrics(cos_with_external(heads, test, item_idx, negs, device, states=Q[perm[item_idx]],
                                                       pos=pos))
            npy(f"correct_{name}_shuf.npy", cs)
            m["shuf_top1"], m["shuf_mrr"] = ms["top1"], ms["mrr"]
        metrics[name] = m
        save()
        if sig is not None:
            sigs[name] = sig
            save_json(sigs, sig_path)
        log(f"[eval] {name} top1={m['top1']:.4f}" + (f" barajado={m['shuf_top1']:.4f}" if "shuf_top1" in m else ""))
    if "geometry" not in metrics:
        metrics["geometry"] = geometry(heads, test, device, ecfg.n_geo, ecfg.hub_k)
        save_json(metrics, summ_path)
    if "hidden" not in metrics:
        metrics["hidden"] = int(heads.cfg.get("hidden_size") or test.q_all().shape[1])
        save_json(metrics, summ_path)
    return metrics


@torch.no_grad()
def quick_val(heads, val, device, bs: int = 4096, cache: dict | None = None,
              mine_idx: np.ndarray | None = None) -> dict:
    """Validación rápida durante el entrenamiento (val completo en RAM).
    val_*      : T-gem-val (oro + 10 HN de Gemini). Desde v0.3 sólo es diagnóstico (tiene el atajo).
    valmine_*  : val-mine (oro + 10 respuestas minadas de otros documentos de val), métrica de selección de
                 la enmienda A1.2. Sólo si se pasa `mine_idx`."""
    if cache is not None and "q" not in cache:
        cache["q"], cache["a"] = val.q_all(), val.a_all()
        cache["n"] = val.negatives(np.arange(val.n))
    q, a, n = (cache["q"], cache["a"], cache["n"]) if cache is not None else (val.q_all(), val.a_all(), val.negatives(np.arange(val.n)))
    heads.eval()
    rows = []
    for s in range(0, len(q), bs):
        e = slice(s, s + bs)
        zs = l2n(project(heads.state_head, q[e], device))
        b, K, H = n[e].shape
        cands = np.concatenate([a[e][:, None, :], n[e]], 1).reshape(b * (K + 1), H)
        zc = l2n(project(heads.action_head, cands, device)).view(b, K + 1, -1)
        rows.append(torch.einsum("bd,bkd->bk", zs, zc).cpu().numpy())
    out = {}
    if mine_idx is not None:
        zs = l2n(project(heads.state_head, q, device))
        za = l2n(project(heads.action_head, a, device))
        I = torch.from_numpy(np.asarray(mine_idx, np.int64)).to(device)
        Cm = torch.cat([(zs * za).sum(-1, keepdim=True), torch.einsum("nd,nkd->nk", zs, za[I])], 1).cpu().numpy()
        mm, _, _ = rank_metrics(Cm)
        out.update(valmine_top1=mm["top1"], valmine_mrr=mm["mrr"])
    heads.train()
    C = np.concatenate(rows)
    m, _, _ = rank_metrics(C)
    ab = abstention_metrics(C)
    return {"val_top1": m["top1"], "val_mrr": m["mrr"], "val_abst_auroc": ab["abst_auroc_maxcos"],
            "val_margin": m["margin_mean"], **out}
