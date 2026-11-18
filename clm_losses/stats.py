"""Estadística: todos-contra-todos (15 brazos -> 105 pares) y contrastes confirmatorios C1–C4.

Diseño:
  * Unidad = corrida (brazo x semilla). Los ítems de test son los mismos para todos los brazos
    (comparación pareada por ítem) y están agrupados por documento.
  * Bootstrap jerárquico: pesos de Poisson por DOCUMENTO (compartidos entre brazos -> pareado)
    + remuestreo de semillas dentro de cada brazo (varianza de entrenamiento).
  * Para cada par: diferencia, IC95 %, p (aprox. normal sobre la distribución bootstrap),
    p ajustada por Holm sobre TODOS los pares de la métrica, equivalencia TOST (IC90 dentro de
    ±ROPE) y probabilidades tipo ROPE: P(A mejor), P(equivalente), P(B mejor).
  * Ranking multi-tarea: Friedman + post-hoc de Nemenyi (diferencia crítica) con un bloque por
    tarea (top-1 en cada conjunto de test y AUROC de abstención; media sobre semillas).
  * AUROC de abstención: la misma matriz de pesos por documento para todos los brazos y semillas
    en cada réplica (las semillas comparten el test; pesos independientes subestimarían la varianza).
"""
from __future__ import annotations

import itertools
import math
import os

import numpy as np
from scipy import stats as st

from .utils import load_json, save_json


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def clusters_from_docs(doc_ids) -> np.ndarray:
    _, inv = np.unique(np.asarray(doc_ids, dtype=object).astype(str), return_inverse=True)
    return inv.astype(np.int64)


# --------------------------------------------------------------------------------------
def bootstrap_means(items: dict, clusters: np.ndarray, B: int = 2000, seed: int = 0,
                    chunk: int = 100) -> tuple[dict, dict]:
    """items: brazo -> matriz (S x N) de valores por ítem (p. ej. acierto 0/1).
    Devuelve (medias bootstrap por brazo [B], estimación puntual por brazo)."""
    rs = np.random.default_rng(seed)
    n_cl = int(clusters.max()) + 1
    boots = {a: np.empty(B) for a in items}
    for b0 in range(0, B, chunk):
        bs = min(chunk, B - b0)
        w = rs.poisson(1.0, size=(bs, n_cl)).astype(np.float32)[:, clusters]    # bs x N
        den = np.maximum(w.sum(1, keepdims=True), 1e-9)
        for a, X in items.items():
            X = np.asarray(X, np.float32)
            per_seed = (w @ X.T) / den                                             # bs x S
            pick = rs.integers(0, X.shape[0], size=(bs, X.shape[0]))
            boots[a][b0:b0 + bs] = np.take_along_axis(per_seed, pick, 1).mean(1)
    point = {a: float(np.asarray(X, np.float64).mean()) for a, X in items.items()}
    return boots, point


def bootstrap_seed_level(values: dict, B: int = 2000, seed: int = 0) -> tuple[dict, dict]:
    """Para métricas no descomponibles por ítem (ECE, geometría): sólo remuestreo de semillas."""
    rs = np.random.default_rng(seed)
    boots, point = {}, {}
    for a, v in values.items():
        v = np.asarray(v, float)
        pick = rs.integers(0, len(v), size=(B, len(v)))
        boots[a] = v[pick].mean(1)
        point[a] = float(v.mean())
    return boots, point


def weighted_auroc_reps(pos: np.ndarray, neg: np.ndarray, clusters: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AUROC ponderado para cada réplica (fila de W = pesos por documento); pos[i] y neg[i] vienen
    del mismo ítem i. Los empates cuentan 1/2 (igual que el AUROC puntual por rangos)."""
    s = np.r_[pos, neg]
    lab = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    cl = np.r_[clusters, clusters]
    order = np.argsort(s, kind="stable")
    s_o, lab_o, cl_o = s[order], lab[order], cl[order]
    starts = np.r_[0, np.nonzero(np.diff(s_o))[0] + 1]        # grupos de valores iguales
    out = np.empty(len(W))
    for b in range(len(W)):
        w = W[b][cl_o].astype(np.float64)
        wn = np.where(~lab_o, w, 0.0)
        wp = np.where(lab_o, w, 0.0)
        wn_g, wp_g = np.add.reduceat(wn, starts), np.add.reduceat(wp, starts)
        below = np.cumsum(wn_g) - wn_g
        out[b] = (wp_g * (below + 0.5 * wn_g)).sum() / max(wp.sum() * wn.sum(), 1e-12)
    return out


def weighted_auroc_boot(pos: np.ndarray, neg: np.ndarray, clusters: np.ndarray, B: int, seed: int) -> np.ndarray:
    """Compatibilidad: AUROC bootstrap con pesos propios (sólo para una corrida aislada)."""
    W = np.random.default_rng(seed).poisson(1.0, size=(B, int(clusters.max()) + 1)).astype(np.float32)
    return weighted_auroc_reps(pos, neg, clusters, W)


# --------------------------------------------------------------------------------------
def pairwise_table(boots: dict, point: dict, rope: float, alpha: float = 0.05,
                   higher_is_better: bool = True) -> list[dict]:
    arms = list(point)
    rows = []
    for a, b in itertools.combinations(arms, 2):
        d = boots[a] - boots[b]
        est = point[a] - point[b]
        se = float(d.std(ddof=1))
        z = est / se if se > 0 else (math.inf if est != 0 else 0.0)
        p = float(2 * st.norm.sf(abs(z))) if math.isfinite(z) else 0.0
        lo95, hi95 = np.percentile(d, [2.5, 97.5])
        lo90, hi90 = np.percentile(d, [5, 95])
        sign = 1 if higher_is_better else -1
        rows.append({"A": a, "B": b, "diff": est, "ci95_lo": float(lo95), "ci95_hi": float(hi95),
                     "p": p, "tost_equivalent": bool(lo90 > -rope and hi90 < rope),
                     "P_A_better": float(np.mean(sign * d > rope)),
                     "P_equivalent": float(np.mean(np.abs(d) <= rope)),
                     "P_B_better": float(np.mean(sign * d < -rope))})
    if rows:
        adj = holm(np.array([r["p"] for r in rows]))
        for r, pa in zip(rows, adj):
            r["p_holm"] = float(pa)
            r["significant"] = bool(pa < alpha)
    return rows


def friedman_nemenyi(matrix: np.ndarray, arms: list, alpha: float = 0.05, higher_is_better: bool = True) -> dict:
    """matrix: bloques x brazos. Rango 1 = mejor."""
    M = np.asarray(matrix, float)
    N, k = M.shape
    ranks = np.vstack([st.rankdata(-row if higher_is_better else row) for row in M])
    avg = ranks.mean(0)
    out = {"arms": list(arms), "avg_rank": avg.tolist(), "n_blocks": int(N), "k": int(k)}
    if N >= 2 and k >= 3:
        chi = st.friedmanchisquare(*[M[:, j] for j in range(k)])
        out["friedman_chi2"], out["friedman_p"] = float(chi.statistic), float(chi.pvalue)
        q = st.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
        out["nemenyi_q"] = float(q)
        out["critical_difference"] = float(q * math.sqrt(k * (k + 1) / (6 * N)))
    return out


# --------------------------------------------------------------------------------------
def plot_pairwise_heatmap(rows: list[dict], arms: list, title: str, path: str, scale: float = 100.0,
                          unit: str = "pp") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(arms)
    idx = {a: i for i, a in enumerate(arms)}
    D = np.full((k, k), np.nan)
    mark = np.full((k, k), "", dtype=object)
    for r in rows:
        i, j = idx[r["A"]], idx[r["B"]]
        D[i, j], D[j, i] = scale * r["diff"], -scale * r["diff"]
        m = "*" if r.get("significant") else ("≈" if r["tost_equivalent"] else "")
        mark[i, j] = mark[j, i] = m
    lim = np.nanmax(np.abs(D)) if np.isfinite(D).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * k, 0.8 + 0.5 * k))
    im = ax.imshow(D, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(k), arms, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(k), arms, fontsize=8)
    for i in range(k):
        for j in range(k):
            if i != j and np.isfinite(D[i, j]):
                ax.text(j, i, f"{D[i, j]:.1f}{mark[i, j]}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label=f"fila − columna ({unit})")
    ax.set_title(f"{title}\n* Holm p<0.05   ≈ equivalente (TOST)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_cd_diagram(fr: dict, path: str, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "critical_difference" not in fr:
        return
    arms, avg, cd = fr["arms"], np.array(fr["avg_rank"]), fr["critical_difference"]
    order = np.argsort(avg)
    k = len(arms)
    fig, ax = plt.subplots(figsize=(8, 0.35 * k + 1.2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1, k + 1)
    ax.hlines(k, 1, k, color="k", lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, k - 0.1, k + 0.1, color="k", lw=0.8)
        ax.text(r, k + 0.3, str(r), ha="center", fontsize=7)
    for pos, i in enumerate(order):
        y = k - 1 - pos
        ax.plot([avg[i], avg[i]], [k, y], color="0.4", lw=0.7)
        ax.text(avg[i] + 0.05, y, f"{arms[i]} ({avg[i]:.2f})", va="center", fontsize=7)
    # grupos sin diferencia significativa
    y = -0.5
    for a in range(k):
        b = a
        while b + 1 < k and avg[order[b + 1]] - avg[order[a]] <= cd:
            b += 1
        if b > a:
            ax.hlines(y, avg[order[a]] - 0.03, avg[order[b]] + 0.03, lw=3, color="tab:blue")
            y -= 0.25
    ax.hlines(k + 0.8, 1, 1 + cd, color="tab:red", lw=2)
    ax.text(1 + cd / 2, k + 1.0, f"CD={cd:.2f}", ha="center", fontsize=7, color="tab:red")
    ax.axis("off")
    ax.set_title(title or "Rango promedio (1 = mejor); barras azules = sin diferencia significativa", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def collect_runs(runs_root: str, phase: str, arms: list) -> dict:
    """brazo -> lista de (semilla, run_dir) terminadas con status ok."""
    out = {}
    for a in arms:
        base = os.path.join(runs_root, phase, a)
        lst = []
        if os.path.isdir(base):
            for d in sorted(os.listdir(base)):
                rd = os.path.join(base, d)
                done = load_json(os.path.join(rd, "DONE.json"))
                if d.startswith("seed_") and done and done.get("status") == "ok":
                    lst.append((int(done["seed"]), rd))
        if lst:
            out[a] = lst
    return out


def collect_failures(runs_root: str, phase: str, arms: list) -> dict:
    """brazo -> semillas que terminaron con status distinto de ok (p. ej. pérdida no finita)."""
    out = {}
    for a in arms:
        base = os.path.join(runs_root, phase, a)
        if not os.path.isdir(base):
            continue
        bad = []
        for d in sorted(os.listdir(base)):
            done = load_json(os.path.join(base, d, "DONE.json"))
            if d.startswith("seed_") and done and done.get("status") != "ok":
                bad.append(int(done.get("seed", -1)))
        if bad:
            out[a] = bad
    return out


ABST_SOURCES = {"tgem": ("scores_tgem.npy", "tgem", "abst_auroc"),
                 "tmine": ("scores_tmine.npy", "tabst_mine", "abst_auroc_tmine")}


def all_vs_all(runs: dict, testsets: list, doc_ids: np.ndarray, out_dir: str, rope_pp: float = 1.0,
               B: int = 2000, seed: int = 0, abst: bool = True, log=print, gen2_items: np.ndarray | None = None,
               failed: dict | None = None, items_map: dict | None = None, abst_sets=("tgem",)) -> dict:
    """Análisis completo todos-contra-todos. Guarda CSV/JSON y figuras en out_dir.
    items_map: {conjunto: índices de ítems del test} para conjuntos que usan un subconjunto (panel T-gen,
    macro); sirve para verificar que todas las corridas usaron los mismos ítems y para agrupar por documento."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    arms = list(runs)
    clusters_all = clusters_from_docs(doc_ids)
    items_map = dict(items_map or {})
    if gen2_items is not None:
        items_map.setdefault("tgen2", gen2_items)
    report = {"arms": arms, "rope_pp": rope_pp, "B": B, "metrics": {}, "failed_seeds": failed or {}}
    if failed:
        log(f"[stats] ATENCIÓN: semillas fallidas (excluidas de las medias): {failed}")
    rope = rope_pp / 100.0
    block_values = {}
    for ts in testsets:
        items, ok = {}, True
        want = items_map.get(ts)
        for a in arms:
            mats = []
            for s, rd in runs[a]:
                p = os.path.join(rd, "eval", f"correct_{ts}.npy")
                if not os.path.exists(p):
                    ok = False
                    break
                pi = os.path.join(rd, "eval", f"items_{ts}.npy")
                if os.path.exists(pi) and want is not None and not np.array_equal(np.load(pi), np.asarray(want)):
                    log(f"[stats] {ts}: {a}/seed {s} se evaluó con otros ítems (el conjunto cambió); re-evalúa con "
                        f"add_missing_evals")
                    ok = False
                    break
                mats.append(np.load(p).astype(np.float32))
            if not ok:
                break
            items[a] = np.vstack(mats)
        if not ok:
            log(f"[stats] {ts}: faltan resultados en algún brazo; se omite")
            continue
        n = items[arms[0]].shape[1]
        if want is not None and n == len(want):
            clusters = clusters_from_docs(np.asarray(doc_ids)[np.asarray(want)])
        elif n == len(clusters_all):
            clusters = clusters_all
        else:
            clusters = np.arange(n)
        boots, point = bootstrap_means(items, clusters, B=B, seed=seed)
        # bloque del ranking multi-tarea = conjunto de test (media sobre semillas); las semillas
        # NO son bloques independientes (serían pseudo-réplicas del mismo test)
        block_values[ts] = {a: float(items[a].mean()) for a in arms}
        rows = pairwise_table(boots, point, rope)
        name = f"{ts}_top1"
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"pairwise_{name}.csv"), index=False)
        summary = pd.DataFrame({"arm": list(point), "mean": list(point.values()),
                                "ci95_lo": [float(np.percentile(boots[a], 2.5)) for a in point],
                                "ci95_hi": [float(np.percentile(boots[a], 97.5)) for a in point],
                                "n_seeds": [items[a].shape[0] for a in point],
                                "n_failed": [len((failed or {}).get(a, [])) for a in point]}
                               ).sort_values("mean", ascending=False)
        summary.to_csv(os.path.join(out_dir, f"summary_{name}.csv"), index=False)
        plot_pairwise_heatmap(rows, list(summary["arm"]), f"top-1 en {ts}: diferencias por pares",
                              os.path.join(out_dir, f"heatmap_{name}.png"))
        report["metrics"][name] = {"summary": summary.to_dict("records"),
                                   "n_significant": int(sum(r["significant"] for r in rows)),
                                   "n_equivalent": int(sum(r["tost_equivalent"] for r in rows)),
                                   "n_pairs": len(rows)}
        log(f"[stats] {name}: {len(rows)} pares, {report['metrics'][name]['n_significant']} significativos (Holm), "
            f"{report['metrics'][name]['n_equivalent']} equivalentes (TOST ±{rope_pp} pp)")
    for src in (abst_sets if abst else ()):
        fname, mkey, label = ABST_SOURCES[src]
        boots, point = {}, {}
        Bab = max(200, B // 4)
        rs_w = np.random.default_rng(seed + 7919)
        W = rs_w.poisson(1.0, size=(Bab, int(clusters_all.max()) + 1)).astype(np.float32)   # compartida
        rs_seed = np.random.default_rng(seed + 104729)
        for a in arms:
            per_seed, pts = [], []
            for s, rd in runs[a]:
                p = os.path.join(rd, "eval", fname)
                mj = (load_json(os.path.join(rd, "eval", "metrics.json")) or {}).get(mkey) or {}
                if not os.path.exists(p) or "abst_auroc_maxcos" not in mj:
                    per_seed = []
                    break
                C = np.load(p).astype(np.float32)
                pos, neg = C[:, :10].max(1), C[:, 1:11].max(1)
                per_seed.append(weighted_auroc_reps(pos, neg, clusters_all, W))
                pts.append(mj["abst_auroc_maxcos"])
            if per_seed:
                stack = np.vstack(per_seed)                                   # S x Bab
                pick = rs_seed.integers(0, len(per_seed), size=(Bab, len(per_seed)))
                boots[a] = np.take_along_axis(stack.T, pick, 1).mean(1)
                point[a] = float(np.mean(pts))
        if len(point) >= 2:
            rows = pairwise_table(boots, point, 0.01)
            import pandas as pd
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"pairwise_{label}.csv"), index=False)
            ttl = "T-abst-mine" if src == "tmine" else "T-abst (HN de Gemini)"
            plot_pairwise_heatmap(rows, sorted(point, key=point.get, reverse=True),
                                  f"AUROC de abstención ({ttl}): diferencias por pares",
                                  os.path.join(out_dir, f"heatmap_{label}.png"), scale=100.0, unit="x100")
            report["metrics"][label] = {"point": point, "n_significant": int(sum(r["significant"] for r in rows))}
            if all(a in point for a in arms):
                block_values[label] = dict(point)
    # ranking multi-tarea: un bloque por tarea (top-1 en cada test + AUROC de abstención)
    blocks = [k for k, v in block_values.items() if all(a in v for a in arms)]
    if len(blocks) >= 2 and len(arms) >= 3:
        M = np.array([[block_values[k][a] for a in arms] for k in blocks])
        fr = friedman_nemenyi(M, arms)
        fr["blocks"] = blocks
        fr["nota"] = ("pocos bloques (tareas) => CD grande; las comparaciones por pares con bootstrap por "
                      "documento son el análisis principal")
        report["friedman"] = fr
        plot_cd_diagram(fr, os.path.join(out_dir, "cd_diagram.png"))
        log(f"[stats] Friedman p={fr.get('friedman_p', float('nan')):.3g}, CD={fr.get('critical_difference', float('nan')):.2f}")
    save_json(report, os.path.join(out_dir, "all_vs_all.json"))
    return report


def contrast(items_a: np.ndarray, items_b: np.ndarray, clusters: np.ndarray, rope: float, B: int = 4000,
             seed: int = 0) -> dict:
    """Contraste confirmatorio A − B (ítems pareados)."""
    boots, point = bootstrap_means({"A": items_a, "B": items_b}, clusters, B=B, seed=seed)
    return pairwise_table(boots, point, rope)[0]
