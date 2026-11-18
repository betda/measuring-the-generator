"""Búsqueda aleatoria de hiperparámetros por brazo (mismo presupuesto para todos), reanudable,
con curvas de desempeño de validación esperado vs número de trials (Dodge et al., 2019)."""
from __future__ import annotations

import copy
import math
import os

import numpy as np

from .config import TrainConfig
from .utils import load_json, rng, save_json

COMMON = {"lr_mult": ("loguniform", 0.25, 4.0), "weight_decay": ("choice", [0.0, 1e-4, 1e-2])}
TAU = {"tau_init": ("choice", [0.03, 0.05, 0.07, 0.1])}

SPACES: dict[str, dict] = {
    "infonce": {**TAU},
    "infonce_hn": {**TAU, "loss.hn_denominator": ("choice", ["hn_only", "hn_plus_inbatch"])},
    "siglip": {"loss.t_init": ("choice", [1.0, 5.0, 10.0, 20.0]), "loss.b_init": ("choice", [-5.0, -10.0, -15.0])},
    "siglip_hn": {"loss.t_init": ("choice", [1.0, 5.0, 10.0, 20.0]), "loss.b_init": ("choice", [-5.0, -10.0, -15.0])},
    "triplet_hn": {"loss.margin": ("loguniform", 0.05, 0.5)},
    "simsiam_x": {"loss.pred_hidden": ("choice", [64, 128, 256])},
    "byol_x": {"loss.tau_base": ("choice", [0.99, 0.996, 0.999]), "loss.pred_hidden": ("choice", [512, 1024, 2048])},
    "barlow_x": {"loss.exp_dim": ("choice", [2048, 4096, 8192]), "loss.lambd": ("loguniform", 1e-3, 2e-2)},
    "vicreg_x": {"loss.exp_dim": ("choice", [2048, 4096, 8192]), "loss.sim": ("loguniform", 5.0, 50.0),
                 "loss.std": ("loguniform", 5.0, 50.0), "loss.cov": ("loguniform", 0.1, 4.0)},
    "mse_sigreg": {"loss.lam": ("loguniform", 0.005, 0.5), "loss.n_slices": ("choice", [128, 256, 1024])},
    "infonce_sigreg": {**TAU, "loss.lam": ("loguniform", 1e-4, 1e-1)},
    "align_unif": {"loss.gamma": ("loguniform", 0.2, 5.0)},
    "infonce_lse": {**TAU, "loss.lam": ("loguniform", 1e-4, 1e-1)},
    "dino_x": {"loss.K": ("choice", [1024, 4096, 16384]), "loss.tau_t_final": ("choice", [0.04, 0.06, 0.07]),
               "loss.m_base": ("choice", [0.99, 0.996])},
    "swav_x": {"loss.K": ("choice", [1024, 3000, 8192]), "loss.eps": ("choice", [0.03, 0.05, 0.1]),
               "loss.tau": ("choice", [0.05, 0.1, 0.2]), "loss.freeze_steps": ("choice", [0, 100])},
}


# Espacios por fase. F1b (enmienda A1.7): el denominador de InfoNCE+HN deja de ser una opción del HPO
# ("HN + in-batch" fijo por diseño; "HN solos" sólo como condición explícita de B1).
PHASE_SPACES: dict[str, dict] = {"F1b": {"infonce_hn": {**TAU}}}


def space_for(arm: str, phase: str | None = None) -> dict:
    sp = (PHASE_SPACES.get(phase or "") or {}).get(arm)
    return {**COMMON, **(sp if sp is not None else SPACES[arm])}


def _sample(spec, g: np.random.Generator):
    kind = spec[0]
    if kind == "loguniform":
        return float(math.exp(g.uniform(math.log(spec[1]), math.log(spec[2]))))
    if kind == "uniform":
        return float(g.uniform(spec[1], spec[2]))
    if kind == "choice":
        v = spec[1][int(g.integers(len(spec[1])))]
        return v.item() if hasattr(v, "item") else v
    raise ValueError(kind)


def sample_trials(arm: str, n: int, seed: int, phase: str | None = None) -> list[dict]:
    """Trial 0 = valores por defecto publicados; el resto, muestreo aleatorio con semilla fija."""
    space = space_for(arm, phase)
    g = rng(seed, sum(map(ord, arm)))
    trials = [{}]
    for _ in range(max(0, n - 1)):
        trials.append({k: _sample(v, g) for k, v in space.items()})
    return trials


def apply_overrides(cfg: TrainConfig, over: dict) -> TrainConfig:
    c = copy.deepcopy(cfg)
    lp = dict(c.loss_params)
    for k, v in over.items():
        if k.startswith("loss."):
            lp[k[5:]] = v
        else:
            setattr(c, k, v)
    c.loss_params = lp
    return c


def expected_max_curve(values, ns=None) -> dict:
    """E[máximo de n trials] a partir de los N valores observados (Dodge et al., 2019)."""
    v = np.sort(np.asarray([x for x in values if np.isfinite(x)], float))
    N = len(v)
    if N == 0:
        return {}
    ns = ns or sorted({1, 2, 4, 8, N} | set(range(1, N + 1, max(1, N // 6))))
    out = {}
    for n in ns:
        if n > N:
            continue
        i = np.arange(1, N + 1)
        w = (i / N) ** n - ((i - 1) / N) ** n
        out[int(n)] = float((w * v).sum())
    return out


def run_hpo(arm: str, base_cfg: TrainConfig, bundle, arm_dir: str, n_trials: int, budget_frac: float,
            seed: int, run_fn, log=print, phase: str | None = None) -> dict:
    """Ejecuta (o reanuda) los trials de un brazo. run_fn(cfg, run_dir, row_fraction) -> resumen.
    La métrica de selección es la de base_cfg.select_metric (val-mine en F1b)."""
    trials = sample_trials(arm, n_trials, seed, phase)
    path = os.path.join(arm_dir, "hpo_summary.json")
    rows = []
    for t, over in enumerate(trials):
        cfg = apply_overrides(base_cfg, over)
        cfg.arm, cfg.seed = arm, 0
        rd = os.path.join(arm_dir, "hpo", f"trial_{t:02d}")
        res = run_fn(cfg, rd, budget_frac)
        val = res.get("val_best")
        val = float(val) if val is not None and np.isfinite(val) else -math.inf
        rows.append({"trial": t, "overrides": over, "val_best": val, "status": res.get("status")})
        log(f"[hpo:{arm}] trial {t:02d} {base_cfg.select_metric}={val:.4f} {over}")
        save_json({"arm": arm, "trials": rows, "n_trials": n_trials, "budget_frac": budget_frac,
                   "select_metric": base_cfg.select_metric, "phase": phase}, path)
    ok = [r for r in rows if np.isfinite(r["val_best"])]
    best = max(ok, key=lambda r: r["val_best"]) if ok else rows[0]
    summary = {"arm": arm, "trials": rows, "best": best, "n_trials": n_trials, "budget_frac": budget_frac,
               "select_metric": base_cfg.select_metric, "phase": phase,
               "evp_curve": expected_max_curve([r["val_best"] for r in rows])}
    save_json(summary, path)
    return summary


def load_best_overrides(arm_dir: str) -> dict | None:
    s = load_json(os.path.join(arm_dir, "hpo_summary.json"))
    return None if s is None or "best" not in s else s["best"]["overrides"]
