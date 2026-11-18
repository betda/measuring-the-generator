"""Notebook pre-envío (v0.3.8, 2026-10-11): lo que se corre antes de enviar el paper a JAIR.

Está FUERA del protocolo: no toca el freeze, no re-entrena nada de F3 y no cambia ningún veredicto confirmatorio.
Diseño: PROJECT_STATE.md (EXPERIMENTS PENDING, v21) e INNOVATION_MAP.md §2, punto 8 (v1.2).

Bloques (cada uno escribe su regla de lectura en results/PRE/<bloque>/REGLA.json, con fecha, ANTES de correr, y no
la reescribe):
    0  extracción de lo que ya está en Drive, sin entrenar: entorno por fase, puntuaciones secundarias de
       SimSiam-x/BYOL-x/DINO-x/SwAV-x, F3 frente a F1b, Friedman sin el bloque de T-gem barajado y el Δ con estados
       barajados del macro de T-gen (B4)
    1  calidad de los sets del split sellado: falsos negativos (juez LLM sobre todos los ítems de T-gem, T-mine y los
       5 generadores del macro, más 200 ítems etiquetados a mano para medir el acuerdo) y fuga entre splits (pares
       pregunta–oro repetidos o casi repetidos en entrenamiento); con eso, sensibilidad de C1, C2 y C2c
    2  condición de borde de B5 (exploratoria; split de test): 6 denominadores de InfoNCE+HN × 3 semillas
    3  referencias sin entrenamiento: Qwen3-Embedding-8B (zero-shot) y BM25, más el coseno crudo de Qwen3-8B

Resultados en <Drive>/results/PRE/ y corridas del bloque 2 en <Drive>/runs/PRE_borde/. Todo es reanudable.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import unicodedata
from collections import Counter

import numpy as np

from .utils import load_json, makedirs, now_iso, read_done, read_jsonl, save_json, write_text

PRE = "PRE"
SEED = 20261011                 # semilla de los muestreos de este notebook (fecha del diseño)
B_BOOT = 4000                   # réplicas del bootstrap (las de C1, C2, C2b y C2c)
ROPE = 0.01
JUDGE = {"key": "olmo31_32b", "repo": "allenai/Olmo-3.1-32B-Instruct", "dtype": "bfloat16", "chat_kwargs": {},
         "license": "apache-2.0", "params_b": 32.23,
         "por_que": ("Ya corrió en este entorno (panel T-gen) con transformers y en bf16 (64 GB, cabe en la RTX PRO 6000). "
                     "No es de la familia del oro (Qwen3) ni de la de los HN de entrenamiento (Gemini, Gemma) ni uno de "
                     "los 5 generadores del macro. gpt-oss-120b, el juez propuesto antes, no corre aquí: sus kernels "
                     "MXFP4 no cubren sm_120 y des-cuantizado no cabe (generators.py, gptoss20b).")}
JUDGE_SETS_FIXED = ("tgem", "tmine")
N_HAND = {"tgem": 30, "tmine": 30, "tgen": 28}      # ítems etiquetados a mano por set (28 por generador del macro)
QC_COL = "es_correcta(0/1)"
FN_MAX = 0.05                   # criterio de DISENO_01 (F0.5): < 5 % de falsos negativos
LEAK_MAX = 0.005                # criterio de DISENO_01 (F0.4): fuga < 0.5 % de los ítems de test
JACCARD_MIN = 0.8


def pdir(study, *p) -> str:
    return study.paths.d("results", PRE, *p)


# --------------------------------------------------------------------------------------
# Utilidades
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / den), min(1.0, (c + h) / den))


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def _num(x, d: int = 6) -> str:
    return "—" if not isinstance(x, (int, float)) else f"{x:.{d}f}"


def set_label(s: str) -> str:
    return {"tgem": "T-gem", "tmine": "T-mine"}.get(s, f"T-gen · {s[len('tgen_'):]}" if s.startswith("tgen_") else s)


def pct(x, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.{d}f}"


def write_rule(study, block: str, rule: dict, log=print) -> dict:
    """Registra la regla de lectura de un bloque ANTES de correrlo. Si ya existe, vale la registrada: una versión
    distinta se guarda aparte, con aviso, y no la reemplaza."""
    p = pdir(study, block, "REGLA.json")
    old = load_json(p)
    if old is None:
        rec = {"creado": now_iso(), **rule}
        save_json(rec, p)
        lines = [f"# Regla del bloque {block} — registrada el {rec['creado']}", ""]
        for k, v in rule.items():
            lines.append(f"- **{k}:** {v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)}")
        write_text(pdir(study, block, "REGLA.md"), "\n".join(lines) + "\n")
        return rec
    if {k: v for k, v in old.items() if k != "creado"} != json.loads(json.dumps(rule)):
        alt = pdir(study, block, f"REGLA_distinta_{now_iso().replace(':', '')}.json")
        save_json({"creado": now_iso(), **rule, "nota": "difiere de REGLA.json; vale la registrada antes"}, alt)
        log(f"[{block}] ATENCIÓN: la regla del código difiere de la registrada el {old.get('creado')}; vale la "
            f"registrada (la nueva quedó en {os.path.basename(alt)})")
    return old


def state(study) -> dict:
    return load_json(pdir(study, "ESTADO.json"), {}) or {}


def mark(study, step: str, status: str, **extra) -> None:
    st = state(study)
    st[step] = {"estado": status, "hora": now_iso(), **extra}
    save_json(st, pdir(study, "ESTADO.json"))


def correr(study, step: str, fn, log=print):
    """Corre un paso del notebook; si falla, deja el error en ESTADO.json y sigue con el siguiente."""
    t0 = time.time()
    try:
        out = fn()
        mark(study, step, "ok", minutos=round((time.time() - t0) / 60, 1))
        return out
    except SystemExit as e:
        mark(study, step, "detenido", motivo=str(e))
        log(f"[{step}] detenido: {e}")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        mark(study, step, "falló", error=f"{type(e).__name__}: {e}", traceback=tb[-4000:])
        log(f"[{step}] FALLÓ: {type(e).__name__}: {e}\n{tb[-2500:]}\nSigue con el siguiente paso.")
    return None


def _confirm_doc_ids(study) -> np.ndarray:
    from .data import EvalSet
    return np.asarray(EvalSet(study.paths.d("data", "eval", "confirm")).doc_ids)


def _runs(study, phase: str, name: str) -> list:
    from .pipeline import iter_run_dirs
    return [r for r in iter_run_dirs(study, phase) if f"/{phase}/{name}/" in r]


def _load_correct(rd: str, name: str):
    p = os.path.join(rd, "eval", f"correct_{name}.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


# ======================================================================================
# BLOQUE 0 — extracción de lo que ya está en Drive (sin entrenar)
ENV_PHASES = ("F0.7", "F0.7b", "F1", "F1b", "F2b_B1", "F2b_B2", "F2b_B3", "F3")
FREEZE_KEYS = ("torch", "transformers", "numpy", "scipy", "pandas", "pyarrow", "huggingface_hub", "accelerate",
               "vllm", "sentence-transformers", "scikit-learn", "tokenizers", "safetensors")


def _env_row(env: dict) -> dict:
    gpu = env.get("gpu") or {}
    row = {"gpu": gpu.get("name"), "mem_gb": gpu.get("mem_gb"), "capability": gpu.get("capability"),
           "python": env.get("python"), "cuda": env.get("cuda_version"), "code_hash": env.get("code_hash"),
           "platform": env.get("platform")}
    for k, v in (env.get("versions") or {}).items():
        row[k] = v
    return row


def _summ(rows: list) -> dict:
    return {k: dict(Counter(str(r.get(k)) for r in rows).most_common())
            for k in sorted({k for r in rows for k in r})}


def b0_entorno(study) -> dict:
    """GPU, Python, CUDA, hash del código y versiones de librerías: por fase (config.json de cada corrida) y por
    sesión (logs/env_*.json, que cubren también F0)."""
    from .pipeline import iter_run_dirs
    fases = {}
    for ph in ENV_PHASES:
        rows = []
        for rd in iter_run_dirs(study, ph):
            env = (load_json(os.path.join(rd, "config.json")) or {}).get("env")
            if env:
                rows.append(_env_row(env))
        if rows:
            fases[ph] = {"corridas": len(rows), "valores": _summ(rows)}
    sesiones = []
    for p in sorted(glob.glob(study.paths.d("logs", "env_*.json"))):
        e = load_json(p) or {}
        sesiones.append({"archivo": os.path.basename(p), "hora": e.get("time"), **_env_row(e)})
    freeze = {}
    fz = study.paths.d("logs", "pip_freeze.txt")
    if os.path.exists(fz):
        for line in open(fz, encoding="utf-8", errors="replace"):
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line.strip())
            if m and m.group(1).lower().replace("_", "-") in {k.replace("_", "-") for k in FREEZE_KEYS}:
                freeze[m.group(1)] = m.group(2)
    return {"por_fase": fases, "sesiones": sesiones, "pip_freeze_primera_sesion": freeze,
            "esta_sesion_nvidia_smi": nvidia_smi(),
            "nota": "El driver de NVIDIA no se registró en las sesiones anteriores (env_snapshot guarda GPU, CUDA y "
                    "versiones de librerías); el de esta sesión queda arriba."}


def nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


ALT_ARMS = ("simsiam_x", "byol_x", "dino_x", "swav_x")


def b0_puntuaciones_secundarias(study, doc_ids: np.ndarray, B: int = B_BOOT) -> list:
    """Puntuaciones secundarias pre-declaradas (predictor en SimSiam-x y BYOL-x; prototipos en DINO-x y SwAV-x):
    se calcularon sólo en T-gem del test sellado (correct_tgem_alt.npy). Se comparan con el coseno, pareado."""
    from .stats import bootstrap_means, clusters_from_docs
    cl = clusters_from_docs(doc_ids)
    out = []
    for arm in ALT_ARMS:
        rds = _runs(study, "F3", arm)
        A = [_load_correct(rd, "tgem_alt") for rd in rds]
        C = [_load_correct(rd, "tgem") for rd in rds]
        seeds = [{"semilla": (read_done(rd) or {}).get("seed"),
                  "alt_top1": (((read_done(rd) or {}).get("test") or {}).get("tgem") or {}).get("alt_top1"),
                  "top1": (((read_done(rd) or {}).get("test") or {}).get("tgem") or {}).get("top1")} for rd in rds]
        rec = {"brazo": arm, "semillas": seeds, "n_corridas": len(rds)}
        if rds and all(x is not None for x in A + C) and len({len(x) for x in A + C}) == 1 and len(A[0]) == len(cl):
            Am, Cm = np.vstack(A), np.vstack(C)
            boots, point = bootstrap_means({"alt": Am, "cos": Cm, "d": Am - Cm}, cl, B=B, seed=0)
            d = boots["d"]                    # misma corrida con las dos puntuaciones: se remuestrea la diferencia
            se = float(d.std(ddof=1))
            rec.update(alt=point["alt"], alt_ci=[float(np.percentile(boots["alt"], 2.5)),
                                                 float(np.percentile(boots["alt"], 97.5))],
                       cos=point["cos"], cos_ci=[float(np.percentile(boots["cos"], 2.5)),
                                                 float(np.percentile(boots["cos"], 97.5))],
                       alt_menos_cos={"diff": point["d"], "ci95_lo": float(np.percentile(d, 2.5)),
                                      "ci95_hi": float(np.percentile(d, 97.5)),
                                      "p": float(2 * _norm_sf(abs(point["d"] / se))) if se > 0 else 0.0})
        else:
            rec["nota"] = "faltan correct_tgem_alt.npy o correct_tgem.npy en alguna corrida de F3"
        out.append(rec)
    return out


def _stage_sig(done: dict, st: str):
    s = ((done or {}).get("stages") or {}).get(st) or {}
    return tuple(s.get(k) for k in ("steps", "best", "best_step", "final_val"))


def b0_f1b_vs_f3(study) -> dict:
    """¿F3 repitió los modelos de F1b? F3 re-entrenó con la misma configuración, los mismos datos y las semillas
    0–4: compara val_best y la trayectoria de las dos etapas para las semillas comunes (0–2). También la variante
    "HN solos" de F3 frente a F2b_B1 (misma configuración)."""
    from .losses import arm_names
    pares = [(a, ("F1b", a), ("F3", a)) for a in arm_names()]
    pares.append(("infonce_hn_hnonly", ("F2b_B1", "infonce_hn_hnonly"), ("F3", "infonce_hn_hnonly")))
    rows = []
    for label, (p1, n1), (p3, n3) in pares:
        for s in study.f1_seeds:
            r1, r3 = study.paths.d("runs", p1, n1, f"seed_{s}"), study.paths.d("runs", p3, n3, f"seed_{s}")
            d1, d3 = read_done(r1), read_done(r3)
            if not d1 or not d3 or d1.get("status") != "ok" or d3.get("status") != "ok":
                rows.append({"brazo": label, "semilla": s, "estado": "falta alguna de las dos corridas"})
                continue
            t1 = (load_json(os.path.join(r1, "config.json")) or {}).get("train") or {}
            t3 = (load_json(os.path.join(r3, "config.json")) or {}).get("train") or {}
            dif = sorted(k for k in set(t1) | set(t3) if t1.get(k) != t3.get(k))
            v1, v3 = d1.get("val_best"), d3.get("val_best")
            if not (isinstance(v1, (int, float)) and isinstance(v3, (int, float))):
                rows.append({"brazo": label, "semilla": s, "estado": "sin val_best en alguna de las dos"})
                continue
            tray = all(_stage_sig(d1, st) == _stage_sig(d3, st) for st in ("pre", "mid"))
            rows.append({"brazo": label, "semilla": s, "val_best_ref": v1, "val_best_F3": v3,
                         "identico": bool(v1 == v3 and tray), "misma_trayectoria": bool(tray),
                         "dif_val_best": float(v3) - float(v1),
                         "config_distinta_en": dif, "codigo_ref": d1.get("code_hash_start"),
                         "codigo_F3": d3.get("code_hash_start")})
    comp = [r for r in rows if "identico" in r]
    return {"filas": rows, "comparadas": len(comp), "identicas": sum(r["identico"] for r in comp),
            "max_abs_dif_val_best": max((abs(r["dif_val_best"]) for r in comp if r["dif_val_best"] is not None),
                                        default=None),
            "config_distinta": sorted({k for r in comp for k in r["config_distinta_en"]})}


def b0_friedman(study) -> dict:
    """Ranking de Friedman de F3 con los 8 bloques pre-especificados (control: tiene que reproducir el guardado) y sin
    el bloque de T-gem con estados barajados, donde más acierto significa más preferencia sin estado."""
    from .stats import friedman_nemenyi
    rep = load_json(study.paths.d("results", "F3", "all_vs_all", "all_vs_all.json"))
    if not rep or "friedman" not in rep:
        return {"nota": "falta results/F3/all_vs_all/all_vs_all.json con su Friedman"}
    arms = list(rep["friedman"]["arms"])
    m = rep["metrics"]

    def block(name):
        if f"{name}_top1" in m:
            d = {x["arm"]: x["mean"] for x in m[f"{name}_top1"]["summary"]}
        else:
            d = m[name]["point"]
        return [d[a] for a in arms]
    b8 = list(rep["friedman"]["blocks"])
    f8 = friedman_nemenyi(np.array([block(b) for b in b8]), arms)
    b7 = [b for b in b8 if b != "tgem_shuf"]
    f7 = friedman_nemenyi(np.array([block(b) for b in b7]), arms)
    ok = bool(np.allclose(f8["avg_rank"], rep["friedman"]["avg_rank"]))
    return {"bloques_8": b8, "bloques_7": b7, "reproduce_el_guardado": ok,
            "rango_8": dict(zip(arms, f8["avg_rank"])), "rango_7": dict(zip(arms, f7["avg_rank"])),
            "friedman_p_8": f8.get("friedman_p"), "friedman_p_7": f7.get("friedman_p"),
            "cd_8": f8.get("critical_difference"), "cd_7": f7.get("critical_difference")}


def b0_b4(study, doc_ids: np.ndarray, B: int = B_BOOT) -> dict:
    """B4: con estados barajados, ¿lo que los HN agregan en el macro de T-gen del sellado se distingue de 0?
    Descriptivo. Por par: Δ en el macro barajado y, de referencia, en el macro real y en T-gem barajado."""
    from .generators import load_panel, macro_for_run
    from .stats import clusters_from_docs, contrast
    panel = load_panel(study, "confirm") or {}
    if not panel.get("macro"):
        return {"nota": "no hay panel del confirmatorio"}
    it = np.asarray(panel["macro_items"], np.int64)
    cl_m, cl_all = clusters_from_docs(doc_ids[it]), clusters_from_docs(doc_ids)
    out = []
    for a, b in (("infonce_hn", "infonce"), ("siglip_hn", "siglip")):
        ga, gb = _runs(study, "F3", a), _runs(study, "F3", b)
        if not ga or not gb:
            continue
        for rd in ga + gb:
            macro_for_run(rd, panel)
        rec = {"par": f"{a} − {b}", "semillas": [len(ga), len(gb)]}
        for key, cl, n in (("tgenmacro_shuf", cl_m, len(it)), ("tgenmacro", cl_m, len(it)),
                           ("tgem_shuf", cl_all, len(doc_ids))):
            A = [_load_correct(rd, key) for rd in ga]
            Bm = [_load_correct(rd, key) for rd in gb]
            if any(x is None or len(x) != n for x in A + Bm):
                continue
            c = contrast(np.vstack(A), np.vstack(Bm), cl, ROPE, B=B)
            rec[key] = {k: c[k] for k in ("diff", "ci95_lo", "ci95_hi", "p", "tost_equivalent")}
        out.append(rec)
    return {"pares": out, "panel_version": panel.get("version"), "n_items_macro": int(len(it))}


def b0_markdown(r: dict) -> str:
    L = ["# Bloque 0 — extracción de lo que ya está en Drive", "", f"Generado: {r.get('creado')}", ""]
    env = r.get("entorno") or {}
    L += ["## Entorno por fase (config.json de cada corrida)", "", "| fase | corridas | GPU | CUDA | torch | transformers | numpy | hash del código |",
          "|---|---|---|---|---|---|---|---|"]
    for ph, v in (env.get("por_fase") or {}).items():
        val = v["valores"]
        f = lambda k: ", ".join(f"{x} ({n})" for x, n in (val.get(k) or {}).items())  # noqa: E731
        L.append(f"| {ph} | {v['corridas']} | {f('gpu')} | {f('cuda')} | {f('torch')} | {f('transformers')} | "
                 f"{f('numpy')} | {f('code_hash')} |")
    if env.get("pip_freeze_primera_sesion"):
        L += ["", "Versiones en `logs/pip_freeze.txt` (primera sesión, F0): "
              + ", ".join(f"{k} {v}" for k, v in env["pip_freeze_primera_sesion"].items())]
    L += ["", f"Sesiones registradas en `logs/env_*.json`: {len(env.get('sesiones') or [])} (detalle en extraccion.json). "
          f"{env.get('nota', '')} Esta sesión (nvidia-smi: GPU, driver, memoria): {env.get('esta_sesion_nvidia_smi') or '—'}.", ""]
    ide = r.get("f1b_vs_f3") or {}
    L += ["## ¿F3 repitió los modelos de F1b?", "",
          f"**{ide.get('identicas')} de {ide.get('comparadas')} corridas comparadas son idénticas** (semillas 0–2; mismo "
          f"val_best y misma trayectoria en las dos etapas). Máxima |Δ val_best|: {_num(ide.get('max_abs_dif_val_best'))}. "
          f"Campos de configuración distintos: {ide.get('config_distinta') or 'ninguno'}. Sin comparar (falta alguna "
          f"corrida o su val_best): {sum(1 for x in ide.get('filas', []) if 'identico' not in x)}.", "",
          "Si son idénticas, F3 no es una réplica independiente del entrenamiento: es la misma evaluación sobre datos "
          "nuevos (el sellado). La evidencia confirmatoria no cambia.", ""]
    no = [x for x in ide.get("filas", []) if x.get("identico") is False]
    if no:
        L += ["| brazo | semilla | val_best ref. | val_best F3 | Δ |", "|---|---|---|---|---|"]
        for x in no:
            L.append(f"| {x['brazo']} | {x['semilla']} | {_num(x['val_best_ref'])} | {_num(x['val_best_F3'])} | "
                     f"{_num(x['dif_val_best'])} |")
        L.append("")
    fr = r.get("friedman") or {}
    if "rango_8" in fr:
        L += ["## Friedman de F3 con y sin el bloque de T-gem barajado", "",
              f"Control: la versión de 8 bloques reproduce la guardada: **{'sí' if fr['reproduce_el_guardado'] else 'NO'}**. "
              f"p de Friedman: 8 bloques {fr['friedman_p_8']:.3g}; 7 bloques {fr['friedman_p_7']:.3g}. "
              f"Diferencia crítica (Nemenyi): {fr['cd_8']:.2f} y {fr['cd_7']:.2f}.", "",
              "| brazo | rango (8 bloques) | rango (7, sin T-gem barajado) |", "|---|---|---|"]
        for a in sorted(fr["rango_7"], key=fr["rango_7"].get):
            L.append(f"| {a} | {fr['rango_8'][a]:.2f} | {fr['rango_7'][a]:.2f} |")
        L.append("")
    L += ["## Puntuaciones secundarias pre-declaradas (sólo T-gem del sellado)", "",
          "| brazo | top-1 con la puntuación secundaria | top-1 con el coseno | secundaria − coseno (pp) [IC95] |",
          "|---|---|---|---|"]
    for x in r.get("puntuaciones_secundarias") or []:
        if "alt" in x:
            d = x["alt_menos_cos"]
            L.append(f"| {x['brazo']} | {pct(x['alt'])} [{pct(x['alt_ci'][0])}, {pct(x['alt_ci'][1])}] | {pct(x['cos'])} | "
                     f"{100 * d['diff']:+.1f} [{100 * d['ci95_lo']:+.1f}, {100 * d['ci95_hi']:+.1f}] |")
        else:
            L.append(f"| {x['brazo']} | — | — | {x.get('nota')} |")
    b4 = r.get("b4") or {}
    L += ["", "## B4: lo que agregan los HN con estados barajados (sellado; descriptivo)", "",
          "| par | macro de T-gen barajado (pp) | macro de T-gen real (pp) | T-gem barajado (pp) |", "|---|---|---|---|"]
    for x in b4.get("pares") or []:
        def cell(k):
            c = x.get(k)
            return "—" if not c else f"{100 * c['diff']:+.2f} [{100 * c['ci95_lo']:+.2f}, {100 * c['ci95_hi']:+.2f}]"
        L.append(f"| {x['par']} | {cell('tgenmacro_shuf')} | {cell('tgenmacro')} | {cell('tgem_shuf')} |")
    return "\n".join(L) + "\n"


def bloque0(study, log=print) -> dict:
    """Bloque 0 completo. No entrena ni descarga nada: sólo lee las corridas y los resultados en Drive."""
    out = pdir(study, "b0_extraccion")
    write_rule(study, "b0_extraccion", {
        "que": "Extracción descriptiva de lo que ya está en Drive; no entrena, no descarga y no cambia ningún resultado.",
        "F1b_vs_F3": "Idénticas = mismo val_best y misma trayectoria (pasos, mejor valor, mejor paso, valor final) en "
                     "pre y mid. Si lo son, 'F3 replica F1b' se reporta como réplica sobre datos de evaluación nuevos, "
                     "no como entrenamientos independientes.",
        "Friedman": "Se reporta el pre-especificado (8 bloques) y, como sensibilidad declarada, el de 7 sin T-gem "
                    "barajado (P21).",
        "B4": "Descriptivo: InfoNCE+HN − InfoNCE (y SigLIP+HN − SigLIP) en el macro de T-gen barajado del sellado, "
              "bootstrap pareado por documento (B = 4000, semilla 0).",
        "puntuaciones_secundarias": "Descriptivo, sólo T-gem del sellado (donde se calcularon): top-1 con la puntuación "
                                    "secundaria pre-declarada (predictor en SimSiam-x y BYOL-x; prototipos en DINO-x y "
                                    "SwAV-x) y con el coseno, y su diferencia dentro de cada corrida (bootstrap por "
                                    "documento y semilla, B = 4000)."}, log=log)
    doc_ids = _confirm_doc_ids(study)
    r = {"creado": now_iso(), "entorno": b0_entorno(study)}
    log("[b0] entorno por fase y por sesión: listo")
    r["f1b_vs_f3"] = b0_f1b_vs_f3(study)
    log(f"[b0] F3 frente a F1b: {r['f1b_vs_f3']['identicas']}/{r['f1b_vs_f3']['comparadas']} corridas idénticas")
    r["friedman"] = b0_friedman(study)
    r["puntuaciones_secundarias"] = b0_puntuaciones_secundarias(study, doc_ids)
    r["b4"] = b0_b4(study, doc_ids)
    save_json(r, os.path.join(out, "extraccion.json"))
    md = b0_markdown(r)
    write_text(os.path.join(out, "EXTRACCION.md"), md)
    log(f"[b0] guardado en {out}")
    return r


# ======================================================================================
# BLOQUE 1 — calidad de los sets del sellado: falsos negativos y fuga entre splits
def _tokenizer():
    from transformers import AutoTokenizer
    from .config import ENCODER_MODEL
    return AutoTokenizer.from_pretrained(ENCODER_MODEL)


def textos_split(study, split: str, lookup_fn=None, tok=None, log=print) -> dict:
    """Pregunta, oro y los 10 HN de Gemini (texto) de TODOS los ítems del set de un split, decodificados de los tokens
    públicos de Nemotron-DQA con el tokenizer de Qwen3 (igual que texts.decode_items). Caché en
    results/PRE/textos/<split>.jsonl (sólo agrega lo que falta; el archivo se escribe entero al final, de una vez:
    en Drive, agregar línea por línea es lento). Devuelve {i: {"q", "a", "negs"}}."""
    from .data import EvalSet
    path = pdir(study, "textos", f"{split}.jsonl")
    ev = EvalSet(study.paths.d("data", "eval", split))
    have = {int(r["i"]): r for r in read_jsonl(path)}
    todo = [i for i in range(ev.n) if i not in have]
    if todo:
        if lookup_fn is None:
            from .texts import lookup as lookup_fn
        tok = tok or _tokenizer()
        docs = {ev.doc_ids[i] for i in todo}
        log(f"[textos] {split}: {len(todo)} ítems de {len(docs)} documentos (tokens de Nemotron-DQA)")
        qa = {(r["doc_id"], int(r["pair_idx"])): r for r in
              lookup_fn(study.paths, docs, "qa", ["doc_id", "pair_idx", "question_ids", "answer_ids"])}
        negm: dict = {}
        for r in lookup_fn(study.paths, docs, "neg", ["doc_id", "pair_idx", "neg_idx", "neg_ids"]):
            negm.setdefault((r["doc_id"], int(r["pair_idx"])), {})[int(r["neg_idx"])] = r["neg_ids"]
        recs = []
        for i in todo:
            key = (ev.doc_ids[i], int(ev.pair_idx[i]))
            r = qa.get(key)
            if r is None:
                recs.append({"i": i, "missing": True})
                continue
            negs = [v for _, v in sorted(negm.get(key, {}).items())][: ev.k]
            recs.append({"i": i, "doc_id": key[0], "pair_idx": key[1], "q_ids": r["question_ids"],
                         "a_ids": r["answer_ids"], "n_ids": negs})
        flat = [x for r in recs if not r.get("missing") for x in [r["q_ids"], r["a_ids"], *r["n_ids"]]]
        dec = iter(tok.batch_decode(flat)) if flat else iter(())
        for r in recs:
            if not r.get("missing"):
                r["q"], r["a"] = next(dec), next(dec)
                r["negs"] = [next(dec) for _ in r["n_ids"]]
                for k in ("q_ids", "a_ids", "n_ids"):
                    r.pop(k)
            have[int(r["i"])] = r
        makedirs(os.path.dirname(path))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for i in sorted(have):
                f.write(json.dumps(have[i], ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        miss = sum(1 for r in have.values() if r.get("missing"))
        log(f"[textos] {split}: {len(have) - miss} ítems con texto" + (f"; {miss} sin texto en el dataset" if miss else ""))
    return {i: r for i, r in have.items() if not r.get("missing")}


def textos_tgen(study, split: str, keys) -> tuple:
    """({clave: {i: {"pos", "negs"}}}, {i: {"q", "a"}}) del panel T-gen de un split, verificando que kept.jsonl sea el
    set embebido vigente (mismos ítems y orden que items.npy). "pos" es el positivo propio del set (construcción P;
    None si el positivo es el oro original, como en la construcción E del panel congelado)."""
    from .generators import load_gen, tgen_root
    root = tgen_root(study, split)
    txt = {int(r["i"]): {"q": r.get("question"), "a": r.get("answer")}
           for r in read_jsonl(os.path.join(root, "items_text.jsonl")) if not r.get("missing")}
    out = {}
    for k in keys:
        kept = read_jsonl(os.path.join(root, k, "kept.jsonl"))
        g = load_gen(study, k, split)
        if g is None or [int(r["i"]) for r in kept] != [int(x) for x in g.items]:
            raise RuntimeError(f"{split}/{k}: kept.jsonl no coincide con el set embebido (items.npy)")
        out[k] = {int(r["i"]): {"pos": r.get("positive") or None, "negs": list(r["negatives"])} for r in kept}
    return out, txt


def qc_items(study, tx: dict, tg: dict, ttxt: dict, keys) -> dict:
    """{"<set>:<i>": {"q", "a", "cands"}} de los sets que juzga el juez, en el sellado: T-gem (HN de Gemini), T-mine
    (oros de los ítems minados) y cada generador del macro (sus ítems del macro). La referencia es siempre el oro
    original: la pregunta es si un candidato que el set da por incorrecto responde bien."""
    from .generators import load_panel
    mine = np.load(study.paths.d("data", "eval", "confirm", "mine_idx.npy"))
    panel = load_panel(study, "confirm") or {}
    macro_items = [int(i) for i in panel.get("macro_items") or []]
    out = {}
    for i in sorted(tx):
        r = tx[i]
        if len(r.get("negs") or []) == 10:
            out[f"tgem:{i}"] = {"q": r["q"], "a": r["a"], "cands": list(r["negs"])}
        js = [int(j) for j in mine[i]]
        if all(j in tx for j in js):
            out[f"tmine:{i}"] = {"q": r["q"], "a": r["a"], "cands": [tx[j]["a"] for j in js]}
    for k in keys:
        for i in macro_items:
            if i in tg[k] and (i in tx or i in ttxt):
                q, a = (tx[i]["q"], tx[i]["a"]) if i in tx else (ttxt[i]["q"], ttxt[i]["a"])
                out[f"tgen_{k}:{i}"] = {"q": q, "a": a, "cands": list(tg[k][i]["negs"])}
    return out


def set_of(key: str) -> str:
    return key.split(":", 1)[0]


def export_hand_labels(study, items: dict, keys, log=print) -> str:
    """CSV para etiquetar a mano (200 ítems × 10 candidatos), estratificado por set y sin decir de qué set es cada
    ítem. Se exporta una sola vez, antes de correr el juez. La clave (ítem → set) va en otro archivo."""
    import pandas as pd
    d = makedirs(pdir(study, "b1_calidad", "etiquetas"))
    csv = os.path.join(d, "ETIQUETAR_200_items.csv")
    if os.path.exists(csv):
        return csv
    rs = np.random.default_rng([SEED, 7])
    pick = []
    for s in list(JUDGE_SETS_FIXED) + [f"tgen_{k}" for k in keys]:
        pool = sorted(k for k in items if set_of(k) == s)
        n = N_HAND["tgen"] if s.startswith("tgen_") else N_HAND[s]
        if pool:
            pick += [pool[j] for j in rs.choice(len(pool), size=min(n, len(pool)), replace=False)]
    order = rs.permutation(len(pick))
    rows, clave = [], {}
    for j, o in enumerate(order):
        key = pick[int(o)]
        iid = f"e{j + 1:03d}"
        clave[iid] = key
        it = items[key]
        for n, c in enumerate(it["cands"], 1):
            rows.append({"id": iid, "pregunta": it["q"], "oro": it["a"], "n": n, "candidato": c, QC_COL: "",
                         "notas": ""})
    pd.DataFrame(rows).to_csv(csv, index=False, encoding="utf-8-sig")
    save_json({"creado": now_iso(), "clave": clave, "por_set": dict(Counter(set_of(v) for v in clave.values()))},
              os.path.join(d, "clave.json"))
    write_text(os.path.join(d, "LEEME.md"), HAND_README)
    log(f"[b1] CSV para etiquetar a mano: {csv} ({len(clave)} ítems, {len(rows)} filas)")
    return csv


HAND_README = """# Etiquetas a mano (bloque 1 del notebook pre-envío)

1. Descarga `ETIQUETAR_200_items.csv` (200 ítems, 10 candidatos cada uno: 2,000 filas).
2. En la columna `es_correcta(0/1)` pon **1** si el candidato también responde bien la pregunta (dice lo mismo que el oro,
   con otras palabras o formato, o es correcto por otra razón) y **0** si no. Sin el documento, juzga contra el oro.
3. No mires las etiquetas del juez antes de terminar: el acuerdo sólo vale si las etiquetas son ciegas.
4. Guárdalo como `ETIQUETADAS.csv` o `ETIQUETADAS.xlsx` en esta misma carpeta y vuelve a correr el notebook
   (Ejecutar todo): lo terminado se salta y se calcula el acuerdo.

Este archivo no alimenta ninguna regla del notebook principal (los CSV de `data/eval/tgen/` sí; no los marques).
"""


def judge_raw_path(study) -> str:
    return pdir(study, "b1_calidad", "juez", "juez_raw.jsonl")


def judge_labels(study) -> dict:
    """{clave: [10 etiquetas 0/1]} con la primera respuesta válida de cada ítem."""
    out = {}
    for r in read_jsonl(judge_raw_path(study)):
        if r.get("labels") is not None and r["i"] not in out:
            out[r["i"]] = [int(x) for x in r["labels"]]
    return out


def run_judge(study, items: dict, backend: str = "hf", batch: int = 64, max_attempts: int = 2,
              python: str | None = None, hf_home: str | None = None, purge: bool = True, log=print) -> dict:
    """Corre el juez en un subproceso (clm_losses.genworker judge): un prompt por ítem, decodificación voraz.
    Reanudable: los ítems ya juzgados se saltan. Libera la GPU y borra los pesos al terminar."""
    from .genworker import JUDGE_PROMPT, JUDGE_PROMPT_VERSION, judge_prompt
    from .generators import (check_access, free_gb, model_revision, purge_hf_caches, purge_model_cache,
                             worker_env, _tf_version)
    jdir = makedirs(pdir(study, "b1_calidad", "juez"))
    raw = judge_raw_path(study)
    have = judge_labels(study)
    att = Counter(r["i"] for r in read_jsonl(raw))
    pending = [k for k in items if k not in have and att.get(k, 0) < max_attempts]
    info_p = os.path.join(jdir, "juez_info.json")
    info = load_json(info_p, {}) or {}
    if not pending:
        n_ok = sum(1 for k in items if k in have)
        log(f"[juez] ya juzgado: {n_ok}/{len(items)} ítems con etiquetas válidas")
        info.update(n_items=len(items), n_etiquetados=n_ok)
        save_json(info, info_p)
        return info
    hf_home = hf_home or os.environ.get("HF_HOME") or os.path.join(study.paths.cache_root, "hf_home")
    if backend != "fake":
        ok, why = check_access(JUDGE["repo"])
        if not ok:
            raise RuntimeError(f"no se puede bajar el juez {JUDGE['repo']}: {why}")
        fr = purge_hf_caches(hf_home, keep_repos=(JUDGE["repo"],))
        log(f"[juez] disco: {fr['free_after_gb']} GB libres" + (f" (se borró {', '.join(fr['removed'])})" if fr["removed"] else ""))
        if free_gb(hf_home) < 75:
            log("[juez] ATENCIÓN: el juez ocupa ~65 GB y quedan menos de 75 GB libres")
    prompts_p = os.path.join(study.paths.cache_root, "pre_juez", "prompts.jsonl")
    makedirs(os.path.dirname(prompts_p))
    with open(prompts_p + ".tmp", "w") as f:
        for k in sorted(items):
            it = items[k]
            f.write(json.dumps({"i": k, "prompt": judge_prompt(it["q"], it["a"], it["cands"]), "n": len(it["cands"]),
                                "max_new_tokens": 16 + 8 * len(it["cands"])}, ensure_ascii=False) + "\n")
    os.replace(prompts_p + ".tmp", prompts_p)
    cmd = [python or sys.executable, "-m", "clm_losses.genworker", "judge", "--model", JUDGE["repo"], "--prompts",
           prompts_p, "--out", raw, "--backend", backend, "--batch", str(batch), "--max-attempts", str(max_attempts),
           "--dtype", JUDGE["dtype"], "--chat-kwargs", json.dumps(JUDGE["chat_kwargs"])]
    env = worker_env(hf_home, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log(f"[juez] {JUDGE['repo']}: {len(pending)} ítems pendientes de {len(items)} (backend {backend}, batch {batch})")
    t0, tail = time.time(), []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1,
                            start_new_session=True)
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-40:]
        if line.startswith("[genworker]"):
            log(line.replace("[genworker]", "[juez]"))
    rc = proc.wait()
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    secs = time.time() - t0
    have = judge_labels(study)
    n_ok = sum(1 for k in items if k in have)
    info.update(repo=JUDGE["repo"], revision=model_revision(JUDGE["repo"]) if backend != "fake" else None,
                dtype=JUDGE["dtype"], license=JUDGE["license"], por_que=JUDGE["por_que"],
                prompt_version=JUDGE_PROMPT_VERSION, prompt=JUDGE_PROMPT, decodificacion="voraz (sin muestreo)",
                transformers=_tf_version(), backend=backend, n_items=len(items), n_etiquetados=n_ok,
                updated=now_iso(), seconds_total=round(float(info.get("seconds_total", 0.0)) + secs, 1))
    if rc != 0:
        info["error_tail"] = tail[-15:]
        save_json(info, info_p)
        raise RuntimeError(f"el juez falló (código {rc}). Últimas líneas:\n  " + "\n  ".join(tail[-15:]))
    save_json(info, info_p)
    if purge and backend != "fake":
        gb = purge_model_cache(JUDGE["repo"], hf_home)
        log(f"[juez] borrados {gb:.1f} GB de pesos · libres {free_gb(hf_home):.1f} GB")
    log(f"[juez] listo en {secs / 60:.1f} min: {n_ok}/{len(items)} ítems con etiquetas válidas")
    return info


def fn_rates(labels: dict, sets, n_por_set: dict | None = None) -> dict:
    """Tasa de falsos negativos por set: por ítem (ítems con al menos un candidato marcado) y por negativo, con IC95
    de Wilson (la de negativos ignora el agrupamiento por ítem: es aproximada). n_por_set: ítems del set (para
    reportar cuántos quedaron sin etiqueta válida del juez)."""
    out = {}
    for s in sets:
        L = [v for k, v in labels.items() if set_of(k) == s]
        n = len(L)
        ki = sum(1 for v in L if any(v))
        kn = sum(sum(v) for v in L)
        nn = sum(len(v) for v in L)
        out[s] = {"n_items": n, "items_del_set": (n_por_set or {}).get(s), "items_con_fn": ki,
                  "tasa_item": ki / n if n else None, "ic_item": wilson(ki, n),
                  "negativos_fn": kn, "n_negativos": nn, "tasa_negativo": kn / nn if nn else None,
                  "ic_negativo": wilson(kn, nn)}
    return out


def read_hand_labels(study) -> dict | None:
    """{(clave, n): 0/1} de ETIQUETADAS.csv o .xlsx (las filas sin marca 0/1 se ignoran); None si no hay archivo."""
    import pandas as pd
    d = pdir(study, "b1_calidad", "etiquetas")
    clave = (load_json(os.path.join(d, "clave.json")) or {}).get("clave") or {}
    df = None
    for fn in ("ETIQUETADAS.csv", "ETIQUETADAS.xlsx"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            df = _read_csv_any(p) if fn.endswith(".csv") else pd.read_excel(p, dtype={"id": str})
            break
    if df is None or not clave:
        return None
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    col = next((c for c in df.columns if str(c).strip().lower().startswith("es_correcta")), None)
    if col is None or "id" not in df.columns or "n" not in df.columns:
        raise RuntimeError("ETIQUETADAS: faltan las columnas id, n o es_correcta(0/1)")
    df["_m"] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["_m"].isin([0, 1])]
    out = {}
    for iid, n, m in zip(df["id"].astype(str).str.strip(), df["n"], df["_m"]):
        if iid in clave:
            out[(clave[iid], int(n))] = int(m)
    return out


def _read_csv_any(p: str):
    """CSV guardado por pandas, Excel o Google Sheets: UTF-8 (con o sin BOM) o Latin-1, separado por coma o por punto
    y coma (Excel en español usa punto y coma)."""
    import pandas as pd
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(p, dtype={"id": str}, sep=None, engine="python", encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last = e
    raise RuntimeError(f"no pude leer {p}: {last}")


def agreement(judge: dict, hand: dict, sets) -> dict:
    """Acuerdo juez–humano por set y total (sobre los negativos etiquetados a mano), y la tasa por ítem según las
    etiquetas a mano (sólo ítems con sus 10 candidatos marcados)."""
    out = {}
    groups = {s: [] for s in sets}
    items_h: dict = {}
    for (key, n), h in hand.items():
        items_h.setdefault(key, {})[n] = h
        if key in judge and 1 <= n <= len(judge[key]):
            groups.setdefault(set_of(key), []).append((judge[key][n - 1], h))
    allp = [p for v in groups.values() for p in v]
    for s, pairs in list(groups.items()) + [("total", allp)]:
        j = np.array([a for a, _ in pairs], int)
        h = np.array([b for _, b in pairs], int)
        tp, fp = int(((j == 1) & (h == 1)).sum()), int(((j == 1) & (h == 0)).sum())
        fn, tn = int(((j == 0) & (h == 1)).sum()), int(((j == 0) & (h == 0)).sum())
        rec = {"n_negativos": int(len(pairs)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "sensibilidad": tp / (tp + fn) if tp + fn else None,
               "especificidad": tn / (tn + fp) if tn + fp else None,
               "kappa": cohen_kappa(j, h) if len(pairs) else None}
        if s != "total":
            full = [v for k, v in items_h.items() if set_of(k) == s and len(v) >= 10]
            k = sum(1 for v in full if any(v.values()))
            rec.update(items_completos=len(full), items_con_fn_humano=k, tasa_item_humano=k / len(full) if full else None,
                       ic_item_humano=wilson(k, len(full)))
        out[s] = rec
    return out


# --- fuga entre splits ------------------------------------------------------------------
def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def shingles(s: str, k: int = 3) -> frozenset:
    w = s.split()
    if len(w) < k:
        return frozenset([tuple(w)])
    return frozenset(tuple(w[i:i + k]) for i in range(len(w) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def _iter_qa(paths, docs=None, exclude=None, batch_size: int = 200_000, downloader=None):
    """Lotes (doc_id, pair_idx, question_ids, answer_ids) de las partes de qa_pairs, sólo de los documentos `docs`
    (o de todos menos `exclude`). El filtro se hace en pyarrow antes de pasar a Python: las listas de tokens de las
    filas que no interesan nunca se convierten."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from .texts import QA_PARTS, fetch
    cols = ["doc_id", "pair_idx", "question_ids", "answer_ids"]
    vs = {}
    for p in fetch(paths, QA_PARTS, downloader):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=batch_size, columns=cols):
            d = b.column("doc_id")
            if docs is not None or exclude is not None:
                key = str(d.type)
                if key not in vs:
                    vs[key] = pa.array(sorted(docs if docs is not None else exclude), type=d.type)
                m = pc.is_in(d, value_set=vs[key])
                if docs is None:
                    m = pc.invert(m)
                if not bool(pc.any(m).as_py()):
                    continue
                b = b.filter(m)
            yield (b.column("doc_id").to_pylist(), b.column("pair_idx").to_pylist(),
                   b.column("question_ids").to_pylist(), b.column("answer_ids").to_pylist())


def leak_check(eval_items: dict, train_keys: set, qa_batches, decode, log=print) -> dict:
    """Núcleo de la fuga (sin E/S de Drive). eval_items: {split: [(doc_id, pair_idx)]}; train_keys: pares de
    entrenamiento; qa_batches(docs): iterador de lotes (docs, pidx, q_ids, a_ids) de esos documentos; decode: lista de
    listas de ids -> textos. Un ítem de evaluación tiene fuga si un par de entrenamiento tiene el mismo oro
    normalizado y una pregunta igual o casi igual (Jaccard ≥ 0.8 sobre 3-gramas de palabras); el bloqueo por oro hace
    el Jaccard exacto (no hace falta aproximarlo con MinHash)."""
    want = {}
    for sp, keys in eval_items.items():
        for j, k in enumerate(keys):
            want.setdefault((k[0], int(k[1])), []).append((sp, j))
    ev_txt = {}
    for docs, pidx, qids, aids in qa_batches({k[0] for k in want}):
        sel = [j for j, k in enumerate(zip(docs, pidx)) if (k[0], int(k[1])) in want]
        if sel:
            dq = decode([qids[j] for j in sel])
            da = decode([aids[j] for j in sel])
            for j, q, a in zip(sel, dq, da):
                ev_txt[(docs[j], int(pidx[j]))] = (norm_text(q), norm_text(a), q, a)
    golds = {v[1] for v in ev_txt.values()}
    by_gold: dict = {}                     # oro normalizado -> [(pregunta normalizada, pregunta, oro)] de entrenamiento
    exact_map: dict = {}                   # (oro, pregunta) normalizados -> (pregunta, oro) de entrenamiento
    train_q = set()
    n_train = 0
    for docs, pidx, qids, aids in qa_batches({k[0] for k in train_keys}):
        sel = [j for j, k in enumerate(zip(docs, pidx)) if (k[0], int(k[1])) in train_keys]
        if not sel:
            continue
        n_train += len(sel)
        da = decode([aids[j] for j in sel])
        dq = decode([qids[j] for j in sel])
        for q, a in zip(dq, da):
            nq = norm_text(q)
            train_q.add(nq)
            na = norm_text(a)
            if na in golds:
                by_gold.setdefault(na, []).append((nq, q, a))
                exact_map.setdefault((na, nq), (q, a))
    log(f"[fuga] {n_train} pares de entrenamiento leídos; {len(ev_txt)} ítems de evaluación con texto")
    index: dict = {}                       # por oro: shingles de cada candidato e índice invertido (oros frecuentes)

    def near_hit(na, nq):
        cands = by_gold.get(na) or []
        if not cands:
            return None
        sq = shingles(nq)
        lo, hi = JACCARD_MIN * len(sq), len(sq) / JACCARD_MIN
        if na not in index:
            sh = [shingles(cq) for cq, _, _ in cands]
            post: dict = {}
            if len(cands) > 64:
                for c, S in enumerate(sh):
                    for g in S:
                        post.setdefault(g, []).append(c)
            index[na] = (sh, post)
        sh, post = index[na]
        if post:
            # J ≥ 0.8 exige compartir al menos 0.8·|S| shingles: sólo esos candidatos se miran
            cnt = Counter(c for g in sq for c in post.get(g, ()))
            need = math.ceil(JACCARD_MIN * len(sq) - 1e-9)
            pool = [c for c, m in cnt.items() if m >= need]
        else:
            pool = range(len(cands))
        for c in pool:
            if lo <= len(sh[c]) <= hi and jaccard(sq, sh[c]) >= JACCARD_MIN:
                return cands[c]
        return None
    out = {}
    for sp, keys in eval_items.items():
        exact, near, qany, ej = [], [], [], []
        missing = 0
        for j, k in enumerate(keys):
            t = ev_txt.get((k[0], int(k[1])))
            if t is None:
                missing += 1
                continue
            nq, na, q, a = t
            if nq in train_q:
                qany.append(j)
            hit = None
            if (na, nq) in exact_map:
                hit = ("exacta", *exact_map[(na, nq)])
                exact.append(j)
            else:
                h = near_hit(na, nq)
                if h is not None:
                    hit = ("casi", h[1], h[2])
            if hit:
                near.append(j)
                if len(ej) < 8:
                    ej.append({"tipo": hit[0], "pregunta": q, "oro": a, "pregunta_entrenamiento": hit[1],
                               "oro_entrenamiento": hit[2]})
        n = len(keys) - missing
        leak = sorted(set(exact) | set(near))
        out[sp] = {"n_items": n, "sin_texto": missing, "par_exacto": len(exact), "par_exacto_o_casi": len(leak),
                   "tasa": len(leak) / n if n else None, "ic95": wilson(len(leak), n),
                   "pregunta_igual_cualquier_oro": len(qany), "tasa_pregunta_igual": len(qany) / n if n else None,
                   "items": leak, "ejemplos": ej, "supera_criterio": bool(n and len(leak) / n >= LEAK_MAX)}
    return out


def fuga(study, log=print, batch_size: int = 200_000) -> dict:
    """Fuga entre splits (DISENO_01 F0.4: < 0.5 % de los ítems de test): pares pregunta–oro de val, test y sellado
    repetidos o casi repetidos en los pares de entrenamiento (los de pre y los de mid, sin documentos de evaluación)."""
    from .data import EvalSet, Split, train_masks
    from .pipeline import make_store
    out = makedirs(pdir(study, "b1_calidad", "fuga"))
    res_p = os.path.join(out, "fuga.json")
    if os.path.exists(res_p):
        return load_json(res_p)
    evs = {sp: EvalSet(study.paths.d("data", "eval", sp)) for sp in ("val", "test", "confirm")}
    eval_items = {sp: list(zip(ev.doc_ids.tolist(), ev.pair_idx.tolist())) for sp, ev in evs.items()}
    store = make_store(study)
    sdir = study.paths.d("data", "splits")
    split = Split.load(sdir)
    d = study.data
    masks = train_masks(store, split, d.pre_chunks, sdir, "pre")
    extra = [c for c in d.hn_chunks if c not in d.pre_chunks]           # en el perfil colab, ninguno
    if extra:
        masks.update(train_masks(store, split, extra, sdir, "mid"))
    train_keys = set()
    for c in list(d.pre_chunks) + extra:
        ids, pidx = store.meta(c)
        m = masks[c]
        train_keys.update(zip(ids[m].tolist(), pidx[m].tolist()))
    log(f"[fuga] {len(train_keys)} pares de entrenamiento (pre y mid, sin documentos de evaluación)")
    tok = _tokenizer()

    def decode(lists):
        return tok.batch_decode(lists) if lists else []
    res = leak_check(eval_items, train_keys, lambda docs: _iter_qa(study.paths, docs=docs, batch_size=batch_size),
                     decode, log=log)
    res = {"creado": now_iso(), "criterio": f"fuga < {LEAK_MAX:.1%} de los ítems de test (DISENO_01, F0.4)",
           "metodo": "mismo oro normalizado y pregunta igual o con Jaccard ≥ 0.8 sobre 3-gramas de palabras "
                     "(normalización: NFKC, minúsculas, sin puntuación); entrenamiento = pares de los chunks de pre "
                     "y mid que no son de documentos de val, test ni sellado", "n_pares_entrenamiento": len(train_keys),
           "splits": res}
    save_json(res, res_p)
    return res


# --- sensibilidad de los contrastes de F3 ------------------------------------------------
def contrastes_f3(study, doc_ids: np.ndarray, excl: dict | None = None, B: int = B_BOOT) -> dict:
    """C1 (T-mine y macro de T-gen, Holm entre las dos), C2 (dos pares, Holm), C2c (dos pruebas, Holm) y su control,
    recalculados desde los aciertos por ítem de F3 con el mismo bootstrap (B = 4000, semilla 0). excl = {"tgem":
    ítems, "tmine": ítems, "tgen": {clave: ítems}, "todos": ítems} saca esos ítems (los de "todos", de todo). Sin
    exclusiones tiene que reproducir lo confirmatorio guardado."""
    from .analysis import _at, _tgem_at
    from .generators import load_panel, macro_for_run
    from .pipeline import freeze_path
    from .stats import clusters_from_docs, contrast, holm
    excl = excl or {}
    todos = set(int(x) for x in excl.get("todos", ()))
    x_tgem = set(int(x) for x in excl.get("tgem", ())) | todos
    x_tmine = set(int(x) for x in excl.get("tmine", ())) | todos
    x_tgen = {k: set(int(x) for x in v) | todos for k, v in (excl.get("tgen") or {}).items()}
    N = len(doc_ids)
    ok_tgem = np.array([i not in x_tgem for i in range(N)])
    ok_tmine = np.array([i not in x_tmine for i in range(N)])
    panel = load_panel(study, "confirm") or {}
    keys = list(panel.get("macro") or [])
    mit = np.asarray(panel.get("macro_items") or [], np.int64)
    frz = load_json(freeze_path(study)) or {}
    nc = frz.get("C1_best_noncontrastive", "vicreg_x")
    plain = not todos and not any(x_tgen.get(k) for k in keys)

    def macro(rd, shuf=False):
        if plain:
            # sin exclusiones: el macro guardado de la corrida (el mismo archivo que usaron C1 y C2)
            macro_for_run(rd, panel)
            c = _load_correct(rd, "tgenmacro_shuf" if shuf else "tgenmacro")
            pi = os.path.join(rd, "eval", "items_tgenmacro.npy")
            if c is None or not os.path.exists(pi) or not np.array_equal(np.load(pi), mit):
                return None, None
            return c, np.ones(len(mit), bool)
        cols = [_at(rd, f"tgen_{k}", mit, shuf) for k in keys]
        if any(c is None for c in cols):
            return None, None
        keep = np.stack([np.array([int(i) not in x_tgen.get(k, todos) for i in mit]) for k in keys])
        C = np.stack(cols)
        cnt = keep.sum(0)
        S = (C * keep).sum(0, dtype=np.float32)
        return (S / np.maximum(cnt, 1)).astype(np.float32), cnt > 0

    def stack(rds, name, sel):
        xs = [_load_correct(rd, name) for rd in rds]
        if not rds or any(x is None or len(x) != N for x in xs):
            return None
        return np.vstack(xs)[:, sel]

    def keepc(c):
        return {k: c[k] for k in ("diff", "ci95_lo", "ci95_hi", "p", "tost_equivalent")}

    out = {"exclusiones": {"tgem": int((~ok_tgem).sum()), "tmine": int((~ok_tmine).sum()),
                           "tgen": {k: len(x_tgen.get(k, set()) & set(mit.tolist())) for k in keys}}}
    # C1
    c1 = {}
    ga, gb = _runs(study, "F3", nc), _runs(study, "F3", "infonce")
    A, Bm = stack(ga, "tmine", ok_tmine), stack(gb, "tmine", ok_tmine)
    if A is not None and Bm is not None:
        c1["tmine"] = contrast(A, Bm, clusters_from_docs(doc_ids[ok_tmine]), ROPE, B=B)
        c1["tmine"]["n_items"] = int(ok_tmine.sum())
    if keys and ga and gb:
        Ma = [macro(rd) for rd in ga]
        Mb = [macro(rd) for rd in gb]
        if all(m[0] is not None for m in Ma + Mb):
            kk = np.logical_and.reduce([m[1] for m in Ma + Mb])
            c1["tgenmacro"] = contrast(np.vstack([m[0] for m in Ma])[:, kk], np.vstack([m[0] for m in Mb])[:, kk],
                                       clusters_from_docs(doc_ids[mit[kk]]), ROPE, B=B)
            c1["tgenmacro"]["n_items"] = int(kk.sum())
    if c1:
        for (m, c), p in zip(c1.items(), holm([c["p"] for c in c1.values()])):
            c["p_holm"] = float(p)
        out["C1"] = {"A": nc, "B": "infonce", **{m: {**keepc(c), "p_holm": c["p_holm"], "n_items": c["n_items"]}
                                                 for m, c in c1.items()}}
    # C2
    c2 = []
    if keys:
        for a, b in (("infonce_hn", "infonce"), ("siglip_hn", "siglip")):
            ga, gb = _runs(study, "F3", a), _runs(study, "F3", b)
            Ma, Mb = [macro(rd) for rd in ga], [macro(rd) for rd in gb]
            Ta, Tb = [_tgem_at(rd, mit) for rd in ga], [_tgem_at(rd, mit) for rd in gb]
            if not ga or not gb or any(m[0] is None for m in Ma + Mb) or any(t is None for t in Ta + Tb):
                continue
            kk = np.logical_and.reduce([m[1] for m in Ma + Mb]) & ok_tgem[mit]
            cl = clusters_from_docs(doc_ids[mit[kk]])
            Ta, Tb = np.vstack(Ta)[:, kk], np.vstack(Tb)[:, kk]
            Ma, Mb = np.vstack([m[0] for m in Ma])[:, kk], np.vstack([m[0] for m in Mb])[:, kk]
            c2.append({"par": f"{a} − {b}", "n_items": int(kk.sum()),
                       "Δ T-gem": keepc(contrast(Ta, Tb, cl, ROPE, B=B)),
                       "Δ macro T-gen": keepc(contrast(Ma, Mb, cl, ROPE, B=B)),
                       "C2": contrast(Ta - Ma, Tb - Mb, cl, ROPE, B=B)})
        for r, p in zip(c2, holm([r["C2"]["p"] for r in c2])):
            r["C2"] = {**keepc(r["C2"]), "p_holm": float(p)}
    out["C2"] = c2
    # C2c y su control
    tests = []
    ga, gb = _runs(study, "F3", "infonce_hn_hnonly"), _runs(study, "F3", "infonce_hn")
    for key, sel, sign in (("tgem_shuf", ok_tgem, +1), ("tmine", ok_tmine, -1)):
        A, Bm = stack(ga, key, sel), stack(gb, key, sel)
        if A is not None and Bm is not None:
            tests.append({"metrica": key, "signo": sign, "c": contrast(A, Bm, clusters_from_docs(doc_ids[sel]), ROPE, B=B)})
    for t, p in zip(tests, holm([t["c"]["p"] for t in tests])):
        t["c"] = {**keepc(t["c"]), "p_holm": float(p)}
    out["C2c"] = tests
    gc = _runs(study, "F3", "infonce")
    A, Bm = stack(gb, "tgem_shuf", ok_tgem), stack(gc, "tgem_shuf", ok_tgem)
    if A is not None and Bm is not None:
        out["C2c_control"] = keepc(contrast(A, Bm, clusters_from_docs(doc_ids[ok_tgem]), ROPE, B=B))
    return out


def _close(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    a, b = float(a), float(b)
    if b != 0 and abs(b) < 1e-6:
        return abs(a - b) <= max(1e-12, 1e-3 * abs(b))
    return abs(a - b) <= tol


def check_reproduces(study, rec: dict) -> dict:
    """¿Sin exclusiones, contrastes_f3 reproduce lo guardado por el notebook principal? (C1, C2 y C2c)."""
    out = {}
    av = load_json(study.paths.d("results", "F3", "all_vs_all", "all_vs_all.json")) or {}
    pm = ((av.get("C1") or {}).get("por_metrica")) or {}
    for m, c in ((rec.get("C1") or {}).items()):
        if m in pm:
            s = pm[m]
            out[f"C1 {m}"] = all(_close(c[k], s[k]) for k in ("diff", "ci95_lo", "ci95_hi")) and \
                _close(c["p"], s["p"]) and _close(c["p_holm"], s.get("p_holm_c1"))
    c2 = load_json(study.paths.d("results", "F3", "C2", "C2.json")) or {}
    stored = {r["par"]: r for r in c2.get("pares") or []}
    for r in rec.get("C2") or []:
        s = stored.get(r["par"])
        if s:
            out[f"C2 {r['par']}"] = all(_close(r["C2"][k], s["C2"][k]) for k in ("diff", "ci95_lo", "ci95_hi")) and \
                _close(r["C2"]["p_holm"], s["C2"].get("p_holm"))
    c2c = load_json(study.paths.d("results", "F3", "C2c", "C2c.json")) or {}
    stored = {t["metrica"]: t for t in c2c.get("pruebas") or []}
    for t in rec.get("C2c") or []:
        s = stored.get(t["metrica"])
        if s:
            out[f"C2c {t['metrica']}"] = all(_close(t["c"][k], s[k]) for k in ("diff", "ci95_lo", "ci95_hi")) and \
                _close(t["c"]["p_holm"], s.get("p_holm"))
    ctl = c2c.get("control") or {}
    if rec.get("C2c_control") and ctl:
        out["C2c control"] = all(_close(rec["C2c_control"][k], ctl[k]) for k in ("diff", "ci95_lo", "ci95_hi"))
    return out


def qc_fn_exclusions(labels: dict) -> dict:
    """Ítems con al menos un falso negativo según el juez, por set, en el formato de contrastes_f3."""
    ex = {"tgem": set(), "tmine": set(), "tgen": {}}
    for k, v in labels.items():
        if not any(v):
            continue
        s, i = k.split(":", 1)
        if s in ("tgem", "tmine"):
            ex[s].add(int(i))
        elif s.startswith("tgen_"):
            ex["tgen"].setdefault(s[len("tgen_"):], set()).add(int(i))
    return ex


def bloque1_regla(study, keys, log=print) -> dict:
    return write_rule(study, "b1_calidad", {
        "juez": f"{JUDGE['repo']} ({JUDGE['dtype']}, decodificación voraz, prompt fn-judge-v1). {JUDGE['por_que']}",
        "que_juzga": "Todos los ítems del split sellado de T-gem (HN de Gemini), T-mine (oros minados) y de cada uno de "
                     f"los 5 generadores del macro congelado ({', '.join(keys)}; sus ítems del macro). Para cada uno de "
                     "los 10 candidatos: ¿también responde bien la pregunta según el oro?",
        "tasas": "Por ítem (ítems con al menos un falso negativo; es la que afecta al top-1 y la más exigente, porque el "
                 "diseño no fijó la unidad) y por negativo, con IC95 de Wilson, por set.",
        "regla": f"Si la tasa por ítem del juez supera el {FN_MAX:.0%} en algún set, se reporta la sensibilidad de C1, C2 "
                 "y C2c sin los ítems marcados (descriptiva). Se calcula igual aunque ningún set lo supere, y entonces "
                 "se reporta como no requerida. Los resultados congelados, el macro y los veredictos no cambian.",
        "exclusiones": "T-gem y T-mine: se saca el ítem del set donde el juez marcó al menos un falso negativo (T-gem "
                       "barajado usa los candidatos de T-gem, así que sigue a T-gem). Macro de T-gen: en cada ítem se "
                       "promedian sólo los generadores sin falso negativo en ese ítem; si son todos, el ítem sale. C2 "
                       "saca además los ítems con falso negativo en T-gem. Mismo bootstrap que lo confirmatorio "
                       "(B = 4000, semilla 0, por documento); sin exclusiones tiene que reproducirlo.",
        "etiquetas_a_mano": f"200 ítems estratificados ({N_HAND['tgem']} de T-gem, {N_HAND['tmine']} de T-mine y "
                            f"{N_HAND['tgen']} por generador del macro), exportados antes de correr el juez y etiquetados "
                            "a ciegas por los autores. Si el kappa total juez–humano es < 0.6, la regla se aplica con la "
                            "tasa por ítem de las etiquetas a mano.",
        "fuga": f"Ítems de val, test y sellado cuyo par (pregunta, oro) aparece en entrenamiento: mismo oro normalizado "
                f"y pregunta igual o con Jaccard ≥ {JACCARD_MIN} sobre 3-gramas de palabras. Criterio de F0.4: fuga < "
                f"{LEAK_MAX:.1%} de los ítems de test. Si el test o el sellado lo superan, sensibilidad descriptiva sin "
                "esos ítems."}, log=log)


def bloque1_reporte(study, keys, log=print) -> dict:
    """Tasas de falsos negativos (juez y, si ya están, etiquetas a mano), acuerdo, fuga y sensibilidad de C1, C2 y C2c."""
    out = makedirs(pdir(study, "b1_calidad"))
    sets = list(JUDGE_SETS_FIXED) + [f"tgen_{k}" for k in keys]
    lab = judge_labels(study)
    try:
        cur = b1_items(study, log=log)[0]
        n_por_set = dict(Counter(set_of(k) for k in cur))
        lab = {k: v for k, v in lab.items() if k in cur}        # sólo los ítems vigentes de los sets
    except Exception:  # noqa: BLE001
        n_por_set = None
    rates = fn_rates(lab, sets, n_por_set)
    hand = read_hand_labels(study)
    agr = agreement(lab, hand, sets) if hand else None
    fg = load_json(os.path.join(out, "fuga", "fuga.json"))
    usar_humano = bool(agr and agr.get("total", {}).get("kappa") is not None and agr["total"]["kappa"] < 0.6)

    def rate_of(s):
        if usar_humano:
            return (agr.get(s) or {}).get("tasa_item_humano")
        return (rates.get(s) or {}).get("tasa_item")
    supera = sorted(s for s in sets if (rate_of(s) or 0) > FN_MAX)
    doc_ids = _confirm_doc_ids(study)
    ex_fn = qc_fn_exclusions(lab)
    ex_leak = {"todos": set((((fg or {}).get("splits") or {}).get("confirm") or {}).get("items") or [])}
    base = contrastes_f3(study, doc_ids)
    sens = {"sin exclusiones (control)": base,
            "sin ítems con falsos negativos (juez)": contrastes_f3(study, doc_ids, ex_fn)}
    if ex_leak["todos"]:
        sens["sin ítems con fuga"] = contrastes_f3(study, doc_ids, ex_leak)
        both = {**ex_fn, "todos": ex_leak["todos"]}
        sens["sin falsos negativos ni fuga"] = contrastes_f3(study, doc_ids, both)
    rep = check_reproduces(study, base)
    res = {"creado": now_iso(), "juez": load_json(os.path.join(out, "juez", "juez_info.json")),
           "tasas_juez": rates, "acuerdo": agr, "regla_con_etiquetas_a_mano": usar_humano,
           "sets_que_superan_5pc": supera, "sensibilidad_requerida_por_la_regla": bool(supera),
           "fuga": {sp: {k: v for k, v in r.items() if k != "items"} for sp, r in ((fg or {}).get("splits") or {}).items()},
           "fuga_supera": sorted(sp for sp, r in ((fg or {}).get("splits") or {}).items()
                                 if sp in ("test", "confirm") and r.get("supera_criterio")),
           "reproduce_lo_confirmatorio": rep, "sensibilidad": sens}
    save_json(res, os.path.join(out, "calidad.json"))
    write_text(os.path.join(out, "CALIDAD.md"), bloque1_markdown(res, sets))
    log(f"[b1] guardado en {out}")
    return res


def _c(c: dict | None, holm_key: str = "p_holm") -> str:
    if not c:
        return "—"
    s = f"{100 * c['diff']:+.2f} [{100 * c['ci95_lo']:+.2f}, {100 * c['ci95_hi']:+.2f}]"
    if holm_key in c:
        s += f", p_Holm {c[holm_key]:.2g}"
    return s


def sens_rows(sens: dict) -> list:
    """Filas (contraste, {variante: texto}) para la tabla de sensibilidad."""
    names = list(sens)
    rows = []

    def get(r, path):
        x = r
        for p in path:
            if isinstance(x, list):
                x = next((y for y in x if y.get("par") == p or y.get("metrica") == p), None)
            elif isinstance(x, dict):
                x = x.get(p)
            if x is None:
                return None
        return x
    spec = [("C1, T-mine", ("C1", "tmine")), ("C1, macro de T-gen", ("C1", "tgenmacro")),
            ("C2, InfoNCE", ("C2", "infonce_hn − infonce", "C2")), ("C2, SigLIP", ("C2", "siglip_hn − siglip", "C2")),
            ("C2c, T-gem barajado", ("C2c", "tgem_shuf", "c")), ("C2c, T-mine", ("C2c", "tmine", "c")),
            ("Control de C2c", ("C2c_control",))]
    for label, path in spec:
        rows.append((label, {n: _c(get(sens[n], path)) for n in names}))
    return rows


def bloque1_markdown(res: dict, sets) -> str:
    L = ["# Bloque 1 — calidad de los sets del sellado", "", f"Generado: {res['creado']}", ""]
    j = res.get("juez") or {}
    jtxt = (f"Juez: {j.get('repo')} ({j.get('dtype')}, decodificación {j.get('decodificacion')}); "
            f"{j.get('n_etiquetados')}/{j.get('n_items')} ítems con etiquetas válidas.") if j.get("repo") else \
        "El juez todavía no corrió (paso 1.3)."
    L += ["## Falsos negativos según el juez", "", jtxt, "",
          "| set | ítems juzgados / del set | con FN | tasa por ítem (%) [IC95] | tasa por negativo (%) [IC95] |",
          "|---|---|---|---|---|"]
    for s in sets:
        r = (res.get("tasas_juez") or {}).get(s) or {}
        if r.get("n_items"):
            L.append(f"| {set_label(s)} | {r['n_items']} / {r.get('items_del_set') or '?'} | {r['items_con_fn']} | "
                     f"{pct(r['tasa_item'])} [{pct(r['ic_item'][0])}, {pct(r['ic_item'][1])}] | "
                     f"{pct(r['tasa_negativo'], 2)} [{pct(r['ic_negativo'][0], 2)}, {pct(r['ic_negativo'][1], 2)}] |")
    agr = res.get("acuerdo")
    if agr:
        L += ["", "## Acuerdo con las etiquetas a mano", "",
              "| set | negativos | FN según humanos | sensibilidad | especificidad | kappa | tasa por ítem humana (%) [IC95] |",
              "|---|---|---|---|---|---|---|"]
        for s in list(sets) + ["total"]:
            a = agr.get(s) or {}
            if not a.get("n_negativos"):
                continue
            hum = f"{pct(a.get('tasa_item_humano'))} [{pct((a.get('ic_item_humano') or [None, None])[0])}, " \
                  f"{pct((a.get('ic_item_humano') or [None, None])[1])}]" if s != "total" else "—"
            f = lambda x: "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.2f}"  # noqa: E731
            L.append(f"| {set_label(s)} | {a['n_negativos']} | {a['tp'] + a['fn']} | {f(a['sensibilidad'])} | {f(a['especificidad'])} | "
                     f"{f(a['kappa'])} | {hum} |")
        L.append("")
        L.append("La regla se aplicó con las etiquetas a mano (kappa < 0.6)." if res.get("regla_con_etiquetas_a_mano")
                 else "La regla se aplica con la tasa del juez.")
    else:
        L += ["", "Todavía no hay etiquetas a mano (`etiquetas/ETIQUETADAS.csv`): el acuerdo se calcula cuando estén."]
    sup = res.get("sets_que_superan_5pc") or []
    L += ["", f"**Sets con más de 5 % de ítems con falsos negativos:** {', '.join(set_label(x) for x in sup) if sup else 'ninguno'}."
          + ("" if sup else " La sensibilidad de abajo no la exige la regla; se reporta igual."), ""]
    L += ["## Fuga entre splits", "", "| split | ítems | par exacto | par exacto o casi | tasa (%) [IC95] | pregunta igual, cualquier oro (%) |",
          "|---|---|---|---|---|---|"]
    if not res.get("fuga"):
        L.append("| — | la fuga no se calculó (falta `fuga/fuga.json`) | | | | |")
    for sp, r in (res.get("fuga") or {}).items():
        L.append(f"| {sp} | {r['n_items']} | {r['par_exacto']} | {r['par_exacto_o_casi']} | {pct(r['tasa'], 2)} "
                 f"[{pct(r['ic95'][0], 2)}, {pct(r['ic95'][1], 2)}] | {pct(r['tasa_pregunta_igual'], 1)} |")
    L += ["", f"Criterio de F0.4: < 0.5 % en test. Superan: {', '.join(res.get('fuga_supera') or []) or 'ninguno'}.", ""]
    rep = res.get("reproduce_lo_confirmatorio") or {}
    L += ["## Sensibilidad de los contrastes de F3 (pp; descriptiva)", "",
          "Control: sin exclusiones, el cálculo reproduce lo confirmatorio guardado: "
          + (", ".join(f"{k} {'sí' if v else 'NO'}" for k, v in rep.items()) if rep else
             "no hay resultados confirmatorios guardados con que compararlo"), ""]
    names = list((res.get("sensibilidad") or {}).keys())
    if names:
        L.append("| contraste | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for label, vals in sens_rows(res["sensibilidad"]):
            L.append(f"| {label} | " + " | ".join(vals[n] for n in names) + " |")
        L.append("")
        for n in names[1:]:
            ex = (res["sensibilidad"][n] or {}).get("exclusiones") or {}
            tg = ", ".join(f"{k} {v}" for k, v in (ex.get("tgen") or {}).items()) or "ninguno"
            L.append(f"- **{n}:** ítems fuera de T-gem {ex.get('tgem', 0)}, de T-mine {ex.get('tmine', 0)}; "
                     f"columnas del macro fuera por generador: {tg}.")
    return "\n".join(L) + "\n"


# ======================================================================================
# BLOQUE 2 — condición de borde de B5 (exploratoria, split de test)
BORDE = [
    ("a_hn_only", {"hn_denominator": "hn_only"}, "(a) sólo HN"),
    ("b_hn_inbatch", {"hn_denominator": "hn_plus_inbatch"}, "(b) HN + in-batch"),
    ("c_hn_otherhn", {"hn_denominator": "hn_plus_otherhn"}, "(c) HN + HN ajenos (B − 1)"),
    ("c31_hn_otherhn_k31", {"hn_denominator": "hn_plus_otherhn_k", "extra_k": 31}, "(c31) HN + 31 HN ajenos"),
    ("d31_hn_inbatch_k31", {"hn_denominator": "hn_plus_inbatch_k", "extra_k": 31}, "(d31) HN + 31 oros ajenos"),
    ("e_hn_shared", {"hn_denominator": "hn_shared_inbatch"}, "(e) compartido, como NegCLIP"),
]
BORDE_REF = {"a_hn_only": ("F2b_B1", "infonce_hn_hnonly"), "b_hn_inbatch": ("F1b", "infonce_hn")}
BORDE_PHASE = "PRE_borde"
BORDE_METRICS = ("tgem_shuf", "tgem", "tmine", "tmine50", "trand", "tgenmacro", "tgenmacro_shuf")
REPRO_TOL = 0.005


def bloque2_regla(study, log=print) -> dict:
    return write_rule(study, "b2_borde", {
        "pregunta": "¿Por qué los oros de otros estados quitan la preferencia sin estado que aprende 'sólo HN'? "
                    "Empate de estilo (el rasgo sin estado distingue el estilo del oro del de los HN; con otros oros en el "
                    "denominador, sólo el estado desempata) frente a dilución (cualquier candidato extra reduce el peso de "
                    "los HN propios).",
        "condiciones": {f: {"denominador": lp, "etiqueta": lab} for f, lp, lab in BORDE},
        "entrenamiento": "InfoNCE+HN con los hiperparámetros de F1b (val-mine, receta de F0.7b), semillas 0–2; sólo cambia "
                         "el denominador estado→acción; el acción→estado sigue con in-batch. Split de test (exploratorio).",
        "chequeo_previo": f"(a) y (b) tienen la configuración de F2b_B1/infonce_hn_hnonly y de F1b/infonce_hn: deben dar el "
                          f"mismo val_best con la misma semilla (idéntico si el entrenamiento es determinista; si no, "
                          f"|Δ| ≤ {REPRO_TOL}). Si no, se detiene antes de (c)–(e).",
        "medida_principal": "Acierto top-1 en T-gem con estados barajados (P). R(x) = [P(x) − P(b)] / [P(a) − P(b)]. Si "
                            "P(a) − P(b) < 5 pp, R no se interpreta.",
        "lectura": {"empate de estilo": "R(c) ≥ 0.5 y P(c) − P(b) con IC95 sobre 0",
                    "dilución": "R(c) ≤ 0.25",
                    "intermedio": "en otro caso; se reporta sin elegir explicación",
                    "cuántos oros bastan": "R(d31) ≤ 0.25: pocos oros bastan; R(d31) ≥ 0.5: hace falta el batch completo",
                    "HN compartidos": "R(e) ≥ 0.5: compartir los HN devuelve la preferencia aunque haya oros; R(e) ≤ 0.25: "
                                      "los oros la neutralizan también con HN compartidos (condición de borde frente a "
                                      "NegCLIP)",
                    "c31": "si R(c31) y R(c) caen del mismo lado de los umbrales, la lectura de (c) no depende del número "
                           "de candidatos"},
        "otras_medidas": "T-mine (costo en discriminación real), T-gem real, T-mine-50, T-rand y el macro de T-gen del test "
                         "(real y barajado). Bootstrap pareado por documento sobre las 3 semillas (B = 4000, semilla 0)."},
        log=log)


def borde_cfg_kw(lp: dict) -> dict:
    return {"loss_params": dict(lp)}


def borde_hecho(study, seeds=(0, 1, 2)) -> bool:
    return all((read_done(study.paths.d("runs", BORDE_PHASE, f, f"seed_{s}")) or {}).get("status") == "ok"
               for f, _, _ in BORDE for s in seeds)


def chunks_presentes(study) -> bool:
    from .pipeline import make_store
    st, d = make_store(study), study.data
    return all(st.has("pre", c, k) for c in sorted(set(d.pre_chunks) | set(d.hn_chunks)) for k in ("meta", "q", "a")) \
        and all(st.has("hn", c, k) for c in d.hn_chunks for k in ("nmeta", "ncnt", "n"))


def bloque2(study, seeds=(0, 1, 2), log=print) -> dict:
    """Bloque 2 completo: baja los embeddings de entrenamiento sólo si falta alguna corrida, entrena (reanudable) y
    hace el informe. En una segunda pasada del notebook (p. ej. después de subir las etiquetas a mano) no baja ni
    entrena nada."""
    bloque2_regla(study, log=log)
    if not borde_hecho(study, seeds):
        from .f0 import f0_download
        from .pipeline import load_bundle
        if not chunks_presentes(study):
            asegurar_disco(study, study.data.disk_gb() * 1.05, keep_chunks=True, log=log)
            f0_download(study, log=log)
        bundle = load_bundle(study)
        try:
            bloque2_correr(study, bundle, seeds, log=log)
        except SystemExit:
            bloque2_reporte(study, seeds, log=log)     # informe con lo que hay ((a), (b) y el chequeo) y se detiene
            raise
        finally:
            del bundle
            _free_gpu()
    else:
        log("[b2] las 18 corridas ya están: sólo se rehace el informe")
    return bloque2_reporte(study, seeds, log=log)


def _free_gpu() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def bloque2_correr(study, bundle, seeds=(0, 1, 2), log=print) -> dict:
    """Entrena las 6 condiciones (reanudable). Primero (a) y (b) y el chequeo de reproducción; si no reproducen, no
    corre (c)–(e)."""
    from .pipeline import run_variant
    bloque2_regla(study, log=log)
    out = makedirs(pdir(study, "b2_borde"))
    for folder, lp, lab in BORDE[:2]:
        log(f"[b2] {lab}")
        run_variant(study, bundle, "infonce_hn", BORDE_PHASE, folder, seeds=list(seeds), hp_from="F1b",
                    **borde_cfg_kw(lp))
    chk = bloque2_chequeo(study, seeds)
    save_json(chk, os.path.join(out, "chequeo.json"))
    log(f"[b2] chequeo de (a) y (b): {chk['lectura']}")
    if not chk["ok"]:
        raise SystemExit(f"(a) o (b) no reproducen sus corridas de referencia: {chk['lectura']}. No se corren (c)–(e).")
    for folder, lp, lab in BORDE[2:]:
        log(f"[b2] {lab}")
        run_variant(study, bundle, "infonce_hn", BORDE_PHASE, folder, seeds=list(seeds), hp_from="F1b",
                    **borde_cfg_kw(lp))
    return chk


def bloque2_chequeo(study, seeds=(0, 1, 2)) -> dict:
    rows, ok = [], True
    for folder, (ph, name) in BORDE_REF.items():
        for s in seeds:
            d_new = read_done(study.paths.d("runs", BORDE_PHASE, folder, f"seed_{s}")) or {}
            d_ref = read_done(study.paths.d("runs", ph, name, f"seed_{s}")) or {}
            v1, v2 = d_ref.get("val_best"), d_new.get("val_best")
            if not isinstance(v1, (int, float)):
                rows.append({"condicion": folder, "semilla": s, "estado": f"sin referencia ({ph}/{name})"})
                continue
            if not isinstance(v2, (int, float)):
                rows.append({"condicion": folder, "semilla": s, "estado": "falta la corrida nueva"})
                ok = False
                continue
            same = v1 == v2 and _stage_sig(d_ref, "mid") == _stage_sig(d_new, "mid")
            close = abs(v2 - v1) <= REPRO_TOL
            ok &= bool(same or close)
            rows.append({"condicion": folder, "semilla": s, "val_best_ref": v1, "val_best_nuevo": v2,
                         "identico": bool(same), "dentro_de_tolerancia": bool(close)})
    comp = [r for r in rows if "identico" in r]
    if not comp:
        lectura = "sin corridas de referencia: no se pudo comprobar (se sigue)"
    elif all(r["identico"] for r in comp):
        lectura = f"{len(comp)}/{len(comp)} idénticas a sus referencias"
    elif ok:
        lectura = (f"{sum(r['identico'] for r in comp)}/{len(comp)} idénticas; las demás dentro de ±{REPRO_TOL} "
                   "(diferencias numéricas)")
    else:
        lectura = "alguna corrida se aleja de su referencia más que la tolerancia"
    return {"filas": rows, "ok": bool(ok), "lectura": lectura}


def bloque2_reporte(study, seeds=(0, 1, 2), B: int = B_BOOT, log=print) -> dict:
    from .data import EvalSet
    from .generators import load_panel, macro_for_run
    from .stats import bootstrap_means, clusters_from_docs
    out = makedirs(pdir(study, "b2_borde"))
    panel = load_panel(study, "test") or {}
    runs = {}
    for folder, lp, lab in BORDE:
        rds = []
        for s in seeds:
            rd = study.paths.d("runs", BORDE_PHASE, folder, f"seed_{s}")
            if (read_done(rd) or {}).get("status") == "ok":
                if panel:
                    macro_for_run(rd, panel)
                rds.append(rd)
        if len(rds) == len(seeds):
            runs[folder] = rds
    doc = np.asarray(EvalSet(study.paths.d("data", "eval", "test")).doc_ids)    # = bundle.test.doc_ids
    mit = np.asarray(panel.get("macro_items") or [], np.int64)
    res = {"creado": now_iso(), "condiciones": {f: lab for f, _, lab in BORDE}, "corridas": {f: len(v) for f, v in runs.items()},
           "chequeo": load_json(os.path.join(out, "chequeo.json")), "metricas": {}}
    for met in BORDE_METRICS:
        n = len(mit) if met.startswith("tgenmacro") else len(doc)
        items = {}
        for f, rds in runs.items():
            xs = [_load_correct(rd, met) for rd in rds]
            if all(x is not None and len(x) == n for x in xs):
                items[f] = np.vstack(xs)
        if len(items) < 2:
            continue
        cl = clusters_from_docs(doc[mit] if met.startswith("tgenmacro") else doc)
        boots, point = bootstrap_means(items, cl, B=B, seed=0)
        m = {}
        for f in items:
            rec = {"media": point[f], "ic95": [float(np.percentile(boots[f], 2.5)), float(np.percentile(boots[f], 97.5))]}
            if "b_hn_inbatch" in items and f != "b_hn_inbatch":
                d = boots[f] - boots["b_hn_inbatch"]
                est = point[f] - point["b_hn_inbatch"]
                se = float(d.std(ddof=1))
                rec["menos_b"] = {"diff": est, "ci95_lo": float(np.percentile(d, 2.5)),
                                  "ci95_hi": float(np.percentile(d, 97.5)),
                                  "p": float(2 * _norm_sf(abs(est / se))) if se > 0 else 0.0}
            m[f] = rec
        if met == "tgem_shuf" and {"a_hn_only", "b_hn_inbatch"} <= set(items):
            den_b = boots["a_hn_only"] - boots["b_hn_inbatch"]
            den = point["a_hn_only"] - point["b_hn_inbatch"]
            R = {}
            for f in items:
                if f in ("a_hn_only", "b_hn_inbatch"):
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    rb = (boots[f] - boots["b_hn_inbatch"]) / np.where(np.abs(den_b) < 1e-9, np.nan, den_b)
                R[f] = {"R": (point[f] - point["b_hn_inbatch"]) / den if abs(den) > 1e-12 else None,
                        "ic95": [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))]}
            res["R"] = R
            res["P_a_menos_P_b"] = den
        res["metricas"][met] = m
    res["lectura"] = borde_lectura(res)
    save_json(res, os.path.join(out, "borde.json"))
    write_text(os.path.join(out, "BORDE.md"), bloque2_markdown(res))
    log(f"[b2] guardado en {out}: {res['lectura'].get('principal')}")
    return res


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def borde_lectura(res: dict) -> dict:
    """Aplica la regla registrada (REGLA.json del bloque 2) a los resultados."""
    out = {}
    den = res.get("P_a_menos_P_b")
    R = res.get("R") or {}
    if den is None:
        return {"principal": "faltan (a) o (b): no hay lectura"}
    if den < 0.05:
        return {"principal": f"no interpretable: P(a) − P(b) = {100 * den:.1f} pp < 5 pp ('sólo HN' no reprodujo la "
                             "preferencia sin estado)"}
    rc = (R.get("c_hn_otherhn") or {}).get("R")
    pc = ((res["metricas"].get("tgem_shuf") or {}).get("c_hn_otherhn") or {}).get("menos_b") or {}
    if rc is None:
        out["principal"] = "falta (c): no hay lectura principal"
    elif rc >= 0.5 and pc.get("ci95_lo", -1) > 0:
        out["principal"] = f"empate de estilo (R(c) = {rc:.2f})"
    elif rc <= 0.25:
        out["principal"] = f"dilución (R(c) = {rc:.2f})"
    else:
        out["principal"] = f"intermedio (R(c) = {rc:.2f}): se reporta sin elegir explicación"
    rd = (R.get("d31_hn_inbatch_k31") or {}).get("R")
    if rd is not None:
        out["cuantos_oros"] = ("pocos oros bastan" if rd <= 0.25 else "hace falta el batch completo" if rd >= 0.5
                               else "intermedio") + f" (R(d31) = {rd:.2f})"
    r31 = (R.get("c31_hn_otherhn_k31") or {}).get("R")
    if r31 is not None and rc is not None:
        side = lambda r: "alto" if r >= 0.5 else "bajo" if r <= 0.25 else "medio"  # noqa: E731
        out["c31"] = ("la lectura de (c) no depende del número de candidatos" if side(r31) == side(rc)
                      else "la lectura de (c) depende del número de candidatos") + f" (R(c31) = {r31:.2f})"
    re_ = (R.get("e_hn_shared") or {}).get("R")
    if re_ is not None:
        out["compartidos"] = ("compartir los HN devuelve la preferencia aunque haya oros" if re_ >= 0.5 else
                              "los oros la neutralizan también con HN compartidos" if re_ <= 0.25 else
                              "intermedio") + f" (R(e) = {re_:.2f})"
    return out


def bloque2_markdown(res: dict) -> str:
    L = ["# Bloque 2 — condición de borde de B5 (exploratoria, split de test)", "", f"Generado: {res['creado']}", ""]
    lec = res.get("lectura") or {}
    chk = (res.get("chequeo") or {}).get("lectura")
    L += ["## Lectura (regla registrada antes de correr)", "",
          f"- **chequeo previo** ((a) y (b) frente a F2b_B1 y F1b, mismas semillas): {chk or 'sin dato'}"] \
        + [f"- **{k}:** {v}" for k, v in lec.items()] + [""]
    labels = res.get("condiciones") or {}
    names = {"tgem_shuf": "T-gem barajado", "tgem": "T-gem", "tmine": "T-mine", "tmine50": "T-mine-50", "trand": "T-rand",
             "tgenmacro": "macro T-gen", "tgenmacro_shuf": "macro T-gen barajado"}
    mets = [m for m in BORDE_METRICS if m in res.get("metricas", {})]
    L += ["## Acierto top-1 (%) [IC95], media de 3 semillas", "", "| condición | " + " | ".join(names[m] for m in mets) + " | R(x) |",
          "|---|" + "---|" * (len(mets) + 1)]
    for f, lab in labels.items():
        cells = []
        for m in mets:
            r = res["metricas"][m].get(f)
            cells.append("—" if not r else f"{pct(r['media'])} [{pct(r['ic95'][0])}, {pct(r['ic95'][1])}]")
        rr = (res.get("R") or {}).get(f)
        rtxt = "—" if not rr or rr.get("R") is None else f"{rr['R']:.2f} [{rr['ic95'][0]:.2f}, {rr['ic95'][1]:.2f}]"
        L.append(f"| {lab} | " + " | ".join(cells) + f" | {rtxt} |")
    L += ["", "## Diferencia con (b) HN + in-batch (pp) [IC95]", "", "| condición | " + " | ".join(names[m] for m in mets) + " |",
          "|---|" + "---|" * len(mets)]
    for f, lab in labels.items():
        if f == "b_hn_inbatch":
            continue
        cells = []
        for m in mets:
            d = ((res["metricas"][m].get(f) or {}).get("menos_b"))
            cells.append("—" if not d else f"{100 * d['diff']:+.1f} [{100 * d['ci95_lo']:+.1f}, {100 * d['ci95_hi']:+.1f}]")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


# ======================================================================================
# BLOQUE 3 — referencias sin entrenamiento
QEMB = {"repo": "Qwen/Qwen3-Embedding-8B", "max_len": 512,
        "tarea": "Given a question, retrieve the answer that correctly answers it"}
BM25_PARAMS = {"k1": 1.5, "b": 0.75, "n_corpus": 200_000}
SANITY_Q = ["What is the capital of China?", "Explain gravity"]
SANITY_D = ["The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is "
            "responsible for the movement of planets around the sun."]


def bloque3_regla(study, log=print) -> dict:
    return write_rule(study, "b3_referencias", {
        "que": "Referencias sin entrenamiento, descriptivas, en test y sellado, con estados reales y barajados: "
               "Qwen3-Embedding-8B (zero-shot), BM25 y el coseno crudo de Qwen3-8B (el encoder del estudio).",
        "estado": "El texto del estado es la pregunta sola (no el documento) para Qwen3-Embedding-8B y BM25. El coseno "
                  "crudo usa los embeddings públicos de CLM del estado (G1 sugiere que incluyen más que la pregunta).",
        "qwen3_embedding": f"{QEMB['repo']}; consultas con 'Instruct: {QEMB['tarea']}\\nQuery:' y candidatos sin "
                           f"instrucción; agrupación por el último token y normalización L2 (la del modelo); máx. "
                           f"{QEMB['max_len']} tokens. Antes de usarlo se comprueba con el ejemplo de su ficha.",
        "bm25": f"k1 = {BM25_PARAMS['k1']}, b = {BM25_PARAMS['b']}; IDF = ln(1 + (N − n + 0.5)/(n + 0.5)) y largo medio de "
                f"{BM25_PARAMS['n_corpus']:,} oros de entrenamiento (documentos fuera de val, test y sellado, en el orden "
                "de las partes de qa_pairs); tokens = \\w+ en minúsculas; la pregunta es la consulta.",
        "sets": "T-gem, T-rand, T-mine, T-mine-50 y el macro de T-gen de cada split (sus 5 generadores). Empates: acierto "
                "esperado con desempate al azar (rank_metrics).",
        "cautela": "Los negativos de T-mine se minaron con los embeddings crudos de Qwen3-8B: son difíciles a propósito "
                   "para esa familia, así que en T-mine la comparación con Qwen3-Embedding-8B y con el coseno crudo no es "
                   "neutral."}, log=log)


class QwenEmbedder:
    """Qwen3-Embedding-8B con la receta de su ficha: relleno a la izquierda, <|endoftext|> al final, el estado oculto
    del último token y normalización L2; las consultas llevan 'Instruct: <tarea>\nQuery:' delante y los candidatos
    nada. Usa sentence-transformers (la configuración que publica el autor del modelo) si está instalado y su
    tokenizer agrega <|endoftext|>; si no, la misma receta con transformers. Las dos vías dan lo mismo salvo redondeo
    (tests/test_v038.py, con un modelo chico)."""

    def __init__(self, repo: str = QEMB["repo"], cache_dir: str | None = None, device: str | None = None,
                 max_len: int = QEMB["max_len"], batch: int = 128):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len, self.batch = max_len, batch
        self.st = None
        self.modo = None
        self.err_st = None
        try:
            from sentence_transformers import SentenceTransformer
            kw = dict(device=self.device, cache_folder=cache_dir, tokenizer_kwargs={"padding_side": "left"})
            try:
                st = SentenceTransformer(repo, model_kwargs={"dtype": torch.bfloat16}, **kw)
            except TypeError:
                st = SentenceTransformer(repo, model_kwargs={"torch_dtype": torch.bfloat16}, **kw)
            tok = st.tokenizer
            eos = tok.convert_tokens_to_ids("<|endoftext|>")
            probe = tok("x")["input_ids"]
            if not probe or probe[-1] != eos:
                raise RuntimeError("el tokenizer no agrega <|endoftext|> al final")
            if next(st.parameters()).dtype != torch.bfloat16:
                st = st.to(dtype=torch.bfloat16)                # por si la versión instalada ignoró el dtype
            st.max_seq_length = max_len
            self.st, self.modo = st, "sentence-transformers"
        except Exception as e:  # noqa: BLE001
            self.st = None
            self.err_st = f"{type(e).__name__}: {str(e)[:300]}"
            from transformers import AutoModel, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(repo, padding_side="left", cache_dir=cache_dir)
            try:
                self.model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16, cache_dir=cache_dir)
            except TypeError:
                self.model = AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
            self.model.to(self.device).eval()
            self.eos = self.tok.convert_tokens_to_ids("<|endoftext|>")
            probe = self.tok("x")["input_ids"]
            self.add_eos = not probe or probe[-1] != self.eos
            self.modo = "transformers" + (" (+<|endoftext|>)" if self.add_eos else "")

    def encode(self, texts, query: bool = False, prompt: str | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), np.float32)
        pr = prompt if prompt is not None else (f"Instruct: {QEMB['tarea']}\nQuery:" if query else None)
        if self.st is not None:
            E = self.st.encode(texts, prompt=pr, batch_size=self.batch, convert_to_tensor=True,
                               normalize_embeddings=True, show_progress_bar=False).float()
            return torch_l2(E).cpu().numpy()                    # L2 otra vez en float32 (la de ST es en bf16)
        torch = self.torch
        texts = [(pr or "") + t for t in texts]
        order = np.argsort([-len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), self.model.config.hidden_size), np.float32)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos
        with torch.no_grad():
            for s in range(0, len(texts), self.batch):
                idx = order[s:s + self.batch]
                ids = self.tok([texts[j] for j in idx], truncation=True,
                               max_length=self.max_len - (1 if self.add_eos else 0))["input_ids"]
                if self.add_eos:
                    ids = [x + [self.eos] for x in ids]
                L = max(len(x) for x in ids)
                inp = torch.full((len(ids), L), pad, dtype=torch.long)
                att = torch.zeros((len(ids), L), dtype=torch.long)
                for r, x in enumerate(ids):
                    inp[r, L - len(x):] = torch.tensor(x)
                    att[r, L - len(x):] = 1
                h = self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device)).last_hidden_state
                v = torch.nn.functional.normalize(h[:, -1].float(), dim=-1)
                out[idx] = v.cpu().numpy()
        return out

    def sanity(self) -> dict:
        """Ejemplo de la ficha del modelo (2 consultas, 2 documentos): cada consulta tiene que quedar claramente más
        cerca de su documento que del otro (margen ≥ 0.2 de coseno). Si no, algo del pipeline está mal (agrupación,
        relleno, fin de secuencia) y Qwen3-Embedding-8B no se usa."""
        q = self.encode(SANITY_Q, prompt="Instruct: Given a web search query, retrieve relevant passages that answer "
                                         "the query\nQuery:")
        d = self.encode(SANITY_D)
        S = q @ d.T
        ok = bool(S[0, 0] - S[0, 1] >= 0.2 and S[1, 1] - S[1, 0] >= 0.2)
        return {"modo": self.modo, "error_sentence_transformers": self.err_st, "matriz": S.round(4).tolist(), "ok": ok,
                "criterio": "diagonal − fuera de la diagonal ≥ 0.2 en las dos filas"}


def torch_l2(x):
    import torch
    return torch.nn.functional.normalize(x.float(), dim=-1)


def bm25_tokens(s: str) -> list:
    return re.findall(r"\w+", str(s).lower())


class BM25:
    def __init__(self, df: dict, n_docs: int, avgdl: float, k1: float = BM25_PARAMS["k1"], b: float = BM25_PARAMS["b"]):
        self.df, self.N, self.avgdl, self.k1, self.b = df, int(n_docs), float(avgdl), k1, b

    def idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, q: list, d: list) -> float:
        tf = Counter(d)
        L = len(d)
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if f:
                s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avgdl))
        return s

    @classmethod
    def from_corpus(cls, docs_tokens) -> "BM25":
        df, tot, n = Counter(), 0, 0
        for toks in docs_tokens:
            df.update(set(toks))
            tot += len(toks)
            n += 1
        return cls(dict(df), n, tot / max(1, n))


def bm25_corpus(study, n: int = BM25_PARAMS["n_corpus"], log=print, qa_batches=None, decode=None) -> BM25:
    """IDF y largo medio de n oros de entrenamiento (documentos fuera de val, test y sellado). Caché en Drive."""
    from .data import Split
    p = pdir(study, "b3_referencias", "bm25_corpus.json")
    js = load_json(p)
    if js:
        return BM25(js["df"], js["n_docs"], js["avgdl"])
    hold = Split.load(study.paths.d("data", "splits")).holdout()
    if qa_batches is None:
        qa_batches = lambda: _iter_qa(study.paths, exclude=hold, batch_size=50_000)  # noqa: E731
    if decode is None:
        tok = _tokenizer()
        decode = lambda lists: tok.batch_decode(lists) if lists else []  # noqa: E731
    texts = []
    for docs, pidx, qids, aids in qa_batches():
        sel = [j for j, d in enumerate(docs) if d not in hold][: n - len(texts)]
        texts += decode([aids[j] for j in sel])
        if len(texts) >= n:
            break
    bm = BM25.from_corpus(bm25_tokens(t) for t in texts)
    save_json({"creado": now_iso(), "n_docs": bm.N, "avgdl": bm.avgdl, "df": bm.df, **BM25_PARAMS}, p, indent=None)
    log(f"[bm25] corpus: {bm.N} oros de entrenamiento, largo medio {bm.avgdl:.1f} tokens")
    return bm


def ref_sets(study, split: str, tx: dict, tg: dict, ttxt: dict, keys) -> dict:
    """Ítems y candidatos (texto) de cada set de un split: {set: [(i, [oro, negativos...])]}; los de T-gen, por
    generador del macro. Sólo ítems con todos sus textos."""
    from .evaluate import POOL_FILES
    from .generators import load_panel
    out = {"tgem": [(i, [r["a"]] + list(r["negs"])) for i, r in sorted(tx.items()) if len(r.get("negs") or []) == 10]}
    for name, fn in POOL_FILES.items():
        p = study.paths.d("data", "eval", split, fn)
        if os.path.exists(p):
            I = np.load(p)
            out[name] = [(i, [tx[i]["a"]] + [tx[int(j)]["a"] for j in I[i]]) for i in sorted(tx)
                         if all(int(j) in tx for j in I[i])]
    mit = [int(i) for i in ((load_panel(study, split) or {}).get("macro_items") or [])]
    for k in keys:
        rows = []
        for i in mit:
            if i in tg[k] and (i in tx or i in ttxt):
                gold = tg[k][i]["pos"] or (tx[i]["a"] if i in tx else ttxt[i]["a"])   # el positivo del set
                rows.append((i, [gold] + list(tg[k][i]["negs"])))
        out[f"tgen_{k}"] = rows
    return out


def _score_sets(sets: dict, qscore, perm: np.ndarray) -> dict:
    """{set: (correct_real, correct_barajado, items)} con qscore(i_estado, candidatos) -> puntuaciones."""
    from .evaluate import rank_metrics
    out = {}
    for name, rows in sets.items():
        if not rows:
            continue
        items = np.asarray([i for i, _ in rows], np.int64)
        C = np.asarray([qscore(i, c) for i, c in rows], np.float32)
        Cs = np.asarray([qscore(int(perm[i]), c) for i, c in rows], np.float32)
        out[name] = (rank_metrics(C)[1], rank_metrics(Cs)[1], items)
    return out


def ref_qwen3(sets: dict, questions: dict, emb: "QwenEmbedder", n_split: int, log=print, split: str = "") -> dict:
    """n_split = ítems del set de evaluación del split: la permutación de estados barajados es la misma que en la
    evaluación de las corridas (shuffle_perm(n, 0) sobre todo el split)."""
    from .evaluate import shuffle_perm
    texts_d = sorted({t for rows in sets.values() for _, c in rows for t in c})
    qi = sorted(questions)
    log(f"[qwen3-emb] {split}: {len(qi)} preguntas y {len(texts_d)} candidatos distintos")
    Q = emb.encode([questions[i] for i in qi], query=True)
    D = emb.encode(texts_d)
    qpos = {i: j for j, i in enumerate(qi)}
    dpos = {t: j for j, t in enumerate(texts_d)}
    perm = shuffle_perm(n_split, 0)

    def qscore(i, cands):
        if i not in qpos:
            return [np.nan] * len(cands)
        return D[[dpos[t] for t in cands]] @ Q[qpos[i]]
    return _score_sets(sets, qscore, perm)


def ref_bm25(sets: dict, questions: dict, bm: BM25, n_split: int) -> dict:
    from .evaluate import shuffle_perm
    qt = {i: bm25_tokens(q) for i, q in questions.items()}
    cache: dict = {}

    def toks(t):
        if t not in cache:
            cache[t] = bm25_tokens(t)
        return cache[t]
    perm = shuffle_perm(n_split, 0)

    def qscore(i, cands):
        q = qt.get(i)
        if q is None:
            return [np.nan] * len(cands)
        return [bm.score(q, toks(c)) for c in cands]
    return _score_sets(sets, qscore, perm)


def ref_raw_cosine(study, split: str, keys) -> dict:
    """Coseno crudo de Qwen3-8B (embeddings públicos de CLM, sin entrenar), con estados reales y barajados."""
    from .data import EvalSet
    from .evaluate import POOL_FILES, _l2np, rank_metrics, shuffle_perm
    from .generators import load_panel, load_panel_sets
    ev = EvalSet(study.paths.d("data", "eval", split))
    q, a = _l2np(ev.q_all()), _l2np(ev.a_all())
    perm = shuffle_perm(ev.n, 0)
    out = {}

    def run(name, items, cand_fn):
        C, Cs = [], []
        for s in range(0, len(items), 2048):
            it = items[s:s + 2048]
            cands = cand_fn(it, s)                              # n x (1+K) x H
            C.append(np.einsum("nh,nkh->nk", q[it], cands))
            Cs.append(np.einsum("nh,nkh->nk", q[perm[it]], cands))
        out[name] = (rank_metrics(np.concatenate(C))[1], rank_metrics(np.concatenate(Cs))[1], np.asarray(items))
    allit = np.arange(ev.n)
    run("tgem", allit, lambda it, s: np.concatenate([a[it][:, None], _l2np(ev.negatives(it))], 1))
    for name, fn in POOL_FILES.items():
        p = study.paths.d("data", "eval", split, fn)
        if os.path.exists(p):
            I = np.load(p)
            run(name, allit, lambda it, s, I=I: np.concatenate([a[it][:, None], a[I[it]]], 1))
    panel = load_panel(study, split) or {}
    mit = np.asarray(panel.get("macro_items") or [], np.int64)
    ps = load_panel_sets(study, split)
    for k in keys:
        g = ps.get(f"tgen_{k}")
        if g is None:
            continue
        gi = np.asarray(g[0], np.int64)
        where = {int(x): j for j, x in enumerate(gi)}
        it = np.asarray([i for i in mit if int(i) in where], np.int64)
        rows = np.asarray([where[int(i)] for i in it], np.int64)
        negs, own = g[1], (g[3] if len(g) > 3 else None)      # own: positivo propio del set (construcción P)

        def cands(itb, s, rows=rows, negs=negs, own=own):
            r = rows[s:s + len(itb)]
            gold = a[itb] if own is None else _l2np(np.asarray(own[r], np.float32))
            return np.concatenate([gold[:, None], _l2np(np.asarray(negs[r], np.float32))], 1)
        run(f"tgen_{k}", it, cands)
    return out


def ref_summary(res: dict, doc_ids: np.ndarray, keys, B: int = 2000) -> dict:
    """Media e IC95 (bootstrap por documento) de cada set, real y barajado, y el macro de T-gen (promedio por ítem
    sobre los generadores del macro, en sus ítems comunes)."""
    from .stats import bootstrap_means, clusters_from_docs
    out = {}

    def ci(x, items):
        cl = clusters_from_docs(doc_ids[items])
        boots, point = bootstrap_means({"x": x[None, :]}, cl, B=B, seed=0)
        return {"media": point["x"], "ic95": [float(np.percentile(boots["x"], 2.5)), float(np.percentile(boots["x"], 97.5))],
                "n": int(len(items))}
    for name, (c, cs, items) in res.items():
        out[name] = ci(c, items)
        out[f"{name}_shuf"] = ci(cs, items)
    tk = [f"tgen_{k}" for k in keys if f"tgen_{k}" in res]
    if tk:
        common = None
        for k in tk:
            common = res[k][2] if common is None else np.intersect1d(common, res[k][2])
        if common is not None and len(common):
            M, Ms = [], []
            for k in tk:
                pos = {int(x): j for j, x in enumerate(res[k][2])}
                sel = [pos[int(i)] for i in common]
                M.append(res[k][0][sel])
                Ms.append(res[k][1][sel])
            out["tgenmacro"] = ci(np.mean(M, 0).astype(np.float32), common)
            out["tgenmacro_shuf"] = ci(np.mean(Ms, 0).astype(np.float32), common)
    return out


def save_ref(study, model: str, split: str, res: dict) -> None:
    d = makedirs(pdir(study, "b3_referencias", model, split))
    for name, (c, cs, items) in res.items():
        np.save(os.path.join(d, f"correct_{name}.npy"), np.asarray(c, np.float32))
        np.save(os.path.join(d, f"correct_{name}_shuf.npy"), np.asarray(cs, np.float32))
        np.save(os.path.join(d, f"items_{name}.npy"), np.asarray(items, np.int64))


def load_ref(study, model: str, split: str) -> dict | None:
    d = pdir(study, "b3_referencias", model, split)
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "DONE")):
        return None
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "items_*.npy"))):
        name = os.path.basename(p)[len("items_"):-4]
        out[name] = (np.load(os.path.join(d, f"correct_{name}.npy")), np.load(os.path.join(d, f"correct_{name}_shuf.npy")),
                     np.load(p))
    return out


def done_ref(study, model: str, split: str) -> None:
    write_text(pdir(study, "b3_referencias", model, split, "DONE"), now_iso())


REF_SETS_ORDER = ("tgem", "tgem_shuf", "trand", "tmine", "tmine50", "tgenmacro", "tgenmacro_shuf")
REF_NAMES = {"tgem": "T-gem", "tgem_shuf": "T-gem barajado", "trand": "T-rand", "tmine": "T-mine", "tmine50": "T-mine-50",
             "tgenmacro": "macro T-gen", "tgenmacro_shuf": "macro T-gen barajado"}
REF_MODELS = {"qwen3_emb": "Qwen3-Embedding-8B (zero-shot, pregunta)", "bm25": "BM25 (pregunta)",
              "coseno_crudo": "coseno crudo de Qwen3-8B (estado de CLM)"}


def bloque3_markdown(res: dict) -> str:
    s = res.get("sanity") or {}
    stxt = "no se corrió" if not s else (f"{'ok' if s.get('ok') else 'FALLÓ (no se usa)'}; vía {s.get('modo')}; "
                                         f"cosenos {s.get('matriz')} ({s.get('criterio')})")
    L = ["# Bloque 3 — referencias sin entrenamiento (descriptivas)", "", f"Generado: {res['creado']}", "",
         f"Comprobación de Qwen3-Embedding-8B con el ejemplo de su ficha: {stxt}.", ""]
    for split in ("test", "confirm"):
        tab = (res.get("tablas") or {}).get(split) or {}
        if not tab:
            continue
        L += [f"## {'Test (exploratorio)' if split == 'test' else 'Sellado'} — top-1 (%) [IC95]", "",
              "| referencia | " + " | ".join(REF_NAMES[s] for s in REF_SETS_ORDER) + " |", "|---|" + "---|" * len(REF_SETS_ORDER)]
        for m, lab in REF_MODELS.items():
            t = tab.get(m)
            if not t:
                continue
            cells = [("—" if s not in t else f"{pct(t[s]['media'])} [{pct(t[s]['ic95'][0])}, {pct(t[s]['ic95'][1])}]")
                     for s in REF_SETS_ORDER]
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        ctx = (res.get("contexto_F3") or {}) if split == "confirm" else {}
        if ctx:
            L.append("| InfoNCE entrenado (F3, contexto) | " + " | ".join(
                ("—" if s not in ctx else f"{pct(ctx[s])}") for s in REF_SETS_ORDER) + " |")
        L.append("")
    L += ["Azar: 9.1 % (1 de 11). Cautela: los negativos de T-mine se minaron con Qwen3-8B (ver la regla).", ""]
    return "\n".join(L) + "\n"


def contexto_f3(study) -> dict:
    rep = load_json(study.paths.d("results", "F3", "all_vs_all", "all_vs_all.json")) or {}
    out = {}
    for s in REF_SETS_ORDER:
        for x in ((rep.get("metrics") or {}).get(f"{s}_top1") or {}).get("summary") or []:
            if x["arm"] == "infonce":
                out[s] = x["mean"]
    return out


# ======================================================================================
def resumen(study) -> str:
    """RESUMEN_PREENVIO.md: estado de cada paso y los informes de cada bloque."""
    st = state(study)
    L = [f"# Notebook pre-envío — resumen ({now_iso()})", "", "| paso | estado | hora | detalle |", "|---|---|---|---|"]
    for k, v in st.items():
        det = v.get("motivo") or v.get("error") or (f"{v.get('minutos')} min" if v.get("minutos") is not None else "")
        L.append(f"| {k} | {v.get('estado')} | {v.get('hora')} | {str(det)[:160]} |")
    L.append("")
    for rel in (("b0_extraccion", "EXTRACCION.md"), ("b1_calidad", "CALIDAD.md"), ("b2_borde", "BORDE.md"),
                ("b3_referencias", "REFERENCIAS.md")):
        p = pdir(study, *rel)
        if os.path.exists(p):
            L += ["---", "", open(p, encoding="utf-8").read(), ""]
    md = "\n".join(L)
    write_text(pdir(study, "RESUMEN_PREENVIO.md"), md)
    return md


def free_embedding_chunks(study, log=print) -> float:
    """Borra del disco local los chunks de embeddings de entrenamiento (se vuelven a bajar si hacen falta)."""
    from .config import HN_REPO, PRETRAIN_REPO
    from .data import slug
    from .generators import _dir_bytes
    tot = 0
    for repo in (PRETRAIN_REPO, HN_REPO):
        d = study.paths.c("hf", slug(repo))
        if os.path.isdir(d):
            tot += _dir_bytes(d)
            shutil.rmtree(d, ignore_errors=True)
    if tot:
        log(f"[disco] borrados {tot / 1e9:.1f} GB de chunks de embeddings de entrenamiento (se vuelven a bajar solos "
            "si hacen falta)")
    return tot / 1e9


def hf_home(study) -> str:
    """Caché de modelos de HF de los pasos con modelos grandes (la misma que usa el panel T-gen)."""
    return os.environ.get("HF_HOME") or os.path.join(study.paths.cache_root, "hf_home")


def tokens_prescindibles(study) -> bool:
    """¿Ya no hacen falta los tokens de Nemotron-DQA en el disco local? (textos, fuga y corpus de BM25 en Drive)."""
    return all(os.path.exists(pdir(study, *p)) for p in (("textos", "confirm.jsonl"), ("textos", "test.jsonl"),
                                                       ("b1_calidad", "fuga", "fuga.json"),
                                                       ("b3_referencias", "bm25_corpus.json")))


def asegurar_disco(study, need_gb: float, keep_chunks: bool = True, keep_repos=(), log=print) -> float:
    """Libera el disco local hasta tener need_gb libres, en este orden: pesos de modelos de HF (salvo keep_repos);
    chunks de embeddings de entrenamiento (si keep_chunks=False); tokens de Nemotron-DQA (sólo si ya no hacen falta).
    Nada de esto está sólo en el disco local: todo se vuelve a bajar si hace falta. Devuelve los GB libres."""
    from .generators import free_gb, purge_hf_caches
    from .texts import tokens_dir
    root = study.paths.cache_root
    if free_gb(root) >= need_gb:
        return free_gb(root)
    fr = purge_hf_caches(hf_home(study), keep_repos=tuple(keep_repos))
    if fr["removed"]:
        log(f"[disco] se borraron {', '.join(fr['removed'])}")
    if free_gb(root) < need_gb and not keep_chunks:
        free_embedding_chunks(study, log=log)
    if free_gb(root) < need_gb and tokens_prescindibles(study) and os.path.isdir(tokens_dir(study.paths)):
        shutil.rmtree(tokens_dir(study.paths), ignore_errors=True)
        log("[disco] se borraron los tokens de Nemotron-DQA (textos, fuga y BM25 ya están en Drive)")
    free = free_gb(root)
    if free < need_gb:
        log(f"[disco] ATENCIÓN: quedan {free:.0f} GB libres y este paso necesita ~{need_gb:.0f} GB")
    return free


def macro_keys(study, split: str, log=print) -> list:
    """Generadores del macro congelado de T-gen en un split ([] si no hay panel: los pasos siguen sin T-gen)."""
    from .generators import load_panel
    keys = list((load_panel(study, split) or {}).get("macro") or [])
    if not keys:
        log(f"[pre] ATENCIÓN: no hay macro de T-gen en data/eval/tgen/{split}/PANEL.json: se sigue sin T-gen")
    return keys


# ======================================================================================
# Pasos del notebook (cada uno se llama con correr(); todos son reanudables y no repiten lo ya hecho)
def paso_textos(study, log=print) -> dict:
    """Textos de test y sellado (pregunta, oro y HN de Gemini) y corpus de BM25: lo único que necesita los tokens de
    Nemotron-DQA (10.5 GB), junto con la fuga. Antes, registra la regla del bloque 1 (primer paso del bloque)."""
    bloque1_regla(study, macro_keys(study, "confirm", log=log), log=log)
    out = {}
    for sp in ("confirm", "test"):
        out[sp] = len(textos_split(study, sp, log=log))
        log(f"[textos] {sp}: {out[sp]} ítems con texto")
    bm = bm25_corpus(study, log=log)
    out["bm25_corpus"] = bm.N
    return out


def b1_items(study, log=print) -> tuple:
    keys = macro_keys(study, "confirm")
    tx = textos_split(study, "confirm", log=log)
    tg, ttxt = textos_tgen(study, "confirm", keys)
    return qc_items(study, tx, tg, ttxt, keys), keys


def paso_etiquetas(study, log=print) -> str:
    items, keys = b1_items(study, log=log)
    bloque1_regla(study, keys, log=log)
    log(f"[b1] ítems a juzgar: {dict(Counter(set_of(k) for k in items))} (total {len(items)})")
    return export_hand_labels(study, items, keys, log=log)


def paso_juez(study, backend: str = "hf", batch: int = 128, log=print) -> dict:
    items, keys = b1_items(study, log=log)
    bloque1_regla(study, keys, log=log)
    if not os.path.exists(pdir(study, "b1_calidad", "etiquetas", "clave.json")):
        export_hand_labels(study, items, keys, log=log)          # el CSV ciego sale ANTES del juez, siempre
    if backend != "fake" and len(judge_labels(study)) < len(items):
        asegurar_disco(study, 75, keep_chunks=False, keep_repos=(JUDGE["repo"],), log=log)
    return run_judge(study, items, backend=backend, batch=batch, log=log)


def paso_fuga(study, log=print) -> dict:
    bloque1_regla(study, macro_keys(study, "confirm", log=log), log=log)
    r = fuga(study, log=log)
    for sp, x in (r.get("splits") or {}).items():
        log(f"[fuga] {sp}: {x['par_exacto_o_casi']}/{x['n_items']} ítems ({pct(x['tasa'], 2)} %)")
    return r


def paso_calidad(study, log=print) -> dict:
    keys = macro_keys(study, "confirm")
    bloque1_regla(study, keys, log=log)
    return bloque1_reporte(study, keys, log=log)


def bloque3(study, qwen: bool = True, log=print) -> dict:
    """Bloque 3 completo (reanudable por modelo y split): coseno crudo, BM25 y Qwen3-Embedding-8B en test y sellado.
    qwen=False (sin GPU) salta Qwen3-Embedding-8B."""
    from .data import EvalSet
    from .generators import purge_model_cache
    bloque3_regla(study, log=log)
    out = makedirs(pdir(study, "b3_referencias"))
    splits = ("test", "confirm")
    for sp in splits:                                  # no necesita textos: va primero
        if load_ref(study, "coseno_crudo", sp) is None:
            save_ref(study, "coseno_crudo", sp, ref_raw_cosine(study, sp, macro_keys(study, sp, log=log)))
            done_ref(study, "coseno_crudo", sp)
            log(f"[b3] coseno crudo, {sp}: listo")
    prep = {}
    for sp in splits:
        keys = macro_keys(study, sp, log=log)
        tx = textos_split(study, sp, log=log)
        tg, ttxt = textos_tgen(study, sp, keys)
        ev = EvalSet(study.paths.d("data", "eval", sp))
        prep[sp] = {"keys": keys, "sets": ref_sets(study, sp, tx, tg, ttxt, keys), "n": ev.n,
                    "questions": {i: r["q"] for i, r in tx.items()}, "doc_ids": np.asarray(ev.doc_ids)}
    bm = None
    for sp in splits:
        if load_ref(study, "bm25", sp) is None:
            bm = bm or bm25_corpus(study, log=log)
            save_ref(study, "bm25", sp, ref_bm25(prep[sp]["sets"], prep[sp]["questions"], bm, prep[sp]["n"]))
            done_ref(study, "bm25", sp)
            log(f"[b3] BM25, {sp}: listo")
    sp_path = os.path.join(out, "qwen3_comprobacion.json")
    if not qwen:
        log("[b3] sin GPU: Qwen3-Embedding-8B se salta")
    elif any(load_ref(study, "qwen3_emb", sp) is None for sp in splits):
        asegurar_disco(study, 30, keep_chunks=False, log=log)
        emb = None
        try:
            emb = QwenEmbedder(cache_dir=os.path.join(hf_home(study), "hub"))
            chk = emb.sanity()
            save_json(chk, sp_path)
            log(f"[b3] Qwen3-Embedding-8B ({chk['modo']}): comprobación {'ok' if chk['ok'] else 'FALLÓ'} {chk['matriz']}")
            if chk["ok"]:
                for sp in splits:
                    if load_ref(study, "qwen3_emb", sp) is None:
                        save_ref(study, "qwen3_emb", sp, ref_qwen3(prep[sp]["sets"], prep[sp]["questions"], emb,
                                                                   prep[sp]["n"], log=log, split=sp))
                        done_ref(study, "qwen3_emb", sp)
                        log(f"[b3] Qwen3-Embedding-8B, {sp}: listo")
            else:
                log("[b3] Qwen3-Embedding-8B no pasó la comprobación con el ejemplo de su ficha: no se usa (el resto sigue)")
        finally:
            del emb
            _free_gpu()
            gb = purge_model_cache(QEMB["repo"], hf_home(study))
            if gb:
                log(f"[b3] borrados {gb:.1f} GB de pesos de Qwen3-Embedding-8B")
    tablas = {sp: {} for sp in splits}
    for sp in splits:
        for m in REF_MODELS:
            r = load_ref(study, m, sp)
            if r:
                tablas[sp][m] = ref_summary(r, prep[sp]["doc_ids"], prep[sp]["keys"])
    res = {"creado": now_iso(), "sanity": load_json(sp_path), "tablas": tablas, "contexto_F3": contexto_f3(study)}
    save_json(res, os.path.join(out, "referencias.json"))
    write_text(os.path.join(out, "REFERENCIAS.md"), bloque3_markdown(res))
    log(f"[b3] guardado en {out}")
    return res
