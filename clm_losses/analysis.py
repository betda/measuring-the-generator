"""Análisis de la enmienda 03: uso del estado y transferencia entre generadores (EXP-006 y siguientes).

Lectura pre-especificada (ENMIENDA_03 §6), para un contraste A − B en el macro de T-gen:
    "transfiere"      Δ > +ROPE (1 pp) y el IC95 excluye 0
    "empeora"         Δ < −ROPE y el IC95 excluye 0
    "no transfiere"   equivalentes por TOST dentro de ±ROPE
    "no concluyente"  lo demás
Además se reporta la fracción transferida Δ(T-gen) / Δ(T-gem), sólo si los HN suben T-gem (Δ > ROPE, IC95 > 0).
Secundario (§10): el mismo contraste en el macro de generadores grandes y en el de chicos, y su interacción.
"""
from __future__ import annotations

import os

import numpy as np

from .generators import load_panel, macro_for_run
from .pipeline import iter_run_dirs
from .stats import clusters_from_docs, contrast
from .utils import load_json, makedirs, save_json

METRICS = ("tgem", "tgem_shuf", "tmine", "tmine_shuf", "trand", "tmine50", "tgenmacro", "tgenmacro_shuf",
           "tgenbig", "tgensmall", "tgenall", "tgenall_shuf")
LABELS = {"tgem": "T-gem", "tgem_shuf": "T-gem barajado", "tmine": "T-mine", "tmine_shuf": "T-mine barajado",
          "trand": "T-rand", "tmine50": "T-mine-50", "tgenmacro": "T-gen (macro)",
          "tgenmacro_shuf": "T-gen (macro) barajado", "tgenbig": "T-gen (macro de grandes, secundario)",
          "tgensmall": "T-gen (macro de chicos, secundario)",
          "tgenall": "T-gen (todos los primarios, con atajo; respaldo §3.6)",
          "tgenall_shuf": "T-gen (todos los primarios) barajado",
          "tgenfail": "T-gen, ítems donde el ciego propio falla (primarios; respaldo §3.6)",
          "tgenfail_shuf": "T-gen, ítems donde el ciego falla, barajado"}


def default_groups(study) -> dict:
    """Grupos de corridas: F0.7 y F1 piloto (ya entrenadas) y, si existen, F0.7b y F1b."""
    g = {}
    for lab, d in (("F0.7 · pre (sin HN)", "infonce_pre"), ("F0.7 · HN + in-batch", "infonce_hn_inbatch"),
                   ("F0.7 · HN solos", "infonce_hn_hnonly"), ("F0.7 · HN desde el inicio (sólo HN)", "infonce_hn_start")):
        rds = [r for r in iter_run_dirs(study, "F0.7") if f"/F0.7/{d}/" in r]
        if rds:
            g[lab] = rds
    for rec in ("clm", "actual"):
        for cond in ("pre", "mid_inbatch", "start_inbatch", "mid_hnonly"):
            rds = [r for r in iter_run_dirs(study, "F0.7b") if f"/F0.7b/{rec}/{cond}/" in r]
            if rds:
                g[f"F0.7b · {rec} · {cond}"] = rds
    for ph in ("F1", "F1b"):
        for arm in ("infonce", "infonce_hn", "siglip", "siglip_hn", "triplet_hn"):
            rds = [r for r in iter_run_dirs(study, ph) if f"/{ph}/{arm}/" in r]
            if rds:
                g[f"{ph} · {arm}"] = rds
    return g


DEFAULT_CONTRASTS = [
    ("F0.7 · HN + in-batch", "F0.7 · pre (sin HN)"),
    ("F0.7 · HN solos", "F0.7 · HN + in-batch"),
    ("F0.7 · HN desde el inicio (sólo HN)", "F0.7 · pre (sin HN)"),
    ("F1 · infonce_hn", "F1 · infonce"),
    ("F1 · siglip_hn", "F1 · siglip"),
    ("F0.7b · clm · mid_inbatch", "F0.7b · clm · pre"),
    ("F0.7b · actual · mid_inbatch", "F0.7b · actual · pre"),
    ("F0.7b · actual · mid_hnonly", "F0.7b · actual · mid_inbatch"),
    ("F1b · infonce_hn", "F1b · infonce"),
    ("F1b · siglip_hn", "F1b · siglip"),
]


def _load(rd: str, name: str):
    p = os.path.join(rd, "eval", f"correct_{name}.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


def _stack(rds, name):
    xs = [_load(rd, name) for rd in rds]
    if any(x is None for x in xs):
        return None
    if len({len(x) for x in xs}) != 1:
        return None
    return np.vstack(xs)


def reading(c: dict, rope: float) -> str:
    if c["diff"] > rope and c["ci95_lo"] > 0:
        return "transfiere"
    if c["diff"] < -rope and c["ci95_hi"] < 0:
        return "empeora"
    if c["tost_equivalent"]:
        return "no transfiere (equivalente)"
    return "no concluyente"


def transfer_report(study, bundle, groups: dict | None = None, contrasts=None, out_name: str = "EXP006",
                    B: int = 2000, log=print) -> dict:
    """Tabla por grupo (métricas + cada generador del panel) y contrastes pareados por documento."""
    import pandas as pd
    panel = load_panel(study, bundle.test_split)
    groups = groups or default_groups(study)
    if panel:
        for rds in groups.values():
            for rd in rds:
                macro_for_run(rd, panel)
    gens = sorted((panel or {}).get("generators", {}))
    # subconjunto "el ciego propio falla" de cada generador (respaldo de §3.6): ítems donde la sonda ciega con
    # validación cruzada no acierta; alineado con items.npy del generador
    fail_mask = {}
    for k in gens:
        gd = os.path.join(study.paths.d("data", "eval", "tgen", bundle.test_split), k)
        pc, pi = os.path.join(gd, "blind_own_cv_correct.npy"), os.path.join(gd, "items.npy")
        if os.path.exists(pc) and os.path.exists(pi):
            fail_mask[k] = (np.load(pi), np.load(pc) < 0.5)
    prim = [k for k in (panel or {}).get("todos", [])]

    def fail_pool(rds, shuf=False):
        """Respaldo de §3.6: acierto en los pares (generador primario, ítem) donde el ciego propio de ese generador
        falla, concatenados; con el documento de cada ítem para el bootstrap. None si falta algo."""
        cols, docs = [], []
        for k in prim:
            if k not in fail_mask:
                continue
            X = _stack(rds, f"tgen_{k}" + ("_shuf" if shuf else ""))
            if X is None:
                return None, None
            it_run = np.load(os.path.join(rds[0], "eval", f"items_tgen_{k}.npy"))
            it_gen, mask = fail_mask[k]
            if not np.array_equal(it_run, it_gen):
                return None, None
            cols.append(X[:, mask])
            docs.append(np.asarray(bundle.test.doc_ids)[it_gen[mask]])
        if not cols:
            return None, None
        return np.concatenate(cols, 1), np.concatenate(docs)

    rows = []
    for lab, rds in groups.items():
        row = {"grupo": lab, "semillas": len(rds)}
        for m in METRICS:
            X = _stack(rds, m)
            row[LABELS[m]] = round(float(X.mean()), 4) if X is not None else None
        for nm, sh in (("tgenfail", False), ("tgenfail_shuf", True)):
            X, _ = fail_pool(rds, sh)
            row[LABELS[nm]] = round(float(X.mean()), 4) if X is not None else None
        for k in gens:
            X = _stack(rds, f"tgen_{k}")
            row[f"{k}"] = round(float(X.mean()), 4) if X is not None else None
            Xs = _stack(rds, f"tgen_{k}_shuf")
            row[f"{k} (barajado)"] = round(float(Xs.mean()), 4) if Xs is not None else None
            if X is not None and k in fail_mask:
                it_run = np.load(os.path.join(rds[0], "eval", f"items_tgen_{k}.npy"))
                it_gen, mask = fail_mask[k]
                if np.array_equal(it_run, it_gen) and mask.any():
                    row[f"{k} (ciego propio falla)"] = round(float(X[:, mask].mean()), 4)
        rows.append(row)
    table = pd.DataFrame(rows)
    rope = study.rope_pp / 100
    doc = np.asarray(bundle.test.doc_ids)
    cl_all = clusters_from_docs(doc)
    eff = []
    for a, b in (contrasts or DEFAULT_CONTRASTS):
        if a not in groups or b not in groups:
            continue
        mets = (["tgem", "tgem_shuf", "tmine", "tgenmacro", "tgenmacro_shuf", "tgenbig", "tgensmall", "tgenall",
                 "tgenall_shuf", "tgenfail", "tgenfail_shuf"] + [f"tgen_{k}" for k in gens])
        d_tgem = None
        for m in mets:
            if m.startswith("tgenfail"):
                (A, dA), (Bm, dB) = fail_pool(groups[a], m.endswith("_shuf")), fail_pool(groups[b], m.endswith("_shuf"))
                if A is None or Bm is None or A.shape[1] != Bm.shape[1]:
                    continue
                c = contrast(A, Bm, clusters_from_docs(dA), rope, B=B)
                eff.append({"contraste": f"{a} − {b}", "métrica": LABELS[m], "Δ (pp)": round(100 * c["diff"], 1),
                            "IC95 (pp)": f"[{100 * c['ci95_lo']:.1f}, {100 * c['ci95_hi']:.1f}]", "p": c["p"],
                            "TOST ±1 pp": c["tost_equivalent"],
                            "lectura": reading(c, rope) + " (respaldo §3.6)" if m == "tgenfail" else ""})
                continue
            A, Bm = _stack(groups[a], m), _stack(groups[b], m)
            if A is None or Bm is None or A.shape[1] != Bm.shape[1]:
                continue
            if m.startswith("tgen"):
                it = None
                if m.startswith("tgenmacro") and panel:
                    it = np.asarray(panel["macro_items"], np.int64)
                elif m.startswith("tgenall") and panel:
                    it = np.asarray(panel.get("todos_items") or [], np.int64)
                else:
                    pi = os.path.join(groups[a][0], "eval", f"items_{m.replace('_shuf', '')}.npy")
                    it = np.load(pi) if os.path.exists(pi) else None
                cl = clusters_from_docs(doc[it]) if it is not None and len(it) == A.shape[1] else np.arange(A.shape[1])
            else:
                cl = cl_all
            c = contrast(A, Bm, cl, rope, B=B)
            r = {"contraste": f"{a} − {b}", "métrica": LABELS.get(m, m), "Δ (pp)": round(100 * c["diff"], 1),
                 "IC95 (pp)": f"[{100 * c['ci95_lo']:.1f}, {100 * c['ci95_hi']:.1f}]", "p": c["p"],
                 "TOST ±1 pp": c["tost_equivalent"]}
            if m == "tgem" and c["diff"] > rope and c["ci95_lo"] > 0:
                d_tgem = c["diff"]          # la fracción sólo tiene sentido si los HN sí suben T-gem
            if m == "tgenmacro":
                r["lectura"] = reading(c, rope)
                if d_tgem:
                    r["fracción transferida"] = round(c["diff"] / d_tgem, 2)
            elif m in ("tgenbig", "tgensmall"):
                r["lectura"] = reading(c, rope) + " (secundario)"
            elif m == "tgenall":
                r["lectura"] = reading(c, rope) + " (con atajo; respaldo §3.6)"
            eff.append(r)
        # secundario (ENMIENDA_03 §10): ¿el efecto A − B cambia con el tamaño del generador? Interacción
        # (A − B en el macro de grandes) − (A − B en el macro de chicos), en los ítems comunes a los dos macros.
        Ab, Bb, As_, Bs_ = (_stack(groups[g], m) for g, m in ((a, "tgenbig"), (b, "tgenbig"),
                                                               (a, "tgensmall"), (b, "tgensmall")))
        if all(x is not None for x in (Ab, Bb, As_, Bs_)):
            ib = np.load(os.path.join(groups[a][0], "eval", "items_tgenbig.npy"))
            ism = np.load(os.path.join(groups[a][0], "eval", "items_tgensmall.npy"))
            common = np.intersect1d(ib, ism)
            if len(common) >= 20 and Ab.shape[1] == len(ib) and As_.shape[1] == len(ism):
                pb, ps = np.searchsorted(ib, common), np.searchsorted(ism, common)
                c = contrast(Ab[:, pb] - As_[:, ps], Bb[:, pb] - Bs_[:, ps], clusters_from_docs(doc[common]), rope, B=B)
                eff.append({"contraste": f"{a} − {b}", "métrica": "interacción: efecto en grandes − efecto en chicos",
                            "Δ (pp)": round(100 * c["diff"], 1),
                            "IC95 (pp)": f"[{100 * c['ci95_lo']:.1f}, {100 * c['ci95_hi']:.1f}]", "p": c["p"],
                            "TOST ±1 pp": c["tost_equivalent"],
                            "lectura": f"secundario ({len(common)} ítems comunes)"})
    effects = pd.DataFrame(eff)
    out = makedirs(study.paths.d("results", "transfer", out_name))
    table.to_csv(os.path.join(out, "tabla.csv"), index=False)
    effects.to_csv(os.path.join(out, "contrastes.csv"), index=False)
    save_json({"panel_version": (panel or {}).get("version"), "panel_macro": (panel or {}).get("macro"),
               "panel_build": (panel or {}).get("build"), "panel_todos": (panel or {}).get("todos"),
               "tabla": rows, "contrastes": eff, "rope_pp": study.rope_pp,
               "lectura": "transfiere: Δ macro > +ROPE con IC95 > 0; no transfiere: TOST ±ROPE; si no, no concluyente"},
              os.path.join(out, "transfer.json"))
    log(f"[transfer] guardado en {out}")
    return {"tabla": table, "contrastes": effects, "dir": out}


def state_use_table(study, phase: str = "F1b"):
    """Por brazo (media de semillas): acierto con estados reales y barajados (A1.4) en T-mine, T-gem y el macro
    de T-gen, y abstención en T-abst-mine. "Uso del estado" = real − barajado."""
    import pandas as pd
    from .losses import ARMS, arm_names
    panel = load_panel(study, "confirm" if phase == "F3" else "test")
    rows = []
    for arm in arm_names():
        rds = [r for r in iter_run_dirs(study, phase) if f"/{phase}/{arm}/" in r]
        if not rds:
            continue
        acc = {}
        for rd in rds:
            m = load_json(os.path.join(rd, "eval", "metrics.json"), {}) or {}
            mac = macro_for_run(rd, panel) if panel else None
            vals = {"T-mine": (m.get("tmine") or {}).get("top1"), "T-mine barajado": (m.get("tmine_shuf") or {}).get("top1"),
                    "T-gem": (m.get("tgem") or {}).get("top1"), "T-gem barajado": (m.get("tgem_shuf") or {}).get("top1"),
                    "T-gen macro": (mac or {}).get("top1"), "T-gen macro barajado": (mac or {}).get("shuf_top1"),
                    "AUROC T-abst-mine": (m.get("tabst_mine") or {}).get("abst_auroc_maxcos")}
            for k, v in vals.items():
                if v is not None:
                    acc.setdefault(k, []).append(v)
        row = {"#": ARMS[arm]["num"], "brazo": arm, "semillas": len(rds)}
        row.update({k: round(float(np.mean(v)), 4) for k, v in acc.items()})
        if "T-gem" in row and "T-gem barajado" in row:
            row["uso del estado T-gem (pp)"] = round(100 * (row["T-gem"] - row["T-gem barajado"]), 1)
        if "T-gen macro" in row and "T-gen macro barajado" in row:
            row["uso del estado T-gen (pp)"] = round(100 * (row["T-gen macro"] - row["T-gen macro barajado"]), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("#") if rows else pd.DataFrame()


def f07b_table(study):
    """Tabla de F0.7b: una fila por receta × condición (media de semillas)."""
    import pandas as pd
    res = load_json(study.paths.d("f0", "F0.7b_recipes", "result.json")) or {}
    acc = {}
    for key, r in (res.get("runs") or {}).items():
        rec, cond, _ = key.split("/")
        acc.setdefault((rec, cond), []).append(r)
    rows = []
    for (rec, cond), lst in sorted(acc.items()):
        row = {"receta": rec, "condición": cond, "semillas": len(lst)}
        for k in ("tmine", "tmine_shuf", "tgem", "tgem_shuf", "trand", "tmine50", "abst_tmine", "val_best"):
            xs = [x.get(k) for x in lst if x.get(k) is not None]
            row[k] = round(float(np.mean(xs)), 4) if xs else None
        secs = [x.get("train_seconds") for x in lst if x.get("train_seconds") is not None]
        row["min_entrenamiento"] = round(float(np.mean(secs)) / 60, 1) if secs else None
        rows.append(row)
    return pd.DataFrame(rows), res.get("G2b")


# --------------------------------------------------------------------------------------
# EXP-006c (ENMIENDA_03 §12): ¿la brecha T-gem / T-gen es del generador o de la construcción?
#
# EXP-006 compara la ganancia por HN en T-gem (negativos LIBRES de Gemini) con la del macro de T-gen (negativos
# EDITADOS de otros generadores): la "fracción transferida" mezcla dos cosas, el generador y el tipo de negativo.
# Los sets del piloto (6.0) permiten separarlas sin generar ni entrenar nada: los mismos 3 generadores abiertos con
# las tres construcciones (S: negativos libres con las reglas de texto; P: libres + oro reescrito; E: editados), en
# ítems del split de test, donde T-gem también tiene su respuesta. En los ítems comunes a los 9 sets:
#     efecto del generador    = Δ(T-gem) − Δ(S)   (libres de Gemini frente a libres de otros generadores)
#     efecto de la construcción = Δ(S) − Δ(E)     (libres frente a editados, mismos generadores)
#     efecto del oro reescrito = Δ(P) − Δ(S)      (atajo de "reconocer el oro": P cambia sólo el positivo)
# con Δ = A − B (p. ej. infonce_hn − infonce), bootstrap pareado por documento. Lectura fijada antes de mirar
# (§12.3): cada componente "explica parte de la brecha" si es > +ROPE con IC95 > 0; "no la explica" si es
# equivalente por TOST ±ROPE; si no, no concluyente.

CONSTRUCTION_CONTRASTS = [
    ("F1b · infonce_hn", "F1b · infonce"),                              # primario de EXP-006c
    ("F1b · siglip_hn", "F1b · siglip"),
    ("F0.7b · actual · mid_inbatch", "F0.7b · actual · pre"),
    ("F0.7 · HN + in-batch", "F0.7 · pre (sin HN)"),
    ("F0.7b · actual · mid_hnonly", "F0.7b · actual · mid_inbatch"),   # mecanismo: "HN solos"
    ("F0.7 · HN solos", "F0.7 · HN + in-batch"),
]
PILOT_BUILDS = ("S", "P", "E")


def pilot_set_name(build: str, key: str) -> str:
    return f"pilot{build}_{key}"


def load_pilot_sets(study, split: str = "test") -> dict:
    """{"pilot<B>_<clave>": (ítems, negativos, huella, positivos)} de los sets del piloto (6.0), sólo de los
    generadores que tienen las tres construcciones embebidas."""
    from .generators import build_path, gen_sig, load_gen
    rec = load_json(build_path(study)) or {}
    keys = list((rec.get("pilot") or {}).get("keys") or [])
    out = {}
    for k in keys:
        sets = {b: load_gen(study, k, split, sub=f"pilot_{b}") for b in PILOT_BUILDS}
        if any(g is None for g in sets.values()):
            continue
        for b, g in sets.items():
            out[pilot_set_name(b, k)] = (np.asarray(g.items), g.negs, gen_sig(study, k, split, f"pilot_{b}"), g.pos)
    return out


def _at(rd: str, name: str, items: np.ndarray, shuf: bool = False):
    """Acierto de una corrida en un set externo, reordenado a `items` (None si falta algo)."""
    ev = os.path.join(rd, "eval")
    pc = os.path.join(ev, f"correct_{name}{'_shuf' if shuf else ''}.npy")
    pi = os.path.join(ev, f"items_{name}.npy")
    if not (os.path.exists(pc) and os.path.exists(pi)):
        return None
    c, it = np.load(pc).astype(np.float32), np.load(pi)
    pos = {int(x): j for j, x in enumerate(it)}
    if any(int(x) not in pos for x in items):
        return None
    return c[[pos[int(x)] for x in items]]


def _tgem_at(rd: str, items: np.ndarray, shuf: bool = False):
    p = os.path.join(rd, "eval", f"correct_tgem{'_shuf' if shuf else ''}.npy")
    if not os.path.exists(p):
        return None
    c, it = np.load(p).astype(np.float32), np.asarray(items, np.int64)
    return c[it] if len(it) == 0 or (it.min() >= 0 and it.max() < len(c)) else None


def construction_matrices(groups: dict, keys: list, items: np.ndarray, shuf: bool = False) -> dict:
    """{grupo: {"T-gem"|"S"|"P"|"E": matriz semillas × ítems}}. En S/P/E, el acierto por ítem es el promedio
    sobre los generadores del piloto (como el macro de T-gen). Grupo con algún dato faltante: se omite."""
    out = {}
    for lab, rds in groups.items():
        mats = {"T-gem": [], **{b: [] for b in PILOT_BUILDS}}
        ok = True
        for rd in rds:
            t = _tgem_at(rd, items, shuf)
            if t is None:
                ok = False
                break
            mats["T-gem"].append(t)
            for b in PILOT_BUILDS:
                cols = [_at(rd, pilot_set_name(b, k), items, shuf) for k in keys]
                if any(c is None for c in cols):
                    ok = False
                    break
                mats[b].append(np.mean(cols, 0))
            if not ok:
                break
        if ok and rds:
            out[lab] = {m: np.vstack(v) for m, v in mats.items()}
    return out


def component_reading(c: dict, rope: float) -> str:
    if c["diff"] > rope and c["ci95_lo"] > 0:
        return "explica parte de la brecha"
    if c["diff"] < -rope and c["ci95_hi"] < 0:
        return "en sentido contrario"
    if c["tost_equivalent"]:
        return "no la explica (equivalente)"
    return "no concluyente"


def construction_report(study, bundle, groups: dict | None = None, contrasts=None, out_name: str = "EXP006c",
                        B: int = 2000, evaluate: bool = True, log=print) -> dict:
    """EXP-006c: descompone la brecha T-gem / T-gen de la ganancia por HN en generador y construcción con los
    sets del piloto. Evalúa primero (sin re-entrenar) las corridas que falten en esos sets."""
    import pandas as pd
    from .pipeline import eval_extra
    pil = load_pilot_sets(study, bundle.test_split)
    keys = sorted({n.split("_", 1)[1] for n in pil})
    out = makedirs(study.paths.d("results", "transfer", out_name))
    if len(keys) < 2:
        log("[EXP-006c] faltan los sets del piloto (S, P y E de al menos 2 generadores): no se hace nada")
        return {"tabla": pd.DataFrame(), "contrastes": pd.DataFrame(), "dir": out, "keys": keys}
    groups = groups or default_groups(study)
    contrasts = [c for c in (contrasts or CONSTRUCTION_CONTRASTS) if c[0] in groups and c[1] in groups]
    used = {g for c in contrasts for g in c}
    if evaluate:
        rds = sorted({rd for g in used for rd in groups[g]})
        n = eval_extra(study, bundle, rds, pil, log=log)
        log(f"[EXP-006c] corridas evaluadas en los sets del piloto: {n} (de {len(rds)})")
    items = None
    for v in pil.values():
        items = np.asarray(v[0], np.int64) if items is None else np.intersect1d(items, v[0])
    items = np.sort(np.asarray(items, np.int64))
    sub = {g: groups[g] for g in used}
    M, Ms = construction_matrices(sub, keys, items), construction_matrices(sub, keys, items, shuf=True)
    rows = []
    for lab in sub:
        if lab not in M:
            continue
        r = {"grupo": lab, "semillas": len(groups[lab])}
        for m in ("T-gem", "S", "P", "E"):
            r[m] = round(float(M[lab][m].mean()), 4)
            if lab in Ms:
                r[f"{m} barajado"] = round(float(Ms[lab][m].mean()), 4)
        rows.append(r)
    table = pd.DataFrame(rows)
    rope = study.rope_pp / 100
    cl = clusters_from_docs(np.asarray(bundle.test.doc_ids)[items])
    eff = []
    raw = {}

    def add(a, b, name, A, Bm, kind):
        c = contrast(A, Bm, cl, rope, B=B)
        rec = {"contraste": f"{a} − {b}", "cantidad": name, "Δ (pp)": round(100 * c["diff"], 1),
               "IC95 (pp)": f"[{100 * c['ci95_lo']:.1f}, {100 * c['ci95_hi']:.1f}]", "p": c["p"],
               "TOST ±1 pp": c["tost_equivalent"], "lectura": component_reading(c, rope) if kind == "comp" else ""}
        eff.append(rec)
        raw.setdefault((a, b), {})[name] = c
        return c

    for a, b in contrasts:
        if a not in M or b not in M:
            continue
        d = {}
        for m in ("T-gem", "S", "P", "E"):
            d[m] = add(a, b, f"Δ en {m}", M[a][m], M[b][m], "delta")
        for m in ("T-gem", "S", "E"):
            if a in Ms and b in Ms:
                add(a, b, f"Δ en {m} barajado", Ms[a][m], Ms[b][m], "delta")
        gen = add(a, b, "generador: Δ(T-gem) − Δ(S)", M[a]["T-gem"] - M[a]["S"], M[b]["T-gem"] - M[b]["S"], "comp")
        con = add(a, b, "construcción: Δ(S) − Δ(E)", M[a]["S"] - M[a]["E"], M[b]["S"] - M[b]["E"], "comp")
        add(a, b, "oro reescrito: Δ(P) − Δ(S)", M[a]["P"] - M[a]["S"], M[b]["P"] - M[b]["S"], "comp")
        gap = add(a, b, "brecha total: Δ(T-gem) − Δ(E)", M[a]["T-gem"] - M[a]["E"], M[b]["T-gem"] - M[b]["E"], "comp")
        if gap["diff"] > rope and gap["ci95_lo"] > 0:
            eff.append({"contraste": f"{a} − {b}", "cantidad": "parte de la brecha (descriptivo)",
                        "Δ (pp)": None, "IC95 (pp)": "", "p": None, "TOST ±1 pp": None,
                        "lectura": f"generador {gen['diff'] / gap['diff']:.2f} · construcción {con['diff'] / gap['diff']:.2f}"})
    effects = pd.DataFrame(eff)
    table.to_csv(os.path.join(out, "tabla.csv"), index=False)
    effects.to_csv(os.path.join(out, "contrastes.csv"), index=False)
    save_json({"keys": keys, "n_items": int(len(items)), "items": items.tolist(), "tabla": rows, "contrastes": eff,
               "rope_pp": study.rope_pp, "regla": "ENMIENDA_03 §12.3: un componente explica parte de la brecha si "
               "es > +ROPE con IC95 > 0; no la explica si es equivalente por TOST ±ROPE; si no, no concluyente"},
              os.path.join(out, "construccion.json"))
    log(f"[EXP-006c] {len(keys)} generadores del piloto · {len(items)} ítems comunes · guardado en {out}")
    return {"tabla": table, "contrastes": effects, "dir": out, "keys": keys, "n_items": int(len(items)), "raw": raw}


# --------------------------------------------------------------------------------------
# C2 en F3 (ENMIENDA_03 §12.4; pre-especificado antes del freeze): la caída de la ganancia por HN al pasar de
# los negativos de Gemini (T-gem) a los del panel (macro de T-gen), en el test confirmatorio.
C2_PAIRS = (("infonce_hn", "infonce"), ("siglip_hn", "siglip"))


def c2_reading(d_tgem: dict, c2: dict, p_holm: float, rope: float) -> str:
    """Lectura de C2 (ENMIENDA_03 §12.4). Sólo aplica si los HN suben T-gem en el par."""
    if not (d_tgem["diff"] > rope and d_tgem["ci95_lo"] > 0):
        return "no aplica: los HN no suben T-gem en este par"
    if c2["diff"] > rope and p_holm < 0.05:
        return "H1b se sostiene: la ganancia por HN cae al pasar a otros generadores"
    if c2["tost_equivalent"]:
        return "equivalente: la ganancia no cae"
    if c2["diff"] < -rope and c2["ci95_hi"] < 0:
        return "la ganancia es MAYOR con otros generadores"
    return "no concluyente"


def c2_report(study, bundle, phase: str = "F3", B: int = 4000, log=print) -> dict:
    """Por par (con HN − sin HN): Δ en T-gem y en el macro de T-gen (ítems del macro) y C2 = Δ(T-gem) − Δ(macro),
    pareado por documento, con Holm entre los pares. H1b se sostiene en un par si los HN suben T-gem (Δ > +ROPE,
    IC95 > 0) y C2 > +ROPE con p_Holm < 0.05 (ENMIENDA_03 §12.4)."""
    from .generators import load_panel, macro_for_run
    from .pipeline import iter_run_dirs
    from .stats import holm
    panel = load_panel(study, bundle.test_split) or {}
    if not panel.get("macro"):
        log("[C2] no hay macro de T-gen en este split: no se hace nada")
        return {"pares": [], "panel_version": panel.get("version")}
    it = np.asarray(panel["macro_items"], np.int64)
    cl = clusters_from_docs(np.asarray(bundle.test.doc_ids)[it])
    rope = study.rope_pp / 100
    res = []
    for a, b in C2_PAIRS:
        ga = [r for r in iter_run_dirs(study, phase) if f"/{phase}/{a}/" in r]
        gb = [r for r in iter_run_dirs(study, phase) if f"/{phase}/{b}/" in r]
        if not ga or not gb:
            continue
        for rd in ga + gb:
            macro_for_run(rd, panel)
        T = {g: [_tgem_at(rd, it) for rd in rds] for g, rds in ((a, ga), (b, gb))}
        Mc = {g: [_stack([rd], "tgenmacro") for rd in rds] for g, rds in ((a, ga), (b, gb))}
        if any(x is None for v in T.values() for x in v) or any(x is None for v in Mc.values() for x in v):
            log(f"[C2] {a} − {b}: faltan evaluaciones; se omite")
            continue
        Ta, Tb = np.vstack(T[a]), np.vstack(T[b])
        Ma, Mb = np.vstack([x[0] for x in Mc[a]]), np.vstack([x[0] for x in Mc[b]])
        if Ma.shape[1] != len(it):
            log(f"[C2] {a} − {b}: el macro de la corrida no coincide con el del panel; se omite")
            continue
        dt, dm = contrast(Ta, Tb, cl, rope, B=B), contrast(Ma, Mb, cl, rope, B=B)
        c2 = contrast(Ta - Ma, Tb - Mb, cl, rope, B=B)
        res.append({"par": f"{a} − {b}", "Δ T-gem": dt, "Δ macro T-gen": dm, "C2": c2})
    if res:
        ph = holm([r["C2"]["p"] for r in res])
        for r, p in zip(res, ph):
            r["C2"]["p_holm"] = float(p)
            r["lectura"] = c2_reading(r["Δ T-gem"], r["C2"], float(p), rope)
    out = makedirs(study.paths.d("results", phase, "C2"))
    save_json({"panel_version": panel.get("version"), "macro": panel.get("macro"), "n_items": int(len(it)),
               "pares": res, "rope_pp": study.rope_pp,
               "regla": "C2 = Δ(T-gem) − Δ(macro de T-gen) por par; H1b se sostiene si Δ(T-gem) > +ROPE con IC95 > 0 "
                        "y C2 > +ROPE con p_Holm < 0.05 (Holm entre los pares de C2_PAIRS)"}, os.path.join(out, "C2.json"))
    log(f"[C2] guardado en {out}")
    return {"pares": res, "panel_version": panel.get("version"), "dir": out}


# --------------------------------------------------------------------------------------
# C2b en F3 (ENMIENDA_03 §13; pre-especificado antes del freeze): la descomposición de EXP-006c, repetida en el
# test confirmatorio con los brazos de F3 y sets del piloto generados para ese split (8.2c).
C2B_COMPONENTS = ("generador: Δ(T-gem) − Δ(S)", "construcción: Δ(S) − Δ(E)")


def c2b_reading(gap: dict, comp: dict, p_holm: float, rope: float) -> str:
    """Lectura de un componente de C2b. Sólo aplica si hay brecha (Δ(T-gem) − Δ(E) > +ROPE con IC95 > 0)."""
    if not (gap["diff"] > rope and gap["ci95_lo"] > 0):
        return "no aplica: no hay brecha entre T-gem y E en este par"
    if comp["diff"] > rope and p_holm < 0.05:
        return "se confirma: explica parte de la brecha"
    if comp["diff"] < -rope and p_holm < 0.05:
        return "en sentido contrario"
    if comp["tost_equivalent"]:
        return "no la explica (equivalente)"
    return "no concluyente"


def c2b_report(study, bundle, phase: str = "F3", B: int = 4000, log=print) -> dict:
    """C2b: por par de C2_PAIRS, componentes generador y construcción (EXP-006c) en el test confirmatorio, con
    Holm entre los 4 contrastes. El oro reescrito y las proporciones se reportan como descriptivos."""
    from .pipeline import iter_run_dirs
    from .stats import holm
    groups, contrasts = {}, []
    for a, b in C2_PAIRS:
        ga = [r for r in iter_run_dirs(study, phase) if f"/{phase}/{a}/" in r]
        gb = [r for r in iter_run_dirs(study, phase) if f"/{phase}/{b}/" in r]
        if ga and gb:
            groups[f"{phase} · {a}"], groups[f"{phase} · {b}"] = ga, gb
            contrasts.append((f"{phase} · {a}", f"{phase} · {b}"))
    out = makedirs(study.paths.d("results", phase, "C2b"))
    if not contrasts:
        log(f"[C2b] faltan brazos de {phase}: no se hace nada")
        return {"pruebas": [], "dir": out}
    cr = construction_report(study, bundle, groups=groups, contrasts=contrasts, out_name=f"EXP006c_{phase}", B=B,
                             log=log)
    rope = study.rope_pp / 100
    tests = []
    for a, b in contrasts:
        r = (cr.get("raw") or {}).get((a, b)) or {}
        gap = r.get("brecha total: Δ(T-gem) − Δ(E)")
        for name in C2B_COMPONENTS:
            if gap is not None and name in r:
                tests.append({"par": f"{a} − {b}", "componente": name, "c": r[name], "brecha": gap})
    if tests:
        for t, p in zip(tests, holm([t["c"]["p"] for t in tests])):
            t["p_holm"] = float(p)
            t["lectura"] = c2b_reading(t["brecha"], t["c"], float(p), rope)
    save_json({"phase": phase, "keys": cr.get("keys"), "n_items": cr.get("n_items"),
               "pruebas": [{"par": t["par"], "componente": t["componente"], "diff": t["c"]["diff"],
                            "ci95": [t["c"]["ci95_lo"], t["c"]["ci95_hi"]], "p": t["c"]["p"], "p_holm": t["p_holm"],
                            "brecha_total": t["brecha"]["diff"], "lectura": t["lectura"]} for t in tests],
               "rope_pp": study.rope_pp,
               "regla": "ENMIENDA_03 §13: por par de C2_PAIRS, componentes generador = Δ(T-gem) − Δ(S) y construcción "
                        "= Δ(S) − Δ(E) en los ítems comunes a los sets del piloto del confirmatorio; Holm entre los 4; "
                        "un componente se confirma si es > +ROPE con p_Holm < 0.05 (sólo si la brecha total es "
                        "> +ROPE con IC95 > 0). El oro reescrito y las proporciones son descriptivos."},
              os.path.join(out, "C2b.json"))
    log(f"[C2b] guardado en {out}")
    return {"pruebas": tests, "construccion": cr, "dir": out}


# --------------------------------------------------------------------------------------
# C2c en F3 (ENMIENDA_03 §14; pre-especificado antes del freeze): el mecanismo de H1b. ¿Comparar el oro sólo contra
# sus HN ("HN solos") aprende una preferencia que no depende del estado y cuesta discriminación real? Variante
# registrada en el freeze (F3_variants) frente a infonce_hn ("HN + in-batch"), con los mismos HP y semillas.
C2C_PAIR = ("infonce_hn_hnonly", "infonce_hn")
C2C_TESTS = (("tgem_shuf", "T-gem con estados barajados", +1), ("tmine", "T-mine", -1))
C2C_CONTROL = ("infonce_hn", "infonce")
C2C_DESCRIPTIVE = ("tgem", "trand", "tmine50", "tgenmacro")


def c2c_reading(c: dict, p_holm: float, rope: float, sign: int) -> str:
    """Lectura de una prueba de C2c en la dirección pre-especificada (`sign` = +1: sube; −1: baja)."""
    d = sign * c["diff"]
    if d > rope and p_holm < 0.05:
        return "se confirma"
    if d < -rope and p_holm < 0.05:
        return "en sentido contrario"
    if c["tost_equivalent"]:
        return "no se confirma (equivalente)"
    if p_holm < 0.05:
        return "diferencia significativa pero < ROPE"
    return "no concluyente"


def c2c_control_reading(c: dict, rope: float) -> str:
    """Control de C2c: con in-batch, los HN no crean la preferencia sin estado (equivalencia por TOST ±ROPE)."""
    if c["tost_equivalent"]:
        return "se confirma (equivalente)"
    if c["diff"] > rope and c["ci95_lo"] > 0:
        return "no se confirma: con in-batch los HN también crean la preferencia"
    return "no concluyente"


def c2c_global(readings: list) -> str:
    ok = [r == "se confirma" for r in readings]
    if len(ok) == len(C2C_TESTS) and all(ok):
        return "mecanismo confirmado: 'HN solos' aprende una preferencia sin estado y pierde discriminación real"
    if any(ok):
        return "mecanismo confirmado en parte (ver cada prueba)"
    return "mecanismo no confirmado"


def c2c_report(study, bundle, phase: str = "F3", B: int = 4000, log=print) -> dict:
    """C2c: 'HN solos' − infonce_hn en T-gem con estados barajados (Δ > +ROPE) y en T-mine (Δ < −ROPE), Holm entre
    las 2; control infonce_hn − infonce en T-gem barajado (TOST ±ROPE); T-gem, T-rand, T-mine-50 y el macro de
    T-gen como descriptivos. Pareado por documento."""
    from .stats import holm
    runs = iter_run_dirs(study, phase)
    grp = lambda name: [r for r in runs if f"/{phase}/{name}/" in r]  # noqa: E731
    out = makedirs(study.paths.d("results", phase, "C2c"))
    a, b = C2C_PAIR
    ga, gb = grp(a), grp(b)
    if not ga or not gb:
        log(f"[C2c] faltan corridas de {phase} ({a}: {len(ga)}, {b}: {len(gb)}): no se hace nada")
        return {"pruebas": [], "control": None, "descriptivos": {}, "dir": out}
    n = len(bundle.test.doc_ids)
    cl = clusters_from_docs(bundle.test.doc_ids)
    rope = study.rope_pp / 100

    def pair(ra, rb, key, clusters, n_items):
        A, Bm = _stack(ra, key), _stack(rb, key)
        if A is None or Bm is None or A.shape[1] != n_items or Bm.shape[1] != n_items:
            return None
        return contrast(A, Bm, clusters, rope, B=B)

    tests = []
    for key, label, sign in C2C_TESTS:
        c = pair(ga, gb, key, cl, n)
        if c is None:
            log(f"[C2c] falta {key} en alguna corrida; se omite esa prueba")
            continue
        tests.append({"prueba": label, "metrica": key, "signo": sign, "c": c})
    for t, p in zip(tests, holm([t["c"]["p"] for t in tests])):
        t["p_holm"] = float(p)
        t["lectura"] = c2c_reading(t["c"], float(p), rope, t["signo"])
    ctrl = None
    ca, cb = grp(C2C_CONTROL[0]), grp(C2C_CONTROL[1])
    if ca and cb:
        c = pair(ca, cb, "tgem_shuf", cl, n)
        if c is not None:
            ctrl = {"contraste": f"{C2C_CONTROL[0]} − {C2C_CONTROL[1]}", "metrica": "tgem_shuf", "c": c,
                    "lectura": c2c_control_reading(c, rope)}
    desc = {}
    panel = load_panel(study, bundle.test_split) or {}
    for key in C2C_DESCRIPTIVE:
        if key == "tgenmacro":
            if not panel.get("macro"):
                continue
            it = np.asarray(panel["macro_items"], np.int64)
            for rd in ga + gb:
                macro_for_run(rd, panel)
            c = pair(ga, gb, key, clusters_from_docs(np.asarray(bundle.test.doc_ids)[it]), len(it))
        else:
            c = pair(ga, gb, key, cl, n)
        if c is not None:
            desc[key] = c
    glob = c2c_global([t["lectura"] for t in tests])
    keep = ("diff", "ci95_lo", "ci95_hi", "p", "tost_equivalent")
    from .pipeline import addendum_path, frozen_variants
    _, where = frozen_variants(study)
    desv = (load_json(addendum_path(study)) or {}).get("desviacion") if where == "adenda" else None
    save_json({"phase": phase, "variante": a, "referencia": b, "semillas": {a: len(ga), b: len(gb)},
               "registro": where, "desviacion": desv,
               "pruebas": [{"prueba": t["prueba"], "metrica": t["metrica"], "signo": t["signo"],
                            **{k: t["c"][k] for k in keep}, "p_holm": t["p_holm"], "lectura": t["lectura"]}
                           for t in tests],
               "control": ({"contraste": ctrl["contraste"], "metrica": ctrl["metrica"],
                            **{k: ctrl["c"][k] for k in keep}, "lectura": ctrl["lectura"]} if ctrl else None),
               "descriptivos": {k: {kk: v[kk] for kk in keep} for k, v in desc.items()},
               "lectura_global": glob, "rope_pp": study.rope_pp,
               "regla": "ENMIENDA_03 §14: 'HN solos' − infonce_hn; prueba 1, T-gem con estados barajados, se confirma si "
                        "Δ > +ROPE con p_Holm < 0.05; prueba 2, T-mine, se confirma si Δ < −ROPE con p_Holm < 0.05; Holm "
                        "entre las 2. Control (fuera de Holm): infonce_hn − infonce en T-gem barajado, equivalente por "
                        "TOST ±ROPE. Descriptivos: T-gem, T-rand, T-mine-50 y el macro de T-gen."},
              os.path.join(out, "C2c.json"))
    log(f"[C2c] guardado en {out}")
    return {"pruebas": tests, "control": ctrl, "descriptivos": desc, "lectura_global": glob, "dir": out,
            "registro": where, "desviacion": desv}
