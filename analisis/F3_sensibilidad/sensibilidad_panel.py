"""Sensibilidad del panel T-gen en F3 (post hoc, descriptivo; EXP-011 y DP-08 del EXPERIMENT_LOG).

Por qué: en el test confirmatorio la auditoría de Phi-4 dio 0.312 (umbral 0.30: banda alta) y la diferencia
VICReg − InfoNCE por generador es heterogénea. La regla del freeze manda usar el macro congelado aunque su
auditoría cambie, y eso no se toca: C1 y C2 confirmatorios quedan como están. Aquí sólo se describe cuánto
dependen de cada generador:

- «los N»: el macro congelado. Es el control: tiene que dar exactamente lo mismo que el confirmatorio;
- «sin X»: el macro de los otros generadores, en la intersección de sus ítems;
- «sólo X»: un generador con todos sus ítems.

Usa el mismo bootstrap pareado por documento que el confirmatorio (stats.contrast, semilla 0). No lleva Holm:
no es una prueba. Sólo lee las corridas y escribe results/<fase>/sensibilidad_panel/{sensibilidad.json,
SENSIBILIDAD.md}. Requiere el paquete clm_losses v0.3.6 (no lo modifica).
"""
from __future__ import annotations

import os

import numpy as np

from clm_losses.analysis import C2_PAIRS, _at, _tgem_at
from clm_losses.stats import clusters_from_docs, contrast

MOTIVO = ("Post hoc y descriptivo. En el test confirmatorio la auditoría de Phi-4 dio 0.312 (umbral 0.30: banda "
          "alta) y la diferencia VICReg − InfoNCE por generador es heterogénea. El macro congelado se mantiene "
          "(regla del freeze); esto sólo describe cuánto dependen C1 (parte T-gen) y C2 de cada generador.")


def subsets(keys) -> list:
    """[(nombre, generadores)]: el macro completo, cada uno afuera y cada uno solo."""
    keys = list(keys)
    out = [(f"los {len(keys)} (macro congelado)", keys)]
    if len(keys) > 1:
        out += [(f"sin {k}", [x for x in keys if x != k]) for k in keys]
        out += [(f"sólo {k}", [k]) for k in keys]
    return out


def lectura_c1(c: dict, rope: float, a: str, b: str = "infonce") -> str:
    """Misma regla que el veredicto de C1, más si el IC95 queda entero fuera de ±ROPE."""
    if c["p"] < 0.05 and abs(c["diff"]) >= rope:
        mejor = f"{b} mejor" if c["diff"] < 0 else f"{a} mejor"
        fuera = c["ci95_hi"] < -rope if c["diff"] < 0 else c["ci95_lo"] > rope
        return mejor + (" (IC95 fuera de ±ROPE)" if fuera else " (IC95 entra en ±ROPE)")
    if c["p"] < 0.05:
        return "significativa pero menor que ROPE"
    if c["tost_equivalent"]:
        return "equivalentes (TOST ±ROPE)"
    return "no concluyente"


def compute(groups: dict, keys, items_of, doc_ids, nc, rope: float, at, tgem_at, B: int = 4000,
            pairs=C2_PAIRS) -> list:
    """Núcleo sin lectura ni escritura de archivos.

    groups: {brazo: [carpetas de corridas]}; items_of(generadores) -> ítems (índices del test);
    at(carpeta, generador, ítems) -> acierto por ítem en el set de ese generador (None si falta);
    tgem_at(carpeta, ítems) -> acierto por ítem en T-gem. El acierto de un subconjunto es el promedio por ítem
    sobre sus generadores, como el macro de T-gen."""
    doc_ids = np.asarray(doc_ids)
    rows = []
    for name, ks in subsets(keys):
        items = np.asarray(items_of(ks), np.int64)
        rec = {"subconjunto": name, "generadores": list(ks), "n_items": int(len(items))}
        if len(items) == 0:
            rows.append(rec)
            continue
        cl = clusters_from_docs(doc_ids[items])

        def mat(arm):
            out = []
            for rd in groups.get(arm, []):
                cols = [at(rd, k, items) for k in ks]
                if any(c is None for c in cols):
                    return None
                out.append(np.mean(cols, 0).astype(np.float32))
            return np.vstack(out) if out else None

        def tmat(arm):
            out = [tgem_at(rd, items) for rd in groups.get(arm, [])]
            return np.vstack(out).astype(np.float32) if out and all(o is not None for o in out) else None

        M = {arm: mat(arm) for arm in groups}
        if nc and M.get(nc) is not None and M.get("infonce") is not None:
            c = contrast(M[nc], M["infonce"], cl, rope, B=B)
            c.update(A=nc, B="infonce", lectura=lectura_c1(c, rope, nc))
            rec["C1"] = c
        for a, b in pairs:
            Ta, Tb = tmat(a), tmat(b)
            if any(x is None for x in (M.get(a), M.get(b), Ta, Tb)):
                continue
            rec[f"{a} − {b}"] = {"Δ T-gem": contrast(Ta, Tb, cl, rope, B=B),
                                 "Δ T-gen": contrast(M[a], M[b], cl, rope, B=B),
                                 "C2": contrast(Ta - M[a], Tb - M[b], cl, rope, B=B)}
        rows.append(rec)
    return rows


def _pp(c: dict | None) -> str:
    if not c:
        return "—"
    return f"{100 * c['diff']:+.2f} [{100 * c['ci95_lo']:+.2f}, {100 * c['ci95_hi']:+.2f}]"


def control(rows: list, c1_conf: dict | None, c2_conf: list | None, tol: float = 1e-9) -> dict:
    """¿«los N» reproduce el confirmatorio? Para C1 (T-gen) y C2 por par: la diferencia puntual tiene que ser
    idéntica; el IC95 también, si el confirmatorio usó el mismo número de remuestreos (B=4000 en el estudio real;
    el smoke usa menos, y ahí sólo coincide el punto)."""
    first = rows[0] if rows else {}
    out = {}

    def cmp(mine, conf):
        return {"punto": abs(mine["diff"] - conf["diff"]) <= tol,
                "IC95": all(abs(mine[k] - conf[k]) <= tol for k in ("ci95_lo", "ci95_hi"))}
    if c1_conf and first.get("C1"):
        out["C1"] = cmp(first["C1"], c1_conf)
    for r in c2_conf or []:
        mine = (first.get(r["par"]) or {}).get("C2")
        if mine:
            out[f"C2 {r['par']}"] = cmp(mine, r["C2"])
    return out


def _control_txt(ctrl: dict | None) -> str:
    if not ctrl:
        return "no hay resultados confirmatorios con qué comparar"
    si = lambda x: "sí" if x else "NO"
    return "; ".join(f"{k}: punto {si(v['punto'])}, IC95 {si(v['IC95'])}" for k, v in ctrl.items())


def to_markdown(res: dict) -> str:
    rows = res.get("filas", [])
    pares = [f"{a} − {b}" for a, b in C2_PAIRS]
    nc = res.get("brazo_C1")
    lines = [f"# Sensibilidad del panel T-gen — {res.get('fase')} ({res.get('split')}) — {res.get('creado')}", "",
             res.get("motivo", ""), "",
             f"Panel {res.get('panel_version')} · macro {res.get('macro')} · corridas por brazo {res.get('corridas')}",
             "", "Control (la fila del macro congelado reproduce el confirmatorio): " + _control_txt(res.get("control")), "",
             "Diferencias en pp, IC95 por bootstrap pareado por documento (sin Holm: es descriptivo).", "",
             f"| Subconjunto | ítems | C1: {nc} − infonce | lectura | "
             + " | ".join(f"C2 {p} (Δ T-gen)" for p in pares) + " |",
             "|---|---|---|---|" + "---|" * len(pares)]
    for r in rows:
        c2s = []
        for p in pares:
            d = r.get(p)
            c2s.append(f"{_pp(d['C2'])} ({100 * d['Δ T-gen']['diff']:+.1f})" if d else "—")
        lines.append(f"| {r['subconjunto']} | {r['n_items']} | {_pp(r.get('C1'))} | "
                     f"{(r.get('C1') or {}).get('lectura', '—')} | " + " | ".join(c2s) + " |")
    return "\n".join(lines) + "\n"


def tabla(res: dict):
    """La misma tabla como DataFrame (pp)."""
    import pandas as pd
    out = []
    for r in res.get("filas", []):
        row = {"subconjunto": r["subconjunto"], "ítems": r["n_items"]}
        c = r.get("C1")
        if c:
            row.update({"C1 Δ": round(100 * c["diff"], 2), "C1 IC95": f"[{100 * c['ci95_lo']:.2f}, {100 * c['ci95_hi']:.2f}]",
                        "C1 lectura": c["lectura"]})
        for a, b in C2_PAIRS:
            d = r.get(f"{a} − {b}")
            if d:
                row[f"C2 {a}"] = round(100 * d["C2"]["diff"], 2)
                row[f"Δ T-gen {a}"] = round(100 * d["Δ T-gen"]["diff"], 2)
        out.append(row)
    return pd.DataFrame(out)


def panel_sensitivity(study, phase: str = "F3", split: str = "confirm", B: int = 4000, log=print,
                      meta: dict | None = None) -> dict:
    """Lee el panel congelado del split, las corridas de la fase y los ítems; calcula y guarda la sensibilidad.
    meta (p. ej. el hash del código y del script) se guarda junto con el resultado."""
    from clm_losses.data import EvalSet
    from clm_losses.generators import _intersect, load_panel
    from clm_losses.pipeline import freeze_path, iter_run_dirs
    from clm_losses.utils import load_json, makedirs, now_iso, save_json
    panel = load_panel(study, split) or {}
    keys = list(panel.get("macro") or [])
    if not keys:
        log(f"[sensibilidad] no hay macro de T-gen en {split}: no se hace nada")
        return {}
    nc = (load_json(freeze_path(study)) or {}).get("C1_best_noncontrastive")
    arms = sorted({nc, "infonce", *[x for p in C2_PAIRS for x in p]} - {None})
    runs = iter_run_dirs(study, phase)
    groups = {a: [r for r in runs if f"/{phase}/{a}/" in r.replace(os.sep, "/")] for a in arms}
    doc_ids = EvalSet(study.paths.d("data", "eval", split)).doc_ids
    macro_items = np.asarray(panel["macro_items"], np.int64)

    def items_of(ks):
        if sorted(ks) == sorted(keys):
            return macro_items                     # exactamente los ítems del confirmatorio
        return _intersect(study, ks, split)

    rows = compute(groups, keys, items_of, doc_ids, nc, study.rope_pp / 100,
                   lambda rd, k, items: _at(rd, f"tgen_{k}", items), _tgem_at, B=B)
    avv = load_json(study.paths.d("results", phase, "all_vs_all", "all_vs_all.json")) or {}
    c1_conf = ((avv.get("C1") or {}).get("por_metrica") or {}).get("tgenmacro")
    c2_conf = (load_json(study.paths.d("results", phase, "C2", "C2.json")) or {}).get("pares")
    res = {"motivo": MOTIVO, "fase": phase, "split": split, "panel_version": panel.get("version"), "macro": keys,
           "brazo_C1": nc, "corridas": {a: len(v) for a, v in groups.items()}, "rope_pp": study.rope_pp, "B": B,
           "filas": rows, "control": control(rows, c1_conf, c2_conf), "creado": now_iso(), **(meta or {})}
    d = makedirs(study.paths.d("results", phase, "sensibilidad_panel"))
    save_json(res, os.path.join(d, "sensibilidad.json"))
    with open(os.path.join(d, "SENSIBILIDAD.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(res))
    log(f"[sensibilidad] {len(rows)} subconjuntos · control {res['control']} · guardado en {d}")
    return res
