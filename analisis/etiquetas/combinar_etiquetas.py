"""Combina las etiquetas a mano de varios anotadores (bloque 1, notebook pre-envío) — regla fijada en DP-10.

Cada archivo es un anotador (ETIQUETAR_<Nombre>.xlsx, o el libro completo ETIQUETADAS.xlsx con sólo su parte marcada).
Por candidato (id, n):
  * 1 etiqueta            -> esa ("una_sola");
  * >= 2 y todas iguales  -> esa ("unanime");
  * >= 3 con mayoría      -> la mayoría ("mayoria");
  * si no                 -> la decide el investigador sin ver al juez ("adjudicar" -> "adjudicado", con --adjudicadas).
Acuerdo entre anotadores (descriptivo; techo para el acuerdo juez–humano): % de acuerdo y kappa de Fleiss sobre los
candidatos con exactamente 2 etiquetas, kappa de Cohen por par de anotadores y, por pregunta, acuerdo en "tiene al menos
un falso negativo" (preguntas con sus 10 candidatos marcados por los dos).
Salidas en --out: REPORTE_ACUERDO.md, acuerdo.json, etiquetas_individuales.csv y, según el caso, ADJUDICAR.xlsx
(desacuerdos pendientes) o ETIQUETADAS.xlsx (final, formato que lee preenvio.read_hand_labels).
Uso: python3 -I combinar_etiquetas.py --csv ETIQUETAR_200_items.csv --out salida/ [--adjudicadas ADJUDICADAS.xlsx] archivos...
"""
import argparse
import collections
import itertools
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from etiquetas_xlsx import COLS, QC, asignacion, build_workbook, ids_all, read_csv, textos_adjudicar  # noqa: E402


def read_sheet(path):
    """La pestaña Etiquetar si existe; si no, la primera (CSV también)."""
    if path.lower().endswith((".xlsx", ".xlsm")):
        sheets = pd.ExcelFile(path).sheet_names
        return pd.read_excel(path, sheet_name="Etiquetar" if "Etiquetar" in sheets else 0, dtype={"id": str})
    return pd.read_csv(path, dtype={"id": str}, sep=None, engine="python", encoding="utf-8-sig")


def _norm(x) -> str:
    s = " ".join(str(x).split()).lower()
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():      # Excel convirtió "1993" en 1993.0
        s = s[:-2]
    return s.rstrip(".")


def mismo_candidato(en_archivo, canon) -> str | None:
    """'igual' si el texto coincide; 'editado' si el anotador sólo recortó el candidato y lo que quedó es el final del
    original (p. ej. borró la pregunta repetida y dejó el año, que Excel guardó como número); None si es otro texto."""
    if str(en_archivo) == canon:
        return "igual"
    a, b = _norm(en_archivo), _norm(canon)
    if a == b:
        return "igual"
    if a and b.endswith(a) and (len(a) == len(b) or not b[-len(a) - 1].isalnum()):
        return "editado"
    return None


def read_labels(path, canon):
    """({(id, n): 0/1}, info) de un libro marcado. Comprueba que cada fila marcada sea la del CSV original."""
    df = read_sheet(path)
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    col = next((c for c in df.columns if c.lower().startswith("es_correcta")), None)
    if col is None or "id" not in df.columns or "n" not in df.columns:
        raise SystemExit(f"{path}: faltan las columnas id, n o es_correcta(0/1)")
    m = pd.to_numeric(df[col], errors="coerce")
    raros = df[df[col].notna() & ~m.isin([0, 1])]
    out, malos, editados = {}, [], []
    cands = df["candidato"] if "candidato" in df.columns else [None] * len(df)
    for iid, n, lab, cand in zip(df["id"].astype(str).str.strip(), df["n"], m, cands):
        if lab not in (0, 1) or pd.isna(n):
            continue
        key = (iid, int(n))
        if key not in canon:
            malos.append(f"{iid}/{n} no existe")
            continue
        if cand is not None and not pd.isna(cand):
            st = mismo_candidato(cand, canon[key]["candidato"])
            if st is None:
                malos.append(f"{iid}/{n}: el candidato no coincide con el CSV")
                continue
            if st == "editado":
                editados.append(f"{iid}/{int(n)}")
        out[key] = int(lab)
    if malos:
        raise SystemExit(f"{path}: {len(malos)} filas no coinciden con el CSV original, p. ej. {malos[:3]}")
    notas = df["notas"].dropna().astype(str).str.strip() if "notas" in df.columns else pd.Series([], dtype=str)
    notas = notas[notas != ""]
    ids_archivo = set(df["id"].dropna().astype(str).str.strip())
    info = {"valores_raros": [(str(r["id"]), r["n"], str(r[col])) for _, r in raros.iterrows()],
            "notas": int(len(notas)), "dudas": int(notas.str.lower().str.contains("duda").sum()),
            "ids_archivo": ids_archivo, "candidatos_editados": editados}
    return out, info


def cohen(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def fleiss(counts):
    """counts: lista de (n0, n1) con el mismo número de etiquetas m por candidato."""
    N = len(counts)
    if N == 0:
        return None
    m = sum(counts[0])
    P = [(c0 * c0 + c1 * c1 - m) / (m * (m - 1)) for c0, c1 in counts]
    pbar = sum(P) / N
    p1 = sum(c1 for _, c1 in counts) / (N * m)
    pe = p1 * p1 + (1 - p1) * (1 - p1)
    return None if pe >= 1 else (pbar - pe) / (1 - pe)


def anotador(ids_archivo, asg, i):
    for a, ids in asg.items():
        if set(ids) == ids_archivo:
            return f"alumno{a}"
    return f"archivo{i + 1}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adjudicadas")
    ap.add_argument("--bandas", action="store_true",
                    help="sin adjudicar (DP-11): ETIQUETADAS.xlsx = estricta (1 sólo si todos marcaron 1) y "
                         "ETIQUETADAS_amplia.xlsx = amplia (1 si al menos uno marcó 1)")
    ap.add_argument("archivos", nargs="+")
    a = ap.parse_args(argv)
    df = read_csv(a.csv)
    canon = {(r["id"], int(r["n"])): r for r in df.to_dict("records")}
    asg = asignacion(ids_all(df))
    os.makedirs(a.out, exist_ok=True)

    labs, info = {}, {}
    for i, p in enumerate(a.archivos):
        lab, extra = read_labels(p, canon)
        name = anotador(extra["ids_archivo"], asg, i)
        while name in labs:
            name += "b"
        labs[name] = lab
        q_marked = collections.Counter(k[0] for k in lab)
        esperadas = len(extra["ids_archivo"] & set(ids_all(df)))
        info[name] = {"archivo": os.path.basename(p), "preguntas_en_archivo": esperadas, "filas_marcadas": len(lab),
                      "preguntas_completas": sum(1 for v in q_marked.values() if v == 10),
                      "preguntas_a_medias": sorted(k for k, v in q_marked.items() if 0 < v < 10),
                      "preguntas_sin_marcar": sorted((extra["ids_archivo"] & set(ids_all(df))) - set(q_marked)),
                      "valores_raros": extra["valores_raros"], "notas": extra["notas"], "dudas": extra["dudas"],
                      "candidatos_editados": extra["candidatos_editados"],
                      "proporcion_1": (sum(lab.values()) / len(lab)) if lab else None,
                      "preguntas_con_algun_1": sum(1 for q in q_marked if any(lab.get((q, n)) for n in range(1, 11)))}

    per = collections.defaultdict(dict)                      # (id, n) -> {anotador: etiqueta}
    for name, lab in labs.items():
        for k, v in lab.items():
            per[k][name] = v

    # acuerdo entre anotadores
    two = [k for k in canon if len(per.get(k, {})) == 2]
    pares = [tuple(per[k].values()) for k in two]
    acuerdo = {
        "candidatos_con_2_etiquetas": len(two),
        "pct_acuerdo": (sum(x == y for x, y in pares) / len(pares)) if pares else None,
        "ambos_1": sum(1 for x, y in pares if x == y == 1), "ambos_0": sum(1 for x, y in pares if x == y == 0),
        "desacuerdos": sum(1 for x, y in pares if x != y),
        "kappa_fleiss": fleiss([(2 - x - y, x + y) for x, y in pares]),
        "prevalencia_1": (sum(x + y for x, y in pares) / (2 * len(pares))) if pares else None,
    }
    por_par = {}
    for u, v in itertools.combinations(sorted(labs), 2):
        shared = sorted(set(labs[u]) & set(labs[v]))
        if shared:
            au, av = [labs[u][k] for k in shared], [labs[v][k] for k in shared]
            por_par[f"{u}–{v}"] = {"n": len(shared), "pct_acuerdo": sum(x == y for x, y in zip(au, av)) / len(shared),
                                   "kappa_cohen": cohen(au, av), "solo_1_" + u: sum(1 for x, y in zip(au, av) if x > y),
                                   "solo_1_" + v: sum(1 for x, y in zip(au, av) if y > x)}
    # por pregunta: "tiene al menos un falso negativo", con los 10 candidatos marcados por los dos
    qa, qb = [], []
    for iid in ids_all(df):
        names = [nm for nm, lab in labs.items() if all((iid, n) in lab for n in range(1, 11))]
        if len(names) == 2:
            u, v = names
            qa.append(int(any(labs[u][(iid, n)] for n in range(1, 11))))
            qb.append(int(any(labs[v][(iid, n)] for n in range(1, 11))))
    acuerdo["preguntas_con_2_completas"] = len(qa)
    acuerdo["pregunta_pct_acuerdo"] = (sum(x == y for x, y in zip(qa, qb)) / len(qa)) if qa else None
    acuerdo["pregunta_kappa_cohen"] = cohen(qa, qb)

    # combinación
    adj = {}
    if a.adjudicadas:
        adj, _ = read_labels(a.adjudicadas, canon)
    filas, pendientes = [], []
    cuenta = collections.Counter()
    for r in df.to_dict("records"):
        k = (r["id"], int(r["n"]))
        vals = list(per.get(k, {}).values())
        c = collections.Counter(vals)
        if not vals:
            lab, origen = None, "sin_etiqueta"
        elif len(vals) == 1:
            lab, origen = vals[0], "una_sola"
        elif len(c) == 1:
            lab, origen = vals[0], "unanime"
        elif len(vals) >= 3 and c.most_common(1)[0][1] > len(vals) / 2:
            lab, origen = c.most_common(1)[0][0], "mayoria"
        elif k in adj:
            lab, origen = adj[k], "adjudicado"
        else:
            lab, origen = None, "adjudicar"
            pendientes.append(r)
        cuenta[origen] += 1
        filas.append({**r, QC: lab, "origen": origen, "n_etiquetas": len(vals)})

    long = [{"anotador": nm, "id": k[0], "n": k[1], "etiqueta": v} for nm, lab in labs.items() for k, v in sorted(lab.items())]
    pd.DataFrame(long).to_csv(os.path.join(a.out, "etiquetas_individuales.csv"), index=False)
    res = {"anotadores": info, "acuerdo": acuerdo, "por_par": por_par, "combinacion": dict(cuenta),
           "pendientes_de_adjudicar": len(pendientes)}
    with open(os.path.join(a.out, "acuerdo.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=list)

    final = None
    if a.bandas:
        bandas = {}
        for nombre, regla in (("estricta", all), ("amplia", any)):
            rows_b, q = [], collections.defaultdict(list)
            for r in df.to_dict("records"):
                k = (r["id"], int(r["n"]))
                vals = list(per.get(k, {}).values())
                lab = int(regla(vals)) if vals else None
                origen = ("sin_etiqueta" if not vals else "una_sola" if len(vals) == 1 else
                          "unanime" if len(set(vals)) == 1 else f"desacuerdo_{nombre}")
                rows_b.append({**r, QC: lab, "origen": origen, "n_etiquetas": len(vals)})
                if lab is not None:
                    q[r["id"]].append(lab)
            fn = "ETIQUETADAS.xlsx" if nombre == "estricta" else "ETIQUETADAS_amplia.xlsx"
            with pd.ExcelWriter(os.path.join(a.out, fn)) as w:
                pd.DataFrame(rows_b)[COLS + ["origen", "n_etiquetas"]].to_excel(w, sheet_name="Etiquetar", index=False)
            full = {i: v for i, v in q.items() if len(v) == 10}
            kq = sum(1 for v in full.values() if any(v))
            bandas[nombre] = {"archivo": fn, "candidatos_1": sum(x[QC] or 0 for x in rows_b),
                              "preguntas_completas": len(full), "preguntas_con_algun_1": kq,
                              "tasa_pregunta": kq / len(full) if full else None}
        res["bandas"] = bandas
        with open(os.path.join(a.out, "acuerdo.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=list)
        pendientes = []
        final = os.path.join(a.out, "ETIQUETADAS.xlsx")
    elif pendientes:
        build_workbook(pd.DataFrame(pendientes)[COLS].assign(**{QC: "", "notas": ""}),
                       os.path.join(a.out, "ADJUDICAR.xlsx"), textos_adjudicar())
    else:
        final = os.path.join(a.out, "ETIQUETADAS.xlsx")
        out = pd.DataFrame(filas)[COLS + ["origen", "n_etiquetas"]]
        with pd.ExcelWriter(final) as w:
            out.to_excel(w, sheet_name="Etiquetar", index=False)

    def pct(x):
        return "—" if x is None else f"{100 * x:.1f} %"

    def kap(x):
        return "—" if x is None else f"{x:.3f}"
    L = ["# Etiquetas a mano: acuerdo entre anotadores y combinación", "",
         "Regla (DP-10): cada pregunta la marcan dos anotadores por separado; si coinciden, vale su etiqueta; si no, "
         "decide el investigador sin ver al juez. El acuerdo entre anotadores es descriptivo: es el techo con el que se "
         "compara el acuerdo juez–humano del bloque 1.", "", "## Anotadores", "",
         "| Anotador | Archivo | Preguntas | Filas marcadas | Completas | A medias | Sin marcar | Valores raros | "
         "Proporción de 1 | Preguntas con algún 1 | Notas («duda») | Candidatos editados |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, v in info.items():
        L.append(f"| {nm} | {v['archivo']} | {v['preguntas_en_archivo']} | {v['filas_marcadas']} | "
                 f"{v['preguntas_completas']} | {len(v['preguntas_a_medias'])} | {len(v['preguntas_sin_marcar'])} | "
                 f"{len(v['valores_raros'])} | {pct(v['proporcion_1'])} | {v['preguntas_con_algun_1']} | "
                 f"{v['notas']} ({v['dudas']}) | {len(v['candidatos_editados'])} |")
    A = acuerdo
    L += ["", "## Acuerdo entre anotadores", "",
          f"- Candidatos con 2 etiquetas: {A['candidatos_con_2_etiquetas']:,}; acuerdo {pct(A['pct_acuerdo'])}; "
          f"kappa de Fleiss {kap(A['kappa_fleiss'])}.",
          f"- Ambos 1: {A['ambos_1']}; ambos 0: {A['ambos_0']}; desacuerdos: {A['desacuerdos']}; "
          f"proporción de 1: {pct(A['prevalencia_1'])}. Con pocos 1, el kappa puede salir bajo aunque el acuerdo "
          "sea alto: se reportan los dos.",
          f"- Por pregunta (tiene al menos un falso negativo), {A['preguntas_con_2_completas']} preguntas con los 10 "
          f"candidatos marcados por los dos: acuerdo {pct(A['pregunta_pct_acuerdo'])}; kappa de Cohen "
          f"{kap(A['pregunta_kappa_cohen'])}.", "", "| Par | Candidatos en común | Acuerdo | Kappa de Cohen | 1 sólo del primero | 1 sólo del segundo |",
          "|---|---|---|---|---|---|"]
    for k, v in por_par.items():
        u, w = k.split("–")
        L.append(f"| {k} | {v['n']} | {pct(v['pct_acuerdo'])} | {kap(v['kappa_cohen'])} | {v['solo_1_' + u]} | "
                 f"{v['solo_1_' + w]} |")
    L += ["", "## Combinación", "", "| Origen | Candidatos |", "|---|---|"]
    for k in ("unanime", "mayoria", "adjudicado", "una_sola", "adjudicar", "sin_etiqueta"):
        if cuenta.get(k):
            L.append(f"| {'desacuerdo (sin adjudicar)' if (k == 'adjudicar' and a.bandas) else k} | {cuenta[k]:,} |")
    L.append("")
    if a.bandas:
        L += ["## Sin adjudicar: referencia estricta y amplia (DP-11)", "",
              "| Referencia | Archivo | Candidatos con 1 | Preguntas con al menos un 1 (de las completas) |", "|---|---|---|---|"]
        for nombre, b in res["bandas"].items():
            L.append(f"| {nombre} | {b['archivo']} | {b['candidatos_1']:,} | {b['preguntas_con_algun_1']} de "
                     f"{b['preguntas_completas']} ({pct(b['tasa_pregunta'])}) |")
        L += ["", "**Listo:** `ETIQUETADAS.xlsx` (estricta) va a Drive, a la carpeta de etiquetas; después, Ejecutar todo. "
              "La amplia se compara aparte con las etiquetas del juez."]
    elif pendientes:
        L.append(f"**Faltan {len(pendientes)} desacuerdos por decidir:** están en `ADJUDICAR.xlsx`.")
    else:
        L.append("**Listo:** `ETIQUETADAS.xlsx` va a Drive, a la carpeta de etiquetas; después, Ejecutar todo.")
    with open(os.path.join(a.out, "REPORTE_ACUERDO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return {"res": res, "final": final, "pendientes": len(pendientes)}


if __name__ == "__main__":
    main()
