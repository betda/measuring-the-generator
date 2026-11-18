"""DP-11: el juez frente a las dos referencias humanas (estricta y amplia), en las filas donde los dos anotadores
coinciden, y en los desacuerdos. Usa las funciones del notebook (preenvio.agreement, wilson, cohen_kappa).
Uso: python3 -I juez_vs_humanos.py juez_raw.jsonl clave.json etiquetas_individuales.csv salida.json"""
import collections, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import pandas as pd
from clm_losses import preenvio as pre

raw_p, clave_p, ind_p, out_p = sys.argv[1:5]
SETS = ["tgem", "tmine", "tgen_aya32b", "tgen_glm4_32b", "tgen_gptoss20b", "tgen_mistral_small24b", "tgen_phi4_14b"]
judge = {}
for line in open(raw_p, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("labels") is not None and r["i"] not in judge:
        judge[r["i"]] = [int(x) for x in r["labels"]]
clave = json.load(open(clave_p, encoding="utf-8"))["clave"]
ind = pd.read_csv(ind_p, dtype={"id": str})
per = ind.groupby(["id", "n"])["etiqueta"].agg(list).to_dict()
strict = {(clave[i], int(n)): int(all(v)) for (i, n), v in per.items()}
lenient = {(clave[i], int(n)): int(any(v)) for (i, n), v in per.items()}
res = {"juez_items_con_etiqueta": len(judge)}
for nombre, hand in (("estricta", strict), ("amplia", lenient)):
    res[nombre] = pre.agreement(judge, hand, SETS)


def metrics(pairs):
    j = np.array([a for a, _ in pairs], int); h = np.array([b for _, b in pairs], int)
    tp, fp = int(((j == 1) & (h == 1)).sum()), int(((j == 1) & (h == 0)).sum())
    fn, tn = int(((j == 0) & (h == 1)).sum()), int(((j == 0) & (h == 0)).sum())
    return {"n": len(pairs), "humanos_1": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensibilidad": tp / (tp + fn) if tp + fn else None, "especificidad": tn / (tn + fp) if tn + fp else None,
            "kappa": pre.cohen_kappa(j, h) if pairs else None}


grupos = {"unanimes": collections.defaultdict(list), "una_sola": collections.defaultdict(list),
          "desacuerdo": collections.defaultdict(list)}
for (i, n), v in per.items():
    key = clave[i]
    s = pre.set_of(key)
    if key not in judge:
        continue
    jv = judge[key][int(n) - 1]
    if len(v) == 1:
        grupos["una_sola"][s].append((jv, v[0]))
    elif len(set(v)) == 1:
        grupos["unanimes"][s].append((jv, v[0]))
    else:
        grupos["desacuerdo"][s].append((jv, None))
uni = {s: metrics(p) for s, p in grupos["unanimes"].items()}
uni["total"] = metrics([x for p in grupos["unanimes"].values() for x in p])
res["unanimes"] = uni
# Definición registrada en DP-11: coincidentes + las filas con una sola etiqueta (alumno3 dejó 4 preguntas sin marcar).
ext = {s: metrics(grupos["unanimes"][s] + grupos["una_sola"][s]) for s in set(grupos["unanimes"]) | set(grupos["una_sola"])}
ext["total"] = metrics([x for g in ("unanimes", "una_sola") for p in grupos[g].values() for x in p])
res["unanimes_mas_una_sola"] = ext
dis ={s: {"n": len(p), "juez_1": sum(a for a, _ in p)} for s, p in grupos["desacuerdo"].items()}
dis["total"] = {"n": sum(d["n"] for d in dis.values()), "juez_1": sum(d["juez_1"] for d in dis.values())}
res["desacuerdos"] = dis
json.dump(res, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


def f(x, k=2):
    return "—" if x is None else f"{x:.{k}f}"


print("juez: ítems con etiqueta", len(judge))
for nombre in ("estricta", "amplia"):
    print(f"\n== referencia {nombre}")
    print("set | negativos | FN humanos | sens | esp | kappa | tasa ítem humana [IC95]")
    for s in SETS + ["total"]:
        r = res[nombre][s]
        tasa = "" if s == "total" else f"{100 * r['tasa_item_humano']:.1f} [{100 * r['ic_item_humano'][0]:.1f}, {100 * r['ic_item_humano'][1]:.1f}] (n={r['items_completos']})"
        print(f"{s} | {r['n_negativos']} | {r['tp'] + r['fn']} | {f(r['sensibilidad'])} | {f(r['especificidad'])} | {f(r['kappa'])} | {tasa}")
print("\n== filas donde los dos anotadores coinciden")
print("set | n | humanos 1 | tp | fp | fn | sens | esp | kappa")
for s in SETS + ["total"]:
    r = uni.get(s)
    if r:
        print(f"{s} | {r['n']} | {r['humanos_1']} | {r['tp']} | {r['fp']} | {r['fn']} | {f(r['sensibilidad'])} | {f(r['especificidad'], 3)} | {f(r['kappa'])}")
print("\n== coincidentes + una sola etiqueta (definición de DP-11)")
print("set | n | humanos 1 | tp | fp | fn | sens | esp | kappa")
for s in SETS + ["total"]:
    r = ext.get(s)
    if r:
        print(f"{s} | {r['n']} | {r['humanos_1']} | {r['tp']} | {r['fp']} | {r['fn']} | {f(r['sensibilidad'])} | {f(r['especificidad'], 3)} | {f(r['kappa'])}")
print("\n== desacuerdos (uno marcó 1 y el otro 0): cuántas veces el juez dice 1")
for s in SETS + ["total"]:
    r = dis.get(s)
    if r:
        print(f"{s} | {r['juez_1']} de {r['n']} ({100 * r['juez_1'] / r['n']:.0f} %)")
