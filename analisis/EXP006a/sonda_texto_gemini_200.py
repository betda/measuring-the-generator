"""EXP-006a: calibración de la sonda de texto con los HN de Gemini (200 ítems de QC_negativos_gemini.csv, en Drive) y
comparación a igual tamaño (200 ítems, 5 submuestras) con tres generadores. Resultado del 2026-10-02: Gemini 0.52
(el ciego con embeddings da 0.58 en todo T-gem)."""
import csv, json, sys, collections, numpy as np
sys.path.insert(0, ".")
from sonda_texto import probe, sets_for, K_KEEP
rows = list(csv.DictReader(open("QC_negativos_gemini.csv", encoding="utf-8")))
by = collections.defaultdict(list); gold = {}
for r in rows:
    by[int(r["i"])].append((int(r["neg_idx"]), r["negative"])); gold[int(r["i"])] = r["gold"]
gem = [(i, [gold[i]] + [n for _, n in sorted(by[i])][:K_KEEP]) for i in sorted(by) if len(by[i]) >= K_KEEP]
print("T-gem (Gemini, 200 ítems de QC):", probe(gem))
rs = np.random.default_rng(0)
for k in ("mistral7b", "qwen25_7b", "llama31_8b"):
    for b in ("v1", "v2"):
        S = sets_for(k, b); acc = []
        for rep in range(5):
            sub = [S[j] for j in rs.choice(len(S), 200, replace=False)]
            acc.append(probe(sub)["texto"])
        print(f"{k} {b} (200 ítems, 5 submuestras): texto {np.mean(acc):.3f} ± {np.std(acc):.3f}")
