"""EXP-006a (2026-10-02). Sonda ciega de TEXTO: un proxy barato del "ciego propio" del panel T-gen. Sólo ve las 11
respuestas candidatas (oro + 10 negativos), nunca la pregunta: n-gramas de caracteres (TF-IDF) + rasgos de largo y
formato, regresión logística, validación cruzada de 5 pliegues por ítem. Azar = 1/11.

Compara la construcción v1 (primeros 10 de la unión de intentos, SIN el filtro de coseno, que no se puede reproducir
sin los embeddings de Qwen3-8B) con la construcción S de v0.3.2 (reglas de texto, clm_losses.generators).

Uso (en una carpeta con los archivos del panel bajados de Drive, data/eval/tgen/test/):
    items_text.jsonl  y  <generador>/raw.jsonl  para cada generador
    python sonda_texto.py mistral7b qwen25_7b ...      # salida: textblind_results.json (o $OUT)
Resultados del 2026-10-02 (15 generadores; gpt-oss no: 22 MB, sobre el límite del conector de Drive):
resultados_sonda_texto.json. Requiere scikit-learn y scipy."""
import json, os, sys, collections, re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from clm_losses.generators import item_kind, build_candidates, K_KEEP, n_words, _norm


def _merge_negatives(recs, gold):          # construcción v1 (v0.3.1), copiada para la comparación
    g = _norm(gold)
    seen, out = set(), []
    for r in sorted(recs, key=lambda r: r.get("attempt", 0)):
        for s in r.get("negatives", []):
            low = _norm(s)
            if low and low != g and low not in seen:
                seen.add(low)
                out.append(s)
    return out

texts = {r["i"]: r for r in map(json.loads, open("items_text.jsonl"))}
kinds = {i: item_kind(r["question"], r["answer"]) for i, r in texts.items()}

def dense(c):
    s = c.strip()
    return [np.log1p(len(s)), n_words(s), s.endswith("."), s[:1].isupper(), any(ch.isdigit() for ch in s),
            "(" in s, s.endswith(")"), "," in s, sum(ch.isupper() for ch in s) / max(1, len(s))]

def sets_for(k, build):
    recs = [json.loads(x) for x in open(f"{k}/raw.jsonl")]
    by = collections.defaultdict(list)
    for r in recs: by[r["i"]].append(r)
    out = []
    for i in sorted(by):
        if i not in texts: continue
        g = texts[i]["answer"]
        if build == "v1":
            negs = _merge_negatives(by[i], g)[:K_KEEP]          # todos los tipos de ítem, texto tal cual
        else:
            if kinds[i] != "ok": continue
            negs, _ = build_candidates(g, by[i], K_KEEP)
        if len(negs) >= K_KEEP:
            out.append((i, [g] + negs[:K_KEEP]))
    return out

def probe(sets, seed=0):
    cands = [c for _, cs in sets for c in cs]
    grp = np.repeat(np.arange(len(sets)), 1 + K_KEEP)
    y = np.tile([1] + [0] * K_KEEP, len(sets))
    D = np.asarray([dense(c) for c in cands], np.float32)
    # rasgos relativos dentro del ítem (el ciego ve los 11 candidatos juntos)
    D3 = D.reshape(len(sets), 1 + K_KEEP, -1)
    rel = (D3[:, :, :2] - D3[:, :, :2].mean(1, keepdims=True)) / (D3[:, :, :2].std(1, keepdims=True) + 1e-6)
    D = np.concatenate([D, rel.reshape(len(cands), -1)], 1)
    D = (D - D.mean(0)) / (D.std(0) + 1e-6)
    hits = np.zeros(len(sets), bool)
    for tr, te in GroupKFold(5).split(D, y, grp):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=3, sublinear_tf=True)
        Xtr = sparse.hstack([vec.fit_transform([cands[j] for j in tr]), sparse.csr_matrix(D[tr])]).tocsr()
        Xte = sparse.hstack([vec.transform([cands[j] for j in te]), sparse.csr_matrix(D[te])]).tocsr()
        m = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, y[tr])
        s = m.decision_function(Xte).reshape(-1, 1 + K_KEEP)
        hits[np.unique(grp[te])] = s.argmax(1) == 0
    L = np.asarray([[len(c) for c in cs] for _, cs in sets], float)
    longest = np.mean([(row[0] == row.max()) / (row == row.max()).sum() for row in L])
    per = np.asarray([[c.strip().endswith(".") for c in cs] for _, cs in sets], float)
    period = np.mean([(row[0] == 1) / max(1, row.sum()) if row.sum() else 1 / (1 + K_KEEP) for row in per])
    return {"n": len(sets), "texto": float(hits.mean()), "mas_largo": float(longest), "punto": float(period)}

if __name__ == "__main__":
    gens = sys.argv[1:] or sorted(d for d in os.listdir(".") if os.path.isdir(d))
    res = {}
    for k in gens:
        res[k] = {b: probe(sets_for(k, b)) for b in ("v1", "v2")}
        r = res[k]
        print(f"{k:14s} v1: n={r['v1']['n']:4d} texto={r['v1']['texto']:.3f} largo={r['v1']['mas_largo']:.3f} "
              f"punto={r['v1']['punto']:.3f} | v2: n={r['v2']['n']:4d} texto={r['v2']['texto']:.3f} "
              f"largo={r['v2']['mas_largo']:.3f} punto={r['v2']['punto']:.3f}", flush=True)
    json.dump(res, open(os.environ.get("OUT", "textblind_results.json"), "w"), indent=1)
