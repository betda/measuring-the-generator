"""Tokens de Nemotron-DQA (jackyk02/nemotron-cc-v2.1-hq-dqa-qwen3-tokens): textos, longitudes,
heurística de longitud (F0.6) y negativos de un segundo generador (T-gen2, requiere decisión P3).

Layout verificado: qa_pairs/part_00000{0..5}.parquet (2.9 GB) y neg_pairs/part_00000{0..5}.parquet
(7.6 GB). Columnas qa_pairs: doc_id, pair_idx, question_ids, answer_ids, n_question_tokens,
n_answer_tokens. neg_pairs: doc_id, pair_idx, neg_idx, neg_ids, n_neg_tokens. Token ids de Qwen3
sin tokens especiales. El orden dentro de cada parte NO está normalizado: se une por claves.
"""
from __future__ import annotations

import os

import numpy as np

from .config import ENCODER_MODEL, TOKENS_REPO
from .data import HFDownloader, slug
from .utils import append_jsonl, atomic_np_save, load_json, read_jsonl, save_json

QA_PARTS = [f"qa_pairs/part_{i:06d}.parquet" for i in range(6)]
NEG_PARTS = [f"neg_pairs/part_{i:06d}.parquet" for i in range(6)]


def tokens_dir(paths) -> str:
    return paths.c("hf", slug(TOKENS_REPO))


def fetch(paths, parts, downloader: HFDownloader | None = None) -> list[str]:
    dl = downloader or HFDownloader()
    return [dl.fetch(TOKENS_REPO, p, tokens_dir(paths)) for p in parts]


def lookup(paths, docs, which: str = "qa", columns=None, downloader=None):
    """Filas del dataset de tokens para un conjunto de doc_ids (pyarrow, filtro por columna)."""
    import pyarrow.parquet as pq
    parts = fetch(paths, QA_PARTS if which == "qa" else NEG_PARTS, downloader)
    docs = sorted(set(docs))
    rows = []
    for p in parts:
        t = pq.read_table(p, columns=columns, filters=[("doc_id", "in", docs)])
        rows.extend(t.to_pylist())
    return rows


def eval_lengths(paths, evalset, out_path: str, downloader=None, log=print) -> dict:
    """Longitud (tokens) del oro y de sus negativos para cada ítem de un EvalSet. Se guarda en Drive."""
    if os.path.exists(out_path):
        z = np.load(out_path)
        return {"gold": z["gold"], "neg": z["neg"]}
    docs = set(evalset.doc_ids.tolist())
    log(f"[texts] longitudes para {evalset.n} ítems ({len(docs)} documentos)")
    qa = lookup(paths, docs, "qa", ["doc_id", "pair_idx", "n_answer_tokens"], downloader)
    gold_map = {(r["doc_id"], r["pair_idx"]): r["n_answer_tokens"] for r in qa}
    neg = lookup(paths, docs, "neg", ["doc_id", "pair_idx", "neg_idx", "n_neg_tokens"], downloader)
    neg_map: dict = {}
    for r in neg:
        neg_map.setdefault((r["doc_id"], r["pair_idx"]), {})[r["neg_idx"]] = r["n_neg_tokens"]
    k = evalset.k
    gold = np.full(evalset.n, -1, np.int32)
    negs = np.full((evalset.n, k), -1, np.int32)
    for i, (d, p) in enumerate(zip(evalset.doc_ids, evalset.pair_idx)):
        gold[i] = gold_map.get((d, int(p)), -1)
        m = neg_map.get((d, int(p)), {})
        vals = [m[j] for j in sorted(m)][:k]
        negs[i, : len(vals)] = vals
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, gold=gold, neg=negs)
    os.replace(tmp, out_path)
    return {"gold": gold, "neg": negs}


def length_heuristic(gold: np.ndarray, neg: np.ndarray, rule: str) -> np.ndarray:
    """Acierto esperado (desempate aleatorio) de elegir el candidato más largo/corto."""
    valid = (gold >= 0) & (neg >= 0).all(1)
    acc = np.full(len(gold), np.nan)
    g, n = gold[valid].astype(float), neg[valid].astype(float)
    if rule == "shortest":
        g, n = -g, -n
    best_neg = n.max(1)
    ties = (n == g[:, None]).sum(1)
    acc[valid] = np.where(g > best_neg, 1.0, np.where(g == best_neg, 1.0 / (1 + ties), 0.0))
    return acc


# --------------------------------------------------------------------------------------
def decode_items(paths, evalset, idx, downloader=None, with_negs: bool = True) -> list[dict]:
    """Texto de pregunta, oro y negativos (para QC humano y para generar T-gen2)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    idx = np.asarray(idx)
    docs = set(evalset.doc_ids[idx].tolist())
    qa = {(r["doc_id"], r["pair_idx"]): r for r in
          lookup(paths, docs, "qa", ["doc_id", "pair_idx", "question_ids", "answer_ids"], downloader)}
    negm: dict = {}
    if with_negs:
        for r in lookup(paths, docs, "neg", ["doc_id", "pair_idx", "neg_idx", "neg_ids"], downloader):
            negm.setdefault((r["doc_id"], r["pair_idx"]), {})[r["neg_idx"]] = r["neg_ids"]
    out = []
    for i in idx:
        key = (evalset.doc_ids[i], int(evalset.pair_idx[i]))
        r = qa.get(key)
        if r is None:
            continue
        item = {"i": int(i), "doc_id": key[0], "pair_idx": key[1],
                "question": tok.decode(r["question_ids"]), "answer": tok.decode(r["answer_ids"])}
        if with_negs:
            item["negatives_gemini"] = [tok.decode(v) for _, v in sorted(negm.get(key, {}).items())]
        out.append(item)
    return out


def export_qc_sample(paths, evalset, out_csv: str, n: int = 200, seed: int = 0, downloader=None) -> str:
    import pandas as pd
    idx = np.sort(np.random.default_rng(seed).choice(evalset.n, size=min(n, evalset.n), replace=False))
    items = decode_items(paths, evalset, idx, downloader)
    rows = []
    for it in items:
        for j, neg in enumerate(it.get("negatives_gemini", [])):
            rows.append({"i": it["i"], "question": it["question"], "gold": it["answer"], "neg_idx": j,
                         "negative": neg, "is_actually_correct(0/1)": "", "notes": ""})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


# --------------------------------------------------------------------------------------
GEN2_PROMPT = (
    "You are writing hard negative answers for a question-answering benchmark.\n"
    "Question: {q}\nCorrect answer: {a}\n"
    "Write {k} different answers that are plausible, on-topic, and similar in style and length to the "
    "correct answer, but factually INCORRECT. Do not copy or trivially negate the correct answer.\n"
    "Return exactly {k} lines, one answer per line, with no numbering and no extra text."
)


def hf_generate_fn(model_name: str, max_new_tokens: int = 400, temperature: float = 0.8, batch: int = 8):
    """Generador con transformers (chat template del modelo). Devuelve fn(prompts)->list[str]."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()

    @torch.no_grad()
    def gen(prompts: list[str]) -> list[str]:
        outs = []
        for s in range(0, len(prompts), batch):
            chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in prompts[s:s + batch]]
            enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                               top_p=0.95, pad_token_id=tok.pad_token_id)
            outs.extend(tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs
    return gen


def _parse(text: str, gold: str, k: int) -> list[str]:
    seen, out = set(), []
    g = gold.strip().lower()
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
            line = line[2:].strip()
        low = line.lower()
        if not line or low == g or low in seen:
            continue
        seen.add(low)
        out.append(line)
    return out[:k]


def generate_gen2(paths, evalset, out_dir: str, generate_fn, n_items: int = 10_000, k: int = 10,
                  seed: int = 0, batch: int = 64, generator_name: str = "", downloader=None, log=print) -> dict:
    """Paso 1 de T-gen2: genera k negativos por ítem con OTRO generador (P3). Reanudable (JSONL)."""
    os.makedirs(out_dir, exist_ok=True)
    idx = np.sort(np.random.default_rng(seed).choice(evalset.n, size=min(n_items, evalset.n), replace=False))
    jl = os.path.join(out_dir, "gen2_raw.jsonl")
    done = {r["i"] for r in read_jsonl(jl)}
    todo = [int(i) for i in idx if int(i) not in done]
    for s in range(0, len(todo), batch):
        items = decode_items(paths, evalset, todo[s:s + batch], downloader, with_negs=False)
        texts = generate_fn([GEN2_PROMPT.format(q=it["question"], a=it["answer"], k=k) for it in items])
        for it, t in zip(items, texts):
            append_jsonl(jl, {"i": it["i"], "question": it["question"], "answer": it["answer"],
                              "raw": t, "negatives": _parse(t, it["answer"], k)})
        log(f"[gen2] {min(len(todo), s + batch)}/{len(todo)}")
    info = {"requested": int(len(idx)), "generated": len(read_jsonl(jl)), "k": k, "seed": seed,
            "generator": generator_name, "prompt": GEN2_PROMPT, "items": idx.tolist()}
    save_json(info, os.path.join(out_dir, "gen2_info.json"))
    return {kk: v for kk, v in info.items() if kk != "items"}


def embed_gen2(out_dir: str, encoder, k: int = 10, log=print) -> dict:
    """Paso 2 de T-gen2: embebe los negativos generados (mismo pipeline que validó G1)."""
    info = load_json(os.path.join(out_dir, "gen2_info.json"))
    recs = {r["i"]: r for r in read_jsonl(os.path.join(out_dir, "gen2_raw.jsonl"))}
    keep = [i for i in info["items"] if len(recs.get(int(i), {}).get("negatives", [])) >= k]
    texts = [n for i in keep for n in recs[int(i)]["negatives"][:k]]
    E = encoder.embed_texts(texts, keep="head").astype(np.float16).reshape(len(keep), k, -1)
    atomic_np_save(np.asarray(keep, np.int64), os.path.join(out_dir, "gen2_items.npy"))
    atomic_np_save(E, os.path.join(out_dir, "gen2_negs.npy"))
    info.update(complete=int(len(keep)), dropped_incomplete=int(len(info["items"]) - len(keep)))
    save_json(info, os.path.join(out_dir, "gen2_info.json"))
    log(f"[gen2] embebidos {len(keep)} ítems completos ({info['dropped_incomplete']} descartados por < {k} negativos)")
    return {kk: v for kk, v in info.items() if kk != "items"}


def export_gen2_qc(out_dir: str, n: int = 200, seed: int = 0) -> str:
    """CSV para revisar a mano falsos negativos de T-gen2 (criterio del diseño: < 5 %)."""
    import pandas as pd
    recs = read_jsonl(os.path.join(out_dir, "gen2_raw.jsonl"))
    rs = np.random.default_rng(seed)
    pick = rs.choice(len(recs), size=min(n, len(recs)), replace=False)
    rows = []
    for j in pick:
        r = recs[int(j)]
        for neg in r["negatives"]:
            rows.append({"item": r["i"], "question": r["question"], "gold": r["answer"], "negative": neg,
                         "is_actually_correct(0/1)": "", "notes": ""})
    p = os.path.join(out_dir, "QC_negativos_gen2.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def load_gen2(out_dir: str):
    p_i, p_n = os.path.join(out_dir, "gen2_items.npy"), os.path.join(out_dir, "gen2_negs.npy")
    if os.path.exists(p_i) and os.path.exists(p_n):
        return np.load(p_i), np.load(p_n, mmap_mode="r")
    return None
