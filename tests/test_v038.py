"""Pruebas de v0.3.8: denominadores nuevos de InfoNCE+HN (condición de borde de B5), el juez de falsos negativos del
worker y los bloques del notebook pre-envío (clm_losses.preenvio)."""
import json
import math
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clm_losses import f0
from clm_losses import preenvio as pre
from clm_losses.analysis import c2_report, c2c_report
from clm_losses.data import EvalSet
from clm_losses.evaluate import rank_metrics, raw_cosine_scores, shuffle_perm
from clm_losses.generators import load_panel, macro_for_run
from clm_losses.genworker import JUDGE_MARK, judge_prompt, parse_judge
from clm_losses.heads import Heads
from clm_losses.losses import InfoNCE, NEG, build_loss
from clm_losses.pipeline import base_cfg, freeze_path, load_bundle, setup
from clm_losses.smoke import SMOKE_LOSS_PARAMS
from clm_losses.stats import clusters_from_docs, contrast, holm
from clm_losses.train import Trainer, run_training
from clm_losses.utils import load_json, save_json

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUIET = dict(log=lambda *a, **k: None)
H, D = 32, 16
NEW = ("hn_plus_inbatch_k", "hn_plus_otherhn", "hn_plus_otherhn_k", "hn_shared_inbatch")


def make_ctx(heads, B=12, K=5, seed=0, step=3):
    g = torch.Generator().manual_seed(seed)
    es = F.normalize(torch.randn(B, H, generator=g), dim=-1)
    ea = F.normalize(es @ torch.randn(H, H, generator=g) * 0.3 + torch.randn(B, H, generator=g), dim=-1)
    en = F.normalize(torch.randn(B, K, H, generator=g), dim=-1)
    nmask = torch.ones(B, K, dtype=torch.bool)
    nmask[0, 3:] = False
    zn = heads.za(en.view(B * K, H)).view(B, K, -1)
    dup = torch.zeros(B, B, dtype=torch.bool)
    dup[1, 2] = dup[2, 1] = True
    return types.SimpleNamespace(es=es, ea=ea, zs=heads.zs(es), za=heads.za(ea), en=en, zn=zn, nmask=nmask, dup_a=dup,
                                 dup_s=dup, step=step)


def _loss(den, ctx, heads, **p):
    m = build_loss("infonce_hn", heads, {"hn_denominator": den, "extra_seed": 7, **p}, 10, "native")
    return m(ctx, heads)


@pytest.fixture(scope="module")
def heads():
    torch.manual_seed(0)
    return Heads(hidden=H, width=24, depth=3, proj=D)


# --------------------------------------------------------------------------------------
# Pérdidas
def test_existing_denominators_are_unchanged(heads):
    """'HN solos' y 'HN + in-batch' dan exactamente la fórmula de antes."""
    ctx = make_ctx(heads)
    s, a, n = F.normalize(ctx.zs, dim=-1), F.normalize(ctx.za, dim=-1), F.normalize(ctx.zn, dim=-1)
    t = heads.scale()
    pos = t * (s * a).sum(-1)
    hn = (t * torch.einsum("bd,bkd->bk", s, n)).masked_fill(~ctx.nmask, NEG)
    inb = (t * s @ a.t()).masked_fill(ctx.dup_a | torch.eye(len(s), dtype=torch.bool), NEG)
    for den, cand in (("hn_only", torch.cat([pos[:, None], hn], 1)),
                      ("hn_plus_inbatch", torch.cat([pos[:, None], inb, hn], 1))):
        _, logs = _loss(den, ctx, heads)
        ref = F.cross_entropy(cand, torch.zeros(len(s), dtype=torch.long))
        assert math.isclose(logs["l_s2a"], float(ref.detach()), rel_tol=1e-6), den
        assert "n_extra" not in logs


def test_inbatch_k_with_all_columns_equals_hn_plus_inbatch(heads):
    ctx = make_ctx(heads)
    _, full = _loss("hn_plus_inbatch", ctx, heads)
    _, k_all = _loss("hn_plus_inbatch_k", ctx, heads, extra_k=10_000)
    assert math.isclose(full["l_s2a"], k_all["l_s2a"], rel_tol=1e-6)


def test_number_of_extra_candidates(heads):
    B, K = 12, 5
    ctx = make_ctx(heads, B=B, K=K)
    dup_rows = 2                                     # filas 1 y 2: oros duplicados entre sí (dup_a)
    assert _loss("hn_plus_inbatch_k", ctx, heads, extra_k=4)[1]["n_extra"] == 4
    assert math.isclose(_loss("hn_plus_inbatch_k", ctx, heads, extra_k=10_000)[1]["n_extra"],
                        (B * (B - 1) - dup_rows) / B, rel_tol=1e-6)
    assert _loss("hn_plus_otherhn", ctx, heads)[1]["n_extra"] == B - 1          # un HN de cada otro estado
    assert _loss("hn_plus_otherhn_k", ctx, heads, extra_k=4)[1]["n_extra"] == 4
    hn_valid = int(ctx.nmask.sum())
    other_hn = sum(hn_valid - int(ctx.nmask[i].sum()) for i in range(B))           # HN de los otros estados
    inb = B * (B - 1) - dup_rows
    assert math.isclose(_loss("hn_shared_inbatch", ctx, heads)[1]["n_extra"], (inb + other_hn) / B, rel_tol=1e-6)


def test_new_denominators_are_deterministic_and_ignore_the_global_rng(heads):
    ctx = make_ctx(heads)
    for den in NEW:
        torch.manual_seed(1)
        l1, _ = _loss(den, ctx, heads, extra_k=3)
        torch.rand(1000)                             # mover el RNG global no cambia nada
        l2, _ = _loss(den, ctx, heads, extra_k=3)
        assert torch.equal(l1, l2), den
    a, _ = _loss("hn_plus_otherhn_k", ctx, heads, extra_k=3)
    ctx2 = make_ctx(heads, step=4)
    b, _ = _loss("hn_plus_otherhn_k", ctx2, heads, extra_k=3)
    assert not torch.equal(a, b)                     # otro paso, otra muestra


def test_foreign_hn_that_duplicates_the_gold_is_masked(heads):
    B, K = 8, 4
    ctx = make_ctx(heads, B=B, K=K)
    ctx.en[5] = ctx.ea[3][None, :].expand(K, H)     # todos los HN del estado 5 = el oro del estado 3
    ctx.zn = heads.za(ctx.en.view(B * K, H)).view(B, K, -1)
    s, n = F.normalize(ctx.zs, dim=-1), F.normalize(ctx.zn, dim=-1)
    logits = heads.scale() * s @ F.normalize(ctx.za, dim=-1).t()
    for den in ("hn_plus_otherhn", "hn_shared_inbatch"):
        m = build_loss("infonce_hn", heads, {"hn_denominator": den, "extra_seed": 7}, 10, "native")
        cols, _ = m._extra_candidates(ctx, s, n, logits, heads.scale())
        if den == "hn_plus_otherhn":
            assert cols[3, 5] == NEG and cols[4, 5] != NEG
        else:
            blk = cols[:, B:].reshape(B, B, K)       # después de los B oros in-batch
            assert bool((blk[3, 5] == NEG).all()) and bool((blk[4, 5] != NEG).all())
            assert bool((blk[torch.arange(B), torch.arange(B)] == NEG).all())   # los propios no se repiten


def test_new_denominators_backprop_and_bad_name(heads):
    ctx = make_ctx(heads)
    for den in NEW:
        heads.zero_grad()
        c = make_ctx(heads)
        loss, logs = _loss(den, c, heads, extra_k=3)
        loss.backward()
        assert torch.isfinite(loss) and logs["n_extra"] > 0
        assert sum(float(p.grad.abs().sum()) for p in heads.state_head.parameters() if p.grad is not None) > 0
    with pytest.raises(ValueError):
        build_loss("infonce_hn", heads, {"hn_denominator": "hn_plus_everything"}, 10, "native")
    assert set(NEW) | {"hn_only", "hn_plus_inbatch"} == set(InfoNCE.DENOMINATORS)
    del ctx


# --------------------------------------------------------------------------------------
# Juez del worker
def test_parse_judge_formats():
    assert parse_judge("1: 0\n2: 1\n3: 0", 3) == [0, 1, 0]
    assert parse_judge("[1] - 1\n[2] 0\n3. 1", 3) == [1, 0, 1]
    assert parse_judge("```\nCandidate 1: 0\n**2: 1**\n3) 0\n```", 3) == [0, 1, 0]
    assert parse_judge("Sure:\n1: 0\n2: 1\n1: 1\n3: 0\nDone.", 3) == [0, 1, 0]          # primera aparición
    assert parse_judge("1: 0\n2: 1", 3) is None and parse_judge("", 2) is None
    assert parse_judge("1: yes\n2: 0", 2) is None


def test_judge_prompt_lists_candidates():
    p = judge_prompt("What  is\nX?", "the gold", ["a", "b\nc"])
    assert p.startswith(JUDGE_MARK) and "[1] a\n[2] b c" in p and "exactly 2 lines" in p and "What is X?" in p


def test_genworker_judge_with_retry_and_resume(tmp_path):
    p, out = tmp_path / "prompts.jsonl", tmp_path / "raw.jsonl"
    with open(p, "w") as f:
        for i in range(40):
            f.write(json.dumps({"i": f"tgem:{i}", "prompt": judge_prompt(f"q{i}", f"g{i}", [f"c{i}_{j}" for j in range(10)]),
                                "n": 10, "max_new_tokens": 96}) + "\n")
    env = dict(os.environ, PYTHONPATH=PKG_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, "-m", "clm_losses.genworker", "judge", "--model", "x", "--prompts", str(p), "--out", str(out),
           "--backend", "fake", "--batch", "16", "--max-attempts", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    recs = [json.loads(x) for x in open(out)]
    first = [x for x in recs if x["attempt"] == 0]
    assert len(first) == 40 and any(x["labels"] is None for x in first)          # el falso a veces no respeta el formato
    ok = {x["i"] for x in recs if x["labels"] is not None}
    assert ok == {f"tgem:{i}" for i in range(40)}                                # el recordatorio lo arregla
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    assert r2.returncode == 0 and "pendientes 0" in r2.stdout
    assert len([json.loads(x) for x in open(out)]) == len(recs)                  # reanudar no repite nada


# --------------------------------------------------------------------------------------
# Notebook pre-envío
@pytest.fixture(scope="module")
def study(tmp_path_factory):
    base = tmp_path_factory.mktemp("drive")
    cache = tmp_path_factory.mktemp("cache")
    st = setup("smoke", drive_base=str(base), cache_root=str(cache), mount=False,
               train_overrides={"loss_params": SMOKE_LOSS_PARAMS})
    f0.f0_4_splits(st, **QUIET)
    f0.f0_5_eval_sets(st, lengths=False, qc=False, **QUIET)
    f0.f0_5b_pools(st, **QUIET)
    return st


def test_small_trainings_with_the_new_denominators(study):
    bundle = load_bundle(study, localize=False)
    for den in NEW:
        cfg = base_cfg(study, "infonce_hn", max_steps_pre=4, max_steps_mid=6, eval_every=3,
                       loss_params={"hn_denominator": den, "extra_k": 7})
        r = run_training(cfg, bundle, study.paths.d("runs", "test_v038", den), ecfg=study.eval, evaluate_test=False)
        assert r["status"] == "ok" and math.isfinite(r["val_best"]), den


class Boom(Exception):
    pass


def test_resume_is_exact_with_a_sampled_denominator(study, monkeypatch):
    bundle = load_bundle(study, localize=False)
    cfg = base_cfg(study, "infonce_hn", max_steps_pre=10, max_steps_mid=20, eval_every=5, ckpt_every_steps=3,
                   patience=100, loss_params={"hn_denominator": "hn_plus_otherhn_k", "extra_k": 5})
    da, db = study.paths.d("runs", "test_v038_res", "A"), study.paths.d("runs", "test_v038_res", "B")
    run_training(cfg, bundle, da, ecfg=study.eval, evaluate_test=False)
    orig, calls = Trainer._step, {"n": 0}

    def boom(self, batch, step):
        calls["n"] += 1
        if calls["n"] == 19:                      # a mitad de mid
            raise Boom()
        return orig(self, batch, step)
    monkeypatch.setattr(Trainer, "_step", boom)
    with pytest.raises(Boom):
        run_training(cfg, bundle, db, ecfg=study.eval, evaluate_test=False)
    monkeypatch.setattr(Trainer, "_step", orig)
    run_training(cfg, bundle, db, ecfg=study.eval, evaluate_test=False)
    a = torch.load(os.path.join(da, "best_mid.pt"), weights_only=False)["heads"]
    b = torch.load(os.path.join(db, "best_mid.pt"), weights_only=False)["heads"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_wilson_kappa_and_rule_is_not_overwritten(study):
    lo, hi = pre.wilson(5, 100)
    assert 0.02 < lo < 0.05 < hi < 0.12 and all(math.isnan(x) for x in pre.wilson(0, 0))
    assert math.isclose(pre.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
    assert abs(pre.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) + 1.0) < 1e-9
    r1 = pre.write_rule(study, "b_prueba", {"umbral": 0.5}, **QUIET)
    r2 = pre.write_rule(study, "b_prueba", {"umbral": 0.9}, **QUIET)
    assert r2["umbral"] == 0.5 and r1["creado"] == r2["creado"]
    d = pre.pdir(study, "b_prueba")
    assert any(f.startswith("REGLA_distinta_") for f in os.listdir(d))


def test_leak_check_finds_exact_and_near_duplicates():
    vocab = {}

    def ids(s):
        return [vocab.setdefault(w, len(vocab)) for w in s.split(" ")]
    inv = lambda: {v: k for k, v in vocab.items()}  # noqa: E731
    train = {("t1", 0): ("What year did the war end?", "1945"),
             ("t2", 0): ("Who wrote the novel about the white whale in 1851?", "Herman Melville"),
             ("t3", 0): ("What is the capital of France?", "Paris"),
             ("t4", 0): ("Name the largest planet.", "Jupiter")}
    ev = {("e1", 0): ("what year did the war END", "1945."),                      # exacto tras normalizar
          ("e2", 0): ("Who wrote the novel about the white whale in 1851 originally?", "Herman Melville"),   # casi
          ("e3", 0): ("Who wrote the famous novel about the great white whale?", "Herman Melville"),   # Jaccard bajo
          ("e4", 0): ("What is the capital of France?", "Lyon"),                   # misma pregunta, otro oro
          ("e5", 0): ("Which planet is the largest?", "Jupiter")}                  # mismo oro, otra pregunta
    rows = {**train, **ev}
    for q, a in rows.values():
        ids(q), ids(a)

    def batches(docs):
        keys = [k for k in rows if k[0] in docs]
        yield ([k[0] for k in keys], [k[1] for k in keys], [ids(rows[k][0]) for k in keys], [ids(rows[k][1]) for k in keys])

    def decode(lists):
        iv = inv()
        return [" ".join(iv[t] for t in x) for x in lists]
    r = pre.leak_check({"confirm": list(ev)}, set(train), batches, decode, **QUIET)["confirm"]
    assert r["par_exacto"] == 1 and r["par_exacto_o_casi"] == 2 and r["items"] == [0, 1]
    assert r["pregunta_igual_cualquier_oro"] == 2                               # e1 (tras normalizar) y e4
    assert pre.jaccard(pre.shingles(pre.norm_text("a b c d e f")), pre.shingles(pre.norm_text("a b c d e g"))) == 0.6


def test_iter_qa_filters_by_document_in_pyarrow(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    paths = types.SimpleNamespace(c=lambda *p: str(tmp_path.joinpath(*p)))
    files = []
    for part in range(2):
        t = pa.table({"doc_id": [f"d{part}{i}" for i in range(5)], "pair_idx": list(range(5)),
                      "question_ids": [[1, 2, i] for i in range(5)], "answer_ids": [[3, i] for i in range(5)]})
        f = tmp_path / f"p{part}.parquet"
        pq.write_table(t, f)
        files.append(str(f))

    class DL:
        def fetch(self, repo, name, local_dir):
            return files[int(name.split("_")[-1].split(".")[0]) % 2]
    got = [d for docs, *_ in pre._iter_qa(paths, docs={"d01", "d13"}, downloader=DL()) for d in docs]
    assert sorted(set(got)) == ["d01", "d13"]
    rest = [d for docs, *_ in pre._iter_qa(paths, exclude={"d00"}, downloader=DL()) for d in docs]
    assert "d00" not in rest and "d01" in rest


def _fake_items(n_per_set=40, keys=("g1", "g2")):
    items = {}
    for s in ["tgem", "tmine"] + [f"tgen_{k}" for k in keys]:
        for i in range(n_per_set):
            items[f"{s}:{i}"] = {"q": f"q {s} {i}", "a": f"a{i}", "cands": [f"{s} c{i}_{j}" for j in range(10)]}
    return items


def test_hand_labels_export_read_and_agreement(study):
    import pandas as pd
    keys = ("g1", "g2")
    items = _fake_items(keys=keys)
    csv = pre.export_hand_labels(study, items, keys, **QUIET)
    df = pd.read_csv(csv, dtype={"id": str})
    clave = load_json(os.path.join(os.path.dirname(csv), "clave.json"))
    assert clave["por_set"] == {"tgem": 30, "tmine": 30, "tgen_g1": 28, "tgen_g2": 28}
    assert len(df) == 116 * 10 and set(df.columns) >= {"id", "pregunta", "oro", "n", "candidato", pre.QC_COL}
    assert not any(":" in str(x) for x in df["id"])                  # el CSV no dice de qué set es cada ítem
    assert pre.export_hand_labels(study, items, keys, **QUIET) == csv and \
        pd.read_csv(csv, dtype={"id": str}).equals(df)              # no se reexporta
    # etiquetas: el humano marca el candidato 1 de los ítems pares; el juez, el 1 de los múltiplos de 4
    inv = clave["clave"]
    marks = [1 if (r.n == 1 and int(inv[r.id].split(":")[1]) % 2 == 0) else 0 for r in df.itertuples()]
    df[pre.QC_COL] = [""] * 3 + marks[3:]                           # las 3 primeras sin marcar: se ignoran
    df.to_csv(os.path.join(os.path.dirname(csv), "ETIQUETADAS.csv"), index=False, sep=";", encoding="utf-8-sig")
    hand = pre.read_hand_labels(study)                               # como lo guarda Excel en español: ; y BOM
    assert len(hand) == 116 * 10 - 3
    judge = {k: [1 if (j == 0 and int(k.split(":")[1]) % 4 == 0) else 0 for j in range(10)] for k in items}
    sets = ["tgem", "tmine", "tgen_g1", "tgen_g2"]
    agr = pre.agreement(judge, hand, sets)
    tot = agr["total"]
    assert tot["n_negativos"] == 116 * 10 - 3 and tot["fp"] == 0 and tot["fn"] > 0 and 0 < tot["kappa"] < 1
    rates = pre.fn_rates(judge, sets, {s: 40 for s in sets})
    assert rates["tgem"]["n_items"] == 40 and rates["tgem"]["items_con_fn"] == 10 and rates["tgem"]["items_del_set"] == 40


def test_run_judge_with_the_fake_backend(study):
    items = _fake_items(n_per_set=6)
    info = pre.run_judge(study, items, backend="fake", batch=8, **QUIET)
    lab = pre.judge_labels(study)
    assert set(lab) == set(items) and all(len(v) == 10 for v in lab.values())
    assert info["n_etiquetados"] == len(items) and info["prompt_version"] == "fn-judge-v1"
    again = pre.run_judge(study, items, backend="fake", **QUIET)               # nada pendiente: no corre nada
    assert again["n_etiquetados"] == len(items)
    ex = pre.qc_fn_exclusions(lab)
    assert set(ex) == {"tgem", "tmine", "tgen"}


def _confirm_panel_and_runs(study, keys=("g1", "g2")):
    """Panel del sellado con un macro de 2 generadores y corridas falsas de F3 para C1, C2 y C2c."""
    ev = EvalSet(study.paths.d("data", "eval", "confirm"))
    n = ev.n
    rs = np.random.default_rng(3)
    mit = np.sort(rs.choice(n, size=n // 2, replace=False)).astype(np.int64)
    root = study.paths.d("data", "eval", "tgen", "confirm")
    os.makedirs(root, exist_ok=True)
    save_json({"version": "v-prueba", "macro": list(keys), "macro_items": mit.tolist(),
               "generators": {k: {"negs_sha": None} for k in keys}}, os.path.join(root, "PANEL.json"))
    save_json({"C1_best_noncontrastive": "vicreg_x"}, freeze_path(study))
    probs = {"vicreg_x": 0.6, "infonce": 0.7, "infonce_hn": 0.72, "siglip": 0.65, "siglip_hn": 0.7,
             "infonce_hn_hnonly": 0.6}
    for arm, p in probs.items():
        for s in (0, 1, 2):
            rd = study.paths.d("runs", "F3", arm, f"seed_{s}")
            os.makedirs(os.path.join(rd, "eval"), exist_ok=True)
            save_json({"status": "ok", "seed": s}, os.path.join(rd, "DONE.json"))
            for key in ("tmine", "tgem", "tgem_shuf"):
                np.save(os.path.join(rd, "eval", f"correct_{key}.npy"), (rs.random(n) < p).astype(np.float32))
            for k in keys:
                it = np.sort(np.concatenate([mit, rs.choice(np.setdiff1d(np.arange(n), mit), 5, replace=False)]))
                np.save(os.path.join(rd, "eval", f"items_tgen_{k}.npy"), it)
                np.save(os.path.join(rd, "eval", f"correct_tgen_{k}.npy"), (rs.random(len(it)) < p - 0.1).astype(np.float32))
                np.save(os.path.join(rd, "eval", f"correct_tgen_{k}_shuf.npy"), (rs.random(len(it)) < 0.2).astype(np.float32))
    return ev, mit


def test_contrastes_f3_reproduce_the_confirmatory_reports(study):
    ev, mit = _confirm_panel_and_runs(study)
    doc_ids = np.asarray(ev.doc_ids)
    fake_bundle = types.SimpleNamespace(test=ev, test_split="confirm")
    c2 = c2_report(study, fake_bundle, "F3", B=4000, **QUIET)
    c2c = c2c_report(study, fake_bundle, "F3", B=4000, **QUIET)
    # C1 como lo calcula run_all_vs_all (pipeline): macro de cada corrida, contraste por métrica y Holm entre las 2
    panel = load_panel(study, "confirm")
    for arm in ("vicreg_x", "infonce"):
        for s in (0, 1, 2):
            macro_for_run(study.paths.d("runs", "F3", arm, f"seed_{s}"), panel)
    per = {}
    for m, cl in (("tmine", clusters_from_docs(doc_ids)), ("tgenmacro", clusters_from_docs(doc_ids[mit]))):
        A = np.vstack([np.load(study.paths.d("runs", "F3", "vicreg_x", f"seed_{s}", "eval", f"correct_{m}.npy"))
                       for s in (0, 1, 2)])
        Bm = np.vstack([np.load(study.paths.d("runs", "F3", "infonce", f"seed_{s}", "eval", f"correct_{m}.npy"))
                        for s in (0, 1, 2)])
        per[m] = contrast(A, Bm, cl, 0.01, B=4000)
    for (m, c), p in zip(per.items(), holm(np.array([c["p"] for c in per.values()]))):
        c["p_holm_c1"] = float(p)
    save_json({"C1": {"por_metrica": per}}, study.paths.d("results", "F3", "all_vs_all", "all_vs_all.json"))
    rec = pre.contrastes_f3(study, doc_ids)
    chk = pre.check_reproduces(study, rec)
    assert set(chk) == {"C1 tmine", "C1 tgenmacro", "C2 infonce_hn − infonce", "C2 siglip_hn − siglip",
                        "C2c tgem_shuf", "C2c tmine", "C2c control"}
    assert all(chk.values()), chk
    assert len(c2["pares"]) == 2 and len(c2c["pruebas"]) == 2
    # con exclusiones: cambia el n y sigue corriendo
    lab = {f"tgem:{i}": [1] + [0] * 9 for i in range(0, ev.n, 7)}
    lab.update({f"tgen_g1:{int(i)}": [0, 1] + [0] * 8 for i in mit[::3]})
    rec2 = pre.contrastes_f3(study, doc_ids, pre.qc_fn_exclusions(lab))
    assert rec2["exclusiones"]["tgem"] == len(range(0, ev.n, 7)) and rec2["exclusiones"]["tgen"]["g1"] == len(mit[::3])
    assert rec2["C2"][0]["n_items"] < len(mit) and rec2["C1"]["tgenmacro"]["n_items"] == len(mit)
    assert rec2["C2c"][0]["c"]["diff"] != rec["C2c"][0]["c"]["diff"]
    rec3 = pre.contrastes_f3(study, doc_ids, {"todos": set(range(0, ev.n, 2))})
    assert rec3["C1"]["tgenmacro"]["n_items"] < len(mit)


def test_b0_friedman_and_f1b_identity(study):
    arms = ["a1", "a2", "a3", "a4"]
    rs = np.random.default_rng(0)
    blocks = ["tmine", "tgem", "tgem_shuf", "abst_auroc"]
    vals = {b: dict(zip(arms, rs.random(4))) for b in blocks}
    from clm_losses.stats import friedman_nemenyi
    fr = friedman_nemenyi(np.array([[vals[b][a] for a in arms] for b in blocks]), arms)
    fr["blocks"] = blocks
    metrics = {f"{b}_top1": {"summary": [{"arm": a, "mean": v} for a, v in vals[b].items()]} for b in blocks[:3]}
    metrics["abst_auroc"] = {"point": vals["abst_auroc"]}
    save_json({"friedman": fr, "metrics": metrics}, study.paths.d("results", "F3", "all_vs_all", "all_vs_all.json"))
    r = pre.b0_friedman(study)
    assert r["reproduce_el_guardado"] and r["bloques_7"] == ["tmine", "tgem", "abst_auroc"]
    # F1b frente a F3: una corrida idéntica y otra no
    for ph, v in (("F1b", 0.9), ("F3", 0.9)):
        rd = study.paths.d("runs", ph, "infonce", "seed_0")
        os.makedirs(rd, exist_ok=True)
        save_json({"status": "ok", "seed": 0, "val_best": v, "stages": {"mid": {"steps": 5, "best": v}}},
                  os.path.join(rd, "DONE.json"))
        save_json({"train": {"lr": 1e-3}}, os.path.join(rd, "config.json"))
    for ph, v in (("F1b", 0.8), ("F3", 0.81)):
        rd = study.paths.d("runs", ph, "siglip", "seed_0")
        os.makedirs(rd, exist_ok=True)
        save_json({"status": "ok", "seed": 0, "val_best": v}, os.path.join(rd, "DONE.json"))
    r = pre.b0_f1b_vs_f3(study)
    rows = {(x["brazo"], x["semilla"]): x for x in r["filas"]}
    assert rows[("infonce", 0)]["identico"] and not rows[("siglip", 0)]["identico"]
    assert r["comparadas"] == 2 and r["identicas"] == 1


def test_borde_reading_rule():
    base = {"metricas": {"tgem_shuf": {"c_hn_otherhn": {"menos_b": {"ci95_lo": 0.02}}}}}
    r = pre.borde_lectura({**base, "P_a_menos_P_b": 0.2, "R": {"c_hn_otherhn": {"R": 0.8}, "d31_hn_inbatch_k31": {"R": 0.1},
                                                                "c31_hn_otherhn_k31": {"R": 0.7}, "e_hn_shared": {"R": 0.1}}})
    assert r["principal"].startswith("empate de estilo") and r["cuantos_oros"].startswith("pocos oros bastan")
    assert r["c31"].startswith("la lectura de (c) no depende") and r["compartidos"].startswith("los oros la neutralizan")
    assert pre.borde_lectura({**base, "P_a_menos_P_b": 0.2, "R": {"c_hn_otherhn": {"R": 0.1}}})["principal"].startswith("dilución")
    assert pre.borde_lectura({**base, "P_a_menos_P_b": 0.03, "R": {}})["principal"].startswith("no interpretable")
    assert pre.borde_lectura({**base, "P_a_menos_P_b": 0.2, "R": {"c_hn_otherhn": {"R": 0.4}}})["principal"].startswith("intermedio")


def test_borde_reproduction_check(study):
    for (ph, name), new, v_new in ((("F2b_B1", "infonce_hn_hnonly"), "a_hn_only", 0.5),
                                   (("F1b", "infonce_hn"), "b_hn_inbatch", 0.6)):
        for s in (0, 1, 2):
            for p, n, v in ((ph, name, 0.5 if new == "a_hn_only" else 0.6), (pre.BORDE_PHASE, new, v_new)):
                rd = study.paths.d("runs", p, n, f"seed_{s}")
                os.makedirs(rd, exist_ok=True)
                save_json({"status": "ok", "seed": s, "val_best": v, "stages": {"mid": {"steps": 3}}},
                          os.path.join(rd, "DONE.json"))
    chk = pre.bloque2_chequeo(study)
    assert chk["ok"] and chk["lectura"].startswith("6/6 idénticas")
    rd = study.paths.d("runs", pre.BORDE_PHASE, "b_hn_inbatch", "seed_1")
    save_json({"status": "ok", "seed": 1, "val_best": 0.66, "stages": {"mid": {"steps": 3}}}, os.path.join(rd, "DONE.json"))
    assert not pre.bloque2_chequeo(study)["ok"]


def test_raw_cosine_reference_matches_the_evaluation_code(study):
    res = pre.ref_raw_cosine(study, "test", keys=[])
    ev = EvalSet(study.paths.d("data", "eval", "test"))
    want = rank_metrics(raw_cosine_scores(ev, "tgem"))[1]
    assert np.mean(res["tgem"][0] == want) > 0.99          # mismo cálculo salvo redondeo (otro orden de suma)
    I = np.load(study.paths.d("data", "eval", "test", "mine_idx.npy"))
    assert np.mean(res["tmine"][0] == rank_metrics(raw_cosine_scores(ev, "pool", pool_idx=I))[1]) > 0.99
    assert res["tgem"][1].mean() < res["tgem"][0].mean()                  # con estados barajados baja


def test_text_scoring_uses_the_shuffled_state_of_the_evaluation():
    n = 50
    perm = shuffle_perm(n, 0)
    sets = {"x": [(i, [f"gold{i}"] + [f"neg{i}_{j}" for j in range(3)]) for i in range(n)]}
    res = pre._score_sets(sets, lambda i, c: [1.0 if t == f"gold{i}" else 0.0 for t in c], perm)
    real, shuf, items = res["x"]
    assert real.mean() == 1.0 and shuf.mean() < 0.5 and np.array_equal(items, np.arange(n))
    bm = pre.BM25.from_corpus([["the", "cat"], ["a", "dog"], ["the", "bird"]])
    assert bm.score(["cat"], ["the", "cat"]) > bm.score(["cat"], ["the", "dog"]) == 0.0
    assert math.isclose(bm.idf("the"), math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)))
    r = pre.ref_bm25({"x": [(0, ["paris is the capital", "lyon"]), (1, ["the sky is blue", "red"])]},
                     {0: "capital of france", 1: "color of the sky"}, bm, n_split=2)
    assert r["x"][0].tolist() == [1.0, 1.0]


def test_reports_of_every_block_render(study):
    """Bloques 0–3 y el resumen escriben sus informes con lo que haya (corridas falsas del smoke)."""
    r0 = pre.bloque0(study, **QUIET)
    assert os.path.exists(pre.pdir(study, "b0_extraccion", "EXTRACCION.md")) and "f1b_vs_f3" in r0
    assert os.path.exists(pre.pdir(study, "b0_extraccion", "REGLA.json"))
    r1 = pre.paso_calidad(study, **QUIET)
    md1 = open(pre.pdir(study, "b1_calidad", "CALIDAD.md"), encoding="utf-8").read()
    assert "sin exclusiones (control)" in md1 and r1["acuerdo"] is not None and "tgem" in r1["tasas_juez"]
    # bloque 2: 6 condiciones × 3 semillas con aciertos falsos en el test
    n = EvalSet(study.paths.d("data", "eval", "test")).n
    rs = np.random.default_rng(5)
    p_shuf = {"a_hn_only": 0.40, "b_hn_inbatch": 0.10, "c_hn_otherhn": 0.35, "c31_hn_otherhn_k31": 0.33,
              "d31_hn_inbatch_k31": 0.12, "e_hn_shared": 0.11}
    for f, p in p_shuf.items():
        for s in (0, 1, 2):
            rd = study.paths.d("runs", pre.BORDE_PHASE, f, f"seed_{s}")
            os.makedirs(os.path.join(rd, "eval"), exist_ok=True)
            save_json({"status": "ok", "seed": s, "val_best": 0.5}, os.path.join(rd, "DONE.json"))
            for key, q in (("tgem_shuf", p), ("tgem", 0.6), ("tmine", 0.9), ("tmine50", 0.8), ("trand", 0.95)):
                np.save(os.path.join(rd, "eval", f"correct_{key}.npy"), (rs.random(n) < q).astype(np.float32))
    assert pre.borde_hecho(study)
    r2 = pre.bloque2(study, **QUIET)                       # todo hecho: no baja ni entrena nada
    assert r2["lectura"]["principal"].startswith("empate de estilo") and "pocos oros" in r2["lectura"]["cuantos_oros"]
    assert 0.6 < r2["R"]["c_hn_otherhn"]["R"] < 1.1 and r2["R"]["c_hn_otherhn"]["ic95"][0] < r2["R"]["c_hn_otherhn"]["R"]
    assert "R(x)" in open(pre.pdir(study, "b2_borde", "BORDE.md"), encoding="utf-8").read()
    # bloque 3: resumen de resultados guardados
    ev = EvalSet(study.paths.d("data", "eval", "test"))
    res = pre.ref_raw_cosine(study, "test", keys=[])
    pre.save_ref(study, "coseno_crudo", "test", res)
    pre.done_ref(study, "coseno_crudo", "test")
    back = pre.load_ref(study, "coseno_crudo", "test")
    assert set(back) == set(res) and np.array_equal(back["tgem"][0], res["tgem"][0])
    tab = pre.ref_summary(back, np.asarray(ev.doc_ids), keys=[])
    md3 = pre.bloque3_markdown({"creado": "x", "sanity": None, "tablas": {"test": {"coseno_crudo": tab}}, "contexto_F3": {}})
    assert "coseno crudo" in md3 and "no se corrió" in md3
    md = pre.resumen(study)
    assert "Bloque 0" in md and "Bloque 2" in md


def _tiny_qwen3_embedding(d, eos_in_tokenizer=True):
    """Modelo Qwen3 chico con la estructura de Qwen3-Embedding en sentence-transformers (último token + L2)."""
    from sentence_transformers import SentenceTransformer, models as stm
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    corpus = ["What is the capital of China?", "The capital of China is Beijing.", "Explain gravity",
              "Gravity is a force that attracts two bodies", "Instruct: Given a web search query Query:"] * 30
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<|endoftext|>"]))
    if eos_in_tokenizer:
        eos = tk.token_to_id("<|endoftext|>")
        tk.post_processor = processors.TemplateProcessing(single="$A <|endoftext|>", special_tokens=[("<|endoftext|>", eos)])
    raw = os.path.join(d, "raw")
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                            unk_token="<unk>").save_pretrained(raw)
    torch.manual_seed(0)
    Qwen3Model(Qwen3Config(vocab_size=tk.get_vocab_size(), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                           max_position_embeddings=256)).save_pretrained(raw)
    t = stm.Transformer(raw, tokenizer_args={"padding_side": "left"})
    out = os.path.join(d, "st")
    SentenceTransformer(modules=[t, stm.Pooling(t.get_word_embedding_dimension(), pooling_mode="lasttoken"),
                                 stm.Normalize()]).save(out)
    return out


@pytest.mark.parametrize("eos_in_tokenizer", [True, False])
def test_qwen_embedder_paths_agree(tmp_path, monkeypatch, eos_in_tokenizer):
    st_mod = pytest.importorskip("sentence_transformers")
    repo = _tiny_qwen3_embedding(str(tmp_path), eos_in_tokenizer)
    e1 = pre.QwenEmbedder(repo=repo, device="cpu", batch=3)
    assert e1.modo == ("sentence-transformers" if eos_in_tokenizer else "transformers (+<|endoftext|>)")
    monkeypatch.setattr(st_mod, "SentenceTransformer", None)          # fuerza la vía de transformers
    e2 = pre.QwenEmbedder(repo=repo, device="cpu", batch=2)
    assert e2.modo.startswith("transformers")
    texts = ["The capital of China is Beijing.", "Gravity is a force", "x", "a longer text about gravity and China"]
    for q in (False, True):
        A, B = e1.encode(texts, query=q), e2.encode(texts, query=q)
        assert A.shape == B.shape == (4, 32) and np.abs(A - B).max() < 0.01
        assert np.allclose(np.linalg.norm(A, axis=1), 1, atol=1e-5)
    s = e2.sanity()
    assert set(s) >= {"modo", "matriz", "ok", "criterio"} and len(s["matriz"]) == 2


def test_textos_split_decodes_everything_once_and_orders_negatives(study):
    ev = EvalSet(study.paths.d("data", "eval", "confirm"))
    calls = {"n": 0}
    rs = np.random.default_rng(0)

    def fake_lookup(paths, docs, which, columns):
        calls["n"] += 1
        rows = []
        for d, p in zip(ev.doc_ids.tolist(), ev.pair_idx.tolist()):
            if d not in docs:
                continue
            h = abs(hash((d, int(p)))) % 10_000
            if which == "qa":
                rows.append({"doc_id": d, "pair_idx": int(p), "question_ids": [1, h], "answer_ids": [2, h]})
            else:
                for j in rs.permutation(10):                      # desordenados: hay que ordenarlos por neg_idx
                    rows.append({"doc_id": d, "pair_idx": int(p), "neg_idx": int(j), "neg_ids": [3, h, int(j)]})
        return rows

    class Tok:
        def batch_decode(self, lists):
            return ["t" + "_".join(map(str, x)) for x in lists]
    tx = pre.textos_split(study, "confirm", lookup_fn=fake_lookup, tok=Tok(), **QUIET)
    assert len(tx) == ev.n and calls["n"] == 2
    r = tx[0]
    assert r["q"].startswith("t1_") and r["a"].startswith("t2_") and len(r["negs"]) == 10
    assert [int(x.rsplit("_", 1)[1]) for x in r["negs"]] == list(range(10))
    again = pre.textos_split(study, "confirm", lookup_fn=fake_lookup, tok=Tok(), **QUIET)
    assert calls["n"] == 2 and again == tx                            # de la caché, sin volver a buscar
    items = pre.qc_items(study, tx, {}, {}, [])
    assert sum(k.startswith("tgem:") for k in items) == ev.n and sum(k.startswith("tmine:") for k in items) == ev.n
    mine = np.load(study.paths.d("data", "eval", "confirm", "mine_idx.npy"))
    assert items["tmine:3"]["cands"] == [tx[int(j)]["a"] for j in mine[3]]


def test_leak_check_with_a_frequent_gold_uses_the_index_and_is_fast():
    import time as _t
    rs = np.random.default_rng(0)
    words = [f"w{i}" for i in range(400)]
    train = {(f"t{i}", 0): (" ".join(rs.choice(words, 9)), "yes") for i in range(30_000)}
    base = train[("t123", 0)][0]
    ev = {("e0", 0): (base + " today", "Yes"),                                   # casi: un 3-grama más
          ("e1", 0): (" ".join(rs.choice(words, 9)), "yes")}
    ev.update({(f"e{i}", 0): (" ".join(rs.choice(words, 9)), "yes") for i in range(2, 2000)})
    rows = {**train, **ev}

    def batches(docs):
        keys = [k for k in rows if k[0] in docs]
        yield ([k[0] for k in keys], [0] * len(keys), [rows[k][0] for k in keys], [rows[k][1] for k in keys])
    t0 = _t.time()
    r = pre.leak_check({"confirm": list(ev)}, set(train), batches, lambda xs: list(xs), **QUIET)["confirm"]
    assert _t.time() - t0 < 30
    assert 0 in r["items"] and r["par_exacto"] == 0 and r["par_exacto_o_casi"] >= 1
    assert r["ejemplos"][0]["pregunta_entrenamiento"] == base


def test_hf_backend_greedy_is_deterministic(tmp_path):
    """La vía voraz del juez (transformers) corre y no depende de la semilla."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("t31", os.path.join(PKG_ROOT, "tests", "test_v031.py"))
    t31 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t31)
    from transformers import MistralConfig, MistralForCausalLM
    from clm_losses.genworker import HFBackend
    d = str(tmp_path / "tiny")
    hf = t31._tiny_tokenizer(d)
    cfg = MistralConfig(vocab_size=len(hf), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                        bos_token_id=hf.bos_token_id, eos_token_id=hf.eos_token_id, pad_token_id=hf.pad_token_id)
    torch.manual_seed(0)
    MistralForCausalLM(cfg).save_pretrained(d)
    b = HFBackend(d, "float32", {}, False, {})
    ps = [judge_prompt("q one", "Paris", ["Lyon", "Rome"]), judge_prompt("q two is longer", "Rome", ["Paris", "x"])]
    a1 = b.generate(ps, 12, 1.0, 1.0, 0, greedy=True)
    a2 = b.generate(ps, 12, 1.0, 1.0, 123, greedy=True)
    assert a1 == a2 and len(a1) == 2


def test_bloque3_glue_with_cached_texts(study, monkeypatch):
    """bloque3 de punta a punta con textos ya en la caché, sin panel T-gen y sin Qwen3 (sin GPU)."""
    for sp in ("test", "confirm"):
        ev = EvalSet(study.paths.d("data", "eval", sp))
        p = pre.pdir(study, "textos", f"{sp}.jsonl")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for i in range(ev.n):
                f.write(json.dumps({"i": i, "q": f"question about topic {i % 37} item {i}", "a": f"answer {i % 37} item {i}",
                                    "negs": [f"wrong {j} topic {(i + j) % 37}" for j in range(10)]}) + "\n")
    monkeypatch.setattr(pre, "macro_keys", lambda study, split, log=print: [])
    monkeypatch.setattr(pre, "bm25_corpus", lambda study, log=print: pre.BM25.from_corpus(
        [pre.bm25_tokens(f"answer {k} item") for k in range(50)]))
    res = pre.bloque3(study, qwen=False, **QUIET)
    t = res["tablas"]
    assert set(t["test"]) == {"coseno_crudo", "bm25"} and set(t["confirm"]) == {"coseno_crudo", "bm25"}
    b = t["test"]["bm25"]
    assert b["tgem"]["media"] > b["tgem_shuf"]["media"] + 0.3          # la pregunta comparte palabras con su oro
    md = open(pre.pdir(study, "b3_referencias", "REFERENCIAS.md"), encoding="utf-8").read()
    assert "BM25" in md and "Sellado" in md
    again = pre.bloque3(study, qwen=False, **QUIET)                     # reanudable: no recalcula
    assert again["tablas"]["confirm"]["bm25"]["tgem"] == t["confirm"]["bm25"]["tgem"]


def test_bloque1_steps_glue_with_fake_judge(study, monkeypatch):
    """Pasos 1.2, 1.3 y 1.5 de punta a punta: CSV ciego antes del juez, juez falso sobre todos los ítems e informe."""
    monkeypatch.setattr(pre, "macro_keys", lambda study, split, log=print: [])
    ev = EvalSet(study.paths.d("data", "eval", "confirm"))
    p = pre.pdir(study, "textos", "confirm.jsonl")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for i in range(ev.n):
                f.write(json.dumps({"i": i, "q": f"q{i}", "a": f"a{i}", "negs": [f"n{i}_{j}" for j in range(10)]}) + "\n")
    items, keys = pre.b1_items(study, **QUIET)
    assert keys == [] and len(items) == 2 * ev.n
    pre.paso_etiquetas(study, **QUIET)
    info = pre.paso_juez(study, backend="fake", batch=32, **QUIET)
    assert info["n_items"] == len(items) and info["n_etiquetados"] == len(items)
    r = pre.paso_calidad(study, **QUIET)
    assert r["tasas_juez"]["tgem"]["n_items"] == ev.n and r["tasas_juez"]["tgem"]["items_del_set"] == ev.n
    assert "sin ítems con falsos negativos (juez)" in r["sensibilidad"]
