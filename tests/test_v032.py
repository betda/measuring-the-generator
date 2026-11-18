"""Pruebas de v0.3.2: construcciones del panel T-gen (S, P, E), piloto automático y positivos propios
(ENMIENDA_03 §11)."""
import json
import os

import numpy as np
import pytest

from clm_losses import f0
from clm_losses.evaluate import cos_with_external, evaluate_run
from clm_losses.generators import (BUILDS, K_KEEP, active_build, audit_generator, build_candidates, build_path,
                                   canonical, choose_positive, construct_sets, contains_gold, decide_build,
                                   embed_generator, export_qc, freeze_panel, gen_dir, gen_status, is_edit_of,
                                   item_kind, load_gen, load_panel_sets, match_format, needs_embedding, para_target,
                                   pick_by_length, qc_exclusions, run_tasks, save_build, set_dir, synthetic_panel,
                                   synthetic_pilot)
from clm_losses.pipeline import base_cfg, load_bundle, setup
from clm_losses.smoke import SMOKE_LOSS_PARAMS
from clm_losses.train import Trainer, run_training
from clm_losses.utils import load_json, read_jsonl, save_json

QUIET = dict(log=lambda *a, **k: None)


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


@pytest.fixture(scope="module")
def bundle(study):
    return load_bundle(study, localize=False)


# --------------------------------------------------------------------------------------
# Reglas de texto. Los casos imitan la forma de los del panel real del 2026-10-02; los textos son inventados
# (el repositorio público no redistribuye textos de Nemotron-CC).
def test_item_kind_on_real_cases():
    assert item_kind("Is the river longer than the canal?", "No") == "si_no"
    assert item_kind("Was there a measurable effect?", "No, the effect was negligible.") == "si_no"
    assert item_kind("Which of these is NOT a primary color? A) Red B) Blue C) Yellow", "D) Green") == "opcion_multiple"
    assert item_kind("Is the survey about online or offline learning?",
                     "The survey is about online learning.") == "binaria"
    assert item_kind("Which kind of question asks whether a claim is true or false?", "Yes/No questions.") == "ok"
    assert item_kind("How many chapters does the report have?", "12.") == "ok"
    assert item_kind("What did Prof. B. Jones measure?", "A higher yield.") == "ok"


def test_format_rules():
    assert match_format("Lyon", "Paris.") == "Lyon."
    assert match_format("lyon (the second city)", "Paris") == "Lyon"
    assert match_format("MVC-Sens(Q/R)", "MVC-Sens(P/S)") == "MVC-Sens(Q/R)"   # paréntesis sin espacio: se queda
    assert match_format("Is it Lyon?", "Paris") is None
    assert canonical('"lyon (france)."') == "Lyon"
    assert canonical("27.25%.") == "27.25%"
    assert canonical("What is it?") is None
    assert contains_gold("6 pages", "6") and contains_gold("The text has 35 references, not 37.", "37.")
    assert not contains_gold("16 pages", "6") and not contains_gold("Paris is big and old", "Paris is big and old too")
    assert is_edit_of("STAT3 overexpression enhances DNA repair.", "STAT3 overexpression promotes chemo-resistance.")
    assert not is_edit_of("The STAT3 overexpression promotes chemo-resistance.",
                          "STAT3 overexpression promotes chemo-resistance.")       # sólo cambia una palabra vacía
    assert pick_by_length(["aa", "aaaaaaaaaa", "aaaa", "a"], "aaa", k=2) == ["aa", "aaaa"]


def test_build_candidates_s_and_e():
    gold = "Two hydrogen bonds."
    recs = [{"attempt": 0, "negatives": ["One hydrogen bond", "three hydrogen bonds", "Two hydrogen bonds",
                                         "What about four?", "Four hydrogen bonds (in the dimer)", "Five",
                                         "The two hydrogen bonds"]}]
    negs, st = build_candidates(gold, recs, k=10)
    assert negs == ["One hydrogen bond.", "Three hydrogen bonds.", "Four hydrogen bonds.", "Five."]
    assert st["duplicado"] == 1 and st["no_respuesta"] == 1 and st["contiene_oro"] == 1
    gold = "Two hydrogen bonds link the chains."
    recs = [{"attempt": 0, "negatives": ["Three hydrogen bonds link the chains", "The two hydrogen bonds link the chains",
                                         "Two ionic bonds link the chains", "Two hydrogen bonds link chains"]}]
    negs_e, st_e = build_candidates(gold, recs, k=10, require_edit=True)
    # sin sacar ni poner palabras de contenido no es una edición: es el oro con otras palabras vacías
    assert negs_e == ["Three hydrogen bonds link the chains.", "Two ionic bonds link the chains."]
    assert st_e["no_es_edicion"] == 2
    assert len(build_candidates(gold, recs, k=10)[0]) == 4         # en S sí quedan (no se pide edición)


def test_para_target_balances_length_rank():
    rs = np.random.default_rng(0)
    ranks = []
    for i in range(3000):
        negs = ["x" * int(n) for n in rs.integers(10, 60, K_KEEP)]
        t, tc, tw = para_target("mistral7b", i, negs)
        assert 0 <= t <= K_KEEP and tw >= 1
        ranks.append(sum(1 for x in negs if len(x) > tc))
        assert abs(ranks[-1] - t) <= 1 or len(set(len(x) for x in negs)) < K_KEEP
    h = np.bincount(ranks, minlength=K_KEEP + 1) / len(ranks)
    assert h.max() < 0.15 and h.min() > 0.05                    # ~uniforme en 0..10 (1/11 = 0.091)
    assert para_target("mistral7b", 7, negs) == para_target("mistral7b", 7, negs)          # determinista
    negs = ["a" * n for n in (10, 12, 14, 16, 18, 20, 22, 24, 26, 28)]
    paras = ["b" * 9, "b" * 19, "b" * 40, "Paris", "a" * 18]
    assert choose_positive(paras, "Paris", negs, t=5, target_chars=19.0) == "B" + "b" * 18   # 5 más largos
    assert choose_positive(paras, "Paris", negs, t=10, target_chars=8.0) == "B" + "b" * 8
    assert choose_positive(["Paris.", "paris"], "Paris", negs, 3, 20.0) is None             # copias del oro no


def test_decide_build_rule():
    r = {"S": {"a": 0.45, "b": 0.50, "c": 0.40}, "P": {"a": 0.22, "b": 0.25, "c": 0.30}, "E": {"a": 0.15, "b": 0.20}}
    d = decide_build(r)
    assert d["build"] == "E" and d["medianas"]["P"] == 0.25 and abs(d["medianas"]["E"] - 0.175) < 1e-9
    r["E"] = {"a": 0.245, "b": 0.26}                              # empate (< 1 pp) con P: gana la más barata
    assert decide_build(r)["build"] == "P"
    r["E"] = {"a": 0.01}                                          # 1 solo generador: no compite
    assert decide_build(r)["build"] == "P"
    assert decide_build({"S": {"a": None}})["build"] is None


# --------------------------------------------------------------------------------------
# Generación (backend falso) -> construcciones S, P y E -> piloto -> panel con positivos propios
class FakeEncoder:
    def __init__(self, test, dim):
        self.dim = dim

    def tokenize(self, texts, keep="head"):
        return [[hash(t) % 50000, len(t)] + [7] * (len(t) // 5) for t in texts]

    def embed_ids(self, ids):
        out = np.zeros((len(ids), self.dim), np.float32)
        for r, x in enumerate(ids):
            rs = np.random.default_rng(abs(hash(tuple(x))) % (2 ** 32))
            out[r] = rs.standard_normal(self.dim)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _lookup(test):
    def lookup_fn(paths, docs, which, columns):
        return [{"doc_id": test.doc_ids[i], "pair_idx": int(test.pair_idx[i]), "question_ids": [1, 2, 1000 + i],
                 "answer_ids": [5, 6, 7, 1000 + i]} for i in range(test.n) if test.doc_ids[i] in docs]
    return lookup_fn


def test_three_builds_pilot_and_panel_with_own_positives(study, bundle, tmp_path):
    test = bundle.test
    dec = lambda ids: " ".join(f"t{x}" for x in ids)          # noqa: E731
    kw = dict(backend="fake", batch=16, decoder=dec, lookup_fn=_lookup(test), check=False)
    # "para" sin negativos: se niega y lo dice
    st = run_tasks(study, test, "mistral7b", ["para"], n_items=40, **kw, **QUIET)
    assert st == {"para": "sin_negativos"}
    st = run_tasks(study, test, "mistral7b", ["neg", "para", "edit"], n_items=40, **kw, **QUIET)
    assert st == {"neg": "generado", "para": "generado", "edit": "generado"}
    gd = gen_dir(study, "mistral7b")
    assert os.path.exists(os.path.join(gd, "raw_edit.jsonl")) and os.path.exists(os.path.join(gd, "para.jsonl"))
    tg = load_json(os.path.join(gd, "para_targets.json"))
    assert tg and all(0 <= v["t"] <= K_KEEP for v in tg.values())
    # un piloto con menos ítems no achica lo ya generado
    run_tasks(study, test, "mistral7b", ["neg"], n_items=20, **kw, **QUIET)
    assert load_json(os.path.join(gd, "gen_info.json"))["n_target"] == 40
    enc = FakeEncoder(test, bundle.hidden)
    for b in BUILDS:
        sub = f"pilot_{b}"
        assert needs_embedding(study, "mistral7b", "test", b, sub, 40)
        e = embed_generator(study, test, "mistral7b", enc, build=b, sub=sub, n_items=40, **QUIET)
        assert e["build"] == b and e["items_kept"] > 0 and not needs_embedding(study, "mistral7b", "test", b, sub, 40)
        g = load_gen(study, "mistral7b", sub=sub)
        assert g is not None and (g.pos is not None) == (b == "P")
        kept = read_jsonl(os.path.join(set_dir(study, "mistral7b", sub=sub), "kept.jsonl"))
        if b == "P":
            assert all(r["positive"] for r in kept) and g.pos.shape == (len(g.items), bundle.hidden)
            assert all(not r["negatives"][0].endswith(".") for r in kept)                # formato canónico
        else:
            assert all(r["positive"] is None for r in kept)
    rows, stats = construct_sets(study, "mistral7b", "S", n_limit=20)
    assert stats["items_generated"] <= 20
    # piloto sintético: elige la construcción sin atajo (P) y la deja fija
    lp = os.path.join(study.paths.d("data", "eval", "test"), "lengths.npz")
    if not os.path.exists(lp):
        rs = np.random.default_rng(0)
        np.savez(lp, gold=rs.integers(10, 60, test.n), neg=rs.integers(10, 60, (test.n, 10)))
    assert active_build(study) == "S"
    blind = f0.train_blind(bundle, study.paths.d("f0", "F0.6_shortcuts"), **QUIET)
    rec = synthetic_pilot(study, bundle, blind=blind, n=200, **QUIET)
    assert rec["build"] == "P" and active_build(study) == "P"
    assert rec["medianas"]["S"] > rec["medianas"]["P"]
    # el set S ya embebido de mistral7b no cuenta para el panel P: hay que rehacerlo
    embed_generator(study, test, "mistral7b", enc, build="S", **QUIET)
    assert load_gen(study, "mistral7b") is None and gen_status(study, "mistral7b") == "sin_embeber"
    assert needs_embedding(study, "mistral7b", n_items=40)
    embed_generator(study, test, "mistral7b", enc, n_items=40, **QUIET)
    g = load_gen(study, "mistral7b")
    assert g is not None and g.pos is not None
    assert os.path.exists(os.path.join(gen_dir(study, "mistral7b"), "history", "S_v3_embed_info.json"))
    # panel sintético con la construcción P: positivos propios en la evaluación
    synthetic_panel(study, test, n_items=200)
    for k in ("syn_limpio", "syn_debil", "syn_largo", "syn_fuerte", "syn_control", "mistral7b"):
        audit_generator(study, bundle, k, blind=blind, steps=100, **QUIET)
    panel = freeze_panel(study, **QUIET)
    assert panel["build"] == "P" and "syn_fuerte" not in panel["macro"]
    sets = load_panel_sets(study)
    assert all(v[3] is not None for v in sets.values())
    cfg = base_cfg(study, "infonce", stages=["pre"], max_steps_pre=6, eval_every=3)
    rd = study.paths.d("runs", "tmp_pos")
    run_training(cfg, bundle, rd, ecfg=study.eval, evaluate_test=False)
    tr = Trainer(cfg, bundle, rd)
    tr.load_best("pre")
    it, negs, sig, pos = sets["tgen_syn_limpio"]
    c_pos = cos_with_external(tr.heads, test, it, negs, tr.device, pos=pos)
    c_gold = cos_with_external(tr.heads, test, it, negs, tr.device)
    assert np.allclose(c_pos[:, 1:], c_gold[:, 1:]) and not np.allclose(c_pos[:, 0], c_gold[:, 0])
    c_same = cos_with_external(tr.heads, test, it, negs, tr.device, pos=np.asarray(test.a_all())[it])
    assert np.allclose(c_same, c_gold, atol=1e-5)
    m = evaluate_run(tr.heads, tr.loss, test, bundle.val, os.path.join(rd, "eval"), tr.device, study.eval,
                     extra_sets=sets, **QUIET)
    assert "tgen_syn_limpio" in m and "shuf_top1" in m["tgen_syn_limpio"]
    # QC: CSV con filas de positivos y de negativos; exclusión automática con marcas suficientes
    p = export_qc(study, keys=["syn_limpio", "mistral7b"], n_items=10)
    import pandas as pd
    df = pd.read_csv(p)
    assert {"positivo", "negativo"} <= set(df["tipo"])
    assert qc_exclusions(study) == {}
    df.loc[df["generador"] == "syn_limpio", "es_correcta(0/1)"] = 0
    df.loc[(df["generador"] == "syn_limpio") & (df["tipo"] == "positivo"), "es_correcta(0/1)"] = 1
    sel = df.index[(df["generador"] == "syn_limpio") & (df["tipo"] == "negativo")][:8]
    df.loc[sel, "es_correcta(0/1)"] = 1                       # 8 de 100 negativos en realidad correctos: > 5 %
    df.to_csv(p, index=False)
    ex = qc_exclusions(study)
    assert list(ex) == ["syn_limpio"] and "8.0%" in ex["syn_limpio"]
    assert export_qc(study, keys=["syn_limpio"], n_items=5).endswith("_nuevo.csv")    # no pisa las marcas
    p2 = freeze_panel(study, exclude_qc=ex, **QUIET)
    assert "syn_limpio" not in p2["macro"] and p2["generators"]["syn_limpio"]["excluido_qc"]
    os.remove(build_path(study))                              # deja el estudio como estaba para otros módulos
