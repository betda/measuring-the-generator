"""Pruebas de v0.3 (enmienda A1 + panel T-gen)."""
import dataclasses
import importlib
import json
import math
import os
import shutil
import sys

import numpy as np
import pytest
import torch

from clm_losses import f0
from clm_losses.config import V03_TRAIN_DEFAULTS, TrainConfig, clm_lr, recipe_overrides
from clm_losses.data import EvalSet, Split, Stream, train_masks
from clm_losses.evaluate import evaluate_run, expected_eval_keys, quick_val, rank_metrics, shuffle_perm
from clm_losses.generators import (PANEL, PANEL_BY_KEY, embed_generator, freeze_panel, harmony_final,
                                   load_panel, macro_for_run, parse_negatives, resolve_keys, run_generator,
                                   synthetic_panel, audit_generator, max_new_tokens_for, build_prompt)
from clm_losses.pipeline import (add_missing_evals, base_cfg, iter_run_dirs, load_bundle, make_store,
                                 phase_settings, run_all_vs_all, setup)
from clm_losses.smoke import SMOKE_LOSS_PARAMS
from clm_losses.train import Trainer, run_fingerprint, run_training, stage_lr
from clm_losses.utils import load_json, read_done, read_jsonl, save_json

QUIET = dict(log=lambda *a, **k: None)
V02 = "/tmp/claude-0/-home-claude/41baa85b-be00-526a-8371-aa9735e40df1/scratchpad/v02_backup/clm_losses"


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
def test_fingerprint_backward_compatible(study, bundle, tmp_path):
    """Una config de v0.2 tiene la misma huella con el código v0.3 (las corridas viejas no se rechazan)."""
    if not os.path.isdir(V02):
        pytest.skip("sin copia de v0.2")
    dst = tmp_path / "pkg"
    shutil.copytree(V02, dst / "clm_losses_v02")
    sys.path.insert(0, str(dst))
    try:
        old_cfg_mod = importlib.import_module("clm_losses_v02.config")
        old_train = importlib.import_module("clm_losses_v02.train")
        for arm, kw in (("infonce", {}), ("infonce_hn", {"loss_params": {"hn_denominator": "hn_only"}}),
                        ("infonce_hn", {"hn_from_start": True, "stages": ["mid"]})):
            new = base_cfg(study, arm, **kw)
            d = {k: v for k, v in dataclasses.asdict(new).items() if k not in V03_TRAIN_DEFAULTS}
            old = old_cfg_mod.TrainConfig(**d)
            assert run_fingerprint(new, 0.25, bundle) == old_train.run_fingerprint(old, 0.25, bundle)
        # con un campo nuevo distinto del defecto, la huella sí cambia
        assert run_fingerprint(base_cfg(study, "infonce", lr_schedule="const"), 1.0, bundle) != \
            run_fingerprint(base_cfg(study, "infonce"), 1.0, bundle)
    finally:
        sys.path.remove(str(dst))
        for m in [m for m in sys.modules if m.startswith("clm_losses_v02")]:
            del sys.modules[m]


def test_stream_grouping_unchanged_for_pilot_batches_and_valid_for_big_batches(study):
    store = make_store(study)
    sp = Split.load(study.paths.d("data", "splits"))
    masks = train_masks(store, sp, study.data.pre_chunks, study.paths.d("data", "splits"), "pre")
    d = study.data
    st = Stream(store, d.pre_chunks, masks, 256, d.pre_block_rows, d.pre_group_blocks, seed=1)
    assert st.group_blocks == d.pre_group_blocks             # batch del smoke: no cambia el agrupamiento
    big = Stream(store, d.pre_chunks, masks, 2048, d.pre_block_rows, d.pre_group_blocks, seed=1)
    assert big.group_blocks > d.pre_group_blocks
    nb = big.epoch_batches(0)
    assert nb >= 3 and nb == sum(1 for _ in big.iterate(0))
    with pytest.raises(ValueError):
        Stream(store, d.pre_chunks, masks, 10 ** 7, d.pre_block_rows, d.pre_group_blocks, seed=1)


def test_stage_lr_recipes(study):
    t = TrainConfig()
    assert math.isclose(stage_lr(t, "pre", 2048), clm_lr(1536, 2048))
    assert math.isclose(stage_lr(t, "mid", 512), clm_lr(1536, 512))     # F1 piloto: sin cambios
    clm = TrainConfig(batch_pre=16384, batch_mid=16384, lr_mid_scale=1e-3 / clm_lr(1536, 16384))
    assert math.isclose(stage_lr(clm, "pre", 16384), 0.0065319726474218085, rel_tol=1e-9)   # cfg público de CLM
    assert math.isclose(stage_lr(clm, "mid", 16384), 1e-3, rel_tol=1e-9)
    start = dataclasses.replace(clm, hn_from_start=True)
    assert math.isclose(stage_lr(start, "mid", 16384), 0.0065319726474218085, rel_tol=1e-9)  # desde cero
    rov = recipe_overrides(study, "clm")
    assert rov["lr_schedule"] == "const" and rov["weight_decay"] == 0.0 and rov["evals_per_stage"] == 20
    assert recipe_overrides(study, "actual") == {}


def test_quick_val_reports_valmine(study, bundle):
    assert bundle.val_mine_idx is not None and bundle.val_mine_idx.shape == (bundle.val.n, study.eval.n_mine)
    tr = Trainer(base_cfg(study, "infonce"), bundle, study.paths.d("runs", "tmp_qv"))
    m = tr.validate()
    assert {"val_top1", "valmine_top1", "valmine_mrr"} <= set(m)
    # el pool no contiene el mismo documento
    docs = np.asarray(bundle.val.doc_ids)
    assert not (docs[bundle.val_mine_idx] == docs[:, None]).any()


def test_const_schedule_and_valmine_selection(study, bundle):
    cfg = base_cfg(study, "infonce", stages=["pre"], lr_schedule="const", max_steps_pre=12, evals_per_stage=4,
                   select_metric="valmine_top1", patience=100)
    rd = study.paths.d("runs", "tmp_const")
    r = run_training(cfg, bundle, rd, ecfg=study.eval, evaluate_test=False)
    sj = load_json(os.path.join(rd, "stage_pre.json"))
    lrs = {round(h["lr"], 12) for h in sj["history"] if "lr" in h}
    assert len(lrs) == 1                                         # LR constante
    assert sj["eval_every"] == 3 and sj["select_metric"] == "valmine_top1"
    best = max(h["valmine_top1"] for h in sj["history"] if not h.get("init"))
    assert math.isclose(r["val_best"], best)


def test_init_from_copies_pre_weights(study, bundle):
    pre = base_cfg(study, "infonce", stages=["pre"], max_steps_pre=10, eval_every=5)
    rd_pre = study.paths.d("runs", "tmp_init", "pre")
    run_training(pre, bundle, rd_pre, ecfg=study.eval, evaluate_test=False)
    mid = base_cfg(study, "infonce_hn", stages=["mid"], max_steps_mid=1, init_from=rd_pre,
                   loss_params={"hn_denominator": "hn_plus_inbatch"})
    tr = Trainer(mid, bundle, study.paths.d("runs", "tmp_init", "mid"))
    tr._init_from()
    st = torch.load(os.path.join(rd_pre, "best_pre.pt"), weights_only=False)
    for k, v in st["heads"].items():
        assert torch.equal(tr.heads.state_dict()[k].cpu(), v.cpu())
    r = run_training(mid, bundle, study.paths.d("runs", "tmp_init", "mid2"), ecfg=study.eval, evaluate_test=False)
    assert r["status"] == "ok"
    meta = load_json(study.paths.d("runs", "tmp_init", "mid2", "config.json"))
    assert meta.get("init_from_sha256")


def test_evaluate_run_adds_v03_blocks_without_touching_old(study, bundle):
    cfg = base_cfg(study, "infonce", stages=["pre"], max_steps_pre=10, eval_every=5)
    rd = study.paths.d("runs", "tmp_eval")
    run_training(cfg, bundle, rd, ecfg=study.eval, evaluate_test=False)
    tr = Trainer(cfg, bundle, rd)
    tr.load_best("pre")
    ev = os.path.join(rd, "eval")
    # simula una evaluación de v0.2: sólo tgem/trand/tmine sin barajados ni pools nuevos
    m0 = evaluate_run(tr.heads, tr.loss, bundle.test, bundle.val, ev, tr.device, study.eval, pools_dir=None,
                      shuffle=False, **QUIET)
    from clm_losses.evaluate import cos_with_pool
    I = np.load(os.path.join(bundle.pools_dir, "mine_idx.npy"))
    old = dict(m0)
    old["tmine"] = rank_metrics(cos_with_pool(tr.heads, bundle.test, I, tr.device))[0]
    save_json(old, os.path.join(ev, "metrics.json"))
    before = json.dumps(old["tgem"], sort_keys=True)
    m = evaluate_run(tr.heads, tr.loss, bundle.test, bundle.val, ev, tr.device, study.eval,
                     pools_dir=bundle.pools_dir, **QUIET)
    for k in expected_eval_keys(bundle.pools_dir):
        assert k in m, k
    assert json.dumps(m["tgem"], sort_keys=True) == before          # lo viejo no se rehace
    assert os.path.exists(os.path.join(ev, "scores_tmine.npy"))
    # estados barajados = la permutación de las celdas 5.4/5.5
    assert np.array_equal(shuffle_perm(bundle.test.n), np.random.default_rng(0).permutation(bundle.test.n))
    c_real = np.load(os.path.join(ev, "correct_tgem.npy"))
    c_shuf = np.load(os.path.join(ev, "correct_tgem_shuf.npy"))
    assert c_real.shape == c_shuf.shape
    # T-abst-mine: misma construcción (oro + 9 vs 10) que T-abst
    assert 0.0 <= m["tabst_mine"]["abst_auroc_maxcos"] <= 1.0
    assert m["tmine50"]["n"] == bundle.test.n


# --------------------------------------------------------------------------------------
def test_parse_negatives_and_harmony():
    txt = ("Here are 14 incorrect answers:\n1. The moon is made of iron.\n2) Water boils at 50 C\n"
           "- \"Paris is in Spain\"\n• Answer 4: Light is slower than sound\nThe correct answer\n"
           "The moon is made of iron.\n\n<think>razonamiento</think>Cats can fly")
    out = parse_negatives(txt, gold="The correct answer", k=14)
    assert out == ["The moon is made of iron.", "Water boils at 50 C", "Paris is in Spain",
                   "Light is slower than sound", "Cats can fly"]
    h = ("<|channel|>analysis<|message|>We need wrong answers.<|end|><|start|>assistant<|channel|>final"
         "<|message|>Alpha claim\nBeta claim<|return|>")
    assert harmony_final(h) == "Alpha claim\nBeta claim"
    assert parse_negatives(h, "x", harmony=True) == ["Alpha claim", "Beta claim"]
    assert parse_negatives("<|channel|>analysis<|message|>sin final", "x", harmony=True) == []
    # si el oro empieza con "Answer:", los negativos conservan ese prefijo (si no, sería un atajo de estilo)
    assert parse_negatives("Answer: Rome is the capital\n2. Answer: Madrid is", "Answer: Paris is the capital") == \
        ["Answer: Rome is the capital", "Answer: Madrid is"]
    assert parse_negatives("Answer: Rome is the capital", "Paris") == ["Rome is the capital"]
    assert parse_negatives("ok\x00 line�", "x") == ["ok line"]
    assert max_new_tokens_for(20) >= 256 and max_new_tokens_for(20, harmony=True) == max_new_tokens_for(20) + 1024
    p = build_prompt("¿Q?", "uno dos tres", 14)
    assert "around 3 words" in p and "Return exactly 14 lines" in p


def test_panel_registry():
    keys = [g.key for g in PANEL]
    assert len(keys) == len(set(keys)) == 23          # 16 originales + 7 grandes (v0.3.1, ENMIENDA_03 §10)
    assert {g.role for g in PANEL} == {"primario", "control_gemini", "control_qwen", "control_oro"}
    assert set(resolve_keys("sin_gated")) == {g.key for g in PANEL if not g.gated}
    assert PANEL_BY_KEY["qwen3_30b"].n_items == 1000 and PANEL_BY_KEY["gptoss20b"].harmony
    with pytest.raises(KeyError):
        resolve_keys("no_existe")


class FakeEncoder:
    """Tokeniza por palabras y embebe con un hash (determinista); una respuesta que sea el oro tiene coseno 1."""

    def __init__(self, test, dim):
        self.test, self.dim = test, dim
        self.gold_text = {}

    def tokenize(self, texts, keep="head"):
        return [[hash(w) % 50000 for w in t.split()][:2048] or [0] for t in texts]

    def embed_ids(self, ids):
        out = np.zeros((len(ids), self.dim), np.float32)
        for r, x in enumerate(ids):
            rs = np.random.default_rng(abs(hash(tuple(x))) % (2 ** 32))
            out[r] = rs.standard_normal(self.dim)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_generation_pipeline_with_fake_backend(study, bundle):
    test = bundle.test

    def lookup_fn(paths, docs, which, columns):
        return [{"doc_id": test.doc_ids[i], "pair_idx": int(test.pair_idx[i]), "question_ids": [1, 2, 1000 + i],
                 "answer_ids": [5, 6, 7, 1000 + i]} for i in range(test.n) if test.doc_ids[i] in docs]
    dec = lambda ids: " ".join(f"t{x}" for x in ids)
    info = run_generator(study, test, "mistral7b", n_items=40, batch=16, backend="fake", decoder=dec,
                         lookup_fn=lookup_fn, check=False, **QUIET)
    assert info["status"] == "generado" and info["items_requested"] == 40
    raw = read_jsonl(os.path.join(study.paths.d("data", "eval", "tgen", "test", "mistral7b"), "raw.jsonl"))
    att = {}
    for r in raw:
        att[r["i"]] = att.get(r["i"], 0) + 1
        assert f"t5 t6 t7 t{1000 + r['i']}" not in [n.lower() for n in r["negatives"]]   # nunca la copia del oro
    assert max(att.values()) == 2                                           # hubo reintentos
    again = run_generator(study, test, "mistral7b", n_items=40, batch=16, backend="fake", decoder=dec,
                          lookup_fn=lookup_fn, check=False, **QUIET)
    assert len(read_jsonl(os.path.join(study.paths.d("data", "eval", "tgen", "test", "mistral7b"), "raw.jsonl"))) == len(raw)
    assert again["status"] == "generado"
    e = embed_generator(study, test, "mistral7b", FakeEncoder(test, bundle.hidden), **QUIET)
    assert e["items_kept"] > 0
    negs = np.load(os.path.join(study.paths.d("data", "eval", "tgen", "test", "mistral7b"), "negs.npy"))
    assert negs.shape[1:] == (10, bundle.hidden) and negs.dtype == np.float16


def test_synthetic_panel_audit_macro_and_stats(study, bundle):
    # longitudes sintéticas para que la auditoría ejercite las heurísticas de longitud
    lp = os.path.join(study.paths.d("data", "eval", "test"), "lengths.npz")
    if not os.path.exists(lp):
        rs = np.random.default_rng(0)
        np.savez(lp, gold=rs.integers(10, 60, bundle.test.n), neg=rs.integers(10, 60, (bundle.test.n, 10)))
    synthetic_panel(study, bundle.test, n_items=200)
    blind = f0.train_blind(bundle, study.paths.d("f0", "F0.6_shortcuts"), **QUIET)
    for k in ("syn_limpio", "syn_debil", "syn_largo", "syn_fuerte", "syn_control"):
        a = audit_generator(study, bundle, k, blind=blind, steps=150, **QUIET)
        assert a["banda"] in ("despreciable", "moderado", "alto")
    aud = {k: load_json(os.path.join(study.paths.d("data", "eval", "tgen", "test", k), "audit.json"))
           for k in ("syn_limpio", "syn_fuerte", "syn_largo")}
    assert aud["syn_fuerte"]["blind_own_cv"] > aud["syn_limpio"]["blind_own_cv"] + 0.1   # el artefacto se detecta
    assert aud["syn_largo"]["longest"] > 0.5                                             # sesgo de longitud
    panel = freeze_panel(study, **QUIET)
    assert "syn_fuerte" not in panel["macro"] and "syn_control" not in panel["macro"]
    assert set(panel["macro"]) <= {"syn_limpio", "syn_debil", "syn_largo", "mistral7b"}
    # dos corridas cortas evaluadas con el panel, macro y todos contra todos
    rds = []
    for arm in ("infonce", "infonce_hn"):
        for s in (0, 1):
            cfg = base_cfg(study, arm, seed=s, max_steps_pre=8, max_steps_mid=4, eval_every=4)
            rd = study.paths.d("runs", "F1b", arm, f"seed_{s}")
            run_training(cfg, bundle, rd, ecfg=study.eval, evaluate_test=False)
            rds.append(rd)
    from clm_losses.utils import mark_done
    for rd in rds:
        d = read_done(rd)
        mark_done(rd, d)
    fin = {rd: read_done(rd)["finished_at"] for rd in rds}
    assert add_missing_evals(study, bundle, ["F1b"], **QUIET) == 4
    for rd in rds:
        assert read_done(rd)["finished_at"] == fin[rd] and "evals_added_at" in read_done(rd)
        mm = macro_for_run(rd, load_panel(study))
        assert mm is not None and mm["n"] == panel["n_macro_items"]
        assert os.path.exists(os.path.join(rd, "eval", "correct_tgenmacro_shuf.npy"))
    assert add_missing_evals(study, bundle, ["F1b"], **QUIET) == 0     # idempotente
    # secundario (§10): macro por tamaño del generador y su interacción en EXP-006
    pnl = load_panel(study)
    assert pnl["macro_por_tamano"]["grande"]["keys"] == ["syn_limpio"]
    assert pnl["macro_por_tamano"]["chico"]["keys"] == ["syn_debil"]
    for rd in rds:
        for nm in ("tgenbig", "tgensmall"):
            assert os.path.exists(os.path.join(rd, "eval", f"correct_{nm}.npy"))
    from clm_losses.analysis import transfer_report
    tr = transfer_report(study, bundle, groups={"A": rds[:2], "B": rds[2:]}, contrasts=[("B", "A")], B=200, **QUIET)
    mets = set(tr["contrastes"]["métrica"])
    assert {"T-gen (macro de grandes, secundario)", "T-gen (macro de chicos, secundario)",
            "interacción: efecto en grandes − efecto en chicos"} <= mets
    from clm_losses.stats import all_vs_all, collect_runs
    from clm_losses.pipeline import items_map_of
    runs = collect_runs(study.paths.runs, "F1b", ["infonce", "infonce_hn"])
    rep = all_vs_all(runs, ["tmine", "tgenmacro", "tgem_shuf"], bundle.test.doc_ids,
                     study.paths.d("results", "tmp_ava"), B=200, items_map=items_map_of(bundle, panel=load_panel(study)),
                     abst_sets=("tmine", "tgem"), **QUIET)
    assert {"tmine_top1", "tgenmacro_top1", "tgem_shuf_top1", "abst_auroc_tmine"} <= set(rep["metrics"])


def test_phase_settings_require_recipe(study):
    with pytest.raises(RuntimeError):
        phase_settings(study, "F1b")
    save_json({"receta": "clm"}, study.paths.d("config", "RECETA_F1b.json"))
    try:
        s = phase_settings(study, "F1b")
        assert s["select_metric"] == "valmine_top1" and s["loss_params"]["hn_denominator"] == "hn_plus_inbatch"
        assert s["lr_schedule"] == "const"
        c = base_cfg(study, "infonce_hn", phase="F1b")
        assert c.loss_params["hn_denominator"] == "hn_plus_inbatch" and "exp_dim" in c.loss_params
        from clm_losses.hpo import sample_trials
        assert all("loss.hn_denominator" not in t for t in sample_trials("infonce_hn", 8, 1, "F1b"))
        assert any("loss.hn_denominator" in t for t in sample_trials("infonce_hn", 8, 1))     # F1 piloto
    finally:
        os.remove(study.paths.d("config", "RECETA_F1b.json"))


# --------------------------------------------------------------------------------------
# Regresiones de la revisión independiente (2026-10-01)
def test_parse_keeps_numbers_and_real_answers():
    lines = ["2.5 billion years ago", "12.5% of the population", "-40 degrees Celsius", "10 - 20 percent of cases",
             "3:00 pm on Tuesday", "1. 2.5 million people", "Sure, the capital is Rome", "OK, it is 42",
             "Here are 14 incorrect answers:", "Of course it melts at 100 C"]
    out = parse_negatives("\n".join(lines), gold="something else", k=20)
    assert out == ["2.5 billion years ago", "12.5% of the population", "-40 degrees Celsius", "10 - 20 percent of cases",
                   "3:00 pm on Tuesday", "2.5 million people", "Sure, the capital is Rome", "OK, it is 42",
                   "Of course it melts at 100 C"]


def test_append_jsonl_recovers_truncated_line(tmp_path):
    from clm_losses.utils import append_jsonl
    p = str(tmp_path / "x.jsonl")
    append_jsonl(p, {"i": 1})
    with open(p, "a") as f:
        f.write('{"i": 2, "trunc')                     # corte a mitad de una línea
    append_jsonl(p, {"i": 3})
    assert [r["i"] for r in read_jsonl(p)] == [1, 3]


def test_macro_requires_complete_generation_and_clears_stale(study, bundle, tmp_path):
    from clm_losses.generators import gen_dir, load_gen
    keys = synthetic_panel(study, bundle.test, n_items=200)
    blind = f0.train_blind(bundle, study.paths.d("f0", "F0.6_shortcuts"), **QUIET)
    for k in keys:
        audit_generator(study, bundle, k, blind=blind, steps=100, **QUIET)
    # una generación interrumpida (status != generado) no entra al macro aunque tenga negativos y auditoría
    gi = os.path.join(gen_dir(study, "syn_limpio"), "gen_info.json")
    info = load_json(gi)
    save_json({**info, "status": "fallido"}, gi)
    p = freeze_panel(study, exclude_qc={"syn_debil": "QC manual"}, **QUIET)
    assert "syn_limpio" not in p["macro"] and "syn_debil" not in p["macro"]
    assert p["generators"]["syn_debil"]["excluido_qc"] == "QC manual"
    save_json(info, gi)
    p = freeze_panel(study, **QUIET)
    assert "syn_limpio" in p["macro"] and set(p["todos"]) >= set(p["macro"]) and "syn_fuerte" in p["todos"]
    # macro_for_run: si la corrida no cubre los ítems del macro, no se usa (y no quedan archivos viejos)
    rd = str(tmp_path / "run")
    ev = os.path.join(rd, "eval")
    os.makedirs(ev)
    np.save(os.path.join(ev, "correct_tgenmacro.npy"), np.ones(3, np.float32))     # archivo viejo
    save_json({"version": "vieja"}, os.path.join(ev, "tgenmacro.json"))
    for k in p["macro"]:
        it = load_gen(study, k)[0][:5]                                              # cubre sólo 5 ítems
        np.save(os.path.join(ev, f"items_tgen_{k}.npy"), it)
        np.save(os.path.join(ev, f"correct_tgen_{k}.npy"), np.ones(len(it), np.float32))
    assert macro_for_run(rd, p) is None
    assert not os.path.exists(os.path.join(ev, "correct_tgenmacro.npy"))


def test_extra_set_signature_forces_reevaluation(study, bundle):
    cfg = base_cfg(study, "infonce", stages=["pre"], max_steps_pre=6, eval_every=3)
    rd = study.paths.d("runs", "tmp_sig")
    run_training(cfg, bundle, rd, ecfg=study.eval, evaluate_test=False)
    tr = Trainer(cfg, bundle, rd)
    tr.load_best("pre")
    ev = os.path.join(rd, "eval")
    items = np.arange(20)
    negs = np.asarray(bundle.test.negatives(items))
    m1 = evaluate_run(tr.heads, tr.loss, bundle.test, bundle.val, ev, tr.device, study.eval,
                      extra_sets={"tgen_x": (items, negs, "sig1")}, **QUIET)
    negs2 = negs[:, ::-1].copy()                       # mismos ítems, otros negativos
    c_before = np.load(os.path.join(ev, "correct_tgen_x.npy")).copy()
    evaluate_run(tr.heads, tr.loss, bundle.test, bundle.val, ev, tr.device, study.eval,
                 extra_sets={"tgen_x": (items, negs2[:, :, ::-1].copy() * 0 + negs2, "sig2")}, **QUIET)
    assert load_json(os.path.join(ev, "extra_sigs.json"))["tgen_x"] == "sig2"
    assert "tgen_x" in m1 and c_before.shape == (20,)


def test_estimate_budget_f1b_proxies(tmp_path):
    """4.17 sin corridas de F1b: con la receta 'actual' usa el piloto (misma receta); con la de CLM, pre + mid de
    F0.7b escalado por brazo con los tiempos relativos del piloto."""
    from clm_losses.pipeline import estimate_budget
    st = setup("smoke", drive_base=str(tmp_path / "d"), cache_root=str(tmp_path / "c"), mount=False)

    def done(rel, arm, tr, ev=0.0):
        rd = os.path.join(st.paths.runs, rel)
        os.makedirs(rd, exist_ok=True)
        save_json({"status": "ok", "arm": arm, "train_seconds": tr, "eval_seconds": ev}, os.path.join(rd, "DONE.json"))
        save_json({"env": {"gpu": {"name": "GPU-X"}}}, os.path.join(rd, "config.json"))

    done("F1/infonce/seed_0", "infonce", 100, 10)
    done("F1/infonce_hn/seed_0", "infonce_hn", 200, 10)
    done("F1/infonce/hpo/trial_000", "infonce", 30)
    done("F0.7b/clm/pre/seed_0", "infonce", 300, 20)
    done("F0.7b/clm/mid_inbatch/seed_0", "infonce_hn", 500, 20)
    save_json({"receta": "actual"}, st.paths.d("config", "RECETA_F1b.json"))
    b = estimate_budget(st, phase="F1b")
    assert b["medido_en"].startswith("F1 piloto (misma receta")
    t = b["tabla"].set_index("brazo")["min_por_corrida_completa"]
    assert t["infonce"] == round(110 / 60, 1) and t["infonce_hn"] == round(210 / 60, 1)
    save_json({"receta": "clm"}, st.paths.d("config", "RECETA_F1b.json"))
    b = estimate_budget(st, phase="F1b")
    assert b["medido_en"].startswith("F0.7b, receta clm") and "escalado" in b["medido_en"]
    t = b["tabla"].set_index("brazo")["min_por_corrida_completa"]
    full = 300 + 500 + 20
    assert t["infonce_hn"] == round(full / 60, 1)
    assert t["infonce"] == round(full * 110 / 210 / 60, 1)
    assert b["gpu_de_las_mediciones"] == ["GPU-X"]
