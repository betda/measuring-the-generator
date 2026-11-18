import os

import numpy as np
import pytest
import torch

from clm_losses import f0
from clm_losses.data import EvalSet, Split, Stream, make_synthetic_store, train_masks
from clm_losses.pipeline import base_cfg, load_bundle, make_store, setup
from clm_losses.smoke import SMOKE_LOSS_PARAMS
from clm_losses.train import Trainer, run_training


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    base = tmp_path_factory.mktemp("drive")
    cache = tmp_path_factory.mktemp("cache")
    st = setup("smoke", drive_base=str(base), cache_root=str(cache), mount=False,
               train_overrides={"loss_params": SMOKE_LOSS_PARAMS})
    f0.f0_4_splits(st, log=lambda *a, **k: None)
    f0.f0_5_eval_sets(st, lengths=False, qc=False, log=lambda *a, **k: None)
    return st


def test_no_leakage_and_alignment(study):
    r = f0.f0_4_splits(study)
    assert r["leaked_rows"] == 0 and r["all_aligned"]
    sp = Split.load(study.paths.d("data", "splits"))
    assert not (sp.test_docs & sp.val_docs)


def test_eval_set_matches_store(study):
    store = make_store(study)
    ev = EvalSet(study.paths.d("data", "eval", "test"))
    assert ev.n > 0
    c = study.data.hn_chunks[0]
    meta = __import__("json").load(open(study.paths.d("data", "eval", "test", f"c{c:04d}_meta.json")))
    first = np.nonzero(ev.part == 0)[0][:5]                 # ítems que vienen del primer chunk
    rows = np.asarray(meta["row"])[ev.local[first]]
    q = np.asarray(store.q(c)[rows])
    assert np.array_equal(ev.q_all()[first], q)
    assert list(ev.doc_ids[first]) == [meta["doc_id"][i] for i in ev.local[first]]
    n = ev.negatives(np.arange(3))
    assert n.shape == (3, study.data.k_neg, q.shape[1])


def test_stream_skip_is_exact(study):
    store = make_store(study)
    sp = Split.load(study.paths.d("data", "splits"))
    masks = train_masks(store, sp, study.data.pre_chunks, study.paths.d("data", "splits"), "pre")
    st = Stream(store, study.data.pre_chunks, masks, 128, 1024, 2, seed=5)
    full = [b["es"].copy() for b in st.iterate(1)]
    k = len(full) // 2 + 1
    tail = [b["es"].copy() for b in st.iterate(1, start_batch=k)]
    assert len(tail) == len(full) - k
    assert all(np.array_equal(x, y) for x, y in zip(full[k:], tail))
    assert len(full) == st.epoch_batches(1)


class Boom(Exception):
    pass


def test_resume_is_exact(study, monkeypatch):
    bundle = load_bundle(study, localize=False)
    cfg = base_cfg(study, "infonce_hn", max_steps_pre=30, max_steps_mid=20, eval_every=10,
                   ckpt_every_steps=5, patience=100)
    da = study.paths.d("runs", "test_resume", "A")
    db = study.paths.d("runs", "test_resume", "B")
    run_training(cfg, bundle, da, ecfg=study.eval, evaluate_test=False)
    orig = Trainer._step
    calls = {"n": 0}

    def boom(self, batch, step):
        calls["n"] += 1
        if calls["n"] == 38:          # corte a mitad de la etapa mid
            raise Boom()
        return orig(self, batch, step)
    monkeypatch.setattr(Trainer, "_step", boom)
    with pytest.raises(Boom):
        run_training(cfg, bundle, db, ecfg=study.eval, evaluate_test=False)
    monkeypatch.setattr(Trainer, "_step", orig)
    assert os.path.exists(os.path.join(db, "ckpt_last.pt"))
    run_training(cfg, bundle, db, ecfg=study.eval, evaluate_test=False)
    a = torch.load(os.path.join(da, "best_mid.pt"), weights_only=False)["heads"]
    b = torch.load(os.path.join(db, "best_mid.pt"), weights_only=False)["heads"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_confirm_split_is_sealed_and_disjoint(study):
    sp = Split.load(study.paths.d("data", "splits"))
    assert sp.confirm_docs and not (sp.confirm_docs & (sp.test_docs | sp.val_docs))
    with pytest.raises(PermissionError):
        load_bundle(study, localize=False, test_split="confirm")


def test_eval_set_cap_per_doc(study):
    ev = EvalSet(study.paths.d("data", "eval", "test"))
    _, counts = np.unique(ev.doc_ids, return_counts=True)
    assert counts.max() <= study.data.max_per_doc


def test_resume_across_stage_boundary_with_augmentation(study, monkeypatch):
    """Corte en el primer paso de mid con aumentación aleatoria: el resultado debe ser idéntico."""
    bundle = load_bundle(study, localize=False)
    cfg = base_cfg(study, "infonce_hn", max_steps_pre=12, max_steps_mid=10, eval_every=5,
                   ckpt_every_steps=4, patience=100, aug_noise_std=0.5)
    da = study.paths.d("runs", "test_resume2", "A")
    db = study.paths.d("runs", "test_resume2", "B")
    run_training(cfg, bundle, da, ecfg=study.eval, evaluate_test=False)
    orig = Trainer._step
    calls = {"n": 0}

    def boom(self, batch, step):
        calls["n"] += 1
        if calls["n"] == 13:          # primer paso de la etapa mid
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


def test_run_folder_refuses_other_config(study):
    from clm_losses.train import ConfigMismatch
    bundle = load_bundle(study, localize=False)
    d = study.paths.d("runs", "test_fp", "X")
    cfg = base_cfg(study, "infonce", max_steps_pre=3, max_steps_mid=2, eval_every=2)
    run_training(cfg, bundle, d, ecfg=study.eval, evaluate_test=False)
    with pytest.raises(ConfigMismatch):
        run_training(base_cfg(study, "infonce", max_steps_pre=3, max_steps_mid=2, eval_every=2, lr_mult=8.0),
                     bundle, d, ecfg=study.eval, evaluate_test=False)


def test_embedding_server_accepts_requests():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from clm_losses.encoder_server import build_app

    class Fake:
        pad_id = 0

        def tok(self, x, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

        def embed_ids(self, ids):
            return np.ones((len(ids), 4), np.float32)
    c = TestClient(build_app(Fake()))
    assert c.post("/v1/embeddings", json={"input": ["hola"]}).status_code == 200
    assert c.post("/v1/embeddings", json={"input": [5, 6], "encoding_format": "base64"}).status_code == 200


def test_export_head_from_finished_run(study):
    from clm_losses.heads import heads_from_clm_checkpoint, load_torch
    from clm_losses.pipeline import export_head_from_run
    bundle = load_bundle(study, localize=False)
    d = study.paths.d("runs", "test_export", "seed_0")
    cfg = base_cfg(study, "siglip", max_steps_pre=4, max_steps_mid=4, eval_every=2)
    run_training(cfg, bundle, d, ecfg=study.eval, evaluate_test=True, pools_dir=bundle.pools_dir)
    assert sorted(f for f in os.listdir(d) if f.endswith(".pt")) == ["best_mid.pt"]   # Drive liviano
    p = export_head_from_run(study, d)
    h = heads_from_clm_checkpoint(load_torch(p))
    x = torch.randn(3, bundle.hidden)
    assert h.zs(x).shape == (3, study.train.proj)


def test_f0_8_collapse_diagnostic_runs(study):
    bundle = load_bundle(study, localize=False)
    r = f0.f0_8_collapse(study, bundle, arms=["vicreg_x"], aug_levels=(0.0,), max_steps=6, row_fraction=0.5,
                         log=lambda *a, **k: None)
    v = r["vicreg_x|aug=0.0"]
    assert v["status"] == "ok" and "rankme_zs" in v and "collapsed" in v


def test_setup_freezes_study_config(tmp_path):
    kw = dict(drive_base=str(tmp_path / "d"), cache_root=str(tmp_path / "c"), mount=False)
    s1 = setup("mini", hpo_overrides={"n_trials": 12}, **kw)
    s2 = setup("mini", hpo_overrides={"n_trials": 12}, **kw)          # misma petición: carga la guardada
    assert s2.hpo.n_trials == 12 and s1.to_dict()["data"] == s2.to_dict()["data"]
    assert setup("mini", **kw).hpo.n_trials == 12                      # sin overrides: la guardada
    with pytest.raises(RuntimeError):
        setup("mini", hpo_overrides={"n_trials": 24}, **kw)            # cambio silencioso: se detiene
    s3 = setup("mini", hpo_overrides={"n_trials": 24}, allow_config_change=True, **kw)
    assert s3.hpo.n_trials == 24 and os.listdir(os.path.join(s3.paths.drive_root, "config", "history"))


def test_f0_6_shortcuts_without_gen2(study):
    bundle = load_bundle(study, localize=False)
    r = f0.f0_6_shortcuts(study, bundle, log=lambda *a, **k: None)
    assert {"tgem", "trand", "tmine"} <= set(r["blind"]) and "tgem" in r["raw_cosine"]
    assert 0.0 <= r["blind"]["tgem"] <= 1.0 and r["G3"]


def test_planning_helpers(study, tmp_path):
    from clm_losses.pipeline import estimate_budget, profile_disk_table, run_arm, tunability_report
    t = profile_disk_table(str(tmp_path / "c"))
    assert list(t["perfil"]) == ["mini", "colab", "colab_large"]
    assert abs(float(t.loc[t.perfil == "colab", "GB_embeddings"].iloc[0]) - 65.5) < 0.5
    bundle = load_bundle(study, localize=False)
    for arm in ("infonce", "siglip"):
        run_arm(study, bundle, arm, n_trials=3, seeds=[0])
    b = estimate_budget(study, cu_per_hour=1.7)
    assert b["horas_restantes_F1"] >= 0 and "unidades_F3" in b
    r = tunability_report(study, ns=(1, 2, 3))
    assert set(r["evp"].index) == {"infonce", "siglip"} and r["n_referencia"] == 3
