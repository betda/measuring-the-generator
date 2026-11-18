"""Pruebas de v0.3.5: C2c (mecanismo de H1b en F3, ENMIENDA_03 §14) y la variante "HN solos" registrada en el
freeze (F3_variants)."""
import os

import numpy as np
import pytest

from clm_losses import f0
from clm_losses.analysis import C2C_PAIR, c2c_control_reading, c2c_reading, c2c_report
from clm_losses.pipeline import F3_VARIANTS, confirm_variant_cfg, load_bundle, setup
from clm_losses.smoke import SMOKE_LOSS_PARAMS
from clm_losses.utils import load_json, save_json

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
def test_c2c_reading_rule():
    up = {"diff": 0.15, "ci95_lo": 0.12, "ci95_hi": 0.18, "tost_equivalent": False}
    assert c2c_reading(up, 1e-6, 0.01, +1) == "se confirma"            # sube, como se predijo
    assert c2c_reading(up, 1e-6, 0.01, -1) == "en sentido contrario"   # se predijo que baja
    assert c2c_reading(up, 0.2, 0.01, +1) == "no concluyente"
    down = {"diff": -0.06, "ci95_lo": -0.08, "ci95_hi": -0.04, "tost_equivalent": False}
    assert c2c_reading(down, 1e-4, 0.01, -1) == "se confirma"
    eq = {"diff": 0.001, "ci95_lo": -0.004, "ci95_hi": 0.006, "tost_equivalent": True}
    assert c2c_reading(eq, 0.7, 0.01, +1) == "no se confirma (equivalente)"
    small = {"diff": 0.006, "ci95_lo": 0.003, "ci95_hi": 0.009, "tost_equivalent": False}
    assert c2c_reading(small, 0.001, 0.01, +1) == "diferencia significativa pero < ROPE"


def test_c2c_control_reading():
    assert c2c_control_reading({"diff": 0.002, "ci95_lo": -0.003, "ci95_hi": 0.007, "tost_equivalent": True},
                               0.01) == "se confirma (equivalente)"
    assert c2c_control_reading({"diff": 0.05, "ci95_lo": 0.03, "ci95_hi": 0.07, "tost_equivalent": False},
                               0.01).startswith("no se confirma")
    assert c2c_control_reading({"diff": 0.01, "ci95_lo": -0.01, "ci95_hi": 0.03, "tost_equivalent": False},
                               0.01) == "no concluyente"


def test_confirm_variant_cfg_only_changes_the_denominator(study):
    """La variante usa los ajustes congelados de la fase y los HP congelados del brazo; sólo cambia lo registrado."""
    frz = {"phase_settings": {"select_metric": "valmine_top1", "loss_params": {"hn_denominator": "hn_plus_inbatch"}},
           "arms": {"infonce_hn": {"overrides": {"lr_mult": 0.5, "loss.tau_init": 0.05}}}}
    var = F3_VARIANTS[C2C_PAIR[0]]
    assert var["arm"] == C2C_PAIR[1]
    base = confirm_variant_cfg(study, frz, "infonce_hn", {}, seed=3)
    cv = confirm_variant_cfg(study, frz, "infonce_hn", var["changes"], seed=3)
    assert base.loss_params["hn_denominator"] == "hn_plus_inbatch" and cv.loss_params["hn_denominator"] == "hn_only"
    assert cv.lr_mult == base.lr_mult == 0.5 and cv.loss_params["tau_init"] == base.loss_params["tau_init"] == 0.05
    assert cv.select_metric == base.select_metric == "valmine_top1" and cv.seed == 3 and cv.arm == "infonce_hn"
    a, b = base.__dict__.copy(), cv.__dict__.copy()
    la, lb = a.pop("loss_params"), b.pop("loss_params")
    assert a == b and {k: v for k, v in la.items() if k != "hn_denominator"} == \
        {k: v for k, v in lb.items() if k != "hn_denominator"}
    assert F3_VARIANTS[C2C_PAIR[0]]["changes"] == {"loss_params": {"hn_denominator": "hn_only"}}   # no se modificó


def _fake_runs(study, name, n, seeds, rs, **probs):
    for s in seeds:
        rd = study.paths.d("runs", "F3", name, f"seed_{s}")
        os.makedirs(os.path.join(rd, "eval"), exist_ok=True)
        save_json({"status": "ok", "seed": s}, os.path.join(rd, "DONE.json"))
        for key, p in probs.items():
            np.save(os.path.join(rd, "eval", f"correct_{key}.npy"), (rs.random(n) < p).astype(np.float32))


def test_c2c_report_reads_the_mechanism(study, bundle):
    n = len(bundle.test.doc_ids)
    rs = np.random.default_rng(0)
    _fake_runs(study, "infonce_hn_hnonly", n, (0, 1), rs, tgem_shuf=0.30, tmine=0.80, tgem=0.65, trand=0.85)
    _fake_runs(study, "infonce_hn", n, (0, 1), rs, tgem_shuf=0.08, tmine=0.95, tgem=0.60, trand=0.95)
    _fake_runs(study, "infonce", n, (0, 1), rs, tgem_shuf=0.08, tmine=0.95, tgem=0.38, trand=0.95)
    r = c2c_report(study, bundle, "F3", B=300, **QUIET)
    lect = {t["metrica"]: t["lectura"] for t in r["pruebas"]}
    assert lect == {"tgem_shuf": "se confirma", "tmine": "se confirma"}
    assert r["lectura_global"].startswith("mecanismo confirmado:")
    assert r["control"]["lectura"] in ("se confirma (equivalente)", "no concluyente")   # n chico en el smoke
    assert set(r["descriptivos"]) >= {"tgem", "trand"}
    js = load_json(os.path.join(r["dir"], "C2c.json"))
    assert js["variante"] == "infonce_hn_hnonly" and len(js["pruebas"]) == 2 and js["semillas"]["infonce_hn"] == 2
