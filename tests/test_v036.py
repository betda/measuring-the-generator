"""Pruebas de v0.3.6: disco en la generación (pesos y caché de Xet, diagnóstico) y adenda del freeze para C2c
cuando el freeze se hizo con un código que no lo registraba (ENMIENDA_03 §14, DP-07)."""
import os

import pytest

from clm_losses.generators import _dir_bytes, deleted_open_files, purge_hf_caches, purge_model_cache
from clm_losses.pipeline import (F3_VARIANTS, addendum_path, c2c_spec_text, freeze_addendum, freeze_path,
                                 frozen_variants, setup)
from clm_losses.smoke import SMOKE_LOSS_PARAMS
from clm_losses.utils import load_json, save_json


def _fake_model(hub, repo, mb=2):
    d = os.path.join(hub, "models--" + repo.replace("/", "--"))
    os.makedirs(os.path.join(d, "blobs"), exist_ok=True)
    os.makedirs(os.path.join(d, "snapshots", "abc"), exist_ok=True)
    blob = os.path.join(d, "blobs", "h1")
    with open(blob, "wb") as f:
        f.write(b"\0" * (mb * 1_000_000))
    os.symlink(blob, os.path.join(d, "snapshots", "abc", "model.safetensors"))
    return d


def test_purge_counts_symlinked_weights_once(tmp_path):
    hub = tmp_path / "hf" / "hub"
    d = _fake_model(str(hub), "org/m", mb=3)
    assert _dir_bytes(d) == 3_000_000
    assert purge_model_cache("org/m", str(tmp_path / "hf")) == pytest.approx(0.003)   # antes daba el doble
    assert not os.path.exists(d)


def test_purge_hf_caches_keeps_current_repo_and_drops_xet(tmp_path):
    hf = tmp_path / "hf"
    _fake_model(str(hf / "hub"), "org/old")
    keep = _fake_model(str(hf / "hub"), "org/now")
    os.makedirs(hf / "xet" / "chunk-cache")
    (hf / "xet" / "chunk-cache" / "x").write_bytes(b"1" * 1000)
    r = purge_hf_caches(str(hf), keep_repos=("org/now",))
    assert "org/old" in r["removed"] and any(x.startswith("xet") for x in r["removed"])
    assert os.path.isdir(keep) and not (hf / "xet").exists()
    assert r["free_after_gb"] >= r["free_before_gb"] - 0.01
    assert purge_hf_caches(str(tmp_path / "no_existe"))["removed"] == []


def test_deleted_open_files_sees_a_held_file(tmp_path):
    p = tmp_path / "grande.bin"
    p.write_bytes(b"\0" * 2_000_000)
    with open(p, "rb"):
        os.remove(p)
        rows = deleted_open_files(min_mb=1)
        assert any(pid == os.getpid() and gb >= 0.002 for pid, _, gb in rows)


@pytest.fixture()
def study(tmp_path):
    return setup("smoke", drive_base=str(tmp_path / "drive"), cache_root=str(tmp_path / "cache"), mount=False,
                 train_overrides={"loss_params": SMOKE_LOSS_PARAMS})


def test_freeze_addendum_registers_c2c_once(study):
    assert freeze_addendum(study) is None                        # sin freeze no hay adenda
    frz = {"created": "2026-10-06T02:33:41", "code_hash": "94211b491b3f5d44", "rope_pp": 1.0, "seeds": [0, 1],
           "secondary_confirmatory": {"C2": "x", "C2b": "y"}}
    save_json(frz, freeze_path(study))
    add = freeze_addendum(study)
    assert add["F3_variants"] == F3_VARIANTS and add["secondary_confirmatory"]["C2c"] == c2c_spec_text(1.0)
    assert add["freeze_code_hash"] == "94211b491b3f5d44" and "2026-10-06T02:33:41" in add["motivo"]
    assert "desviación" in add["desviacion"] or "declarar" in add["desviacion"]
    assert os.path.exists(os.path.join(os.path.dirname(addendum_path(study)), "PROTOCOL_FREEZE_ADDENDUM.md"))
    again = freeze_addendum(study)                               # una sola vez: no se reescribe
    assert again["created"] == add["created"]
    assert load_json(freeze_path(study)) == frz                  # el freeze no se toca
    var, where = frozen_variants(study)
    assert where == "adenda" and set(var) == set(F3_VARIANTS)


def test_freeze_with_variants_needs_no_addendum(study):
    save_json({"created": "x", "code_hash": "y", "F3_variants": F3_VARIANTS}, freeze_path(study))
    assert freeze_addendum(study) is None and not os.path.exists(addendum_path(study))
    assert frozen_variants(study)[1] == "freeze"


def test_trex_skips_finished_arms_without_servers(study, monkeypatch):
    import clm_losses.trex as tx
    from clm_losses.pipeline import trex_for_arms
    od = study.paths.d("results", "F3", "trex", "infonce")
    os.makedirs(od, exist_ok=True)
    for m in ("shield", "noshield"):
        save_json({"summary": {}}, os.path.join(od, f"trex_{m}.json"))
    save_json({"shield": {"deaths": 0}}, os.path.join(od, "trex_summary.json"))
    monkeypatch.setattr(tx, "setup_clm_repo", lambda *a, **k: "/tmp/no_repo")
    monkeypatch.setattr(tx, "stop_all", lambda *a, **k: None)

    def boom(*a, **k):
        raise AssertionError("no debía correr T-Rex otra vez")
    monkeypatch.setattr(tx, "run_trex", boom)
    out = trex_for_arms(study, arms=["infonce"], log=lambda *a, **k: None)
    assert out == {"infonce": {"shield": {"deaths": 0}}}
