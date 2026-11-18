"""Pruebas de v0.3.7: limpieza de los pesos de los generadores con la caché compartida de huggingface_hub
(«shared blobs»: los archivos reales viven en hub/blobs/<xx>/<hash> y models--X/blobs/* son enlaces simbólicos).
Hasta v0.3.6 se borraban sólo los enlaces y el disco no se liberaba (rev. 4.6: "borrados 0.0 GB")."""
import os

import pytest

from clm_losses.generators import disk_diagnostics, purge_hf_caches, purge_model_cache, worker_env


def _shared_model(hf, repo, xet_hash, mb=2, shared=True):
    """Un repo en la caché de HF con el diseño nuevo (shared=True) o el clásico."""
    hub = os.path.join(hf, "hub")
    d = os.path.join(hub, "models--" + repo.replace("/", "--"))
    os.makedirs(os.path.join(d, "blobs"), exist_ok=True)
    os.makedirs(os.path.join(d, "snapshots", "rev"), exist_ok=True)
    etag = os.path.join(d, "blobs", "etag_" + xet_hash[:6])
    if shared:
        real = os.path.join(hub, "blobs", xet_hash[:2], xet_hash)
        os.makedirs(os.path.dirname(real), exist_ok=True)
        if not os.path.exists(real):
            with open(real, "wb") as f:
                f.write(b"\0" * (mb * 1_000_000))
            open(real + ".refs", "w").write("x")
        os.symlink(os.path.relpath(real, os.path.dirname(etag)), etag)
    else:
        with open(etag, "wb") as f:
            f.write(b"\0" * (mb * 1_000_000))
    os.symlink(os.path.relpath(etag, os.path.join(d, "snapshots", "rev")),
               os.path.join(d, "snapshots", "rev", "model.safetensors"))
    return d


def test_purge_model_cache_removes_the_shared_files(tmp_path):
    hf = str(tmp_path / "hf")
    _shared_model(hf, "org/a", "ab" + "1" * 62, mb=3)
    real = os.path.join(hf, "hub", "blobs", "ab", "ab" + "1" * 62)
    assert purge_model_cache("org/a", hf) == pytest.approx(0.003, abs=1e-6)   # antes daba 0.0 (sólo enlaces)
    assert not os.path.exists(real) and not os.path.exists(real + ".refs")
    assert not os.path.exists(os.path.join(hf, "hub", "models--org--a"))


def test_purge_model_cache_keeps_shared_files_used_by_another_repo(tmp_path):
    hf = str(tmp_path / "hf")
    h = "cd" + "2" * 62
    _shared_model(hf, "org/a", h)
    _shared_model(hf, "org/b", h)                            # mismo archivo deduplicado
    assert purge_model_cache("org/a", hf) == 0.0
    assert os.path.exists(os.path.join(hf, "hub", "blobs", "cd", h))
    assert os.path.exists(os.path.join(hf, "hub", "models--org--b", "snapshots", "rev", "model.safetensors"))


def test_purge_model_cache_classic_layout_still_works(tmp_path):
    hf = str(tmp_path / "hf")
    _shared_model(hf, "org/c", "ef" + "3" * 62, mb=2, shared=False)
    assert purge_model_cache("org/c", hf) == pytest.approx(0.002, abs=1e-6)


def test_purge_hf_caches_frees_the_shared_store_but_not_what_the_current_repo_uses(tmp_path):
    hf = str(tmp_path / "hf")
    old, now = "aa" + "4" * 62, "bb" + "5" * 62
    _shared_model(hf, "org/old", old)
    _shared_model(hf, "org/now", now)
    os.makedirs(os.path.join(hf, "xet"))
    r = purge_hf_caches(hf, keep_repos=("org/now",))
    assert "org/old" in r["removed"] and any(x.startswith("hub/blobs") for x in r["removed"])
    assert not os.path.exists(os.path.join(hf, "hub", "blobs", "aa", old))
    assert os.path.exists(os.path.join(hf, "hub", "blobs", "bb", now))
    assert os.path.exists(os.path.join(hf, "hub", "models--org--now", "snapshots", "rev", "model.safetensors"))
    assert not os.path.exists(os.path.join(hf, "xet"))


def test_worker_env_keeps_the_cache_inside_hf_home():
    env = worker_env("/c/hf_home", "/pkg", base={"PYTHONPATH": "/x", "HF_HUB_CACHE": "/otro", "HF_TOKEN": "t"})
    assert env["HF_HOME"] == "/c/hf_home" and "HF_HUB_CACHE" not in env
    assert env["HF_HUB_DISABLE_SHARED_BLOBS"] == "1" and env["HF_XET_CHUNK_CACHE_SIZE_BYTES"] == "0"
    assert env["PYTHONPATH"].startswith("/pkg" + os.pathsep) and env["HF_TOKEN"] == "t"


def test_disk_diagnostics_reports_a_shared_store(tmp_path):
    big = tmp_path / "cache" / "hf_home" / "hub" / "blobs" / "ab" / ("ab" + "6" * 62)
    big.parent.mkdir(parents=True)
    with open(big, "wb") as f:
        f.truncate(600_000_000)                              # archivo disperso: no ocupa disco real
    lines = disk_diagnostics(roots=(str(tmp_path),))
    assert any("caché compartida" in x and x.strip().startswith("0.6 GB") for x in lines)
