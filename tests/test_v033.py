"""Pruebas de v0.3.3: Gemma 4 (configs por capa), matriz de ciegos sin fuga por ítem, EXP-006c y C2 en F3
(ENMIENDA_03 §12)."""
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

from clm_losses import f0
from clm_losses.analysis import (PILOT_BUILDS, c2_reading, component_reading, construction_matrices,
                                 load_pilot_sets, pilot_set_name)
from clm_losses.generators import (BLIND_TRANSFER_METHOD, blind_correct, blind_transfer_matrix, common_item_folds,
                                   fold_of, synthetic_panel, synthetic_pilot, tgen_root, train_blind_on,
                                   transfer_rows)
from clm_losses.genworker import kv_bytes_per_token
from clm_losses.pipeline import load_bundle, setup
from clm_losses.smoke import SMOKE_LOSS_PARAMS
from clm_losses.utils import load_json

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
class _Ambiguous(Exception):
    """Como AmbiguousGlobalPerLayerAttributeError de transformers: NO es AttributeError."""


class _HetCfg:
    def __init__(self, layers, **glob):
        self.per_layer_config = layers
        self.__dict__.update(glob)

    def __getattribute__(self, k):
        if k in ("num_key_value_heads", "head_dim"):
            raise _Ambiguous(f"'{k}' is a per-layer attribute and may vary across layers")
        return object.__getattribute__(self, k)


def test_kv_bytes_homogeneous_matches_previous_formula():
    cfg = NS(num_hidden_layers=4, num_key_value_heads=2, head_dim=8, num_attention_heads=4, hidden_size=64)
    assert kv_bytes_per_token(cfg) == 2 * 4 * 2 * 8 * 2
    # sin head_dim: hidden / cabezas; sin K/V: las cabezas de atención
    assert kv_bytes_per_token(NS(num_hidden_layers=2, num_attention_heads=4, hidden_size=64)) == 2 * 2 * 4 * 16 * 2


def test_kv_bytes_per_layer_config_like_gemma4():
    het = _HetCfg([NS(num_key_value_heads=2, head_dim=8), NS(num_key_value_heads=4, head_dim=16)],
                  num_attention_heads=8, num_hidden_layers=2)
    with pytest.raises(_Ambiguous):
        het.num_key_value_heads                                  # el acceso global rompe, como en Gemma 4
    assert kv_bytes_per_token(het) == 2 * (2 * 8 + 4 * 16) * 2   # suma capa por capa
    assert kv_bytes_per_token(NS(text_config=het)) == 2 * (2 * 8 + 4 * 16) * 2
    # capas como dict ordenado por índice, y una capa sin datos -> None (el batch lo resuelve la mitad por OOM)
    assert kv_bytes_per_token(NS(per_layer_config={"1": {"num_key_value_heads": 1, "head_dim": 4},
                                                   "0": {"num_key_value_heads": 1, "head_dim": 4}})) == 2 * 8 * 2
    assert kv_bytes_per_token(_HetCfg([NS(head_dim=8)], num_hidden_layers=1)) is None
    assert kv_bytes_per_token(NS()) is None


# --------------------------------------------------------------------------------------
def test_common_item_folds_are_shared_across_sets():
    a, b = np.array([5, 1, 9, 3]), np.array([9, 7, 1])
    u, fu = common_item_folds([a, b], folds=2, seed=0)
    assert list(u) == [1, 3, 5, 7, 9]
    fa, fb = fold_of(a, u, fu), fold_of(b, u, fu)
    assert fa[2] == fb[0] and fa[1] == fb[2]                     # el ítem 9 y el 1, mismo pliegue en los dos sets


def test_blind_transfer_has_no_item_leak():
    """Mismo oro en A y B, negativos propios de cada set con la MISMA distribución que el oro (sin atajo): un
    ciego no puede encontrar el oro salvo memorizándolo. La versión vieja (entrenar con todos los ítems de A y
    evaluar en todos los de B) queda por encima del azar; la nueva, en el azar."""
    N, H = 300, 64
    rs = np.random.default_rng(0)
    nrm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
    gold = nrm(rs.normal(size=(N, H))).astype(np.float32)
    mk = lambda sd: np.concatenate([gold[:, None, :], nrm(np.random.default_rng(sd).normal(size=(N, 10, H)))  # noqa: E731
                                    .astype(np.float32)], 1)
    CA, CB = mk(1), mk(2)
    old = float(blind_correct(train_blind_on(CA, H, steps=200, seed=0, device="cpu"), CB).mean())
    rows = transfer_rows({"A": CA, "B": CB}, {"A": np.arange(N), "B": np.arange(N)}, folds=5, steps=200, seed=0)
    assert old - rows["A"]["B"] >= 0.04
    assert rows["A"]["B"] < 0.13 and rows["A"]["A"] < 0.15


def test_blind_transfer_matrix_replaces_leaky_csv(study, bundle):
    keys = synthetic_panel(study, bundle.test, n_items=120)[:2]
    root = tgen_root(study, "test")
    with open(os.path.join(root, "blind_transfer.csv"), "w") as f:
        f.write(",x\nx,0.9\n")                                   # CSV de v0.3.2 (sin info del método)
    df = blind_transfer_matrix(study, bundle, keys, steps=20, folds=2)
    assert list(df.columns) == keys and len(df) == 2
    assert os.path.exists(os.path.join(root, "history", "blind_transfer_v032_fuga_por_item.csv"))
    assert load_json(os.path.join(root, "blind_transfer_info.json"))["method"] == BLIND_TRANSFER_METHOD
    blind_transfer_matrix(study, bundle, keys, steps=20, folds=2)  # la segunda vez no mueve nada
    assert len(os.listdir(os.path.join(root, "history"))) >= 1


# --------------------------------------------------------------------------------------
def test_pilot_sets_and_construction_matrices(study, bundle, tmp_path):
    synthetic_pilot(study, bundle, n=120, **QUIET)
    pil = load_pilot_sets(study)
    keys = sorted({n.split("_", 1)[1] for n in pil})
    assert len(keys) == 3 and len(pil) == 9
    assert all(pilot_set_name(b, k) in pil for b in PILOT_BUILDS for k in keys)
    # corridas falsas: el acierto por set se guarda con los ítems en otro orden; T-gem por ítem del test
    items = np.sort(np.asarray(pil[pilot_set_name("S", keys[0])][0]))[:10]
    n_test = bundle.test.n
    rds = []
    for s in range(2):
        rd = tmp_path / f"run{s}"
        ev = rd / "eval"
        ev.mkdir(parents=True)
        tg = np.zeros(n_test, np.float32)
        tg[items] = 1.0
        np.save(ev / "correct_tgem.npy", tg)
        np.save(ev / "correct_tgem_shuf.npy", np.zeros(n_test, np.float32))
        for b in PILOT_BUILDS:
            for j, k in enumerate(keys):
                it = items[::-1]                                  # orden distinto al pedido
                val = np.full(len(it), 1.0 if (b == "S" and j == 0) else 0.0, np.float32)
                np.save(ev / f"items_{pilot_set_name(b, k)}.npy", it)
                np.save(ev / f"correct_{pilot_set_name(b, k)}.npy", val)
                np.save(ev / f"correct_{pilot_set_name(b, k)}_shuf.npy", np.zeros(len(it), np.float32))
        rds.append(str(rd))
    M = construction_matrices({"g": rds}, keys, items)
    assert M["g"]["T-gem"].shape == (2, len(items)) and float(M["g"]["T-gem"].mean()) == 1.0
    assert np.allclose(M["g"]["S"], 1 / 3) and float(M["g"]["E"].sum()) == 0.0   # promedio sobre generadores
    assert construction_matrices({"g": rds}, keys, np.r_[items, 10 ** 7]) == {}  # ítem sin evaluar: se omite


def test_component_reading_rule():
    assert component_reading({"diff": 0.05, "ci95_lo": 0.01, "ci95_hi": 0.09, "tost_equivalent": False}, 0.01) \
        == "explica parte de la brecha"
    assert component_reading({"diff": 0.0, "ci95_lo": -0.004, "ci95_hi": 0.004, "tost_equivalent": True}, 0.01) \
        == "no la explica (equivalente)"
    assert component_reading({"diff": 0.02, "ci95_lo": -0.01, "ci95_hi": 0.05, "tost_equivalent": False}, 0.01) \
        == "no concluyente"
    assert component_reading({"diff": -0.03, "ci95_lo": -0.05, "ci95_hi": -0.01, "tost_equivalent": False}, 0.01) \
        == "en sentido contrario"


def test_c2_reading_rule():
    up = {"diff": 0.20, "ci95_lo": 0.18}
    big = {"diff": 0.15, "ci95_lo": 0.10, "ci95_hi": 0.20, "tost_equivalent": False}
    assert c2_reading(up, big, 0.001, 0.01).startswith("H1b se sostiene")
    assert c2_reading(up, big, 0.20, 0.01) == "no concluyente"                       # Holm no alcanza
    assert c2_reading({"diff": -0.003, "ci95_lo": -0.02}, big, 0.001, 0.01).startswith("no aplica")
    eq = {"diff": 0.001, "ci95_lo": -0.005, "ci95_hi": 0.006, "tost_equivalent": True}
    assert c2_reading(up, eq, 0.9, 0.01).startswith("equivalente")
    neg = {"diff": -0.05, "ci95_lo": -0.08, "ci95_hi": -0.02, "tost_equivalent": False}
    assert c2_reading(up, neg, 0.01, 0.01).startswith("la ganancia es MAYOR")
