"""Pruebas de v0.3.4: firma de las generaciones por contenido (sin re-embebidos por el mtime de Drive) y regla
de C2b (ENMIENDA_03 §13)."""
import os
import time

from clm_losses.analysis import c2b_reading
from clm_losses.generators import _raw_sigs, gen_dir, raw_sigs_match
from clm_losses.pipeline import setup
from clm_losses.smoke import SMOKE_LOSS_PARAMS


def test_raw_sigs_match_old_and_new_formats():
    old = {"edit": [1000, 1696259612.123]}                      # formato hasta v0.3.3: [tamaño, mtime]
    assert raw_sigs_match(old, {"edit": [1000, "abc"]})         # mismo tamaño, otro mtime: no se re-embebe
    assert not raw_sigs_match(old, {"edit": [1200, "abc"]})     # se agregaron generaciones
    new = {"edit": [1000, "abc"]}
    assert raw_sigs_match(new, {"edit": [1000, "abc"]})
    assert not raw_sigs_match(new, {"edit": [1000, "abd"]})     # mismo tamaño, otro contenido
    assert not raw_sigs_match(new, {"edit": [1000, "abc"], "neg": [5, "x"]})
    assert not raw_sigs_match(None, new)
    assert raw_sigs_match({"edit": None}, {"edit": None})


def test_raw_sigs_ignore_mtime_but_see_content(tmp_path):
    st = setup("smoke", drive_base=str(tmp_path / "drive"), cache_root=str(tmp_path / "cache"), mount=False,
               train_overrides={"loss_params": SMOKE_LOSS_PARAMS})
    d = gen_dir(st, "gen_x", "test")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "raw_edit.jsonl")
    with open(p, "w") as f:
        f.write('{"i": 0, "negatives": ["a"]}\n')
    s1 = _raw_sigs(st, "gen_x", "test", "E")
    t = time.time() + 1000
    os.utime(p, (t, t))                                         # Drive cambia el mtime entre sesiones
    s2 = _raw_sigs(st, "gen_x", "test", "E")
    assert raw_sigs_match(s1, s2) and s1 == s2
    with open(p, "a") as f:
        f.write('{"i": 1, "negatives": ["b"]}\n')
    assert not raw_sigs_match(s1, _raw_sigs(st, "gen_x", "test", "E"))


def test_c2b_reading_rule():
    gap = {"diff": 0.20, "ci95_lo": 0.14}
    comp = {"diff": 0.15, "ci95_lo": 0.09, "ci95_hi": 0.21, "tost_equivalent": False}
    assert c2b_reading(gap, comp, 0.001, 0.01).startswith("se confirma")
    assert c2b_reading(gap, comp, 0.2, 0.01) == "no concluyente"
    assert c2b_reading({"diff": 0.005, "ci95_lo": -0.02}, comp, 0.001, 0.01).startswith("no aplica")
    neg = {"diff": -0.05, "ci95_lo": -0.08, "ci95_hi": -0.02, "tost_equivalent": False}
    assert c2b_reading(gap, neg, 0.01, 0.01) == "en sentido contrario"
    eq = {"diff": 0.001, "ci95_lo": -0.004, "ci95_hi": 0.006, "tost_equivalent": True}
    assert c2b_reading(gap, eq, 0.8, 0.01).startswith("no la explica")
