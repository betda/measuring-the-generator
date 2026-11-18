"""Pruebas del análisis de sensibilidad del panel en F3 (analisis/F3_sensibilidad; post hoc, no cambia el paquete)."""
import importlib.util
import os

import numpy as np
import pytest

from clm_losses.stats import clusters_from_docs, contrast

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analisis", "F3_sensibilidad",
                  "sensibilidad_panel.py")
_spec = importlib.util.spec_from_file_location("sensibilidad_panel", _P)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

KEYS = ["g1", "g2", "g3", "g4", "g5"]
N = 300                                        # ítems del test


def _fake(seed=0, n_runs=3, drop=None):
    """Corridas falsas: acierto por ítem en T-gem y en cada generador; cada generador cubre ítems distintos."""
    rng = np.random.default_rng(seed)
    cover = {k: np.sort(rng.choice(N, 220, replace=False)) for k in KEYS}
    arms = {"vicreg_x": 0.30, "infonce": 0.35, "infonce_hn": 0.45, "siglip": 0.33, "siglip_hn": 0.44}
    data = {}
    for arm, p in arms.items():
        for s in range(n_runs):
            rd = f"/runs/F3/{arm}/seed_{s}"
            data[rd] = {"tgem": (rng.random(N) < p + 0.2).astype(np.float32)}
            for j, k in enumerate(KEYS):
                data[rd][k] = (rng.random(N) < p + 0.03 * j).astype(np.float32)
    groups = {a: [f"/runs/F3/{a}/seed_{s}" for s in range(n_runs)] for a in arms}

    def items_of(ks):
        it = None
        for k in ks:
            it = cover[k] if it is None else np.intersect1d(it, cover[k])
        return it

    def at(rd, k, items):
        if drop and (rd, k) == drop:
            return None
        if not np.isin(items, cover[k]).all():
            return None
        return data[rd][k][items]

    def tgem_at(rd, items):
        return data[rd]["tgem"][items]

    docs = np.array([f"d{i // 2}" for i in range(N)], dtype=object)      # 2 ítems por documento
    return groups, items_of, at, tgem_at, docs, data


def test_subsets_cover_full_leave_one_out_and_single():
    s = sp.subsets(KEYS)
    assert s[0] == ("los 5 (macro congelado)", KEYS)
    assert [n for n, _ in s[1:6]] == [f"sin {k}" for k in KEYS] and all(len(g) == 4 for _, g in s[1:6])
    assert [g for _, g in s[6:]] == [[k] for k in KEYS]
    assert sp.subsets(["solo"]) == [("los 1 (macro congelado)", ["solo"])]


def test_full_row_equals_direct_contrast_on_the_macro():
    groups, items_of, at, tgem_at, docs, data = _fake()
    rows = sp.compute(groups, KEYS, items_of, docs, "vicreg_x", 0.01, at, tgem_at, B=500)
    it = items_of(KEYS)
    cl = clusters_from_docs(docs[it])
    M = {a: np.vstack([np.mean([data[rd][k][it] for k in KEYS], 0).astype(np.float32) for rd in groups[a]])
         for a in groups}
    want = contrast(M["vicreg_x"], M["infonce"], cl, 0.01, B=500)
    got = rows[0]["C1"]
    assert all(got[k] == pytest.approx(want[k], abs=1e-12) for k in ("diff", "ci95_lo", "ci95_hi", "p"))
    T = {a: np.vstack([data[rd]["tgem"][it] for rd in groups[a]]) for a in groups}
    c2 = contrast(T["infonce_hn"] - M["infonce_hn"], T["infonce"] - M["infonce"], cl, 0.01, B=500)
    assert rows[0]["infonce_hn − infonce"]["C2"]["diff"] == pytest.approx(c2["diff"], abs=1e-12)
    assert len(rows) == 11 and rows[1]["generadores"] == KEYS[1:]
    assert rows[6]["n_items"] == 220                          # «sólo g1» usa todos sus ítems


def test_missing_generator_skips_only_the_rows_that_need_it():
    groups, items_of, at, tgem_at, docs, _ = _fake(drop=("/runs/F3/vicreg_x/seed_0", "g3"))
    rows = {r["subconjunto"]: r for r in sp.compute(groups, KEYS, items_of, docs, "vicreg_x", 0.01, at, tgem_at, B=200)}
    assert "C1" not in rows["los 5 (macro congelado)"] and "C1" not in rows["sólo g3"]
    assert "C1" in rows["sin g3"] and "C1" in rows["sólo g1"]
    assert "infonce_hn − infonce" in rows["los 5 (macro congelado)"]     # C2 no usa vicreg_x


def test_reading_and_control():
    c = {"diff": -0.022, "ci95_lo": -0.036, "ci95_hi": -0.008, "p": 0.002, "tost_equivalent": False}
    assert sp.lectura_c1(c, 0.01, "vicreg_x") == "infonce mejor (IC95 entra en ±ROPE)"
    c2 = dict(c, ci95_hi=-0.012)
    assert sp.lectura_c1(c2, 0.01, "vicreg_x") == "infonce mejor (IC95 fuera de ±ROPE)"
    assert sp.lectura_c1(dict(c, diff=-0.005, ci95_lo=-0.009, ci95_hi=-0.001), 0.01, "x") == \
        "significativa pero menor que ROPE"
    rows = [{"C1": dict(c), "a − b": {"C2": {"diff": 0.1, "ci95_lo": 0.05, "ci95_hi": 0.15}}}]
    ok = sp.control(rows, dict(c), [{"par": "a − b", "C2": {"diff": 0.1, "ci95_lo": 0.04, "ci95_hi": 0.15}}])
    assert ok == {"C1": {"punto": True, "IC95": True}, "C2 a − b": {"punto": True, "IC95": False}}
    assert sp.control(rows, dict(c, diff=-0.03), None) == {"C1": {"punto": False, "IC95": True}}


def test_markdown_renders_rows_and_control():
    groups, items_of, at, tgem_at, docs, _ = _fake()
    rows = sp.compute(groups, KEYS, items_of, docs, "vicreg_x", 0.01, at, tgem_at, B=200)
    res = {"filas": rows, "brazo_C1": "vicreg_x", "fase": "F3", "split": "confirm", "creado": "x", "motivo": "m",
           "control": sp.control(rows, rows[0]["C1"], None)}
    md = sp.to_markdown(res)
    assert "C1: punto sí, IC95 sí" in md and md.count("\n| ") == 1 + len(rows)
    assert list(sp.tabla(res)["subconjunto"])[:2] == ["los 5 (macro congelado)", "sin g1"]
