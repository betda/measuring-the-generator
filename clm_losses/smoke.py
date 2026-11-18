"""Prueba de humo con datos sintéticos (mismo layout de archivos que los datos reales).

Recorre, en pequeño, TODO el flujo del notebook v0.3: F0.4–F0.6, F0.5b (val-mine y T-mine-50), F0.8,
el piloto de construcción del panel (sintético: S con atajo de longitud, P limpio, E con artefacto) y un panel
T-gen SINTÉTICO con la construcción elegida (generadores con artefactos de distinta fuerza), su auditoría y su macro,
F0.7b (receta actual vs la de CLM, escalada) y la decisión de receta, F1 piloto (4 brazos), F1b (los 15
brazos con HPO), el análisis todos contra todos con T-mine y el macro de T-gen, el análisis de
transferencia, F2b (B3), el PROTOCOL FREEZE y un F3 reducido sobre el test sellado.

Los datos sintéticos traen un "artefacto del generador" en los HN de Gemini y en algunos generadores del
panel. Así el smoke comprueba que el pipeline DETECTA atajos (ciego, longitud, estados barajados).
No dice nada sobre las pérdidas ni sobre generadores reales."""
from __future__ import annotations

import os
import time

import numpy as np

from .utils import save_json

SMOKE_LOSS_PARAMS = {"exp_dim": 256, "K": 256, "pred_hidden": 64, "n_slices": 64, "freeze_steps": 5}


def synthetic_lengths(study, split: str, k: int = 10, seed: int = 0) -> str:
    """Longitudes de prueba (el smoke no tiene textos): el oro tiende a ser más largo que los HN de "Gemini",
    como en los datos reales, para ejercitar la heurística de longitud."""
    from .data import EvalSet
    d = study.paths.d("data", "eval", split)
    p = os.path.join(d, "lengths.npz")
    if not os.path.exists(p):
        ev = EvalSet(d)
        rs = np.random.default_rng(seed + (0 if split == "test" else 7))
        gold = rs.integers(15, 60, ev.n)
        neg = np.maximum(1, gold[:, None] - rs.integers(-5, 20, (ev.n, k)))
        np.savez(p, gold=gold.astype(np.int32), neg=neg.astype(np.int32))
    return p


def run_smoke(drive_base: str, cache_root: str = "/content/cache", arms=None, n_trials: int = 2,
              seeds=(0, 1), mount: bool = True, full_flow: bool = True, log=print) -> dict:
    from . import f0
    from .analysis import c2_report, c2b_report, c2c_report, construction_report, transfer_report
    from .generators import (audit_generator, blind_transfer_matrix, freeze_panel, panel_table, synthetic_panel,
                             synthetic_pilot, synthetic_pilot_sets)
    from .losses import arm_names
    from .pipeline import (add_missing_evals, compare_variants, export_summary_md, load_bundle, plot_val_curves,
                           progress, protocol_freeze, run_all_vs_all, run_arm, run_confirm_arm, run_confirm_variant,
                           run_variant, setup)
    t0 = time.time()
    study = setup("smoke", drive_base=drive_base, cache_root=cache_root, mount=mount,
                  train_overrides={"loss_params": SMOKE_LOSS_PARAMS})
    out = {"profile": "smoke", "drive_root": study.paths.drive_root}
    out["F0.4"] = f0.f0_4_splits(study, log=log)
    out["F0.5"] = f0.f0_5_eval_sets(study, lengths=False, qc=False, log=log)
    for sp in ("test", "val"):
        synthetic_lengths(study, sp)
    out["F0.5b"] = f0.f0_5b_pools(study, log=log)
    bundle = load_bundle(study, localize=False)
    out["F0.6"] = f0.f0_6_shortcuts(study, bundle, log=log)
    out["F0.8"] = f0.f0_8_collapse(study, bundle, arms=["simsiam_x", "vicreg_x"], max_steps=40, log=log)
    # piloto de construcción (sintético) + panel T-gen sintético con la construcción elegida + auditoría + macro
    blind = f0.train_blind(bundle, study.paths.d("f0", "F0.6_shortcuts"), log=lambda *a, **k: None)
    pil = synthetic_pilot(study, bundle, blind=blind, n=300, log=log)
    out["piloto"] = {"build": pil["build"], "medianas": pil["medianas"], "tabla": pil["pilot"]["table"]}
    keys = synthetic_panel(study, bundle.test, n_items=300)
    for k in keys:
        audit_generator(study, bundle, k, blind=blind, steps=150, log=log)
    blind_transfer_matrix(study, bundle, keys, blind=blind, steps=150)
    panel = freeze_panel(study, log=log)
    out["panel"] = {"macro": panel["macro"], "n_macro_items": panel["n_macro_items"], "build": panel["build"],
                    "tabla": panel_table(study).to_dict("records")}
    # F0.7b: receta actual vs la de CLM (escalada al smoke) + decisión pre-especificada
    r7b = f0.f0_7b_recipes(study, bundle, seeds=tuple(seeds), log=log)
    out["F0.7b"] = r7b["G2b"]
    out["receta_F1b"] = (f0.decide_recipe(study, r7b, log=log) or {}).get("receta")
    arms = list(arms or arm_names())
    # F1 piloto (sólo algunos brazos: el código de F1 no cambió) y F1b (los 15)
    for arm in [a for a in ("infonce", "infonce_hn", "siglip", "siglip_hn") if a in arms]:
        run_arm(study, bundle, arm, phase="F1", n_trials=n_trials, seeds=list(seeds))
    for arm in arms:
        run_arm(study, bundle, arm, phase="F1b", n_trials=n_trials, seeds=list(seeds))
    out["progress_F1b"] = progress(study, "F1b").to_dict("records")
    add_missing_evals(study, bundle, ["F0.7b", "F1", "F1b"], log=log)
    rep = run_all_vs_all(study, bundle, "F1b", B=300)
    out["all_vs_all_F1b"] = {k: v for k, v in rep.items() if k in ("arms", "friedman", "C1")}
    out["resumen_F1b"] = export_summary_md(study, "F1b")
    tr = transfer_report(study, bundle, B=300, log=log)
    out["transfer"] = tr["contrastes"].to_dict("records")
    cr = construction_report(study, bundle, B=300, log=log)        # EXP-006c con los sets del piloto sintético
    out["construccion"] = {"n_items": cr.get("n_items"), "contrastes": cr["contrastes"].to_dict("records")}
    if full_flow and {"infonce_hn", "infonce_sigreg", "infonce_lse"} <= set(arms):
        b3 = {"InfoNCE+HN (F1b)": ("F1b", "infonce_hn"),
              "InfoNCE+HN desde el inicio": ("F2b_B3", "infonce_hn_start"),
              "InfoNCE+SIGReg+HN desde el inicio": ("F2b_B3", "infonce_sigreg_hn_start"),
              "InfoNCE+LSE+HN desde el inicio": ("F2b_B3", "infonce_lse_hn_start")}
        run_variant(study, bundle, "infonce_hn", "F2b_B3", "infonce_hn_start", seeds=[0], hp_from="F1b",
                    hn_from_start=True)
        run_variant(study, bundle, "infonce_sigreg", "F2b_B3", "infonce_sigreg_hn_start", seeds=[0], hp_from="F1b",
                    hn_from_start=True, hn_mode="native")
        run_variant(study, bundle, "infonce_lse", "F2b_B3", "infonce_lse_hn_start", seeds=[0], hp_from="F1b",
                    hn_from_start=True, hn_mode="native")
        run_variant(study, bundle, "infonce_hn", "F2b_B1", "infonce_hn_hnonly", seeds=[0], hp_from="F1b",
                    loss_params={"hn_denominator": "hn_only"})
        out["F2b_B3"] = {k: v for k, v in compare_variants(study, bundle, b3, "B3", B=300, results_phase="F2b").items()
                         if k in ("arms", "metrics")}
        out["F2b_B3_curvas"] = plot_val_curves(study, b3, study.paths.d("results", "F2b", "B3", "curvas_val.png"),
                                               metric="valmine_top1")
    if full_flow and len(arms) == len(arm_names()):
        # PROTOCOL FREEZE (desde F1b) + panel del confirmatorio + F3 reducido sobre el test SELLADO
        frz = protocol_freeze(study, phase="F1b", note="smoke: sólo prueba de código")
        bundle_c = load_bundle(study, localize=False, test_split="confirm")
        synthetic_lengths(study, "confirm")
        kc = synthetic_panel(study, bundle_c.test, split="confirm", n_items=300)
        for k in kc:
            audit_generator(study, bundle_c, k, split="confirm", blind=blind, steps=150, log=log)
        freeze_panel(study, split="confirm", force_macro=(frz.get("tgen_panel") or {}).get("macro") or [], log=log)
        synthetic_pilot_sets(study, bundle_c.test, split="confirm")     # sets del piloto en el confirmatorio (C2b)
        for arm in sorted({"infonce", frz["C1_best_noncontrastive"], "siglip", "infonce_hn", "siglip_hn"}):
            run_confirm_arm(study, bundle_c, arm, seeds=list(seeds))
        rep3 = run_all_vs_all(study, bundle_c, phase="F3", B=300)
        out["F3"] = {k: v for k, v in rep3.items() if k in ("arms", "C1")}
        c2 = c2_report(study, bundle_c, "F3", B=300, log=log)          # C2 pre-especificado (secundario)
        out["C2_F3"] = [{"par": r["par"], "C2": r["C2"]["diff"], "p_holm": r["C2"]["p_holm"], "lectura": r["lectura"]}
                        for r in c2["pares"]]
        c2b = c2b_report(study, bundle_c, "F3", B=300, log=log)         # C2b pre-especificado (secundario)
        out["C2b_F3"] = [{"par": t["par"], "componente": t["componente"], "diff": t["c"]["diff"],
                          "p_holm": t["p_holm"], "lectura": t["lectura"]} for t in c2b["pruebas"]]
        for name in frz.get("F3_variants") or {}:                       # variante de C2c ("HN solos")
            run_confirm_variant(study, bundle_c, name, seeds=list(seeds))
        c2c = c2c_report(study, bundle_c, "F3", B=300, log=log)         # C2c pre-especificado (secundario)
        out["C2c_F3"] = {"pruebas": [{"prueba": t["prueba"], "diff": t["c"]["diff"], "p_holm": t["p_holm"],
                                      "lectura": t["lectura"]} for t in c2c["pruebas"]],
                         "control": (c2c["control"] or {}).get("lectura"), "lectura_global": c2c["lectura_global"]}
        out["resumen_F3"] = export_summary_md(study, "F3")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    from .utils import code_hash
    out["code_hash"] = code_hash()
    save_json(out, study.paths.d("results", "SMOKE_SUMMARY.json"))
    log(f"[smoke] OK en {out['minutes']} min -> {study.paths.d('results')}")
    return out
