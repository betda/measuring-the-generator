"""Funciones de alto nivel que llama cada celda del notebook."""
from __future__ import annotations

import dataclasses
import os
import shutil
import sys
from dataclasses import dataclass

import numpy as np

from .config import (HN_REPO, PRETRAIN_REPO, EvalConfig, HPOConfig, Paths, StudyConfig, TrainConfig,
                     data_config, recipe_overrides)
from .data import (EvalSet, HFDownloader, Split, Store, make_synthetic_store, slug, train_masks)
from .hpo import apply_overrides, run_hpo
from .losses import ARMS, NONCONTRASTIVE_POOL, arm_names
from .train import run_training
from .utils import (RunLogger, disk_free_gb, env_snapshot, gpu_info, load_json, makedirs, now_iso,
                    pip_freeze, ram_gb, read_done, save_json)

# Fases que usan la configuración de la enmienda A1 (selección por val-mine, receta de F0.7b, "HN + in-batch").
A1_PHASES = ("F1b", "F2b_B1", "F2b_B2", "F2b_B3")
SELECT_A1 = "valmine_top1"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _config_diff(a: dict, b: dict, prefix: str = "") -> list:
    out = []
    for k in sorted(set(a) | set(b)):
        if k == "paths":
            continue
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _config_diff(va, vb, f"{prefix}{k}.")
        elif va != vb:
            out.append(f"{prefix}{k}: {va!r} -> {vb!r}")
    return out


def setup(profile: str = "colab", drive_base: str = "/content/drive/MyDrive/CLM_losses",
          cache_root: str = "/content/cache", mount: bool = True, train_overrides: dict | None = None,
          hpo_overrides: dict | None = None, study_overrides: dict | None = None,
          allow_config_change: bool = False, **data_overrides) -> StudyConfig:
    """Monta Drive, crea carpetas y carga/guarda la configuración del estudio.
    Cada perfil vive en su propia carpeta: <drive_base>/<profile>.

    La configuración queda congelada en <drive>/config/study.json la primera vez. En sesiones
    posteriores se carga la guardada y los overrides se aplican ENCIMA de ella: si piden algo distinto,
    se detiene (para no mezclar corridas con configuraciones distintas) salvo allow_config_change=True,
    que archiva la anterior en config/history/."""
    import copy
    if mount and in_colab():
        from google.colab import drive
        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")
    paths = Paths(drive_root=os.path.join(drive_base, profile), cache_root=cache_root).make()
    cfg_path = paths.d("config", "study.json")
    saved = None
    if os.path.exists(cfg_path):
        saved = StudyConfig.load(cfg_path)
        saved.paths = paths
        study = copy.deepcopy(saved)
    else:
        study = StudyConfig(paths=paths, data=data_config(profile))
        if profile == "smoke":
            study.train = TrainConfig(width=128, depth=3, proj=64, batch_pre=256, batch_mid=64, eval_every=25,
                                      patience=4, pre_epochs=2.0, mid_epochs=1.0, ckpt_every_steps=50)
            study.hpo = HPOConfig(n_trials=3, budget_frac=0.5)
            study.eval = EvalConfig(n_geo=500, eval_batch=512)
            study.f1_seeds = [0, 1]
    if data_overrides:
        study.data = dataclasses.replace(study.data, **data_overrides)
    if train_overrides:
        study.train = dataclasses.replace(study.train, **train_overrides)
    if hpo_overrides:
        study.hpo = dataclasses.replace(study.hpo, **hpo_overrides)
    for k, v in (study_overrides or {}).items():
        if k not in ("f1_seeds", "f3_seeds", "rope_pp"):
            raise KeyError(f"study_overrides admite f1_seeds, f3_seeds y rope_pp; no {k!r}")
        setattr(study, k, v)
    if saved is None:
        study.save(cfg_path)
    else:
        diff = _config_diff(saved.to_dict(), study.to_dict())
        if diff and not allow_config_change:
            raise RuntimeError(
                "La configuración pedida difiere de la guardada en Drive (" + cfg_path + "):\n  "
                + "\n  ".join(diff)
                + "\nPara no mezclar corridas incompatibles me detengo. Opciones: (a) deja los parámetros como "
                  "estaban; (b) usa otro DRIVE_BASE o perfil; (c) allow_config_change=True si sabes lo que haces "
                  "(las corridas ya hechas NO se rehacen solas).")
        if diff:
            save_json(saved.to_dict(), paths.d("config", "history", f"study_{now_iso().replace(':', '')}.json"))
            study.save(cfg_path)
    snap = env_snapshot()
    snap.update(disk_free_cache_gb=round(disk_free_gb(cache_root), 1), profile=profile)
    save_json(snap, paths.d("logs", f"env_{now_iso().replace(':', '')}.json"))
    fz = paths.d("logs", "pip_freeze.txt")
    if not os.path.exists(fz):
        with open(fz, "w") as f:
            f.write(pip_freeze())
    return study


def resources(study: StudyConfig) -> dict:
    """Chequeo de recursos antes de descargar."""
    need = study.data.disk_gb() if not study.data.synthetic else 0.1
    free = disk_free_gb(study.paths.cache_root)
    r = {"gpu": gpu_info(), "ram_gb": round(ram_gb(), 1), "disco_libre_gb": round(free, 1),
         "disco_necesario_gb": round(need, 1), "alcanza": free > need * 1.05}
    if not r["alcanza"]:
        r["sugerencia"] = "usa un perfil menor (mini) o reduce pre_chunks; ver config.PROFILES"
    return r


# --------------------------------------------------------------------------------------
def make_store(study: StudyConfig) -> Store:
    p, d = study.paths, study.data
    if d.synthetic:
        pre_root, hn_root = p.c("synthetic", "pre"), p.c("synthetic", "hn")
        info = make_synthetic_store(pre_root, hn_root, d.shard, n_chunks=max(d.pre_chunks) + 1,
                                    hn_chunks=d.hn_chunks, k=d.k_neg, **d.syn)
        fp = p.d("data", "SYNTHETIC_fingerprint.json")
        old = load_json(fp) if os.path.exists(fp) else None
        if old is not None and old != info:
            raise RuntimeError(f"Los datos sintéticos cambiaron respecto de los derivados guardados en {p.d('data')}. "
                               f"Borra la carpeta {p.drive_root} (sólo es el smoke) y vuelve a correr.")
        if old is None:
            save_json(info, fp)
        return Store(pre_root, hn_root, d.shard)
    return Store(p.c("hf", slug(PRETRAIN_REPO)), p.c("hf", slug(HN_REPO)), d.shard, downloader=HFDownloader(),
                 pre_repo=PRETRAIN_REPO, hn_repo=HN_REPO)


@dataclass
class DataBundle:
    study: StudyConfig
    store: Store
    split: Split
    masks_pre: dict
    masks_mid: dict
    val: EvalSet
    test: EvalSet
    hidden: int
    test_split: str = "test"      # "test" (exploratorio) o "confirm" (sellado; sólo tras el freeze)
    val_mine_idx: np.ndarray | None = None   # pool de val-mine (A1.2); None si falta F0.5b

    @property
    def dcfg(self):
        return self.study.data

    @property
    def pools_dir(self) -> str:
        return self.study.paths.d("data", "eval", self.test_split)

    @property
    def gen2_dir(self) -> str:
        return self.study.paths.d("data", "eval", "gen2" if self.test_split == "test" else f"gen2_{self.test_split}")

    @property
    def tgen_dir(self) -> str:
        return self.study.paths.d("data", "eval", "tgen", self.test_split)


def _localize(src: str, dst: str) -> str:
    """Copia un set de evaluación de Drive a disco local (lecturas más rápidas)."""
    if os.path.exists(os.path.join(dst, "manifest.json")):
        return dst
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    os.replace(tmp, dst)
    return dst


def freeze_path(study: StudyConfig) -> str:
    return study.paths.d("config", "PROTOCOL_FREEZE.json")


def load_bundle(study: StudyConfig, localize: bool = True, test_split: str = "test") -> DataBundle:
    """Carga splits, máscaras y sets de evaluación. test_split='confirm' sólo después del freeze."""
    p, d = study.paths, study.data
    if test_split not in ("test", "confirm"):
        raise ValueError("test_split debe ser 'test' o 'confirm'")
    if test_split == "confirm" and not os.path.exists(freeze_path(study)):
        raise PermissionError("El test confirmatorio está SELLADO: primero ejecuta la celda de PROTOCOL FREEZE "
                              "(protocol_freeze). Después del freeze no se optimiza nada con este test.")
    store = make_store(study)
    sdir = p.d("data", "splits")
    split = Split.load(sdir)
    masks_pre = train_masks(store, split, d.pre_chunks, sdir, "pre")
    masks_mid = train_masks(store, split, d.hn_chunks, sdir, "mid")
    vdir, tdir = p.d("data", "eval", "val"), p.d("data", "eval", test_split)
    if localize and not d.synthetic:
        vdir = _localize(vdir, p.c("eval_local", "val"))
        tdir_local = _localize(tdir, p.c("eval_local", test_split))
    else:
        tdir_local = tdir
    val, test = EvalSet(vdir), EvalSet(tdir_local)
    hidden = int(val.q_all().shape[1])
    # los pools se leen de Drive (la copia local de val puede ser anterior a F0.5b)
    pv = p.d("data", "eval", "val", "mine_idx.npy")
    vm = np.load(pv) if os.path.exists(pv) else None
    return DataBundle(study, store, split, masks_pre, masks_mid, val, test, hidden, test_split, vm)


def extra_sets(bundle: DataBundle) -> dict:
    """Conjuntos externos que se evalúan en cada corrida: el panel T-gen (todos sus generadores con
    negativos) y, si existiera, el T-gen2 de v0.2."""
    from .generators import load_panel_sets
    from .texts import load_gen2
    out = {}
    g = load_gen2(bundle.gen2_dir)
    if g is not None:
        out["tgen2"] = g
    out.update(load_panel_sets(bundle.study, bundle.test_split))
    return out


def items_map_of(bundle: DataBundle, ex: dict | None = None, panel: dict | None = None) -> dict:
    ex = extra_sets(bundle) if ex is None else ex
    m = {k: np.asarray(v[0]) for k, v in ex.items()}
    if panel and panel.get("macro"):
        m["tgenmacro"] = np.asarray(panel["macro_items"], np.int64)
    return m


# --------------------------------------------------------------------------------------
def recipe_decision(study) -> dict | None:
    """Decisión de receta para F1b (regla pre-especificada de A1.5, la escribe f0_7b_recipes)."""
    return load_json(study.paths.d("config", "RECETA_F1b.json"))


def phase_settings(study, phase: str | None) -> dict:
    """Overrides de TrainConfig de cada fase (se aplican antes del HPO).
    F1 / F0.7 / F2 (v0.2): ninguno. F1b y F2b (enmienda A1): selección por val-mine, receta decidida en
    F0.7b y "HN + in-batch" en las pérdidas InfoNCE con HN. F3: lo congelado en PROTOCOL_FREEZE.json."""
    if phase in A1_PHASES:
        dec = recipe_decision(study)
        if dec is None:
            raise RuntimeError("Falta la decisión de receta de F1b (config/RECETA_F1b.json): corre F0.7b (celda 3.8b).")
        s = dict(recipe_overrides(study, dec["receta"]))
        s["select_metric"] = SELECT_A1
        s["loss_params"] = {"hn_denominator": "hn_plus_inbatch"}
        return s
    if phase == "F3":
        frz = load_json(freeze_path(study)) or {}
        return dict(frz.get("phase_settings") or {})
    return {}


def _apply(c: TrainConfig, kw: dict) -> TrainConfig:
    for k, v in kw.items():
        if k == "loss_params":
            c.loss_params = {**c.loss_params, **(v or {})}
        elif not hasattr(c, k):
            raise AttributeError(f"TrainConfig no tiene {k!r}")
        else:
            setattr(c, k, v)
    return c


def base_cfg(study: StudyConfig, arm: str, phase: str | None = None, **kw) -> TrainConfig:
    """Config base de un brazo: la del estudio + ajustes de la fase + kw. `loss_params` se fusiona."""
    c = dataclasses.replace(study.train, arm=arm, loss_params=dict(study.train.loss_params))
    if phase:
        _apply(c, phase_settings(study, phase))
    return _apply(c, kw)


def run_one(study, bundle, cfg: TrainConfig, phase: str, name: str, evaluate: bool = True,
            row_fraction: float = 1.0, cleanup: bool = False) -> dict:
    rd = study.paths.d("runs", phase, name)
    return run_training(cfg, bundle, rd, ecfg=study.eval, evaluate_test=evaluate, row_fraction=row_fraction,
                        pools_dir=bundle.pools_dir if evaluate else None,
                        extra_sets=extra_sets(bundle) if evaluate else None, cleanup_heavy=cleanup)


def run_arm(study, bundle, arm: str, phase: str = "F1", seeds=None, n_trials: int | None = None,
            budget_frac: float | None = None, **cfg_kw) -> dict:
    """Una celda = un brazo: búsqueda de HP (reanudable) + semillas finales + evaluación."""
    if arm not in ARMS:
        raise KeyError(f"brazo desconocido {arm!r}; opciones: {arm_names()}")
    arm_dir = study.paths.d("runs", phase, arm)
    makedirs(arm_dir)
    log = RunLogger(arm_dir, "arm_log.jsonl", prefix=f"[{phase}:{arm}] ")
    base = base_cfg(study, arm, phase=phase, **cfg_kw)
    nt = n_trials if n_trials is not None else study.hpo.n_trials
    bf = budget_frac if budget_frac is not None else study.hpo.budget_frac

    def trial_fn(cfg, rd, frac):
        return run_training(cfg, bundle, rd, ecfg=study.eval, evaluate_test=False, row_fraction=frac,
                            cleanup_heavy=True, export_clm=False)
    hpo = run_hpo(arm, base, bundle, arm_dir, nt, bf, study.hpo.seed, trial_fn, log=log.log,
                  phase=phase if phase in A1_PHASES else None)
    best = hpo["best"]["overrides"]
    log.log("mejor configuración", overrides=best, val=hpo["best"]["val_best"], select_metric=base.select_metric)
    results = []
    for s in (seeds if seeds is not None else study.f1_seeds):
        cfg = apply_overrides(base, best)
        cfg.arm, cfg.seed = arm, int(s)
        rd = os.path.join(arm_dir, f"seed_{s}")
        res = run_training(cfg, bundle, rd, ecfg=study.eval, evaluate_test=True, pools_dir=bundle.pools_dir,
                           extra_sets=extra_sets(bundle))
        results.append(res)
        t = res.get("test", {})
        log.log(f"semilla {s}", status=res.get("status"), tmine=(t.get("tmine") or {}).get("top1"),
                tgem=(t.get("tgem") or {}).get("top1"), tgem_barajado=(t.get("tgem_shuf") or {}).get("top1"))
    summary = {"arm": arm, "phase": phase, "best_overrides": best, "hpo_best_val": hpo["best"]["val_best"],
               "select_metric": base.select_metric, "evp_curve": hpo["evp_curve"], "seeds": results}
    save_json(summary, os.path.join(arm_dir, "ARM_SUMMARY.json"))
    return summary


def best_overrides(study, arm: str, phase: str = "F1") -> dict:
    """Hiperparámetros elegidos en validación para un brazo (hpo_summary.json de la fase)."""
    s = load_json(study.paths.d("runs", phase, arm, "hpo_summary.json"))
    if s is None:
        raise FileNotFoundError(f"{arm}: falta runs/{phase}/{arm}/hpo_summary.json (corre primero su celda de {phase})")
    return dict(s["best"]["overrides"])


def run_variant(study, bundle, arm: str, phase: str, name: str, seeds=None, hp_from: str | None = "F1",
                evaluate: bool = True, **cfg_kw) -> list:
    """Una variante (F2/F2b: regímenes de HN, currículo, batch, ...) con los HP de `hp_from` (sin re-tuning).
    Con hp_from="F1b" hereda también los ajustes de la enmienda A1 (val-mine, receta, "HN + in-batch").
    Cada semilla queda en runs/<phase>/<name>/seed_<s> y se reanuda sola."""
    base = base_cfg(study, arm, phase=hp_from if hp_from in A1_PHASES else None)
    if hp_from:
        base = apply_overrides(base, best_overrides(study, arm, hp_from))
    _apply(base, cfg_kw)
    out = []
    for s in (seeds if seeds is not None else study.f1_seeds):
        cfg = dataclasses.replace(base, arm=arm, seed=int(s), loss_params=dict(base.loss_params))
        rd = study.paths.d("runs", phase, name, f"seed_{s}")
        out.append(run_training(cfg, bundle, rd, ecfg=study.eval, evaluate_test=evaluate,
                                pools_dir=bundle.pools_dir if evaluate else None,
                                extra_sets=extra_sets(bundle) if evaluate else None))
    return out


TOP_ARMS_RULE = ("B1 régimen R2 (decidido el 2026-10-02, antes de F1b): los 2 brazos SIN HN con mayor val_best "
                 "(top-1 en val-mine) de F1b; se excluyen los que quedan a ≤ 2 pp del azar (1/11), es decir, colapsados.")


def top_arms_rule(study, k: int = 2, phase: str = "F1b") -> dict:
    """Brazos para B1-R2 por la regla fija TOP_ARMS_RULE. Se guarda en results/F2b/B1/TOP_ARMS.json la primera vez
    que hay datos de los 15 brazos y después no cambia. {"arms": [...]} o {"arms": [], "nota": ...}."""
    from .losses import arm_names
    p = study.paths.d("results", "F2b", "B1", "TOP_ARMS.json")
    old = load_json(p)
    if old and old.get("arms"):
        return old
    vals, falta = {}, []
    for a in arm_names():
        if ARMS[a]["hn"]:
            continue
        s = load_json(study.paths.d("runs", phase, a, "hpo_summary.json"))
        v = ((s or {}).get("best") or {}).get("val_best")
        if v is None:
            falta.append(a)
        else:
            vals[a] = float(v)
    if falta:
        return {"arms": [], "nota": f"faltan brazos sin HN de {phase}: {falta}"}
    ok = {a: v for a, v in vals.items() if v > 1.0 / 11 + 0.02}
    arms = sorted(ok, key=lambda a: -ok[a])[:k]
    rec = {"arms": arms, "val_best": vals, "excluidos_por_colapso": sorted(set(vals) - set(ok)), "rule": TOP_ARMS_RULE,
           "phase": phase, "created": now_iso()}
    save_json(rec, p)
    return rec


def protocol_freeze(study, phase: str = "F1b", note: str = "", force: bool = False) -> dict:
    """PROTOCOL FREEZE (§7.8): congela HP por brazo, ajustes de la fase (receta, selección), brazo no
    contrastivo de C1, métricas, panel T-gen y estadística. Después sólo se corre F3 (sin tuning) y recién
    entonces se abre el test confirmatorio."""
    from .generators import load_panel
    from .utils import code_hash
    fp = freeze_path(study)
    if os.path.exists(fp) and not force:
        return load_json(fp)
    arms = {}
    missing = []
    for arm in arm_names():
        try:
            arms[arm] = {"overrides": best_overrides(study, arm, phase),
                         "val_best": (load_json(study.paths.d("runs", phase, arm, "hpo_summary.json")) or {})
                         .get("best", {}).get("val_best")}
        except FileNotFoundError:
            missing.append(arm)
    if missing:
        raise RuntimeError(f"No se puede congelar: faltan brazos de {phase}: {missing}")
    pool = [a for a in NONCONTRASTIVE_POOL if a in arms]
    best_nc = max(pool, key=lambda a: arms[a]["val_best"] if arms[a]["val_best"] is not None else -1)
    settings = phase_settings(study, phase)
    a1 = phase in A1_PHASES
    panel = load_panel(study, "test") or {}
    frz = {"created": now_iso(), "code_hash": code_hash(), "source_phase": phase, "note": note,
           "study_config": study.to_dict(), "arms": arms, "phase_settings": settings,
           "recipe": (recipe_decision(study) or {}).get("receta") if a1 else "actual (F1 piloto)",
           "select_metric": settings.get("select_metric", study.train.select_metric),
           "C1_best_noncontrastive": best_nc,
           "C1_rule": f"mejor no contrastiva elegida por val_best ({settings.get('select_metric', 'val_top1')}) en {phase}",
           "primary_metric": ("top-1 en T-mine (test confirmatorio); co-primaria: macro de T-gen con los generadores "
                              "del panel congelado, Holm entre ambas (ENMIENDA_03, A1.1)") if a1
           else "top-1 en T-gem (test confirmatorio)",
           "tgen_panel": {"version": panel.get("version"), "macro": panel.get("macro"),
                          "n_macro_items_test": panel.get("n_macro_items"), "build": panel.get("build"),
                          "build_txt": panel.get("build_txt"), "rules": panel.get("rules")},
           "secondary_metrics": ["top-1 en T-gem (con estados barajados), T-rand, T-mine-50 y cada generador del panel",
                                 "AUROC de abstención en T-abst-mine (primaria de H2) y T-abst de Gemini",
                                 "ECE listwise con temperature scaling", "MRR"],
           "seeds": list(study.f3_seeds), "rope_pp": study.rope_pp,
           "statistics": "bootstrap jerárquico pareado (documentos; semillas dentro de brazo), Holm sobre los "
                         "105 pares por métrica, TOST ±ROPE, Friedman + Nemenyi; C1 como contraste confirmatorio",
           "success_criterion_C1": f"|Δ| ≥ {study.rope_pp} pp con p_Holm < 0.05 (superioridad/inferioridad) "
                                   f"o equivalencia por TOST dentro de ±{study.rope_pp} pp",
           "secondary_confirmatory": {
               "C2": ("H1b (secundario pre-especificado; ENMIENDA_03 §12.4): por par (infonce_hn − infonce; "
                      "siglip_hn − siglip), C2 = Δ(T-gem) − Δ(macro de T-gen) en los ítems del macro del test "
                      "confirmatorio, bootstrap pareado por documento; H1b se sostiene en un par si los HN suben T-gem "
                      f"(Δ > +{study.rope_pp} pp, IC95 > 0) y C2 > +{study.rope_pp} pp con p_Holm < 0.05 (Holm entre "
                      "los dos pares). No cambia C1. C2 mezcla "
                      "generador y tipo de negativo (libre frente a editado); C2b los separa."),
               "C2b": ("Descomposición de C2 (secundario pre-especificado; ENMIENDA_03 §13): sets S, P y E de los "
                       "mismos generadores del piloto, generados para los primeros ítems del panel del test "
                       "confirmatorio (8.2c). Por par de C2: generador = Δ(T-gem) − Δ(S) y construcción = Δ(S) − Δ(E) "
                       "en los ítems comunes, bootstrap pareado por documento, Holm entre los 4 contrastes. Un "
                       f"componente se confirma si es > +{study.rope_pp} pp con p_Holm < 0.05, sólo si la brecha "
                       f"total Δ(T-gem) − Δ(E) es > +{study.rope_pp} pp con IC95 > 0. El oro reescrito "
                       "(Δ(P) − Δ(S)) y las proporciones son descriptivos."),
               "C2c": c2c_spec_text(study.rope_pp)},
           "F3_variants": {k: {kk: (copy_changes(vv) if kk == "changes" else vv) for kk, vv in v.items()}
                           for k, v in F3_VARIANTS.items()}}
    save_json(frz, fp)
    lines = [f"# PROTOCOL FREEZE — {frz['created']}", "", f"code_hash: `{frz['code_hash']}` · fase de origen: "
             f"{phase} · receta: {frz['recipe']} · selección: {frz['select_metric']}", "",
             f"Métrica primaria: {frz['primary_metric']}", "",
             f"C1: **{best_nc}** vs **infonce** (elegido en validación de {phase})", "",
             f"Panel T-gen: versión {panel.get('version')} · construcción {panel.get('build')} "
             f"({panel.get('build_txt')}) · macro {panel.get('macro')}", "",
             f"Secundario pre-especificado: {frz['secondary_confirmatory']['C2']}", "",
             f"Secundario pre-especificado: {frz['secondary_confirmatory']['C2b']}", "",
             f"Secundario pre-especificado: {frz['secondary_confirmatory']['C2c']}", "",
             "Variantes de F3 (no entran al todos contra todos ni a C1): "
             + "; ".join(f"`{k}` = HP congelados de {v['arm']} + `{v['changes']}`" for k, v in frz["F3_variants"].items()),
             "", "| brazo | val_best | HP congelados |", "|---|---|---|"]
    for a, v in arms.items():
        lines.append(f"| {a} | {v['val_best']} | `{v['overrides']}` |")
    with open(study.paths.d("config", "PROTOCOL_FREEZE.md"), "w") as f:
        f.write("\n".join(lines))
    return frz


# Variantes de F3 registradas en el freeze (v0.3.5). C2c (ENMIENDA_03 §14): "HN solos" de infonce_hn, con los HP
# congelados de infonce_hn y sólo el denominador cambiado (el oro compite sólo contra sus HN). No entran al todos
# contra todos ni a C1.
F3_VARIANTS = {"infonce_hn_hnonly": {"arm": "infonce_hn", "changes": {"loss_params": {"hn_denominator": "hn_only"}},
                                     "uso": "C2c (mecanismo de H1b): 'HN solos' frente a infonce_hn ('HN + in-batch')"}}


def c2c_spec_text(rope_pp: float) -> str:
    """Regla de C2c tal como se fijó en ENMIENDA_03 §14 (la misma en el freeze y en la adenda)."""
    return ("Mecanismo de H1b (secundario pre-especificado; ENMIENDA_03 §14): en F3 se entrena además la "
            "variante 'HN solos' de infonce_hn (F3_variants: mismos HP congelados y semillas; sólo el "
            "denominador pasa a hn_only) y se compara con infonce_hn ('HN + in-batch'), bootstrap pareado "
            "por documento. Prueba 1: top-1 en T-gem con estados barajados, se confirma si "
            f"Δ > +{rope_pp} pp con p_Holm < 0.05 (aprende una preferencia que no depende del estado). "
            f"Prueba 2: top-1 en T-mine, se confirma si Δ < −{rope_pp} pp con p_Holm < 0.05 (pierde "
            "discriminación real). Holm entre las 2. Control, fuera de Holm: infonce_hn − infonce en T-gem "
            f"con estados barajados, equivalente por TOST ±{rope_pp} pp (con in-batch, los HN no "
            "crean esa preferencia). T-gem, T-rand, T-mine-50 y el macro de T-gen son descriptivos.")


def addendum_path(study) -> str:
    return study.paths.d("config", "PROTOCOL_FREEZE_ADDENDUM.json")


ADDENDUM_REASON = (
    "El PROTOCOL FREEZE se creó el {created} con el código {code} (notebook rev. 4.4), que todavía no registraba "
    "C2c. C2c quedó especificado en ENMIENDA_03 §14 el 2026-10-05, antes del freeze y antes de abrir el test "
    "confirmatorio, y el investigador confirmó el freeze sabiendo que lo incluía. Esta adenda lo registra sin "
    "modificar el freeze, antes de entrenar la variante.")
ADDENDUM_DEVIATION = (
    "C2c queda registrado fuera del JSON del freeze y se ejecuta después de ver los resultados de F3 de los 15 "
    "brazos. Su especificación (variante, pruebas, umbrales y Holm) es la de ENMIENDA_03 §14, sin cambios. "
    "Al reportarlo hay que declarar esta desviación.")


def freeze_addendum(study) -> dict | None:
    """Adenda del freeze (v0.3.6): si el freeze no registra las variantes de F3 (se congeló con v0.3.4), registra
    una sola vez C2c y su variante tal como los fija ENMIENDA_03 §14. No toca el freeze; si la adenda ya existe, la
    devuelve sin cambios. None si no hay freeze o si el freeze ya trae las variantes."""
    from .utils import code_hash
    frz = load_json(freeze_path(study))
    if frz is None or frz.get("F3_variants"):
        return None
    ap = addendum_path(study)
    old = load_json(ap)
    if old is not None:
        return old
    add = {"created": now_iso(), "code_hash": code_hash(), "freeze_created": frz.get("created"),
           "freeze_code_hash": frz.get("code_hash"),
           "motivo": ADDENDUM_REASON.format(created=frz.get("created"), code=frz.get("code_hash")),
           "desviacion": ADDENDUM_DEVIATION, "registro": "ENMIENDA_03 §14 (2026-10-05)",
           "secondary_confirmatory": {"C2c": c2c_spec_text(frz.get("rope_pp", study.rope_pp))},
           "F3_variants": {k: {kk: (copy_changes(vv) if kk == "changes" else vv) for kk, vv in v.items()}
                           for k, v in F3_VARIANTS.items()}}
    save_json(add, ap)
    lines = [f"# ADENDA AL PROTOCOL FREEZE — {add['created']}", "",
             f"Freeze: {add['freeze_created']} · código congelado `{add['freeze_code_hash']}` · código de la adenda "
             f"`{add['code_hash']}`", "", f"**Motivo.** {add['motivo']}", "", f"**Desviación.** {add['desviacion']}", "",
             f"Secundario pre-especificado: {add['secondary_confirmatory']['C2c']}", "",
             "Variantes de F3 (no entran al todos contra todos ni a C1): "
             + "; ".join(f"`{k}` = HP congelados de {v['arm']} + `{v['changes']}`" for k, v in add["F3_variants"].items())]
    with open(study.paths.d("config", "PROTOCOL_FREEZE_ADDENDUM.md"), "w") as f:
        f.write("\n".join(lines))
    return add


def frozen_variants(study) -> tuple:
    """(variantes de F3, dónde quedaron registradas: "freeze" | "adenda" | None)."""
    frz = load_json(freeze_path(study)) or {}
    if frz.get("F3_variants"):
        return frz["F3_variants"], "freeze"
    add = load_json(addendum_path(study)) or {}
    if add.get("F3_variants"):
        return add["F3_variants"], "adenda"
    return {}, None


def confirm_variant_cfg(study, frz: dict, arm: str, changes: dict, seed: int) -> TrainConfig:
    """Config de una corrida de F3: ajustes congelados de la fase + HP congelados de `arm` + `changes`."""
    cfg = apply_overrides(_apply(base_cfg(study, arm), dict(frz.get("phase_settings") or {})), frz["arms"][arm]["overrides"])
    _apply(cfg, copy_changes(changes))
    return dataclasses.replace(cfg, arm=arm, seed=int(seed), loss_params=dict(cfg.loss_params))


def copy_changes(changes: dict | None) -> dict:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in (changes or {}).items()}


def run_confirm_variant(study, bundle_c, name: str, seeds=None) -> dict:
    """F3: una variante registrada en el freeze (`F3_variants`), con los HP congelados de su brazo y sólo los
    cambios registrados. Queda en runs/F3/<name>/seed_<s> y se reanuda sola."""
    if bundle_c.test_split != "confirm":
        raise ValueError("usa bundle_c = load_bundle(study, test_split='confirm')")
    from .utils import code_hash
    frz = load_json(freeze_path(study))
    variants, where = frozen_variants(study)
    var = variants.get(name)
    if var is None:
        raise KeyError(f"{name!r} no está registrada en el freeze ni en su adenda (F3_variants): no se puede correr en F3")
    if name in ARMS:
        raise ValueError(f"{name!r} es un brazo: las variantes de F3 necesitan otro nombre")
    var_dir = study.paths.d("runs", "F3", name)
    makedirs(var_dir)
    h = code_hash()
    if h != frz.get("code_hash"):
        print(f"[F3] ADVERTENCIA: el código actual ({h}) no es el congelado ({frz.get('code_hash')}). "
              "Si el cambio toca entrenamiento o evaluación, F3 deja de ser confirmatorio; regístralo.")
    results = []
    for s in (seeds if seeds is not None else frz["seeds"]):
        cfg = confirm_variant_cfg(study, frz, var["arm"], var["changes"], s)
        res = run_training(cfg, bundle_c, os.path.join(var_dir, f"seed_{s}"), ecfg=study.eval, evaluate_test=True,
                           pools_dir=bundle_c.pools_dir, extra_sets=extra_sets(bundle_c), export_clm=False)
        results.append(res)
    summary = {"variant": name, "arm": var["arm"], "phase": "F3", "frozen_overrides": frz["arms"][var["arm"]]["overrides"],
               "changes": var["changes"], "registro": where, "code_hash": h, "code_hash_frozen": frz.get("code_hash"),
               "seeds": results}
    save_json(summary, os.path.join(var_dir, "VARIANT_SUMMARY.json"))
    return summary


def run_confirm_arm(study, bundle_c, arm: str, seeds=None) -> dict:
    """F3 (confirmatorio): HP congelados, sin tuning, evaluación en el test sellado."""
    if bundle_c.test_split != "confirm":
        raise ValueError("usa bundle_c = load_bundle(study, test_split='confirm')")
    from .utils import code_hash
    frz = load_json(freeze_path(study))
    ov = frz["arms"][arm]["overrides"]
    arm_dir = study.paths.d("runs", "F3", arm)
    makedirs(arm_dir)
    h = code_hash()
    if h != frz.get("code_hash"):
        print(f"[F3] ADVERTENCIA: el código actual ({h}) no es el congelado ({frz.get('code_hash')}). "
              "Si el cambio toca entrenamiento o evaluación, F3 deja de ser confirmatorio; regístralo.")
    results = []
    for s in (seeds if seeds is not None else frz["seeds"]):
        cfg = apply_overrides(_apply(base_cfg(study, arm), dict(frz.get("phase_settings") or {})), ov)
        cfg.arm, cfg.seed = arm, int(s)
        res = run_training(cfg, bundle_c, os.path.join(arm_dir, f"seed_{s}"), ecfg=study.eval, evaluate_test=True,
                           pools_dir=bundle_c.pools_dir, extra_sets=extra_sets(bundle_c),
                           export_clm=(int(s) == int(frz["seeds"][0])))   # T-Rex usa la primera semilla
        results.append(res)
    summary = {"arm": arm, "phase": "F3", "frozen_overrides": ov, "code_hash": h,
               "code_hash_frozen": frz.get("code_hash"), "seeds": results}
    save_json(summary, os.path.join(arm_dir, "ARM_SUMMARY.json"))
    return summary


def export_head_from_run(study, run_dir: str) -> str:
    """Exporta clm_head.pt (formato clm-serve) desde best_<última etapa>.pt de una corrida terminada."""
    from .config import train_config_from_dict
    from .train import Trainer, export_clm_head
    cfg = train_config_from_dict(load_json(os.path.join(run_dir, "config.json"))["train"])

    class _B:   # el Trainer sólo necesita la dimensión de entrada para construir los cabezales
        hidden = int(load_json(os.path.join(run_dir, "eval", "metrics.json"), {}).get("hidden", 4096) or 4096)
        dcfg = study.data
    tr = Trainer(cfg, _B(), run_dir)
    if not tr.load_best(tr.stages[-1]):
        raise FileNotFoundError(f"{run_dir}: falta best_{tr.stages[-1]}.pt")
    return export_clm_head(tr.heads, load_json(os.path.join(run_dir, "eval", "metrics.json")), run_dir)


def iter_run_dirs(study, phase: str) -> list:
    """Carpetas seed_* terminadas (status ok) de una fase, a cualquier profundidad (F0.7/<cond>/seed_s,
    F0.7b/<receta>/<cond>/seed_s, F1/<brazo>/seed_s), sin los trials del HPO."""
    base = study.paths.d("runs", phase)
    out = []
    if not os.path.isdir(base):
        return out
    for root, dirs, _ in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d != "hpo")
        if os.path.basename(root).startswith("seed_"):
            done = read_done(root)
            if done and done.get("status") == "ok":
                out.append(root)
            dirs[:] = []
    return sorted(out)


def add_missing_evals(study, bundle, phase="F1", arms=None, log=print) -> int:
    """Agrega a corridas YA terminadas lo que su evaluación no tenga (v0.3: estados barajados, T-abst-mine,
    T-mine-50 y el panel T-gen). No re-entrena: carga el mejor checkpoint de la última etapa y evalúa sólo
    los bloques que faltan (los ya hechos no se tocan). `phase` puede ser una lista de fases."""
    from .config import train_config_from_dict
    from .evaluate import evaluate_run, expected_eval_keys
    from .generators import load_panel, macro_for_run
    from .train import Trainer
    ex = extra_sets(bundle)
    panel = load_panel(study, bundle.test_split)
    want = expected_eval_keys(bundle.pools_dir, list(ex))
    n = 0
    for ph in ([phase] if isinstance(phase, str) else list(phase)):
        for rd in iter_run_dirs(study, ph):
            rel = os.path.relpath(rd, study.paths.d("runs"))
            if arms and not any(f"/{a}/" in f"/{rel}/" for a in arms):
                continue
            have = load_json(os.path.join(rd, "eval", "metrics.json"), {}) or {}
            sigs = load_json(os.path.join(rd, "eval", "extra_sigs.json"), {}) or {}
            stale = [k for k in ex if k in have and os.path.exists(os.path.join(rd, "eval", f"items_{k}.npy"))
                     and not np.array_equal(np.load(os.path.join(rd, "eval", f"items_{k}.npy")), np.asarray(ex[k][0]))]
            stale += [k for k in ex if k in have and len(ex[k]) > 2 and ex[k][2] is not None and sigs.get(k) != ex[k][2]
                      and k not in stale]
            need = [k for k in want if k not in have or (k in ex and "shuf_top1" not in (have.get(k) or {}))] + stale
            if need:
                cfg = train_config_from_dict(load_json(os.path.join(rd, "config.json"))["train"])
                tr = Trainer(cfg, bundle, rd)
                if not tr.load_best(tr.stages[-1]):
                    log(f"[re-eval] {rel}: falta best_{tr.stages[-1]}.pt; se omite")
                    continue
                metrics = evaluate_run(tr.heads, tr.loss, bundle.test, bundle.val, os.path.join(rd, "eval"),
                                       tr.device, study.eval, pools_dir=bundle.pools_dir, extra_sets=ex,
                                       log=lambda *a, **k: None)
                done = read_done(rd)
                done["test"] = {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
                                for k, v in metrics.items() if isinstance(v, dict)}
                done["evals_added"] = sorted(set(done.get("evals_added", [])) | set(need))
                done["evals_added_at"] = now_iso()          # finished_at (fin del entrenamiento) no se toca
                save_json(done, os.path.join(rd, "DONE.json"))
                n += 1
                log(f"[re-eval] {rel}: agregado {len(need)} bloques")
            if panel:
                macro_for_run(rd, panel)
    return n


def eval_extra(study, bundle, run_dirs, ex: dict, log=print) -> int:
    """Evalúa corridas YA terminadas en conjuntos adicionales `ex` ({nombre: (ítems, negativos[, huella,
    positivos])}), sin re-entrenar y sólo lo que falte (o lo que cambió de ítems o de contenido). Lo usan los
    análisis que miran sets fuera del panel (p. ej. los del piloto en EXP-006c). No toca DONE.json: estos sets
    no son parte de la evaluación oficial de la corrida."""
    from .config import train_config_from_dict
    from .evaluate import evaluate_run
    from .train import Trainer
    n = 0
    for rd in run_dirs:
        ev = os.path.join(rd, "eval")
        have = load_json(os.path.join(ev, "metrics.json"), {}) or {}
        sigs = load_json(os.path.join(ev, "extra_sigs.json"), {}) or {}
        need = []
        for k, v in ex.items():
            p_items = os.path.join(ev, f"items_{k}.npy")
            if (k not in have or "shuf_top1" not in (have.get(k) or {}) or not os.path.exists(p_items)
                    or not os.path.exists(os.path.join(ev, f"correct_{k}_shuf.npy"))
                    or not np.array_equal(np.load(p_items), np.asarray(v[0]))
                    or (len(v) > 2 and v[2] is not None and sigs.get(k) != v[2])):
                need.append(k)
        if not need:
            continue
        cfg = train_config_from_dict(load_json(os.path.join(rd, "config.json"))["train"])
        tr = Trainer(cfg, bundle, rd)
        if not tr.load_best(tr.stages[-1]):
            log(f"[eval-extra] {os.path.relpath(rd, study.paths.d('runs'))}: falta best_{tr.stages[-1]}.pt; se omite")
            continue
        evaluate_run(tr.heads, tr.loss, bundle.test, bundle.val, ev, tr.device, study.eval,
                     pools_dir=bundle.pools_dir, extra_sets={k: ex[k] for k in need}, log=lambda *a, **k: None)
        n += 1
    return n


def progress(study, phase: str = "F1"):
    """Tabla de avance por brazo (para retomar después de un corte)."""
    import pandas as pd
    rows = []
    for arm in arm_names():
        ad = study.paths.d("runs", phase, arm)
        hpo = load_json(os.path.join(ad, "hpo_summary.json"), {}) or {}
        seeds = [d for d in (os.listdir(ad) if os.path.isdir(ad) else []) if d.startswith("seed_")]
        done = [d for d in seeds if (read_done(os.path.join(ad, d)) or {}).get("status") == "ok"]
        tgem = [(read_done(os.path.join(ad, d)) or {}).get("test", {}).get("tgem", {}).get("top1") for d in done]
        tgem = [x for x in tgem if x is not None]
        rows.append({"#": ARMS[arm]["num"], "brazo": arm, "familia": ARMS[arm]["family"],
                     "trials_hpo": len(hpo.get("trials", [])), "semillas_ok": len(done),
                     "top1_tgem_media": round(float(np.mean(tgem)), 4) if tgem else None})
    return pd.DataFrame(rows).sort_values("#")


def _verdict(c: dict, rope_pp: float, a: str, b: str = "infonce", p_key: str = "p") -> str:
    rope = rope_pp / 100
    sig = c.get(p_key, 1.0) < 0.05
    if sig and abs(c["diff"]) >= rope:
        return f"{a} mejor" if c["diff"] > 0 else f"{b} mejor"
    if sig:
        return "diferencia significativa pero < ROPE (sin relevancia práctica)"
    if c["tost_equivalent"]:
        return f"equivalentes (TOST ±{rope_pp} pp)"
    return "no concluyente"


def default_testsets(phase: str) -> tuple:
    """Métrica primaria primero (A1.1): T-mine y el macro de T-gen; después las secundarias."""
    if phase in A1_PHASES or phase == "F3":
        return ("tmine", "tgenmacro", "trand", "tgem", "tgem_shuf", "tmine50", "tgen2")
    return ("tgem", "trand", "tmine", "tgen2")


def run_all_vs_all(study, bundle, phase: str = "F1", testsets=None, B: int = 2000, with_panel_sets: bool = False):
    """Todos contra todos de una fase. En F1b/F3 (enmienda A1): C1 en T-mine y en el macro de T-gen, con
    Holm entre las dos (co-primarias); el brazo no contrastivo se elige por val-mine (F1b) o es el congelado
    (F3). En F1 (piloto): C1 en T-gem, como en v0.2."""
    from .generators import load_panel, macro_for_run
    from .stats import all_vs_all, collect_failures, collect_runs, contrast, clusters_from_docs, holm
    if (phase == "F3") != (bundle.test_split == "confirm"):
        raise ValueError("F3 se analiza con el bundle confirmatorio y F1/F2 con el exploratorio")
    runs = collect_runs(study.paths.runs, phase, arm_names())
    failed = collect_failures(study.paths.runs, phase, arm_names())
    out_dir = study.paths.d("results", phase, "all_vs_all")
    ex = extra_sets(bundle)
    panel = load_panel(study, bundle.test_split)
    for lst in runs.values():
        for _, rd in lst:
            if panel:
                macro_for_run(rd, panel)
    ts = list(testsets or default_testsets(phase))
    if with_panel_sets:
        ts += sorted(k for k in ex if k.startswith("tgen_"))
    a1 = phase in A1_PHASES or (phase == "F3" and (load_json(freeze_path(study)) or {}).get("source_phase") in A1_PHASES)
    report = all_vs_all(runs, ts, bundle.test.doc_ids, out_dir, rope_pp=study.rope_pp, B=B,
                        failed=failed, items_map=items_map_of(bundle, ex, panel),
                        abst_sets=("tmine", "tgem") if a1 else ("tgem",))
    report["code_hashes"] = {a: sorted({(load_json(os.path.join(rd, "config.json"), {}) or {}).get("env", {})
                                        .get("code_hash") for _, rd in lst} - {None}) for a, lst in runs.items()}
    mixed = {a: h for a, h in report["code_hashes"].items() if len(h) > 1}
    if len({h for hs in report["code_hashes"].values() for h in hs}) > 1:
        print(f"[stats] ATENCIÓN: las corridas usan más de una versión del código: {report['code_hashes']}")
    report["code_mixed_within_arm"] = mixed
    report["panel_version"] = (panel or {}).get("version")
    # C1 (H1a): mejor no contrastiva (elegida en VALIDACIÓN) vs InfoNCE.
    pool = [a for a in NONCONTRASTIVE_POOL if a in runs]
    if "infonce" in runs and pool:
        sel_phase = "F1b" if phase in A1_PHASES else phase
        if phase == "F3":
            sel_phase = f"PROTOCOL_FREEZE ({(load_json(freeze_path(study)) or {}).get('source_phase')})"

        def val_of(a):
            s = load_json(study.paths.d("runs", sel_phase, a, "hpo_summary.json"), {}) or {}
            return (s.get("best") or {}).get("val_best", -1)
        best_nc = load_json(freeze_path(study))["C1_best_noncontrastive"] if phase == "F3" else max(pool, key=val_of)
        metrics_c1 = ("tmine", "tgenmacro") if a1 else ("tgem",)
        per = {}
        for m in metrics_c1:
            files_a = [os.path.join(rd, "eval", f"correct_{m}.npy") for _, rd in runs[best_nc]]
            files_b = [os.path.join(rd, "eval", f"correct_{m}.npy") for _, rd in runs["infonce"]]
            if m == "tgenmacro" and not (panel and panel.get("macro")):
                continue
            if not all(os.path.exists(f) for f in files_a + files_b):
                continue
            A = np.vstack([np.load(f) for f in files_a]).astype(np.float32)
            Bm = np.vstack([np.load(f) for f in files_b]).astype(np.float32)
            if m == "tgenmacro":
                cl = clusters_from_docs(np.asarray(bundle.test.doc_ids)[np.asarray(panel["macro_items"])])
            else:
                cl = clusters_from_docs(bundle.test.doc_ids)
            per[m] = contrast(A, Bm, cl, study.rope_pp / 100, B=4000)
            per[m].update(A=best_nc, B="infonce")
        if a1 and "tgenmacro" not in per:
            report["C1_aviso"] = ("falta la co-primaria T-gen (macro) en alguna corrida de C1 o no hay panel: C1 queda sólo "
                                  "en T-mine. Corre 6.5 (o 8.2b + 8.18 en F3) para agregarla.")
            print("[C1] ATENCIÓN:", report["C1_aviso"])
        if per:
            adj = holm(np.array([c["p"] for c in per.values()]))
            for (m, c), pa in zip(per.items(), adj):
                for k in ("p_holm", "significant"):    # eran de una sola comparación: se reemplazan por los de C1
                    c.pop(k, None)
                c["p_holm_c1"] = float(pa)
                c["significant_c1"] = bool(pa < 0.05)
                c["verdict"] = _verdict(c, study.rope_pp, best_nc, p_key="p_holm_c1")
            prim = per.get(metrics_c1[0]) or next(iter(per.values()))
            c1 = dict(prim)
            c1.update(A=best_nc, B="infonce", best_noncontrastive=best_nc, confirmatory=(phase == "F3"),
                      metrica=metrics_c1[0] if metrics_c1[0] in per else next(iter(per)),
                      por_metrica=per, selection_phase=sel_phase,
                      holm_between=list(per) if a1 else None)
            report["C1"] = c1
        save_json(report, os.path.join(out_dir, "all_vs_all.json"))
    return report


# --------------------------------------------------------------------------------------
# Utilidades para mostrar resultados en el notebook
def summarize_arm(summary: dict):
    """Tabla compacta de un brazo: una fila por semilla."""
    import pandas as pd
    rows = []
    for r in summary.get("seeds", []):
        t = r.get("test", {})
        tg = [v.get("top1") for k, v in t.items() if k.startswith("tgen_") and isinstance(v, dict)]
        row = {"semilla": r.get("seed"), "estado": r.get("status"), "val_best": r.get("val_best"),
               "top1_tmine": t.get("tmine", {}).get("top1"), "top1_tgem": t.get("tgem", {}).get("top1"),
               "top1_tgem_barajado": t.get("tgem_shuf", {}).get("top1"),
               "top1_trand": t.get("trand", {}).get("top1"), "top1_tmine50": t.get("tmine50", {}).get("top1"),
               "top1_tgen_media": float(np.mean(tg)) if tg else None,
               "abst_auroc_tmine": t.get("tabst_mine", {}).get("abst_auroc_maxcos"),
               "abst_auroc_tgem": t.get("tgem", {}).get("abst_auroc_maxcos"),
               "ece_ts": t.get("tgem", {}).get("ts_ece"),
               "pasos_mid": r.get("stages", {}).get("mid", {}).get("steps"),
               "mejor_paso_mid": r.get("stages", {}).get("mid", {}).get("best_step")}
        rows.append({k: v for k, v in row.items() if v is not None or k in ("semilla", "estado")})
    return pd.DataFrame(rows)


def compare_variants(study, bundle, variants: dict, out_name: str, testsets=None, B: int = 2000,
                     results_phase: str = "F2") -> dict:
    """Todos contra todos entre variantes con nombre: {etiqueta: (fase, nombre_de_carpeta)}.
    Por defecto: T-mine, macro de T-gen, T-rand, T-gem (y barajado) y T-mine-50."""
    from .generators import load_panel, macro_for_run
    from .stats import all_vs_all
    runs = {}
    for label, (phase, name) in variants.items():
        lst = []
        base = study.paths.d("runs", phase, name)
        if os.path.isdir(base):
            for d in sorted(os.listdir(base)):
                rd = os.path.join(base, d)
                done = read_done(rd)
                if d.startswith("seed_") and done and done.get("status") == "ok":
                    lst.append((int(done["seed"]), rd))
        if lst:
            runs[label] = lst
    missing = [k for k in variants if k not in runs]
    if missing:
        print(f"[comparar] sin corridas terminadas: {missing}")
    if len(runs) < 2:
        return {"runs": list(runs), "nota": "se necesitan al menos 2 variantes terminadas"}
    ex = extra_sets(bundle)
    panel = load_panel(study, bundle.test_split)
    if panel:
        for lst in runs.values():
            for _, rd in lst:
                macro_for_run(rd, panel)
    ts = list(testsets or ("tmine", "tgenmacro", "trand", "tgem", "tgem_shuf", "tmine50", "tgen2"))
    return all_vs_all(runs, ts, bundle.test.doc_ids, study.paths.d("results", results_phase, out_name),
                      rope_pp=study.rope_pp, B=B, items_map=items_map_of(bundle, ex, panel),
                      abst_sets=("tmine", "tgem"))


def plot_val_curves(study, variants: dict, out_png: str, metric: str = "val_top1"):
    """Curvas de validación por variante (B3: ¿sube y luego cae con HN desde el inicio?)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (phase, name) in variants.items():
        base = study.paths.d("runs", phase, name)
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base)):
            if not d.startswith("seed_"):
                continue
            off = 0
            for stg in ("pre", "mid"):
                sj = load_json(os.path.join(base, d, f"stage_{stg}.json"))
                if not sj:
                    continue
                h = [x for x in sj["history"] if metric in x]
                ax.plot([off + x["step"] for x in h], [x[metric] for x in h], alpha=0.7,
                        label=f"{label} ({d}, {stg})")
                off += sj.get("steps", 0)
    ax.set_xlabel("paso (acumulado entre etapas)")
    ax.set_ylabel(metric)
    ax.legend(fontsize=6)
    fig.tight_layout()
    makedirs(os.path.dirname(out_png))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def trex_for_arms(study, arms=None, phase: str = "F3", seed: int = 0, modes=("shield", "noshield"), log=print):
    """T-Rex (D2: con y sin shield) con el cabezal exportado de cada brazo. Reanudable por brazo y modo."""
    from .trex import run_trex, setup_clm_repo, stop_all
    repo = setup_clm_repo()
    out = {}
    try:
        for arm in (arms or arm_names()):
            od = study.paths.d("results", phase, "trex", arm)
            if all(os.path.exists(os.path.join(od, f"trex_{m}.json")) for m in modes) \
                    and os.path.exists(os.path.join(od, "trex_summary.json")):
                out[arm] = load_json(os.path.join(od, "trex_summary.json"))     # ya hecho: no levanta servidores
                continue
            rd = study.paths.d("runs", phase, arm, f"seed_{seed}")
            ck = os.path.join(rd, "clm_head.pt")
            if not os.path.exists(ck) and os.path.exists(os.path.join(rd, "DONE.json")):
                ck = export_head_from_run(study, rd)
            if not os.path.exists(ck):
                log(f"[trex] {arm}: falta {ck}; se omite")
                continue
            out[arm] = run_trex(ck, study.paths.d("results", phase, "trex", arm), repo=repo, modes=modes, log=log)
    finally:
        stop_all()
    save_json(out, study.paths.d("results", phase, "trex", "TREX_SUMMARY.json"))
    return out


def export_summary_md(study, phase: str = "F1") -> str:
    """Resumen en Markdown (para EXPERIMENT_LOG / PROJECT_STATE del proyecto)."""
    rep = load_json(study.paths.d("results", phase, "all_vs_all", "all_vs_all.json"), {}) or {}
    lines = [f"# Resumen {phase} — perfil {study.data.profile} — {now_iso()}", ""]
    for name, m in (rep.get("metrics") or {}).items():
        if "summary" not in m:
            continue
        lines += [f"## {name}", "", "| brazo | media | IC95 | semillas |", "|---|---|---|---|"]
        for r in m["summary"]:
            lines.append(f"| {r['arm']} | {r['mean']:.4f} | [{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}] | {r['n_seeds']} |")
        lines += ["", f"Pares significativos (Holm): {m['n_significant']}/{m['n_pairs']}; "
                      f"equivalentes (TOST ±{rep.get('rope_pp')} pp): {m['n_equivalent']}", ""]
    if "C1" in rep:
        c = rep["C1"]
        lines += ["## C1 (H1a)", "", f"Brazo no contrastivo elegido en validación: **{c.get('best_noncontrastive')}** "
                  f"({c.get('selection_phase', phase)}); confirmatorio = {c.get('confirmatory')}", ""]
        for met, cc in (c.get("por_metrica") or {c.get("metrica", "tgem"): c}).items():
            ph = f", p_Holm(C1) = {cc['p_holm_c1']:.3g}" if "p_holm_c1" in cc and c.get("holm_between") else ""
            lines.append(f"- {met}: {c.get('best_noncontrastive')} − infonce: Δ = {cc.get('diff', float('nan')):.4f} "
                         f"IC95 [{cc.get('ci95_lo', float('nan')):.4f}, {cc.get('ci95_hi', float('nan')):.4f}], "
                         f"p = {cc.get('p', float('nan')):.3g}{ph}, veredicto = {cc.get('verdict')}")
        lines.append("")
    if rep.get("panel_version"):
        lines += [f"Panel T-gen (macro): versión {rep['panel_version']}", ""]
    md = "\n".join(lines)
    p = study.paths.d("results", phase, f"RESUMEN_{phase}.md")
    with open(p, "w") as f:
        f.write(md)
    return p


# --------------------------------------------------------------------------------------
# Planificación: disco por perfil, tiempo / unidades de cómputo y control de tunability
QWEN3_8B_GB = 16.4        # pesos bf16 de Qwen3-8B (≈ 8.2e9 parámetros × 2 bytes) [INFERENCE]
TOKENS_GB = 10.5          # qa_pairs (2.9 GB) + neg_pairs (7.6 GB) de los tokens de Nemotron (HF, 2026-09-30)


def profile_disk_table(cache_root: str = "/content/cache", margin: float = 1.10):
    """¿Qué perfil cabe en el disco local de ESTA máquina? (no crea ningún estudio)."""
    import pandas as pd
    os.makedirs(cache_root, exist_ok=True)
    free = disk_free_gb(cache_root)
    rows = []
    for name in ("mini", "colab", "colab_large"):
        d = data_config(name)
        emb = d.disk_gb()
        rows.append({"perfil": name, "pares_pre_training": len(d.pre_chunks) * 100_000,
                     "pares_con_HN": len(d.hn_chunks) * 100_000, "GB_embeddings": round(emb, 1),
                     "GB_si_además_corre_Qwen3_y_tokens": round(emb + QWEN3_8B_GB + TOKENS_GB, 1),
                     "cabe_para_entrenar": free >= emb * margin,
                     "cabe_con_todo": free >= (emb + QWEN3_8B_GB + TOKENS_GB) * margin})
    df = pd.DataFrame(rows)
    df.attrs["disco_libre_gb"] = round(free, 1)
    return df


def _done_runs(base: str, pattern: str):
    import glob
    out = []
    for p in glob.glob(os.path.join(base, pattern, "DONE.json")):
        d = load_json(p) or {}
        if d.get("status") == "ok" and d.get("train_seconds") is not None:
            cfg = load_json(os.path.join(os.path.dirname(p), "config.json"), {}) or {}
            tr, ev = float(d["train_seconds"]), float(d.get("eval_seconds", 0.0))
            out.append({"arm": d.get("arm"), "secs": tr + ev, "train": tr, "eval": ev,
                        "gpu": ((cfg.get("env") or {}).get("gpu") or {}).get("name")})
    return out


def _med(rs, key: str = "secs") -> float:
    return float(np.median([r[key] for r in rs]))


def estimate_budget(study, cu_per_hour: float | None = None, phase: str = "F1"):
    """Proyecta horas (y unidades de cómputo si das cu_per_hour) de lo que falta de una fase de brazos y de F3,
    a partir del tiempo MEDIDO de corridas ya terminadas. Si la fase no empezó, usa un proxy y lo dice:
    F1b con la receta 'actual' -> el F1 piloto (misma receta); F1b con la receta de CLM -> pre + mid de
    InfoNCE+HN en F0.7b, escalado por brazo con los tiempos relativos del piloto; si no, F0.7."""
    import pandas as pd
    runs = study.paths.runs
    full = _done_runs(runs, f"{phase}/*/seed_*")
    trials = _done_runs(runs, f"{phase}/*/hpo/trial_*")
    source, rel = phase, {}
    if not full and phase == "F1b":
        rec = (recipe_decision(study) or {}).get("receta")
        pilot = _done_runs(runs, "F1/*/seed_*")
        pre = _done_runs(runs, f"F0.7b/{rec}/pre/seed_*") if rec else []
        mid = _done_runs(runs, f"F0.7b/{rec}/mid_inbatch/seed_*") if rec else []
        if rec == "actual" and pilot:
            full, trials = pilot, _done_runs(runs, "F1/*/hpo/trial_*")
            source = "F1 piloto (misma receta que F1b: 'actual')"
        elif pre and mid:
            # una corrida completa = pre + mid (+ una evaluación); en F0.7b son dos corridas separadas
            t = _med(pre, "train") + _med(mid, "train") + _med(mid, "eval")
            full = [{"arm": "infonce_hn", "secs": t, "gpu": r["gpu"]} for r in pre + mid]
            if pilot:
                p = {a: _med([r for r in pilot if r["arm"] == a]) for a in {r["arm"] for r in pilot}}
                ref = p.get("infonce_hn") or float(np.median(list(p.values())))
                rel = {a: v / ref for a, v in p.items()} if ref > 0 else {}
            trials = []
            source = (f"F0.7b, receta {rec} (pre + mid de InfoNCE+HN"
                      + (", escalado por brazo con los tiempos relativos del F1 piloto)" if rel else ")"))
        elif pilot:
            full, trials = pilot, _done_runs(runs, "F1/*/hpo/trial_*")
            source = "F1 piloto (proxy; la receta de F1b puede tardar distinto)"
    if not full:
        full = _done_runs(runs, "F0.7/*/seed_*")
        source = "F0.7 (proxy: InfoNCE)"
    if not full:
        return {"nota": "Todavía no hay corridas medidas. Corre F0.7 o una celda de F1 y vuelve a ejecutar esta."}
    med = _med(full)
    if rel:
        by_arm_full = {a: med * f for a, f in rel.items()}
    else:
        by_arm_full = {a: _med([r for r in full if r["arm"] == a]) for a in {r["arm"] for r in full}}
    by_arm_trial = {a: _med([r for r in trials if r["arm"] == a]) for a in {r["arm"] for r in trials}}
    med_trial = _med(trials) if trials else None
    rows = []
    for arm in arm_names():
        ad = study.paths.d("runs", phase, arm)
        n_tr = len(_done_runs(ad, "hpo/trial_*"))
        n_sd = len(_done_runs(ad, "seed_*"))
        t_full = by_arm_full.get(arm, med)
        t_trial = by_arm_trial.get(arm, med_trial if med_trial is not None else t_full * study.hpo.budget_frac)
        rem = max(0, study.hpo.n_trials - n_tr) * t_trial + max(0, len(study.f1_seeds) - n_sd) * t_full
        rows.append({"brazo": arm, "trials_hechos": n_tr, "semillas_hechas": n_sd,
                     "min_por_corrida_completa": round(t_full / 60, 1), "horas_restantes_F1": round(rem / 3600, 2)})
    df = pd.DataFrame(rows)
    f1_h = float(df["horas_restantes_F1"].sum())
    f3_h = sum(by_arm_full.get(a, med) for a in arm_names()) * len(study.f3_seeds) / 3600
    gpus = sorted({r["gpu"] for r in full if r.get("gpu")})
    out = {"medido_en": source, "gpu_de_las_mediciones": gpus, "min_por_corrida_completa_mediana": round(med / 60, 1),
           "horas_restantes_F1": round(f1_h, 1), "horas_F3": round(f3_h, 1), "tabla": df}
    if cu_per_hour:
        out["unidades_restantes_F1"] = round(f1_h * cu_per_hour, 0)
        out["unidades_F3"] = round(f3_h * cu_per_hour, 0)
    return out


def tunability_report(study, phase: str = "F1", ns=(1, 2, 4, 8, 16, 32)):
    """Control de tunability (Dodge et al., 2019 [PENDING VERIFICATION]): desempeño de validación
    ESPERADO con n trials (de los trials al presupuesto reducido) y si el ranking de brazos cambia
    entre n = 8, 16 y 32."""
    import pandas as pd
    from scipy.stats import kendalltau
    from .hpo import expected_max_curve
    evp = {}
    for arm in arm_names():
        s = load_json(study.paths.d("runs", phase, arm, "hpo_summary.json"))
        if not s:
            continue
        vals = [t["val_best"] for t in s.get("trials", [])]
        c = expected_max_curve(vals, ns=[n for n in ns])
        evp[arm] = {n: c.get(n, np.nan) for n in ns}
    if len(evp) < 2:
        return {"nota": "se necesitan al menos 2 brazos con búsqueda de HP terminada"}
    df = pd.DataFrame(evp).T[list(ns)]
    ranks = df.rank(ascending=False, axis=0)
    n_max = max(n for n in ns if df[n].notna().all()) if any(df[n].notna().all() for n in ns) else None
    taus = {}
    if n_max is not None:
        for n in ns:
            if n < n_max and df[n].notna().all():
                taus[n] = float(kendalltau(ranks[n], ranks[n_max]).statistic)
    out_dir = makedirs(study.paths.d("results", phase, "tunability"))
    df.to_csv(os.path.join(out_dir, "evp_por_brazo.csv"))
    ranks.to_csv(os.path.join(out_dir, "ranking_por_presupuesto.csv"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, row in df.iterrows():
        ax.plot(list(ns), row.values, marker="o", ms=3, label=arm)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nº de trials (presupuesto de tuning)")
    sel = "val-mine" if phase in A1_PHASES else "T-gem-val"
    ax.set_ylabel(f"top-1 de validación esperado ({sel}; mejor de n)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    png = os.path.join(out_dir, "evp_curvas.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    rep = {"n_referencia": n_max, "kendall_tau_vs_ref": taus, "png": png}
    save_json(rep, os.path.join(out_dir, "tunability.json"))
    return {"evp": df, "ranking": ranks, **rep}
