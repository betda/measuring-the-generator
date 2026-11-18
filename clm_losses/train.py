"""Entrenador reanudable de cabezales.

Cada corrida vive en su carpeta de Drive:
    config.json        configuración + hash del código + entorno
    ckpt_last.pt       estado completo (cabezales, pérdida, optimizador, scheduler, RNG, posición del stream)
    best_<stage>.pt    mejores cabezales (según validación) de cada etapa
    log.jsonl          historia
    eval/              métricas y puntuaciones por ítem en test
    DONE.json          resumen; si existe, la celda salta la corrida

Si Colab se corta, volver a ejecutar la celda reanuda desde ckpt_last.pt en el mismo batch.
"""
from __future__ import annotations

import dataclasses
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import V03_TRAIN_DEFAULTS, TrainConfig, clm_lr
from .data import Stream
from .evaluate import quick_val
from .heads import Heads
from .losses import AuxHN, Ctx, build_loss, resolve_hn_mode
from .utils import (RunLogger, atomic_torch_save, env_snapshot, get_rng_state, is_done, load_json,
                    mark_done, read_done, save_json, seed_everything, set_rng_state, set_status)

__all__ = ["Trainer", "clm_lr", "run_training", "run_fingerprint", "stage_lr", "export_clm_head"]


class NonFinite(RuntimeError):
    pass


def stage_lr(cfg: TrainConfig, stage: str, batch: int) -> float:
    """LR de una etapa: fijo (cfg.lr) o la regla de CLM por batch × lr_mult. En "mid" después de "pre"
    (no con HN desde el inicio, que entrena desde cero) se multiplica por lr_mid_scale si está definido."""
    lr = cfg.lr if cfg.lr else clm_lr(cfg.width, batch) * cfg.lr_mult
    if stage == "mid" and not cfg.hn_from_start and cfg.lr_mid_scale is not None:
        lr *= float(cfg.lr_mid_scale)
    return float(lr)


class Trainer:
    def __init__(self, cfg: TrainConfig, bundle, run_dir: str, device: str | None = None,
                 row_fraction: float = 1.0, log: RunLogger | None = None):
        self.cfg, self.b, self.run_dir = cfg, bundle, run_dir
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.row_fraction = row_fraction
        self.log = log or RunLogger(run_dir, prefix=f"[{cfg.arm}/s{cfg.seed}] ")
        self.hn_mode = resolve_hn_mode(cfg.arm, cfg.hn_mode)
        self.stages = ["mid"] if cfg.hn_from_start else list(cfg.stages)
        self.ckpt = os.path.join(run_dir, "ckpt_last.pt")
        self._val_cache: dict = {}
        seed_everything(cfg.seed)
        H = bundle.hidden
        self.heads = Heads(hidden=H, width=cfg.width, depth=cfg.depth, proj=cfg.proj,
                           activation=cfg.activation, layernorm=cfg.layernorm, residual=cfg.residual,
                           tau_init=cfg.tau_init).to(self.device)
        self.loss = build_loss(cfg.arm, self.heads, cfg.loss_params, 1, self.hn_mode).to(self.device)
        self.aux = AuxHN() if self.hn_mode == "aux" else None
        amp = cfg.amp
        if amp == "bf16" and not (self.device.type == "cuda" and torch.cuda.is_bf16_supported()):
            amp = "fp32"
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and self.device.type == "cuda"))

    # ------------------------------------------------------------------------------
    def stream(self, stage: str) -> Stream:
        d, c = self.b.dcfg, self.cfg
        if stage == "pre":
            return Stream(self.b.store, d.pre_chunks, self.b.masks_pre, c.batch_pre, d.pre_block_rows,
                          d.pre_group_blocks, seed=c.seed, with_hn=False, k=d.k_neg,
                          row_fraction=self.row_fraction)
        need_hn = self.hn_mode in ("native", "aux")
        return Stream(self.b.store, d.hn_chunks, self.b.masks_mid, c.batch_mid, d.mid_block_rows,
                      d.mid_group_blocks, seed=c.seed + 1, with_hn=need_hn, k=d.k_neg,
                      row_fraction=self.row_fraction)

    def stage_steps(self, stage: str, stream: Stream) -> int:
        c = self.cfg
        if c.hn_from_start:
            pre_pairs = c.pre_epochs * self.stream("pre").n_rows
            mid_pairs = c.mid_epochs * stream.n_rows
            steps = int(round((pre_pairs + mid_pairs) / c.batch_mid))
        else:
            ep = c.pre_epochs if stage == "pre" else c.mid_epochs
            steps = int(round(ep * stream.epoch_batches(0)))
        cap = c.max_steps_pre if stage == "pre" else c.max_steps_mid
        if cap:
            steps = min(steps, int(cap))
        return max(1, steps)

    # ------------------------------------------------------------------------------
    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x)).to(self.device, non_blocking=True).float()

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if c.aug_feat_dropout > 0:
            x = F.dropout(x, c.aug_feat_dropout, training=True)
        if c.aug_noise_std > 0:
            x = x + c.aug_noise_std * torch.randn_like(x) / math.sqrt(x.shape[-1])
        if c.aug_feat_dropout > 0 or c.aug_noise_std > 0:
            x = F.normalize(x, dim=-1)
        return x

    def _dup(self, e: torch.Tensor) -> torch.Tensor | None:
        thr = self.cfg.dup_threshold
        if thr is None or thr >= 1.0:
            return None
        en = F.normalize(e, dim=-1)
        m = (en @ en.t()) > thr
        m.fill_diagonal_(False)
        return m

    def _step(self, batch: dict, step: int) -> tuple[float, dict]:
        es, ea = self._tensor(batch["es"]), self._tensor(batch["ea"])
        dup_a, dup_s = self._dup(ea), self._dup(es)
        es, ea = self._augment(es), self._augment(ea)
        en = nmask = None
        if "en" in batch:
            en = self._augment(self._tensor(batch["en"]))
            K = en.shape[1]
            nmask = torch.arange(K, device=self.device)[None, :] < self._tensor(batch["ncnt"]).long()[:, None]
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            zs, za = self.heads.zs(es), self.heads.za(ea)
            zn = None
            if en is not None:
                B, K, H = en.shape
                zn = self.heads.za(en.reshape(B * K, H)).view(B, K, -1)
        ctx = Ctx(es=es, ea=ea, zs=zs.float(), za=za.float(), en=en,
                  zn=zn.float() if zn is not None else None, nmask=nmask, dup_a=dup_a, dup_s=dup_s, step=step)
        loss, logs = self.loss(ctx, self.heads)
        if self.aux is not None and zn is not None:
            aux = self.aux(ctx, self.heads)
            loss = loss + self.cfg.aux_hn_weight * aux
            logs["aux_hn"] = float(aux.detach())
        if not torch.isfinite(loss):
            raise NonFinite(f"pérdida no finita en el paso {step}: {float(loss)}")
        return loss, logs

    # ------------------------------------------------------------------------------
    def _save_last(self, **state) -> None:
        state.update(heads=self.heads.state_dict(), loss=self.loss.state_dict(), rng=get_rng_state(),
                     scaler=self.scaler.state_dict(), cfg=dataclasses.asdict(self.cfg), saved_at=time.time())
        atomic_torch_save(state, self.ckpt)

    def _save_best(self, stage: str, metrics: dict) -> None:
        atomic_torch_save({"heads": self.heads.state_dict(), "loss": self.loss.state_dict(),
                           "metrics": metrics, "stage": stage}, os.path.join(self.run_dir, f"best_{stage}.pt"))

    def load_best(self, stage: str) -> bool:
        p = os.path.join(self.run_dir, f"best_{stage}.pt")
        if not os.path.exists(p):
            return False
        st = torch.load(p, map_location=self.device, weights_only=False)
        self.heads.load_state_dict(st["heads"])
        self.loss.load_state_dict(st["loss"])
        return True

    def validate(self) -> dict:
        self.loss.eval()
        m = quick_val(self.heads, self.b.val, self.device, cache=self._val_cache,
                      mine_idx=getattr(self.b, "val_mine_idx", None))
        self.loss.train()
        return m

    def eval_every(self, total: int) -> int:
        c = self.cfg
        if c.evals_per_stage:
            return max(1, int(total) // int(c.evals_per_stage))
        return int(c.eval_every)

    def _init_from(self) -> None:
        """F0.7b: el mid parte del checkpoint de pre (elegido en validación) de OTRA corrida con la misma
        semilla, en vez de re-entrenar el mismo pre-training en cada condición."""
        p = os.path.join(self.cfg.init_from, "best_pre.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"init_from: falta {p} (corre primero la condición 'pre' de esa semilla)")
        st = torch.load(p, map_location=self.device, weights_only=False)
        self.heads.load_state_dict(st["heads"])
        try:
            self.loss.load_state_dict(st["loss"], strict=False)
        except RuntimeError:
            pass   # la pérdida de pre puede tener otros módulos auxiliares; los cabezales son lo que importa
        self.log.log("inicializado desde", path=p)

    # ------------------------------------------------------------------------------
    def run(self) -> dict:
        if str(self.cfg.select_metric).startswith("valmine") and getattr(self.b, "val_mine_idx", None) is None:
            raise RuntimeError("select_metric usa val-mine pero falta su pool (data/eval/val/mine_idx.npy): "
                               "corre F0.5b (celda 3.6b) antes de entrenar.")
        state = None
        if os.path.exists(self.ckpt):
            state = torch.load(self.ckpt, map_location=self.device, weights_only=False)
            self.heads.load_state_dict(state["heads"])
            self.loss.load_state_dict(state["loss"])
            if "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])
            set_rng_state(state["rng"])     # también si la etapa guardada ya terminó (la siguiente parte igual)
            self.log.log("reanudando", stage=state["stage"], step=state["step"],
                         stages_done=state.get("stages_done", []))
        elif self.cfg.init_from:
            self._init_from()
        stages_done = list(state.get("stages_done", [])) if state else []
        summary = {"stages": {}}
        for stage in self.stages:
            if stage in stages_done:
                summary["stages"][stage] = load_json(os.path.join(self.run_dir, f"stage_{stage}.json"), {})
                continue
            resume = state if (state is not None and state["stage"] == stage and not state.get("stage_complete")) else None
            summary["stages"][stage] = self._run_stage(stage, resume, stages_done)
            stages_done.append(stage)
            state = None
        return summary

    def _run_stage(self, stage: str, resume: dict | None, stages_done: list) -> dict:
        c = self.cfg
        stream = self.stream(stage)
        total = self.stage_steps(stage, stream)
        self.loss.set_total_steps(total)
        batch = c.batch_pre if stage == "pre" else c.batch_mid
        lr = stage_lr(c, stage, batch)
        every = self.eval_every(total)
        params = [p for p in list(self.heads.parameters()) + list(self.loss.parameters()) if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=c.weight_decay)
        # "onecycle" (F1 piloto): OneCycle con pct_start 0.1 y coseno, como train/finetune.py de CLM. Con muy
        # pocos pasos el warm-up tendría 0 pasos y torch divide por cero: se garantiza ≥ 2 pasos de subida
        # (sólo afecta corridas de < 20 pasos). "const": LR constante, la receta de los checkpoints de pre/mid
        # de CLM (lr_schedule "const" en su cfg) [VERIFIED].
        if c.lr_schedule == "onecycle" and total >= 4:
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total,
                                                        pct_start=max(c.warmup_frac, 2.0 / total),
                                                        anneal_strategy="cos")
        elif c.lr_schedule in ("onecycle", "const"):
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        else:
            raise ValueError(f"lr_schedule desconocido {c.lr_schedule!r}")
        step = epoch = bie = 0
        best, stale, history, best_step = -float("inf"), 0, [], 0
        t_stage0, secs_prev = time.time(), 0.0       # tiempo de cómputo acumulado entre reanudaciones
        if resume is not None:
            opt.load_state_dict(resume["opt"])
            sched.load_state_dict(resume["sched"])
            step, epoch, bie = resume["step"], resume["epoch"], resume["batch_in_epoch"]
            best, stale, history = resume["best"], resume["stale"], resume["history"]
            best_step = resume.get("best_step", 0)
            secs_prev = float(resume.get("seconds", 0.0))
            set_rng_state(resume["rng"])
        else:
            m0 = self.validate()
            history.append({"step": 0, "init": True, **m0})
            if c.select_include_init:
                best = m0[c.select_metric]
                self._save_best(stage, m0)
            self.log.log(f"etapa {stage}: {total} pasos, lr={lr:.3g}, batch={batch}, filas={stream.n_rows}, "
                         f"evaluación cada {every} pasos, selección={c.select_metric}", **m0)
        set_status(self.run_dir, "running", stage=stage, step=step, total=total)
        self.heads.train()
        self.loss.train()
        t_ckpt = time.time()
        stop = stale >= c.patience if resume is not None else False   # corte justo tras early stopping
        run_loss, n_loss = 0.0, 0
        while step < total and not stop:
            yielded = False
            for batch_np in stream.iterate(epoch, bie):
                yielded = True
                loss, logs = self._step(batch_np, step)
                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                self.loss.before_optimizer_step(step)    # p. ej. SwAV congela prototipos: antes del clipping
                torch.nn.utils.clip_grad_norm_(params, c.grad_clip)
                self.scaler.step(opt)
                self.scaler.update()
                sched.step()
                self.loss.after_step(self.heads, step)
                step += 1
                bie += 1
                run_loss += float(loss.detach())
                n_loss += 1
                if step % every == 0 or step == total:
                    m = self.validate()
                    rec = {"step": step, "train_loss": run_loss / max(1, n_loss), "lr": sched.get_last_lr()[0], **m,
                           **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
                    history.append(rec)
                    run_loss, n_loss = 0.0, 0
                    if m[c.select_metric] > best + 1e-6:
                        best, stale, best_step = m[c.select_metric], 0, step
                        self._save_best(stage, m)
                    else:
                        stale += 1
                    self.log.log(f"{stage} {step}/{total}", **{k: rec[k] for k in
                                                               ("train_loss", "val_top1", "valmine_top1", "val_abst_auroc")
                                                               if k in rec})
                    if stale >= c.patience:
                        self.log.log(f"early stopping en {stage} (paso {step})")
                        stop = True
                if (step % c.ckpt_every_steps == 0) or (time.time() - t_ckpt > 60 * c.ckpt_every_minutes):
                    self._save_last(stage=stage, step=step, epoch=epoch, batch_in_epoch=bie, total=total,
                                    opt=opt.state_dict(), sched=sched.state_dict(), best=best, stale=stale,
                                    history=history, stages_done=stages_done, best_step=best_step,
                                    seconds=secs_prev + time.time() - t_stage0)
                    set_status(self.run_dir, "running", stage=stage, step=step, total=total)
                    t_ckpt = time.time()
                if stop or step >= total:
                    break
            else:
                if not yielded and bie == 0:
                    raise RuntimeError(f"la época {epoch} de la etapa {stage} no produjo ningún batch "
                                       f"(batch={batch}, filas={stream.n_rows}); revisa el tamaño de batch")
                epoch += 1
                bie = 0
                continue
            break
        # fin de etapa: quedarse con el mejor checkpoint de validación
        if c.save_final:
            atomic_torch_save({"heads": self.heads.state_dict(), "loss": self.loss.state_dict(), "step": step},
                              os.path.join(self.run_dir, f"final_{stage}.pt"))
        final_val = history[-1].get(c.select_metric) if history else None
        self.load_best(stage)
        init_val = history[0].get(c.select_metric) if history else None
        out = {"stage": stage, "steps": step, "total": total, "best": best, "best_step": best_step,
               "init_val": init_val, "final_val": final_val, "history": history,
               "early_stopped": stop, "lr": lr, "rows": stream.n_rows, "eval_every": every,
               "select_metric": c.select_metric, "seconds": round(secs_prev + time.time() - t_stage0, 1)}
        save_json(out, os.path.join(self.run_dir, f"stage_{stage}.json"))
        self._save_last(stage=stage, step=step, epoch=epoch, batch_in_epoch=bie, total=total,
                        opt=opt.state_dict(), sched=sched.state_dict(), best=best, stale=stale,
                        history=history, stages_done=stages_done + [stage], stage_complete=True)
        return out


# --------------------------------------------------------------------------------------
def export_clm_head(heads, metrics: dict | None, run_dir: str) -> str:
    """Exporta los cabezales en el formato de clm-serve. logit_scale = temperatura nativa de la pérdida
    si existe; si no, la de temperature scaling en validación."""
    tg = (metrics or {}).get("tgem", {})
    T = tg.get("native_T") or tg.get("ts_T")
    p = os.path.join(run_dir, "clm_head.pt")
    atomic_torch_save(heads.clm_checkpoint(logit_scale=math.log(T) if T else None), p)
    return p


def _fingerprint_train_dict(cfg: TrainConfig) -> dict:
    """asdict(cfg) sin los campos de v0.3 que están en su valor por defecto: así una corrida de v0.2
    (que no tenía esos campos) conserva exactamente la misma huella con el código nuevo."""
    d = dataclasses.asdict(cfg)
    for k, v in V03_TRAIN_DEFAULTS.items():
        if k in d and d[k] == v:
            d.pop(k)
    return d


def run_fingerprint(cfg: TrainConfig, row_fraction: float, bundle) -> str:
    """Huella de lo que define una corrida (config de entrenamiento, fracción de datos, perfil y split)."""
    import hashlib
    import json
    blob = json.dumps({"train": _fingerprint_train_dict(cfg), "row_fraction": float(row_fraction),
                       "data_profile": bundle.dcfg.profile, "test_split": getattr(bundle, "test_split", "test")},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ConfigMismatch(RuntimeError):
    pass


def run_training(cfg: TrainConfig, bundle, run_dir: str, ecfg=None, evaluate_test: bool = True,
                 row_fraction: float = 1.0, device: str | None = None, pools_dir: str | None = None,
                 extra_sets: dict | None = None, cleanup_heavy: bool = False, export_clm: bool = False) -> dict:
    """Entrena (o reanuda) y evalúa una corrida. Devuelve el resumen de DONE.json.
    Si la carpeta ya tiene una corrida con OTRA configuración, se detiene (no mezcla resultados)."""
    from .utils import code_hash
    fp = run_fingerprint(cfg, row_fraction, bundle)
    cfg_path = os.path.join(run_dir, "config.json")
    old = load_json(cfg_path)
    if old is not None and old.get("fingerprint") not in (None, fp):
        diff = [f"{k}: {old['train'].get(k)!r} -> {v!r}" for k, v in dataclasses.asdict(cfg).items()
                if old.get("train", {}).get(k) != v]
        if old.get("row_fraction") != row_fraction:
            diff.append(f"row_fraction: {old.get('row_fraction')} -> {row_fraction}")
        raise ConfigMismatch(f"{run_dir} ya contiene una corrida con otra configuración ({'; '.join(diff) or 'perfil/split'}). "
                             "Bórrala a mano o usa otra carpeta; no se reutiliza para no mezclar resultados.")
    if is_done(run_dir):
        return read_done(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    if old is None:
        meta = {"train": dataclasses.asdict(cfg), "row_fraction": row_fraction, "env": env_snapshot(),
                "data_profile": bundle.dcfg.profile, "fingerprint": fp}
        if cfg.init_from:      # trazabilidad: qué checkpoint de pre inicializó esta corrida
            from .utils import file_sha256
            p = os.path.join(cfg.init_from, "best_pre.pt")
            meta["init_from_sha256"] = file_sha256(p) if os.path.exists(p) else None
        save_json(meta, cfg_path)
    code_start = (old or {}).get("env", {}).get("code_hash") or code_hash()
    tr = Trainer(cfg, bundle, run_dir, device=device, row_fraction=row_fraction)
    try:
        summary = tr.run()
    except NonFinite as e:
        set_status(run_dir, "failed", error=str(e))
        mark_done(run_dir, {"status": "failed", "error": str(e), "arm": cfg.arm, "seed": cfg.seed,
                            "val_best": -float("inf")})
        tr.log.log("FALLÓ", error=str(e))
        return read_done(run_dir)
    last_stage = tr.stages[-1]
    tr.load_best(last_stage)
    val_best = summary["stages"][last_stage].get("best")
    code_end = code_hash()
    result = {"status": "ok", "arm": cfg.arm, "seed": cfg.seed, "val_best": val_best, "fingerprint": fp,
              "code_hash_start": code_start, "code_hash_end": code_end, "code_changed": code_start != code_end,
              "stages": {k: {kk: v[kk] for kk in ("steps", "total", "best", "best_step", "init_val", "final_val",
                                                  "early_stopped", "seconds") if kk in v}
                         for k, v in summary["stages"].items()},
              "row_fraction": row_fraction,
              "train_seconds": round(sum(float(v.get("seconds", 0.0)) for v in summary["stages"].values()), 1)}
    if evaluate_test:
        from .evaluate import evaluate_run
        t_eval = time.time()
        metrics = evaluate_run(tr.heads, tr.loss, bundle.test, bundle.val, os.path.join(run_dir, "eval"),
                               tr.device, ecfg, pools_dir=pools_dir, extra_sets=extra_sets, log=tr.log.log)
        result["test"] = {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
                          for k, v in metrics.items() if isinstance(v, dict)}
        result["eval_seconds"] = round(time.time() - t_eval, 1)
        if export_clm:
            export_clm_head(tr.heads, metrics, run_dir)
    if cleanup_heavy:
        for fn in os.listdir(run_dir):
            if fn.endswith(".pt"):
                os.remove(os.path.join(run_dir, fn))
    else:
        # Espacio en Drive: sólo queda best_<última etapa>.pt (cabezales + estado de la pérdida). El
        # optimizador y los mejores de etapas anteriores ya no hacen falta para evaluar ni exportar.
        keep = {f"best_{last_stage}.pt", "clm_head.pt"}
        if cfg.save_final:
            keep |= {f"final_{st}.pt" for st in tr.stages}       # diagnóstico de colapso (F0.8)
        for fn in os.listdir(run_dir):
            if fn.endswith(".pt") and fn not in keep:
                os.remove(os.path.join(run_dir, fn))
    set_status(run_dir, "done")
    mark_done(run_dir, result)
    return read_done(run_dir)
