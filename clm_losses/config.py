"""Configuración: rutas, perfiles de datos y valores por defecto.

Regla de almacenamiento:
    * Todo lo que PRODUCIMOS (splits, sets de evaluación, checkpoints, logs, resultados,
      reportes) va a Google Drive  ->  Paths.drive_root
    * Los datos PÚBLICOS grandes y re-descargables (chunks de embeddings de Hugging Face)
      van a la caché local de la VM  ->  Paths.cache_root
      (si tu Drive tiene espacio de sobra puedes apuntar cache_root a Drive; será más lento).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

DRIVE_ROOT_DEFAULT = "/content/drive/MyDrive/CLM_losses"
CACHE_ROOT_DEFAULT = "/content/cache"

PRETRAIN_REPO = "Contrastive-LM/CLM-v0.1-Pretrain-Nemotron"
HN_REPO = "jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings"
TOKENS_REPO = "jackyk02/nemotron-cc-v2.1-hq-dqa-qwen3-tokens"
CLM_HEAD_REPO = "Contrastive-LM/CLM-v0.1-8B"
CLM_STAGE_HEADS_REPO = "jackyk02/qwen3_8b_head"
DEEPSWE_HEADS_REPO = "Contrastive-LM/deepswe-clm-heads-8k"
DEEPSWE_EVAL_CANDIDATE = "tarsur385/deepswe-prm-embeddings-8k"
CLM_GIT_URL = "https://github.com/Contrastive-LM/CLM.git"
CLM_GIT_COMMIT = "bb42c6c"  # commit auditado el 2026-09-30
ENCODER_MODEL = "Qwen/Qwen3-8B"

# Tamaños conocidos (verificados en la API de HF el 2026-09-30)
PRE_CHUNK_GB = 2 * 819_200_128 / 1e9     # q + a de 100K pares (fp16, 4096-d)
HN_CHUNK_GB = 100_000 * 10 * 4096 * 2 / 1e9  # negativos de 100K pares


# --------------------------------------------------------------------------------------
@dataclass
class Paths:
    drive_root: str = DRIVE_ROOT_DEFAULT
    cache_root: str = CACHE_ROOT_DEFAULT

    def d(self, *p: str) -> str:
        """Ruta dentro de Drive (resultados, checkpoints, datos derivados)."""
        return os.path.join(self.drive_root, *p)

    def c(self, *p: str) -> str:
        """Ruta dentro de la caché local (datos públicos re-descargables)."""
        return os.path.join(self.cache_root, *p)

    @property
    def data(self) -> str:
        return self.d("data")

    @property
    def runs(self) -> str:
        return self.d("runs")

    @property
    def results(self) -> str:
        return self.d("results")

    @property
    def logs(self) -> str:
        return self.d("logs")

    def make(self) -> "Paths":
        for p in (self.drive_root, self.data, self.runs, self.results, self.logs, self.cache_root):
            os.makedirs(p, exist_ok=True)
        return self


# --------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    profile: str = "colab"
    synthetic: bool = False
    shard: str = "shard_00"
    pre_chunks: list = field(default_factory=lambda: list(range(20)))
    hn_chunks: list = field(default_factory=lambda: [0, 1, 2, 3])
    k_neg: int = 10
    # Split por documento: SÓLO documentos de hn_chunks son elegibles para test/val/confirm
    # (son los únicos con HN de Gemini). Los cabezales públicos de CLM vieron esos pares:
    # sólo sirven como referencia contaminada.
    test_frac: float = 0.10      # test exploratorio (F1/F2)
    val_frac: float = 0.03       # selección de HP / early stopping
    confirm_frac: float = 0.10   # test confirmatorio SELLADO: sólo se evalúa tras PROTOCOL_FREEZE (F3)
    split_salt: str = "clm-losses-v1"
    max_per_doc: int | None = 2  # ítems de evaluación por documento (diseño: ≤2)
    n_val: int = 10_000          # subconjunto fijo para validación/selección
    n_test_max: int | None = 60_000
    n_confirm_max: int | None = 60_000
    eval_on_drive: bool = True    # sets de evaluación en Drive (sobreviven a reinicios)
    # Streams (bloques contiguos barajados; grupos cargados a RAM)
    pre_block_rows: int = 16_384
    pre_group_blocks: int = 4
    mid_block_rows: int = 4_096
    mid_group_blocks: int = 4
    # Datos sintéticos (perfil smoke)
    syn: dict = field(default_factory=dict)

    def disk_gb(self) -> float:
        extra_q = [c for c in self.hn_chunks if c not in self.pre_chunks]
        return (len(self.pre_chunks) + len(extra_q)) * PRE_CHUNK_GB + len(self.hn_chunks) * HN_CHUNK_GB


PROFILES: dict[str, dict] = {
    # Sin descargas: embeddings sintéticos con el mismo layout de archivos (valida el código en minutos).
    "smoke": dict(synthetic=True, pre_chunks=[0, 1, 2, 3], hn_chunks=[0, 1], n_val=600, n_test_max=None,
                  pre_block_rows=1024, pre_group_blocks=2, mid_block_rows=512, mid_group_blocks=2,
                  syn=dict(rows_per_chunk=3000, hidden=128, pairs_per_doc=4, latent=24, seed=0,
                           hardness=0.25, artifact=0.3)),
    # ~26 GB en disco local.
    "mini": dict(pre_chunks=list(range(6)), hn_chunks=[0, 1], n_val=5_000, n_test_max=30_000),
    # ~66 GB en disco local (recomendado en Colab con runtime A100/L4 o T4 con disco grande).
    "colab": dict(pre_chunks=list(range(20)), hn_chunks=[0, 1, 2, 3]),
    # ~148 GB en disco local.
    "colab_large": dict(pre_chunks=list(range(60)), hn_chunks=list(range(6)), n_test_max=80_000),
}


def data_config(profile: str, **overrides: Any) -> DataConfig:
    if profile not in PROFILES:
        raise ValueError(f"perfil desconocido {profile!r}; opciones: {sorted(PROFILES)}")
    kw = dict(PROFILES[profile])
    kw.update(overrides)
    return DataConfig(profile=profile, **kw)


# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    arm: str = "infonce"
    seed: int = 0
    # Arquitectura común (D6): la de CLM inferida; se confirma en F0.1.
    width: int = 1536
    depth: int = 3
    proj: int = 512
    layernorm: bool = True
    activation: str = "gelu"
    residual: bool = False
    tau_init: float = 0.07
    # Etapas
    stages: list = field(default_factory=lambda: ["pre", "mid"])  # ["mid"] + hn_from_start => régimen B3
    pre_epochs: float = 2.0
    mid_epochs: float = 1.0
    hn_from_start: bool = False       # B3: sólo etapa "mid" (con HN) con el presupuesto de pre+mid
    hn_mode: str = "auto"             # auto | none | native | aux  (R0/R1 = auto, R2 = aux)
    aux_hn_weight: float = 0.5        # peso del término auxiliar de HN (R2)
    batch_pre: int = 2048
    batch_mid: int = 512
    lr_mult: float = 1.0              # multiplica la regla de LR de CLM
    lr: float | None = None           # si se fija, ignora la regla
    weight_decay: float = 0.0
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    max_steps_pre: int | None = None
    max_steps_mid: int | None = None
    # Evaluación / early stopping
    eval_every: int = 100
    patience: int = 8
    select_metric: str = "val_top1"
    # El punto de partida de cada etapa (paso 0) NO compite en la selección: en "pre" son cabezales
    # aleatorios y en "mid" es el final de "pre". Así cada etapa entrega un modelo entrenado en ella
    # (si no, una pérdida que colapsa quedaría reportada como la proyección aleatoria inicial).
    select_include_init: bool = False
    save_final: bool = False          # guarda final_<etapa>.pt (diagnóstico de colapso en F0.8)
    # Checkpoints (reanudación)
    ckpt_every_steps: int = 200
    ckpt_every_minutes: float = 10.0
    # Aumentación en espacio de embedding (factor controlado: igual para todos los brazos)
    aug_noise_std: float = 0.0
    aug_feat_dropout: float = 0.0
    # Falsos negativos in-batch: pares con coseno crudo > umbral se enmascaran
    dup_threshold: float = 0.999
    amp: str = "fp32"                 # fp32 | bf16 | fp16
    loss_params: dict = field(default_factory=dict)
    # --- v0.3 (enmienda A1). Con sus valores por defecto NO cambian nada ni la huella de corridas previas.
    lr_schedule: str = "onecycle"     # onecycle (F1 piloto) | const (receta real de CLM, A1.5)
    lr_mid_scale: float | None = None  # mid (tras pre) usa LR × escala; CLM: 1e-3 / regla(16384) ≈ 0.153
    evals_per_stage: int | None = None  # si se fija: eval_every = pasos de la etapa // evals_per_stage
    init_from: str | None = None      # carpeta de una corrida cuyo best_pre.pt inicializa los cabezales (F0.7b)

    def copy(self, **kw: Any) -> "TrainConfig":
        c = copy.deepcopy(self)
        for k, v in kw.items():
            if not hasattr(c, k):
                raise AttributeError(f"TrainConfig no tiene el campo {k!r}")
            setattr(c, k, v)
        return c


@dataclass
class HPOConfig:
    n_trials: int = 32            # P9 (decidido 2026-09-30): permite el chequeo de tunability 8/16/32
    budget_frac: float = 0.25    # fracción de pasos (pre y mid) usada en cada trial
    seed: int = 12345


@dataclass
class EvalConfig:
    calib_bins: int = 15
    n_geo: int = 5_000
    hub_k: int = 10
    abst_k: int = 9              # conjunto con oro = oro + 9 HN; sin oro = 10 HN (mismo tamaño)
    n_rand: int = 10
    n_mine: int = 10
    mine_fn_threshold: float = 0.95
    eval_batch: int = 2048
    # v0.3
    n_mine_hard: int = 50        # T-mine-50 (secundaria; techo de T-mine)
    shuffle_seed: int = 0        # permutación de estados barajados (la misma de las celdas 5.4 y 5.5)


# Campos de TrainConfig agregados en v0.3 y su valor por defecto. Con ese valor se excluyen de la huella
# de la corrida, así las corridas de v0.2 (F0.7, F1 piloto) conservan su huella y no se rechazan.
V03_TRAIN_DEFAULTS = {"lr_schedule": "onecycle", "lr_mid_scale": None, "evals_per_stage": None, "init_from": None}


def clm_lr(width: int, batch: int) -> float:
    """Regla de LR de CLM: 2e-3·√(1024/width)·√(batch/1024). Con width 1536 y batch 16,384 da 6.53e-3,
    el LR del checkpoint público de pre-training [VERIFIED en su cfg]."""
    return 2e-3 * math.sqrt(1024 / width) * math.sqrt(batch / 1024)


# Recetas de F0.7b / F1b (A1.5). "actual" = la del F1 piloto (lo que dice study.train).
CLM_BATCH = 16_384
CLM_MID_LR = 1e-3


def recipe_overrides(study, name: str) -> dict:
    """Overrides de TrainConfig de cada receta. La de CLM sale de los `cfg` de sus checkpoints de pre y mid:
    batch 16,384, LR constante (pre = regla de CLM, mid = 1e-3), weight decay 0 [VERIFIED]. El tope de épocas
    (pre 4, mid 8) y las 20 evaluaciones por etapa son [SUPUESTO]: CLM no publica cuántas épocas usó. Con
    294K filas con HN, una época de mid a batch 16,384 son sólo 18 pasos: por eso el tope de mid es mayor."""
    if name == "actual":
        return {}
    if name != "clm":
        raise ValueError(f"receta desconocida {name!r}: 'actual' o 'clm'")
    t = study.train
    batch = CLM_BATCH
    if study.data.profile == "smoke":      # datos sintéticos diminutos: se escala el batch (sólo prueba el código)
        batch = 1024
    return {"batch_pre": batch, "batch_mid": batch, "lr_schedule": "const",
            "lr_mid_scale": CLM_MID_LR / clm_lr(t.width, CLM_BATCH), "weight_decay": 0.0,
            "pre_epochs": 4.0, "mid_epochs": 8.0, "evals_per_stage": 20, "patience": 8}


@dataclass
class StudyConfig:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    f1_seeds: list = field(default_factory=lambda: [0, 1, 2])
    f3_seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    rope_pp: float = 1.0          # margen de equivalencia práctica (puntos porcentuales)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        from .utils import save_json
        save_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, d: dict) -> "StudyConfig":
        return _from_dict(cls, d)

    @classmethod
    def load(cls, path: str) -> "StudyConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _from_dict(cls, d):
    if not dataclasses.is_dataclass(cls):
        return d
    kw = {}
    hints = {f.name: f.type for f in dataclasses.fields(cls)}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        sub = {"paths": Paths, "data": DataConfig, "train": TrainConfig, "hpo": HPOConfig,
               "eval": EvalConfig}.get(f.name) if cls is StudyConfig else None
        kw[f.name] = _from_dict(sub, v) if sub is not None else v
    del hints
    return cls(**kw)


def train_config_from_dict(d: dict) -> TrainConfig:
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in d.items() if k in names})
