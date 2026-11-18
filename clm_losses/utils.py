"""Utilidades: guardado atómico (seguro ante cortes de Colab), logs, semillas, entorno."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------
# Archivos
def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def makedirs(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _tmp(path: str) -> str:
    return f"{path}.tmp{os.getpid()}"


def save_json(obj: Any, path: str, indent: int = 1) -> None:
    """Escritura atómica: si Colab se corta a mitad, el archivo anterior queda intacto."""
    makedirs(os.path.dirname(os.path.abspath(path)))
    tmp = _tmp(path)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _ends_with_newline(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return True
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


def append_jsonl(path: str, rec: dict) -> None:
    """Agrega una línea. Si un corte dejó la última línea a medias, primero la cierra (así no se pega a la
    nueva y sólo se pierde el registro cortado)."""
    makedirs(os.path.dirname(os.path.abspath(path)))
    pre = "" if _ends_with_newline(path) else "\n"
    with open(path, "a") as f:
        f.write(pre + json.dumps(rec, default=_json_default) + "\n")
        f.flush()


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # línea truncada por un corte
    return out


def atomic_torch_save(obj: Any, path: str) -> None:
    import torch
    makedirs(os.path.dirname(os.path.abspath(path)))
    tmp = _tmp(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_np_save(arr: np.ndarray, path: str) -> None:
    makedirs(os.path.dirname(os.path.abspath(path)))
    tmp = _tmp(path) + ".npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def file_sha256(path: str, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def write_text(path: str, text: str) -> None:
    """Escritura atómica de texto."""
    makedirs(os.path.dirname(os.path.abspath(path)))
    tmp = _tmp(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
    except ImportError:
        pass
    return str(o)


# --------------------------------------------------------------------------------------
# Estado de corridas (reanudación a nivel de celda)
DONE = "DONE.json"


def is_done(run_dir: str) -> bool:
    return os.path.exists(os.path.join(run_dir, DONE))


def mark_done(run_dir: str, summary: dict) -> None:
    save_json({**summary, "finished_at": now_iso()}, os.path.join(run_dir, DONE))


def read_done(run_dir: str) -> dict | None:
    return load_json(os.path.join(run_dir, DONE))


def set_status(run_dir: str, status: str, **extra: Any) -> None:
    save_json({"status": status, "updated_at": now_iso(), **extra}, os.path.join(run_dir, "status.json"))


class RunLogger:
    """Log JSONL en Drive + eco en pantalla."""

    def __init__(self, run_dir: str, name: str = "log.jsonl", echo: bool = True, prefix: str = ""):
        self.path = os.path.join(run_dir, name)
        self.echo = echo
        self.prefix = prefix
        makedirs(run_dir)

    def log(self, msg: str | None = None, **kw: Any) -> None:
        rec = {"t": now_iso(), **({"msg": msg} if msg else {}), **kw}
        append_jsonl(self.path, rec)
        if self.echo:
            parts = [f"{k}={_short(v)}" for k, v in kw.items()]
            print(f"{self.prefix}{msg or ''} {' '.join(parts)}".strip(), flush=True)


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


# --------------------------------------------------------------------------------------
# Semillas
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_rng_state() -> dict:
    import torch
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    import torch
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception:
            pass  # distinta GPU tras reinicio: el determinismo exacto de CUDA no se garantiza


def rng(*keys: int) -> np.random.Generator:
    """Generador determinista derivado de varias claves enteras (semilla, época, grupo…)."""
    return np.random.default_rng(np.random.SeedSequence([int(k) % (2**32) for k in keys]))


# --------------------------------------------------------------------------------------
# Entorno y trazabilidad
def code_hash(pkg_dir: str = PKG_DIR) -> str:
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(pkg_dir)):
        for fn in sorted(files):
            if fn.endswith(".py"):
                with open(os.path.join(root, fn), "rb") as f:
                    h.update(fn.encode())
                    h.update(f.read())
    return h.hexdigest()[:16]


def gpu_info() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False}
        p = torch.cuda.get_device_properties(0)
        return {"cuda": True, "name": p.name, "mem_gb": round(p.total_memory / 1e9, 1),
                "capability": f"{p.major}.{p.minor}", "bf16": bool(torch.cuda.is_bf16_supported())}
    except Exception as e:  # pragma: no cover
        return {"cuda": False, "error": str(e)}


def disk_free_gb(path: str) -> float:
    makedirs(path)
    return shutil.disk_usage(path).free / 1e9


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return float("nan")


def _version(mod: str) -> str | None:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return None


def env_snapshot() -> dict:
    snap = {
        "time": now_iso(), "python": sys.version.split()[0], "platform": platform.platform(),
        "code_hash": code_hash(), "gpu": gpu_info(), "ram_gb": round(ram_gb(), 1),
        "versions": {m: _version(m) for m in ("torch", "numpy", "scipy", "pyarrow", "pandas",
                                               "transformers", "vllm", "huggingface_hub", "sklearn")},
    }
    try:
        import torch
        snap["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return snap


def pip_freeze() -> str:
    try:
        return subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                              timeout=120).stdout
    except Exception as e:  # pragma: no cover
        return f"pip freeze falló: {e}"


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def s(self) -> float:
        return time.time() - self.t0


def human_time(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"
