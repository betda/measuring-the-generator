"""T-Rex con y sin shield (D2), usando el arnés del repo de CLM sin modificarlo.

Piezas que se levantan en segundo plano:
    1. servidor de embeddings (transformers) en :8090   -> sustituye a vLLM
    2. clm-serve --ckpt <cabezal> en :8700               -> el servidor oficial de CLM
    3. examples/t_rex/run.py --model clm [--no-shield] --out <Drive>

Advertencia científica: las opciones llegan etiquetadas por el planner ("Safe… Best."); aun sin
shield, T-Rex mide "seguir etiquetas", no decidir. Métrica principal: acuerdo con el planner.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from .config import CLM_GIT_COMMIT, CLM_GIT_URL
from .utils import load_json, makedirs, save_json

EMB_PORT, CLM_PORT = 8090, 8700


def setup_clm_repo(dest: str = "/content/CLM", commit: str = CLM_GIT_COMMIT, install: bool = True) -> str:
    if not os.path.isdir(os.path.join(dest, ".git")):
        subprocess.run(["git", "clone", "-q", CLM_GIT_URL, dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "-q", commit], check=True)
    if install:
        # --no-deps: el paquete declara vllm como dependencia y rompería el torch de Colab
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", dest], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "httpx"], check=True)
    return dest


def _wait(url: str, timeout: float = 900.0) -> None:
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"{url} no respondió en {timeout:.0f}s")


_PROCS: dict = {}


def start_embedding_server(log_dir: str, model: str = "Qwen/Qwen3-8B") -> None:
    p = _PROCS.get("emb")
    if p is not None and p.poll() is None:
        return
    makedirs(log_dir)
    f = open(os.path.join(log_dir, "embedding_server.log"), "a")
    _PROCS["emb"] = subprocess.Popen([sys.executable, "-m", "clm_losses.encoder_server", "--port", str(EMB_PORT),
                                      "--model", model], stdout=f, stderr=subprocess.STDOUT)
    _wait(f"http://127.0.0.1:{EMB_PORT}/v1/models")


def start_clm_serve(ckpt: str, log_dir: str) -> None:
    stop_clm_serve()
    makedirs(log_dir)
    f = open(os.path.join(log_dir, "clm_serve.log"), "a")
    _PROCS["clm"] = subprocess.Popen(
        ["clm-serve", "--ckpt", ckpt, "--emb-url", f"http://127.0.0.1:{EMB_PORT}/v1/embeddings",
         "--emb-model", "qwen3-8b", "--port", str(CLM_PORT), "--no-ui"], stdout=f, stderr=subprocess.STDOUT)
    _wait(f"http://127.0.0.1:{CLM_PORT}/health", timeout=300)


def stop_clm_serve() -> None:
    p = _PROCS.pop("clm", None)
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()


def stop_all() -> None:
    stop_clm_serve()
    p = _PROCS.pop("emb", None)
    if p is not None and p.poll() is None:
        p.terminate()


def run_trex(ckpt: str, out_dir: str, repo: str = "/content/CLM", seeds: int = 5, duration: float = 60.0,
             modes=("shield", "noshield"), log=print) -> dict:
    """Corre T-Rex para un cabezal en los modos pedidos. Reanudable por modo."""
    makedirs(out_dir)
    start_embedding_server(out_dir)
    start_clm_serve(ckpt, out_dir)
    res = {}
    try:
        for mode in modes:
            out = os.path.join(out_dir, f"trex_{mode}.json")
            if not os.path.exists(out):
                cmd = [sys.executable, os.path.join(repo, "examples", "t_rex", "run.py"), "--model", "clm",
                       "--seeds", str(seeds), "--duration", str(duration), "--out", out]
                if mode == "noshield":
                    cmd.append("--no-shield")
                env = {**os.environ, "CLM_BASE_URL": f"http://127.0.0.1:{CLM_PORT}"}
                log(f"[trex] {mode}: {' '.join(cmd[1:])}")
                with open(os.path.join(out_dir, f"trex_{mode}.log"), "a") as f:
                    subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            s = (load_json(out) or {}).get("summary", {})
            rows = (load_json(out) or {}).get("results", [])
            res[mode] = {k: s.get(k) for k in ("survived", "deaths", "mean_best_score", "mean_decisions",
                                               "mean_agreement_with_planner", "shield_interventions",
                                               "latency_ms_p50_median", "model_ms_p50_median", "errors")}
            res[mode]["host_stall_seconds"] = sum(r.get("host_stall_seconds_dropped", 0) or 0 for r in rows)
            log(f"[trex] {mode}: acuerdo={res[mode]['mean_agreement_with_planner']} muertes={res[mode]['deaths']} "
                f"stall={res[mode]['host_stall_seconds']:.1f}s")
    finally:
        stop_clm_serve()
    save_json(res, os.path.join(out_dir, "trex_summary.json"))
    return res
