"""F0.10: ¿los embeddings públicos de tarsur385 son el set de evaluación de DeepSWE de CLM?
Se corre el `evaluation/bon_eval.py` original con el cabezal público de DeepSWE. Si reproduce
exactamente 31/38 (81.6 %), la identidad queda confirmada y se habilita la CV sobre 113 tareas."""
from __future__ import annotations

import os
import subprocess
import sys

from .config import DEEPSWE_EVAL_CANDIDATE, DEEPSWE_HEADS_REPO
from .utils import load_json, makedirs, save_json


def f0_deepswe(paths, repo: str = "/content/CLM", candidate: str = DEEPSWE_EVAL_CANDIDATE, log=print) -> dict:
    out_dir = paths.d("f0", "F0.10_deepswe")
    makedirs(out_dir)
    res_path = os.path.join(out_dir, "result.json")
    if os.path.exists(res_path):
        return load_json(res_path)
    from huggingface_hub import snapshot_download
    heads = snapshot_download(DEEPSWE_HEADS_REPO, local_dir=paths.c("hf", "deepswe_heads"))
    out_json = os.path.join(out_dir, "bon_eval.json")
    cmd = [sys.executable, os.path.join(repo, "evaluation", "bon_eval.py"), "--hf-dataset", candidate,
           "--checkpoint", os.path.join(heads, "best_head.pt"),
           "--tasks-file", os.path.join(heads, "heldout_tasks.json"), "--n", "4", "--window", "12",
           "--output", out_json, "--cache-dir", paths.c("deepswe_emb")]
    log("[deepswe] " + " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(out_dir, "bon_eval.log"), "w") as f:
        f.write(proc.stdout + "\n" + proc.stderr)
    res = {"candidate": candidate, "returncode": proc.returncode}
    if proc.returncode == 0 and os.path.exists(out_json):
        r = load_json(out_json)
        sel = next(iter(r.get("selectors", {}).values()), {})
        res.update(n_tasks=r.get("n_tasks"), random_pick=r.get("random_pick"), oracle=r.get("oracle_any"),
                   mixed=r.get("mixed_tasks"), rate=sel.get("rate"), resolved=sel.get("resolved"))
        res["G6_pass"] = bool(res.get("n_tasks") == 38 and res.get("resolved") == 31)
    else:
        res["G6_pass"] = False
        res["error"] = proc.stderr[-2000:]
    save_json(res, res_path)
    log(f"[deepswe] G6={'PASA' if res['G6_pass'] else 'NO PASA'} {res}")
    return res
