"""Servidor /v1/embeddings compatible con el cliente de CLM (clm.embedder.Embedder), usando el
encoder de transformers. Sustituye a `vllm serve Qwen/Qwen3-8B --runner pooling` para correr
T-Rex en Colab sin instalar vLLM.

    python -m clm_losses.encoder_server --port 8090 --model Qwen/Qwen3-8B
"""
# Sin `from __future__ import annotations`: FastAPI necesita resolver el modelo de la petición
# (con anotaciones en texto y el modelo definido dentro de una función, cada POST devolvía 422).
import argparse
import base64

import numpy as np


from typing import List, Optional, Union

from pydantic import BaseModel


class Req(BaseModel):
    model: Optional[str] = None
    input: Union[str, List]
    encoding_format: Optional[str] = "float"
    truncate_prompt_tokens: Optional[int] = None


def build_app(encoder, served_name: str = "qwen3-8b"):
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model"}]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/embeddings")
    def embeddings(req: Req):
        inp = req.input
        if isinstance(inp, str):
            inp = [inp]
        if inp and isinstance(inp[0], int):
            inp = [inp]
        ids = []
        for x in inp:
            t = encoder.tok(x, add_special_tokens=False)["input_ids"] if isinstance(x, str) else list(x)
            if req.truncate_prompt_tokens:
                t = t[-int(req.truncate_prompt_tokens):]   # vLLM trunca por la izquierda
            ids.append(t or [encoder.pad_id])
        E = encoder.embed_ids(ids).astype(np.float32)
        data = []
        for i, v in enumerate(E):
            emb = base64.b64encode(v.tobytes()).decode() if req.encoding_format == "base64" else v.tolist()
            data.append({"object": "embedding", "index": i, "embedding": emb})
        return {"object": "list", "data": data, "model": served_name,
                "usage": {"prompt_tokens": int(sum(len(x) for x in ids)), "total_tokens": int(sum(len(x) for x in ids))}}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--served-name", default="qwen3-8b")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    import uvicorn
    from .encoder import HFEncoder
    enc = HFEncoder(a.model, dtype=a.dtype, max_len=a.max_len)
    uvicorn.run(build_app(enc, a.served_name), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
