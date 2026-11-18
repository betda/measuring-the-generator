"""Encoder Qwen3-8B con last-token pooling y normalización L2 (la receta de los stores de CLM:
token ids "desnudos", sin plantilla de chat ni tokens especiales).

Backends:
    hf    transformers (robusto en Colab; se valida contra los embeddings públicos en F0.3)
    vllm  vLLM runner=pooling (el que usó CLM); opcional, su instalación puede chocar con torch
Requiere GPU con >= 20 GB (A100 o L4). En T4 no cabe en bf16.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class HFEncoder:
    def __init__(self, model: str = "Qwen/Qwen3-8B", dtype: str = "bfloat16", device: str | None = None,
                 max_len: int = 2048, batch_tokens: int = 32_768):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model)
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        kw = dict(torch_dtype=td)
        try:
            self.model = AutoModel.from_pretrained(model, attn_implementation="sdpa", **kw)
        except (TypeError, ValueError):
            self.model = AutoModel.from_pretrained(model, **kw)
        self.model.to(self.device).eval()
        self.max_len, self.batch_tokens = max_len, batch_tokens
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    def tokenize(self, texts: list[str], keep: str = "tail") -> list[list[int]]:
        ids = self.tok(list(texts), add_special_tokens=False)["input_ids"]
        cap = self.max_len
        return [(x[-cap:] if keep == "tail" else x[:cap]) or [self.pad_id] for x in ids]

    @torch.no_grad()
    def embed_ids(self, id_lists: list[list[int]], progress: bool = False) -> np.ndarray:
        lens = np.array([min(len(x), self.max_len) for x in id_lists])
        order = np.argsort(-lens, kind="stable")
        out = np.zeros((len(id_lists), self.model.config.hidden_size), np.float32)
        i = 0
        n = len(order)
        while i < n:
            L = int(lens[order[i]])
            bs = max(1, self.batch_tokens // max(1, L))
            idx = order[i:i + bs]
            ids = torch.full((len(idx), L), self.pad_id, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, j in enumerate(idx):
                x = id_lists[j][: self.max_len]
                ids[r, : len(x)] = torch.tensor(x)
                att[r, : len(x)] = 1
            ids, att = ids.to(self.device), att.to(self.device)
            h = self.model(input_ids=ids, attention_mask=att).last_hidden_state
            last = att.sum(1) - 1
            v = h[torch.arange(len(idx), device=self.device), last].float()
            out[idx] = F.normalize(v, dim=-1).cpu().numpy()
            i += bs
            if progress and (i // bs) % 50 == 0:
                print(f"[encoder] {min(i, n)}/{n}", flush=True)
        return out

    def embed_texts(self, texts: list[str], keep: str = "tail") -> np.ndarray:
        return self.embed_ids(self.tokenize(texts, keep))


class VLLMEncoder:
    def __init__(self, model: str = "Qwen/Qwen3-8B", max_len: int = 2048, gpu_mem: float = 0.85):
        from vllm import LLM
        try:
            self.llm = LLM(model=model, runner="pooling", max_model_len=max_len, gpu_memory_utilization=gpu_mem)
        except TypeError:   # vLLM antiguo
            self.llm = LLM(model=model, task="embed", max_model_len=max_len, gpu_memory_utilization=gpu_mem)
        self.tok = self.llm.get_tokenizer()
        self.max_len = max_len

    def tokenize(self, texts, keep="tail"):
        ids = self.tok(list(texts), add_special_tokens=False)["input_ids"]
        return [(x[-self.max_len:] if keep == "tail" else x[: self.max_len]) or [0] for x in ids]

    def embed_ids(self, id_lists, progress=False) -> np.ndarray:
        from vllm.inputs import TokensPrompt
        outs = self.llm.embed([TokensPrompt(prompt_token_ids=list(x)[: self.max_len]) for x in id_lists],
                              use_tqdm=progress)
        E = np.stack([np.asarray(o.outputs.embedding, np.float32) for o in outs])
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    def embed_texts(self, texts, keep="tail"):
        return self.embed_ids(self.tokenize(texts, keep))


def make_encoder(backend: str = "hf", **kw):
    if backend == "hf":
        return HFEncoder(**kw)
    if backend == "vllm":
        return VLLMEncoder(**{k: v for k, v in kw.items() if k in ("model", "max_len", "gpu_mem")})
    raise ValueError(backend)


def embed_to_files(encoder, id_lists: list[list[int]], out_path: str, chunk: int = 4096, log=print) -> np.ndarray:
    """Embebe por trozos guardando parciales (reanudable). Devuelve fp16 [N, H]."""
    import os
    from .utils import atomic_np_save
    parts = []
    os.makedirs(out_path + ".parts", exist_ok=True)
    for s in range(0, len(id_lists), chunk):
        p = os.path.join(out_path + ".parts", f"{s:09d}.npy")
        if not os.path.exists(p):
            atomic_np_save(encoder.embed_ids(id_lists[s:s + chunk]).astype(np.float16), p)
            log(f"[embed] {min(len(id_lists), s + chunk)}/{len(id_lists)}")
        parts.append(np.load(p))
    E = np.concatenate(parts) if parts else np.zeros((0, 0), np.float16)
    atomic_np_save(E, out_path)
    return E
