"""Cabezales de proyección idénticos a los de CLM, para que cualquier checkpoint nuestro
se pueda servir con `clm-serve --ckpt` (T-Rex) y comparar con los cabezales públicos.

`make_head` reproduce `clm/heads.py` (Contrastive-LM/CLM, Apache-2.0, commit bb42c6c).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 4096


def make_head(width: int, depth: int = 2, proj: int = 512, activation: str = "gelu",
              layernorm: bool = False, residual: bool = False, hidden: int = HIDDEN) -> nn.Module:
    """hidden -> width -> ... -> proj; MLP con LayerNorm / residual opcionales en capas ocultas."""
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity())
                                       for _ in range(depth - 2))
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


class Heads(nn.Module):
    """Cabezal de estado + cabezal de acción + logit_scale (formato CLM)."""

    def __init__(self, hidden: int = HIDDEN, width: int = 1536, depth: int = 3, proj: int = 512,
                 activation: str = "gelu", layernorm: bool = True, residual: bool = False,
                 tau_init: float = 0.07):
        super().__init__()
        self.cfg = dict(hidden_size=hidden, width=width, depth=depth, projection_dim=proj,
                        activation=activation, layernorm=layernorm, residual=residual)
        kw = dict(width=width, depth=depth, proj=proj, activation=activation, layernorm=layernorm,
                  residual=residual, hidden=hidden)
        self.state_head = make_head(**kw)
        self.action_head = make_head(**kw)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / tau_init)))

    def zs(self, es: torch.Tensor) -> torch.Tensor:
        return self.state_head(es)

    def za(self, ea: torch.Tensor) -> torch.Tensor:
        return self.action_head(ea)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def n_params(self) -> int:
        return sum(p.numel() for h in (self.state_head, self.action_head) for p in h.parameters())

    def clm_checkpoint(self, logit_scale: float | None = None) -> dict:
        """Checkpoint en formato CLM (servible con clm-serve / bon_eval.py)."""
        ls = torch.tensor(float(logit_scale)) if logit_scale is not None else self.logit_scale.detach().cpu()
        return {"state_head": {k: v.detach().cpu() for k, v in self.state_head.state_dict().items()},
                "action_head": {k: v.detach().cpu() for k, v in self.action_head.state_dict().items()},
                "logit_scale": ls, "cfg": dict(self.cfg)}


def _infer_layernorm(sd: dict) -> bool:
    return any(k.startswith("norms.") and k.endswith(".weight") for k in sd)


def heads_from_clm_checkpoint(ck: dict, device: str | torch.device = "cpu") -> Heads:
    """Carga un checkpoint CLM (público o nuestro). Si `layernorm` no está en cfg se infiere de
    las claves del state_dict (el servidor y bon_eval.py de CLM tienen defaults distintos)."""
    cfg = dict(ck.get("cfg", {}))
    sd = ck["state_head"]
    hidden = cfg.get("hidden_size") or sd["inp.weight"].shape[1]
    width = cfg.get("width") or sd["inp.weight"].shape[0]
    n_hidden_layers = len({k.split(".")[1] for k in sd if k.startswith("hidden.")})
    depth = cfg.get("depth") or (n_hidden_layers + 2)
    proj = ck.get("projection_dim", cfg.get("projection_dim")) or sd["out.weight"].shape[0]
    layernorm = cfg["layernorm"] if "layernorm" in cfg else _infer_layernorm(sd)
    h = Heads(hidden=hidden, width=width, depth=depth, proj=proj,
              activation=cfg.get("activation", "gelu"), layernorm=layernorm,
              residual=cfg.get("residual", False))
    h.state_head.load_state_dict(ck["state_head"])
    h.action_head.load_state_dict(ck["action_head"])
    with torch.no_grad():
        h.logit_scale.copy_(torch.as_tensor(ck.get("logit_scale", math.log(1 / 0.07))).float())
    return h.to(device).eval()


def load_torch(path: str) -> dict:
    """torch.load con weights_only=True primero (seguro); si falla, lo informa."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # checkpoints con objetos no-tensor
        raise RuntimeError(
            f"{path}: no carga con weights_only=True ({e}). Si confías en la fuente, usa "
            "torch.load(path, weights_only=False) manualmente.") from e


class EMA:
    """Copia EMA de un módulo (targets de BYOL/DINO)."""

    @staticmethod
    @torch.no_grad()
    def update(target: nn.Module, online: nn.Module, tau: float) -> None:
        for pt, po in zip(target.parameters(), online.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1.0 - tau)
        for bt, bo in zip(target.buffers(), online.buffers()):
            bt.copy_(bo)


def cosine_momentum(step: int, total: int, base: float) -> float:
    """Momentum EMA de BYOL/DINO: base -> 1 con coseno."""
    total = max(1, total)
    return 1.0 - (1.0 - base) * (math.cos(math.pi * min(step, total) / total) + 1) / 2


def l2n(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)
