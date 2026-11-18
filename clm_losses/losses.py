"""Las 15 pérdidas del estudio (más el modelo ciego y el término auxiliar de HN).

Convenciones (comunes a todos los brazos, D6):
    es, ea          embeddings del encoder congelado (B x H), normalizados L2
    en, nmask       negativos difíciles (B x K x H) y máscara de válidos (B x K)
    zs, za, zn      salidas de los cabezales (B x d, B x K x d) ANTES de normalizar
    puntuación      cos(zs, za); los módulos extra (predictores, expanders, prototipos,
                    targets EMA) sólo existen en entrenamiento.
    dup_a, dup_s    máscaras de falsos negativos in-batch (acciones / estados casi idénticos)
Todas las pérdidas se calculan en float32.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import EMA, Heads, cosine_momentum, l2n

NEG = -1e9


def _f(x) -> float:
    """float() sin advertencias de autograd."""
    return float(x.detach()) if torch.is_tensor(x) else float(x)


@dataclass
class Ctx:
    es: torch.Tensor
    ea: torch.Tensor
    zs: torch.Tensor
    za: torch.Tensor
    en: torch.Tensor | None = None
    zn: torch.Tensor | None = None
    nmask: torch.Tensor | None = None
    dup_a: torch.Tensor | None = None
    dup_s: torch.Tensor | None = None
    step: int = 0


class Loss(nn.Module):
    name = "base"
    uses_hn = False       # sabe usar HN de forma nativa
    has_scale = False     # usa heads.logit_scale (softmax listwise calibrable de forma nativa)

    def __init__(self, heads: Heads, p: dict, total_steps: int = 1, use_hn: bool = False):
        super().__init__()
        self.p = dict(p)
        self.total_steps = max(1, total_steps)
        self.use_hn = use_hn
        if use_hn and not self.uses_hn:
            raise ValueError(f"{self.name} no admite HN nativos; usa hn_mode='aux'")

    def set_total_steps(self, n: int) -> None:
        self.total_steps = max(1, int(n))

    def forward(self, ctx: Ctx, heads: Heads):  # -> (loss, logs)
        raise NotImplementedError

    def after_step(self, heads: Heads, step: int) -> None:
        pass

    def before_optimizer_step(self, step: int) -> None:
        pass

    def native_scale(self, heads: Heads) -> float | None:
        """Escala para softmax sobre candidatos, si la pérdida la aprende."""
        return _f(heads.scale()) if self.has_scale else None

    def pointwise(self, cos: torch.Tensor) -> torch.Tensor | None:
        """Probabilidad por par (sólo pérdidas sigmoides)."""
        return None


# --------------------------------------------------------------------------------------
def _ce_masked(logits: torch.Tensor, mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if mask is not None:
        logits = logits.masked_fill(mask, NEG)
    return F.cross_entropy(logits, target)


def _hn_logits(s: torch.Tensor, a: torch.Tensor, n: torch.Tensor, nmask: torch.Tensor,
               scale: torch.Tensor) -> torch.Tensor:
    """[pos, HN_1..HN_K] por fila (escalados); negativos inválidos a -inf."""
    pos = (s * a).sum(-1, keepdim=True)
    neg = torch.einsum("bd,bkd->bk", s, n)
    logits = scale * torch.cat([pos, neg], 1)
    valid = torch.cat([torch.ones_like(nmask[:, :1]), nmask], 1)
    return logits.masked_fill(~valid, NEG)


class InfoNCE(Loss):
    """(1) y (2): InfoNCE bidireccional de CLM; en mid-training con HN en s→a.
    hn_denominator='hn_only' sigue la fórmula del README (denominador = positivo + HN);
    'hn_plus_inbatch' añade además los negativos in-batch. null_candidate=True (B2) añade un
    candidato "ninguna" aprendido: en una fracción de filas se quita el oro y el objetivo es null.

    v0.3.8 (condición de borde de B5, notebook pre-envío; INNOVATION_MAP §2.8). Denominadores nuevos, sólo en el
    término estado→acción (el acción→estado sigue con in-batch). Los dos de arriba no cambian:
        'hn_plus_inbatch_k'  el oro, sus HN y `extra_k` oros de otros estados del batch, al azar en cada paso
        'hn_plus_otherhn'    el oro, sus HN y un HN de cada uno de los otros B − 1 estados (mismo número de
                             candidatos extra que 'hn_plus_inbatch', pero escritos por el generador de los HN)
        'hn_plus_otherhn_k'  lo mismo con sólo `extra_k` de esos HN ajenos, al azar en cada paso
        'hn_shared_inbatch'  el oro, todos los HN del batch (los propios y los ajenos) y los oros de los otros estados
                             (como NegCLIP)
    El muestreo usa un generador propio sembrado con (semilla de la corrida, paso): no toca el RNG global y se
    reanuda igual después de un corte. Los HN ajenos casi idénticos al oro de la fila (coseno crudo > dup_threshold)
    quedan fuera, como los oros duplicados in-batch."""
    name = "infonce"
    uses_hn = True
    has_scale = True
    DENOMINATORS = ("hn_only", "hn_plus_inbatch", "hn_plus_inbatch_k", "hn_plus_otherhn", "hn_plus_otherhn_k",
                    "hn_shared_inbatch")

    def __init__(self, heads, p, total_steps=1, use_hn=False):
        super().__init__(heads, p, total_steps, use_hn)
        self.hn_denominator = self.p.get("hn_denominator", "hn_only")
        if self.hn_denominator not in self.DENOMINATORS:
            raise ValueError(f"hn_denominator desconocido {self.hn_denominator!r}; opciones: {self.DENOMINATORS}")
        self.null = bool(self.p.get("null_candidate", False))
        if self.null:
            d = heads.cfg["projection_dim"]
            self.null_vec = nn.Parameter(torch.randn(d) / math.sqrt(d))
            self.p_null = float(self.p.get("p_null", 0.2))
        if self.hn_denominator in ("hn_plus_inbatch_k", "hn_plus_otherhn", "hn_plus_otherhn_k", "hn_shared_inbatch"):
            self.extra_k = int(self.p.get("extra_k", 31))
            self.extra_dup = float(self.p.get("dup_threshold", 0.999))
            # semilla de la corrida: el Trainer llama a seed_everything(cfg.seed) antes de construir la pérdida
            self.extra_seed = int(self.p.get("extra_seed", torch.initial_seed()))

    # ------------------------------------------------------------------------------
    def _gen(self, step: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed((self.extra_seed * 1_000_003 + int(step)) % (2 ** 63 - 1))

    @staticmethod
    def _pick(valid: torch.Tensor, k: int, g: torch.Generator) -> torch.Tensor:
        """Máscara (mismo tamaño que `valid`) con hasta k columnas válidas por fila, elegidas al azar."""
        r = torch.rand(valid.shape, generator=g).to(valid.device)
        r = r.masked_fill(~valid, -1.0)
        k = int(min(k, valid.shape[1]))
        idx = r.topk(k, dim=1).indices
        keep = torch.zeros_like(valid)
        keep.scatter_(1, idx, True)
        return keep & valid

    def _extra_candidates(self, ctx, s, n, inb_logits, scale) -> tuple:
        """Columnas extra del denominador estado→acción para los denominadores de v0.3.8 (B x C) y el número medio
        de candidatos extra válidos por fila."""
        B, K = n.shape[0], n.shape[1]
        dev = s.device
        eye = torch.eye(B, dtype=torch.bool, device=dev)
        g = self._gen(ctx.step)
        if self.hn_denominator == "hn_plus_inbatch_k":
            valid = ~eye if ctx.dup_a is None else ~(eye | ctx.dup_a)
            keep = self._pick(valid, self.extra_k, g)
            return inb_logits.masked_fill(~keep, NEG), keep.sum(1).float().mean()
        raw_a = F.normalize(ctx.ea.float(), dim=-1)
        raw_n = F.normalize(ctx.en.float(), dim=-1)
        if self.hn_denominator in ("hn_plus_otherhn", "hn_plus_otherhn_k"):
            # un HN válido al azar de cada estado del batch
            r = torch.rand((B, K), generator=g).to(dev).masked_fill(~ctx.nmask, -1.0)
            j = r.argmax(1)
            has = ctx.nmask.any(1)
            rows = torch.arange(B, device=dev)
            n1 = n[rows, j]                                              # B x d
            logits = scale * s @ n1.t()                                  # fila i: estado i contra el HN elegido de j
            dup = (raw_a @ raw_n[rows, j].t()) > self.extra_dup
            valid = ~eye & has[None, :] & ~dup
            if self.hn_denominator == "hn_plus_otherhn_k":
                valid = self._pick(valid, self.extra_k, g)
            return logits.masked_fill(~valid, NEG), valid.sum(1).float().mean()
        # hn_shared_inbatch: oros de los otros estados + todos los HN del batch (los propios ya están en `hn`)
        logits = scale * torch.einsum("bd,jkd->bjk", s, n)               # B x B x K
        dup = torch.einsum("bh,jkh->bjk", raw_a, raw_n) > self.extra_dup
        own = eye[:, :, None].expand(B, B, K)
        valid = ctx.nmask[None, :, :].expand(B, B, K) & ~own & ~dup       # los propios van en `hn`, una sola vez
        hn_all = logits.masked_fill(~valid, NEG).reshape(B, B * K)
        inb_valid = ~eye if ctx.dup_a is None else ~(eye | ctx.dup_a)
        cols = torch.cat([inb_logits.masked_fill(~inb_valid, NEG), hn_all], 1)
        return cols, (inb_valid.sum(1) + valid.reshape(B, -1).sum(1)).float().mean()

    def base_logits(self, ctx, heads):
        s, a = l2n(ctx.zs), l2n(ctx.za)
        return s, a, heads.scale() * s @ a.t()

    def forward(self, ctx, heads):
        s, a, logits = self.base_logits(ctx, heads)
        B = logits.shape[0]
        tgt = torch.arange(B, device=logits.device)
        l_a2s = _ce_masked(logits.t(), ctx.dup_s, tgt)
        logs = {}
        if self.use_hn and ctx.zn is not None:
            n = l2n(ctx.zn)
            scale = heads.scale()
            hn = _hn_logits(s, a, n, ctx.nmask, scale)            # B x (1+K)
            if self.hn_denominator == "hn_plus_inbatch":
                inb = logits.masked_fill(ctx.dup_a, NEG) if ctx.dup_a is not None else logits
                diag = torch.eye(B, dtype=torch.bool, device=logits.device)
                full = torch.cat([hn[:, :1], inb.masked_fill(diag, NEG), hn[:, 1:]], 1)
                cand, t = full, torch.zeros(B, dtype=torch.long, device=logits.device)
            elif self.hn_denominator == "hn_only":
                cand, t = hn, torch.zeros(B, dtype=torch.long, device=logits.device)
            else:                                                   # v0.3.8: condición de borde de B5
                extra, n_extra = self._extra_candidates(ctx, s, n, logits, scale)
                cand = torch.cat([hn, extra], 1)
                t = torch.zeros(B, dtype=torch.long, device=logits.device)
                logs["n_extra"] = _f(n_extra)
            if self.null:
                nz = l2n(self.null_vec)
                cand = torch.cat([cand, (scale * s @ nz)[:, None]], 1)
                drop = torch.rand(B, device=cand.device) < self.p_null
                col0 = torch.zeros_like(cand, dtype=torch.bool)
                col0[:, 0] = True
                cand = cand.masked_fill(col0 & drop[:, None], NEG)
                t = torch.where(drop, torch.full_like(t, cand.shape[1] - 1), t)
                logs["null_frac"] = _f(drop.float().mean())
            l_s2a = F.cross_entropy(cand, t)
        else:
            l_s2a = _ce_masked(logits, ctx.dup_a, tgt)
        loss = 0.5 * (l_s2a + l_a2s)
        logs.update(l_s2a=_f(l_s2a), l_a2s=_f(l_a2s), scale=_f(heads.scale()))
        return loss, logs


class SigLIP(Loss):
    """(3) y (4): pérdida sigmoide por pares (Zhai et al., 2023) con t y b aprendidos."""
    name = "siglip"
    uses_hn = True

    def __init__(self, heads, p, total_steps=1, use_hn=False):
        super().__init__(heads, p, total_steps, use_hn)
        self.t_prime = nn.Parameter(torch.tensor(math.log(float(self.p.get("t_init", 10.0)))))
        self.b = nn.Parameter(torch.tensor(float(self.p.get("b_init", -10.0))))

    def t(self):
        return self.t_prime.exp()

    def forward(self, ctx, heads):
        s, a = l2n(ctx.zs), l2n(ctx.za)
        B = s.shape[0]
        logits = self.t() * s @ a.t() + self.b
        eye = torch.eye(B, device=s.device, dtype=torch.bool)
        y = torch.where(eye, 1.0, -1.0)
        valid = torch.ones_like(logits, dtype=torch.bool)
        if ctx.dup_a is not None:
            valid &= ~ctx.dup_a
        if ctx.dup_s is not None:
            valid &= ~ctx.dup_s
        valid |= eye
        ll = F.logsigmoid(y * logits) * valid
        loss = -ll.sum() / B
        if self.use_hn and ctx.zn is not None:
            n = l2n(ctx.zn)
            nl = self.t() * torch.einsum("bd,bkd->bk", s, n) + self.b
            loss = loss - (F.logsigmoid(-nl) * ctx.nmask).sum() / B
        return loss, {"t": _f(self.t()), "b": _f(self.b)}

    def native_scale(self, heads):
        return _f(self.t())

    def pointwise(self, cos):
        return torch.sigmoid(self.t().detach() * cos + self.b.detach())


class Triplet(Loss):
    """(5): margen sobre distancia coseno. s→a: negativo in-batch semi-hard + todos los HN válidos;
    a→s: in-batch semi-hard. Si no hay semi-hard se usa el negativo MÁS LEJANO, como
    `tfa.losses.TripletSemiHardLoss` (usar el más difícil favorece el colapso; FaceNet)."""
    name = "triplet"
    uses_hn = True

    def _dir(self, d, dup, margin):
        B = d.shape[0]
        eye = torch.eye(B, dtype=torch.bool, device=d.device)
        pos = d.diagonal()
        bad = eye if dup is None else (eye | dup)
        dm = d.masked_fill(bad, float("inf"))
        semi = torch.where(dm > pos[:, None], dm, torch.full_like(dm, float("inf"))).min(1).values
        easiest = d.masked_fill(bad, -float("inf")).max(1).values
        neg = torch.where(torch.isfinite(semi), semi, easiest)
        return F.relu(margin + pos - neg), pos

    def forward(self, ctx, heads):
        m = float(self.p.get("margin", 0.2))
        s, a = l2n(ctx.zs), l2n(ctx.za)
        d = 1.0 - s @ a.t()
        l1, pos = self._dir(d, ctx.dup_a, m)
        l2, _ = self._dir(d.t(), ctx.dup_s, m)
        loss_s2a = l1
        if self.use_hn and ctx.zn is not None:
            dn = 1.0 - torch.einsum("bd,bkd->bk", s, l2n(ctx.zn))
            h = F.relu(m + pos[:, None] - dn) * ctx.nmask
            loss_s2a = 0.5 * (l1 + h.sum(1) / ctx.nmask.sum(1).clamp(min=1))
        loss = 0.5 * (loss_s2a.mean() + l2.mean())
        return loss, {"margin": m}


def _mlp(inp, hid, out, bn=True, last_bn=False):
    layers = [nn.Linear(inp, hid)]
    if bn:
        layers.append(nn.BatchNorm1d(hid))
    layers += [nn.ReLU(inplace=True), nn.Linear(hid, out)]
    if last_bn:
        layers.append(nn.BatchNorm1d(out, affine=False))
    return nn.Sequential(*layers)


class SimSiamX(Loss):
    """(6): SimSiam cruzado: p_s(zs) predice sg(za) y p_a(za) predice sg(zs)."""
    name = "simsiam_x"

    def __init__(self, heads, p, total_steps=1, use_hn=False):
        super().__init__(heads, p, total_steps, use_hn)
        d = heads.cfg["projection_dim"]
        h = int(self.p.get("pred_hidden", max(16, d // 4)))
        self.p_s, self.p_a = _mlp(d, h, d), _mlp(d, h, d)

    def forward(self, ctx, heads):
        ls = -F.cosine_similarity(self.p_s(ctx.zs), ctx.za.detach(), dim=-1).mean()
        la = -F.cosine_similarity(self.p_a(ctx.za), ctx.zs.detach(), dim=-1).mean()
        return 0.5 * (ls + la), {}

    @torch.no_grad()
    def alt_score(self, zs, zc):
        """Puntuación secundaria declarada: cos(p_s(zs), zc)."""
        return torch.einsum("bd,bkd->bk", l2n(self.p_s(zs)), l2n(zc))


class BYOLX(Loss):
    """(7): BYOL cruzado: targets EMA por torre; predictores q_s, q_a."""
    name = "byol_x"

    def __init__(self, heads, p, total_steps=1, use_hn=False):
        super().__init__(heads, p, total_steps, use_hn)
        d = heads.cfg["projection_dim"]
        h = int(self.p.get("pred_hidden", 2048))
        self.q_s, self.q_a = _mlp(d, h, d), _mlp(d, h, d)
        self.t_state = copy.deepcopy(heads.state_head).requires_grad_(False)
        self.t_action = copy.deepcopy(heads.action_head).requires_grad_(False)
        self.tau_base = float(self.p.get("tau_base", 0.996))

    def forward(self, ctx, heads):
        with torch.no_grad():
            ts, ta = l2n(self.t_state(ctx.es)), l2n(self.t_action(ctx.ea))
        ls = (2 - 2 * (l2n(self.q_s(ctx.zs)) * ta).sum(-1)).mean()
        la = (2 - 2 * (l2n(self.q_a(ctx.za)) * ts).sum(-1)).mean()
        return 0.5 * (ls + la), {}

    def after_step(self, heads, step):
        tau = cosine_momentum(step, self.total_steps, self.tau_base)
        EMA.update(self.t_state, heads.state_head, tau)
        EMA.update(self.t_action, heads.action_head, tau)

    @torch.no_grad()
    def alt_score(self, zs, zc):
        """Puntuación secundaria declarada: cos(q_s(zs), zc)."""
        return torch.einsum("bd,bkd->bk", l2n(self.q_s(zs)), l2n(zc))


def _expander(d, D):
    return nn.Sequential(nn.Linear(d, D), nn.BatchNorm1d(D), nn.ReLU(inplace=True),
                         nn.Linear(D, D), nn.BatchNorm1d(D), nn.ReLU(inplace=True), nn.Linear(D, D))


def _off_diag(m):
    n = m.shape[0]
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowX(Loss):
    """(8): Barlow Twins entre las dos torres, con expander compartido (sólo entrenamiento)."""
    name = "barlow_x"

    def __init__(self, heads, p, total_steps=1, use_hn=False):
        super().__init__(heads, p, total_steps, use_hn)
        d = heads.cfg["projection_dim"]
        self.D = int(self.p.get("exp_dim", 4096))
        self.E = _expander(d, self.D)
        self.bn = nn.BatchNorm1d(self.D, affine=False)
        self.lambd = float(self.p.get("lambd", 5e-3))

    def forward(self, ctx, heads):
        B = ctx.zs.shape[0]
        xs, xa = self.bn(self.E(ctx.zs)), self.bn(self.E(ctx.za))
        c = xs.t() @ xa / B
        on = (torch.diagonal(c) - 1).pow(2).sum()
        off = _off_diag(c).pow(2).sum()
        return on + self.lambd * off, {"on": _f(on), "off": _f(off)}


class VICRegX(Loss):
    """(9): VICReg entre torres (coeficientes 25/25/1 por defecto, como el código oficial)."""
    name = "vicreg_x"

    def __init__(self, heads, p, total_steps=1, use_hn=False):
        super().__init__(heads, p, total_steps, use_hn)
        d = heads.cfg["projection_dim"]
        self.D = int(self.p.get("exp_dim", 4096))
        self.E = _expander(d, self.D)
        self.sim, self.std, self.cov = (float(self.p.get(k, v)) for k, v in
                                        (("sim", 25.0), ("std", 25.0), ("cov", 1.0)))

    def forward(self, ctx, heads):
        xs, xa = self.E(ctx.zs), self.E(ctx.za)
        B = xs.shape[0]
        inv = F.mse_loss(xs, xa)
        xs, xa = xs - xs.mean(0), xa - xa.mean(0)
        std_s, std_a = torch.sqrt(xs.var(0) + 1e-4), torch.sqrt(xa.var(0) + 1e-4)
        var = F.relu(1 - std_s).mean() / 2 + F.relu(1 - std_a).mean() / 2
        cs, ca = (xs.t() @ xs) / (B - 1), (xa.t() @ xa) / (B - 1)
        cov = _off_diag(cs).pow(2).sum() / self.D + _off_diag(ca).pow(2).sum() / self.D
        loss = self.sim * inv + self.std * var + self.cov * cov
        return loss, {"inv": _f(inv), "var": _f(var), "cov": _f(cov)}


def sigreg(z: torch.Tensor, step: int, n_slices: int = 256, t_max: float = 3.0, n_points: int = 17,
           scale_by_batch: bool = True, seed: int = 0) -> torch.Tensor:
    """SIGReg (LeJEPA): test de Epps–Pulley sobre proyecciones 1-D aleatorias contra N(0,1).
    Direcciones nuevas en cada paso (sincronizadas por `step`). Igual que `lejepa.univariate.EppsPulley`
    (verificado en el código oficial el 2026-09-30): malla en [0, t_max], pesos trapezoidales
    multiplicados por 2 (el integrando es par, así se integra en [−t_max, t_max]), ponderación
    φ(t) = e^{−t²/2} y estadístico multiplicado por el tamaño de batch (por eso λ depende del batch)."""
    B, D = z.shape
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(step))
    A = torch.randn(D, n_slices, generator=g).to(z.device, z.dtype)
    A = A / A.norm(dim=0, keepdim=True)
    h = z @ A                                               # B x S
    t = torch.linspace(0.0, t_max, n_points, device=z.device, dtype=z.dtype)
    th = h.unsqueeze(-1) * t                                # B x S x T
    cos_m, sin_m = th.cos().mean(0), th.sin().mean(0)       # S x T
    phi = torch.exp(-0.5 * t ** 2)
    err = (cos_m - phi) ** 2 + sin_m ** 2
    stat = 2.0 * torch.trapezoid(err * phi, t, dim=-1)      # S  (= ∫_{−t_max}^{t_max})
    if scale_by_batch:
        stat = stat * B
    return stat.mean()


class MSESIGReg(Loss):
    """(10): alineación MSE entre torres + SIGReg por torre (tipo LeJEPA / NOVA), sin negativos.
    El término de predicción es el MSE promediado sobre dimensiones (convención de LeJEPA según
    el revisor; [PENDING VERIFICATION] contra el código oficial), así λ = 0.05 es comparable."""
    name = "mse_sigreg"

    def forward(self, ctx, heads):
        lam = float(self.p.get("lam", 0.05))
        ns = int(self.p.get("n_slices", 256))
        sbb = bool(self.p.get("scale_by_batch", True))
        pred = (ctx.zs - ctx.za).pow(2).mean()
        sig = 0.5 * (sigreg(ctx.zs, ctx.step, ns, scale_by_batch=sbb, seed=1)
                     + sigreg(ctx.za, ctx.step, ns, scale_by_batch=sbb, seed=2))
        return (1 - lam) * pred + lam * sig, {"pred": _f(pred), "sigreg": _f(sig)}


class InfoNCESIGReg(InfoNCE):
    """(11): InfoNCE (sobre z normalizado) + λ·SIGReg (sobre z sin normalizar)."""
    name = "infonce_sigreg"

    def forward(self, ctx, heads):
        base, logs = super().forward(ctx, heads)
        lam = float(self.p.get("lam", 0.01))
        ns = int(self.p.get("n_slices", 256))
        sbb = bool(self.p.get("scale_by_batch", True))
        sig = 0.5 * (sigreg(ctx.zs, ctx.step, ns, scale_by_batch=sbb, seed=1)
                     + sigreg(ctx.za, ctx.step, ns, scale_by_batch=sbb, seed=2))
        logs["sigreg"] = _f(sig)
        return base + lam * sig, logs


class AlignUnif(Loss):
    """(12): alineación + uniformidad (Wang & Isola, 2020; DirectAU) por torre."""
    name = "align_unif"

    @staticmethod
    def _unif(x, t):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self, ctx, heads):
        s, a = l2n(ctx.zs), l2n(ctx.za)
        gamma, t = float(self.p.get("gamma", 1.0)), float(self.p.get("t", 2.0))
        align = (s - a).pow(2).sum(-1).mean()
        unif = 0.5 * (self._unif(s, t) + self._unif(a, t))
        return align + gamma * unif, {"align": _f(align), "unif": _f(unif)}


class InfoNCELSE(InfoNCE):
    """(13): InfoNCE + λ·(logsumexp de cada fila/columna)^2 (regularizador de contrastive RL).
    Control simple para H4 y candidato a fijar la escala absoluta de los logits (H2)."""
    name = "infonce_lse"

    def forward(self, ctx, heads):
        base, logs = super().forward(ctx, heads)
        lam = float(self.p.get("lam", 0.01))
        s, a, logits = self.base_logits(ctx, heads)
        rows = logits.masked_fill(ctx.dup_a, NEG) if ctx.dup_a is not None else logits
        cols = logits.t().masked_fill(ctx.dup_s, NEG) if ctx.dup_s is not None else logits.t()
        reg = 0.5 * (torch.logsumexp(rows, 1).pow(2).mean() + torch.logsumexp(cols, 1).pow(2).mean())
        if self.use_hn and ctx.zn is not None:
            hn = _hn_logits(s, a, l2n(ctx.zn), ctx.nmask, heads.scale())
            reg = 0.5 * (reg + torch.logsumexp(hn, 1).pow(2).mean())
        logs["lse_reg"] = _f(reg)
        return base + lam * reg, logs


class _ProtoHead(nn.Module):
    """Cabezal de prototipos de DINO: MLP -> bottleneck L2 -> capa lineal con pesos normalizados."""

    def __init__(self, d, hidden, bottleneck, K):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.Parameter(torch.randn(K, bottleneck) / math.sqrt(bottleneck))

    def forward(self, z):
        return F.linear(l2n(self.mlp(z)), F.normalize(self.last, dim=1))


class DINOX(Loss):
    """(14): DINO cruzado: el par estado–acción hace de las dos vistas; prototipos compartidos;
    maestro EMA; centering + sharpening."""
    name = "dino_x"

    def __init__(self, heads, p, total_steps=1, use_hn=False):
        super().__init__(heads, p, total_steps, use_hn)
        d = heads.cfg["projection_dim"]
        K = int(self.p.get("K", 4096))
        self.h = _ProtoHead(d, int(self.p.get("head_hidden", 2048)), int(self.p.get("bottleneck", 256)), K)
        self.t_state = copy.deepcopy(heads.state_head).requires_grad_(False)
        self.t_action = copy.deepcopy(heads.action_head).requires_grad_(False)
        self.t_h = copy.deepcopy(self.h).requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, K))
        self.tau_s = float(self.p.get("tau_s", 0.1))
        self.m_base = float(self.p.get("m_base", 0.996))
        self.center_m = float(self.p.get("center_m", 0.9))

    def tau_t(self, step):
        t0, t1 = float(self.p.get("tau_t", 0.04)), float(self.p.get("tau_t_final", 0.07))
        warm = max(1, int(float(self.p.get("tau_t_warm_frac", 0.3)) * self.total_steps))
        return t0 + (t1 - t0) * min(1.0, step / warm)

    def forward(self, ctx, heads):
        ls, la = self.h(ctx.zs) / self.tau_s, self.h(ctx.za) / self.tau_s
        with torch.no_grad():
            tls = self.t_h(self.t_state(ctx.es))
            tla = self.t_h(self.t_action(ctx.ea))
            tt = self.tau_t(ctx.step)
            ps = F.softmax((tls - self.center) / tt, -1)
            pa = F.softmax((tla - self.center) / tt, -1)
            batch_center = torch.cat([tls, tla]).mean(0, keepdim=True)
        loss = 0.5 * (-(pa * F.log_softmax(ls, -1)).sum(-1).mean() - (ps * F.log_softmax(la, -1)).sum(-1).mean())
        if self.training:
            with torch.no_grad():
                self.center.mul_(self.center_m).add_(batch_center, alpha=1 - self.center_m)
        ent = float(-(pa * (pa + 1e-12).log()).sum(-1).mean())
        return loss, {"tau_t": tt, "teacher_entropy": ent}

    def after_step(self, heads, step):
        m = cosine_momentum(step, self.total_steps, self.m_base)
        EMA.update(self.t_state, heads.state_head, m)
        EMA.update(self.t_action, heads.action_head, m)
        EMA.update(self.t_h, self.h, m)

    @torch.no_grad()
    def alt_score(self, zs, zc):
        """Puntuación secundaria declarada: probabilidad de compartir prototipo."""
        b, C, d = zc.shape
        ps = F.softmax(self.h(zs) / self.tau_s, -1)
        pc = F.softmax(self.h(zc.reshape(b * C, d)) / self.tau_s, -1).view(b, C, -1)
        return torch.einsum("bk,bck->bc", ps, pc)


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """Sinkhorn–Knopp de SwAV: asignaciones equipartidas (B x K)."""
    Q = torch.exp((scores - scores.max()) / eps).t()   # K x B
    Q /= Q.sum()
    K, B = Q.shape
    for _ in range(iters):
        Q /= Q.sum(1, keepdim=True)
        Q /= K
        Q /= Q.sum(0, keepdim=True)
        Q /= B
    return (Q * B).t()


class SwAVX(Loss):
    """(15): SwAV cruzado: códigos de Sinkhorn de una torre se predicen desde la otra."""
    name = "swav_x"

    def __init__(self, heads, p, total_steps=1, use_hn=False):
        super().__init__(heads, p, total_steps, use_hn)
        d = heads.cfg["projection_dim"]
        self.K = int(self.p.get("K", 3000))
        self.protos = nn.Parameter(torch.randn(self.K, d) / math.sqrt(d))
        self.eps, self.iters = float(self.p.get("eps", 0.05)), int(self.p.get("iters", 3))
        self.tau = float(self.p.get("tau", 0.1))
        self.freeze_steps = int(self.p.get("freeze_steps", 100))

    def forward(self, ctx, heads):
        C = F.normalize(self.protos, dim=1)
        s, a = l2n(ctx.zs) @ C.t(), l2n(ctx.za) @ C.t()
        qs = sinkhorn(s.detach(), self.eps, self.iters)
        qa = sinkhorn(a.detach(), self.eps, self.iters)
        loss = 0.5 * (-(qa * F.log_softmax(s / self.tau, -1)).sum(-1).mean()
                      - (qs * F.log_softmax(a / self.tau, -1)).sum(-1).mean())
        return loss, {"code_entropy": float(-(qa * (qa + 1e-12).log()).sum(-1).mean())}

    def before_optimizer_step(self, step):
        if step < self.freeze_steps and self.protos.grad is not None:
            self.protos.grad = None

    @torch.no_grad()
    def alt_score(self, zs, zc):
        """Puntuación secundaria declarada: probabilidad de compartir prototipo."""
        C = F.normalize(self.protos, dim=1)
        ps = F.softmax(l2n(zs) @ C.t() / self.tau, -1)
        pc = F.softmax(l2n(zc) @ C.t() / self.tau, -1)
        return torch.einsum("bk,bck->bc", ps, pc)


# --------------------------------------------------------------------------------------
class AuxHN(nn.Module):
    """Término auxiliar estándar de HN (régimen R2): InfoNCE sobre [positivo, HN] con la escala
    de los cabezales. Se suma a pérdidas que no usan negativos de forma nativa."""

    def forward(self, ctx: Ctx, heads: Heads) -> torch.Tensor:
        s, a = l2n(ctx.zs), l2n(ctx.za)
        hn = _hn_logits(s, a, l2n(ctx.zn), ctx.nmask, heads.scale())
        return F.cross_entropy(hn, torch.zeros(len(s), dtype=torch.long, device=s.device))


class BlindScorer(nn.Module):
    """Modelo ciego (F0.6): puntúa acciones SIN ver el estado. Si acierta muy por encima del
    azar en el test con HN, los negativos tienen artefactos explotables."""

    def __init__(self, hidden: int, width: int = 512, depth: int = 2):
        super().__init__()
        from .heads import make_head
        self.f = make_head(width, depth, proj=1, activation="gelu", layernorm=False, hidden=hidden)

    def forward(self, cands: torch.Tensor) -> torch.Tensor:   # (B x C x H) -> (B x C)
        B, C, H = cands.shape
        return self.f(cands.reshape(B * C, H)).view(B, C)


# --------------------------------------------------------------------------------------
ARMS: dict[str, dict] = {
    "infonce":        dict(num=1, cls=InfoNCE, family="contrastiva", hn=False, defaults={}),
    "infonce_hn":     dict(num=2, cls=InfoNCE, family="contrastiva", hn=True,
                           defaults={"hn_denominator": "hn_only"}),
    "siglip":         dict(num=3, cls=SigLIP, family="sigmoide", hn=False,
                           defaults={"t_init": 10.0, "b_init": -10.0}),
    "siglip_hn":      dict(num=4, cls=SigLIP, family="sigmoide", hn=True,
                           defaults={"t_init": 10.0, "b_init": -10.0}),
    "triplet_hn":     dict(num=5, cls=Triplet, family="margen", hn=True, defaults={"margin": 0.2}),
    "simsiam_x":      dict(num=6, cls=SimSiamX, family="predictiva", hn=False, defaults={}),
    "byol_x":         dict(num=7, cls=BYOLX, family="predictiva", hn=False,
                           defaults={"pred_hidden": 2048, "tau_base": 0.996}),
    "barlow_x":       dict(num=8, cls=BarlowX, family="redundancia", hn=False,
                           defaults={"exp_dim": 4096, "lambd": 5e-3}),
    "vicreg_x":       dict(num=9, cls=VICRegX, family="redundancia", hn=False,
                           defaults={"exp_dim": 4096, "sim": 25.0, "std": 25.0, "cov": 1.0}),
    "mse_sigreg":     dict(num=10, cls=MSESIGReg, family="distribucion", hn=False,
                           defaults={"lam": 0.05, "n_slices": 256}),
    "infonce_sigreg": dict(num=11, cls=InfoNCESIGReg, family="distribucion", hn=False,
                           defaults={"lam": 0.01, "n_slices": 256}),
    "align_unif":     dict(num=12, cls=AlignUnif, family="puente", hn=False,
                           defaults={"gamma": 1.0, "t": 2.0}),
    "infonce_lse":    dict(num=13, cls=InfoNCELSE, family="control", hn=False, defaults={"lam": 0.01}),
    "dino_x":         dict(num=14, cls=DINOX, family="prototipos", hn=False,
                           defaults={"K": 4096, "tau_s": 0.1, "tau_t": 0.04, "tau_t_final": 0.07,
                                     "center_m": 0.9, "m_base": 0.996}),
    "swav_x":         dict(num=15, cls=SwAVX, family="prototipos", hn=False,
                           defaults={"K": 3000, "eps": 0.05, "tau": 0.1, "freeze_steps": 100}),
}

NONCONTRASTIVE_POOL = ["simsiam_x", "byol_x", "barlow_x", "vicreg_x", "mse_sigreg", "dino_x", "swav_x"]


def arm_names() -> list[str]:
    return sorted(ARMS, key=lambda k: ARMS[k]["num"])


def resolve_hn_mode(arm: str, hn_mode: str) -> str:
    spec = ARMS[arm]
    if hn_mode == "auto":
        return "native" if spec["hn"] else "none"
    if hn_mode == "native" and not spec["cls"].uses_hn:
        raise ValueError(f"{arm} no admite HN nativos")
    return hn_mode


def build_loss(arm: str, heads: Heads, params: dict | None, total_steps: int, hn_mode: str) -> Loss:
    spec = ARMS[arm]
    p = {**spec["defaults"], **(params or {})}
    return spec["cls"](heads, p, total_steps, use_hn=(hn_mode == "native"))
