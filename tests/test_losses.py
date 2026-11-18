import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clm_losses.heads import Heads, heads_from_clm_checkpoint
from clm_losses.losses import ARMS, AuxHN, Ctx, build_loss, sigreg, sinkhorn

H, D = 32, 16


def make_ctx(heads, B=24, K=5, with_hn=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    es = F.normalize(torch.randn(B, H, generator=g), dim=-1)
    ea = F.normalize(es @ torch.randn(H, H, generator=g) * 0.3 + torch.randn(B, H, generator=g), dim=-1)
    en = F.normalize(torch.randn(B, K, H, generator=g), dim=-1) if with_hn else None
    nmask = torch.ones(B, K, dtype=torch.bool) if with_hn else None
    if with_hn:
        nmask[0, 3:] = False
    zs, za = heads.zs(es), heads.za(ea)
    zn = heads.za(en.view(B * K, H)).view(B, K, -1) if with_hn else None
    dup = torch.zeros(B, B, dtype=torch.bool)
    dup[1, 2] = dup[2, 1] = True
    return Ctx(es=es, ea=ea, zs=zs, za=za, en=en, zn=zn, nmask=nmask, dup_a=dup, dup_s=dup, step=3)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_loss_forward_backward(arm):
    torch.manual_seed(0)
    heads = Heads(hidden=H, width=24, depth=3, proj=D)
    p = {"exp_dim": 32, "K": 16, "pred_hidden": 8, "n_slices": 8}
    mode = "native" if ARMS[arm]["hn"] else "none"
    loss_mod = build_loss(arm, heads, p, total_steps=10, hn_mode=mode)
    loss_mod.train()
    ctx = make_ctx(heads, with_hn=(mode == "native"))
    loss, logs = loss_mod(ctx, heads)
    assert torch.isfinite(loss), arm
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in heads.state_head.parameters() if p.grad is not None)
    assert g > 0, f"{arm}: sin gradiente en el cabezal de estado"
    loss_mod.after_step(heads, 1)
    loss_mod.before_optimizer_step(1)


def test_aux_hn_and_infonce_null():
    heads = Heads(hidden=H, width=24, depth=3, proj=D)
    ctx = make_ctx(heads)
    aux = AuxHN()(ctx, heads)
    assert torch.isfinite(aux)
    m = build_loss("infonce_hn", heads, {"null_candidate": True, "p_null": 0.5}, 10, "native")
    loss, logs = m(ctx, heads)
    assert torch.isfinite(loss) and "null_frac" in logs


def test_infonce_hn_matches_readme_formula():
    """s→a con HN: -log e^{s·a/τ} / (e^{s·a/τ} + Σ_k e^{s·h_k/τ})."""
    heads = Heads(hidden=H, width=24, depth=3, proj=D)
    ctx = make_ctx(heads, B=6, K=4)
    ctx.nmask[:] = True
    ctx.dup_a = ctx.dup_s = None
    m = build_loss("infonce_hn", heads, {"hn_denominator": "hn_only"}, 10, "native")
    loss, logs = m(ctx, heads)
    s, a, n = F.normalize(ctx.zs, dim=-1), F.normalize(ctx.za, dim=-1), F.normalize(ctx.zn, dim=-1)
    t = heads.scale()
    pos = t * (s * a).sum(-1)
    neg = t * torch.einsum("bd,bkd->bk", s, n)
    l_s2a = (-pos + torch.logsumexp(torch.cat([pos[:, None], neg], 1), 1)).mean()
    assert math.isclose(logs["l_s2a"], float(l_s2a), rel_tol=1e-5)


def test_sigreg_gaussian_vs_collapsed():
    z = torch.randn(512, 16)
    zc = torch.ones(512, 16) * 3
    assert float(sigreg(z, 0, 32)) < float(sigreg(zc, 0, 32)) / 10


def test_sinkhorn_marginals():
    # puntuaciones tipo coseno en [-1, 1], como en el uso real
    s = torch.rand(64, 8) * 2 - 1
    q = sinkhorn(s, eps=0.05, iters=50)
    assert torch.allclose(q.sum(1), torch.ones(64), atol=1e-4)          # cada muestra: distribución
    assert torch.allclose(q.sum(0), torch.full((8,), 64 / 8), rtol=0.1)  # equipartición aproximada


def test_clm_checkpoint_roundtrip():
    heads = Heads(hidden=H, width=24, depth=3, proj=D, layernorm=True)
    ck = heads.clm_checkpoint()
    ck["cfg"].pop("layernorm")          # simula un checkpoint sin la clave
    h2 = heads_from_clm_checkpoint(ck)
    x = torch.randn(4, H)
    assert torch.allclose(heads.zs(x), h2.zs(x), atol=1e-6)
    assert h2.cfg["layernorm"] is True


def test_infonce_equals_log_batch_when_scale_is_tiny():
    """Chequeo de F0.8: con logits ≈ 0, InfoNCE ≈ log B."""
    heads = Heads(hidden=H, width=24, depth=3, proj=D)
    with torch.no_grad():
        heads.logit_scale.fill_(math.log(1e-6))
    B = 32
    ctx = make_ctx(heads, B=B, with_hn=False)
    ctx.dup_a = ctx.dup_s = None
    m = build_loss("infonce", heads, {}, 10, "none")
    loss, _ = m(ctx, heads)
    assert abs(float(loss) - math.log(B)) < 1e-3


def test_rank_metrics_ties_count_as_expected_value():
    from clm_losses.evaluate import rank_metrics
    C = np.array([[0.5] * 11,                       # colapso total: 1/11
                  [0.9] + [0.1] * 10,               # acierto
                  [0.1, 0.9] + [0.0] * 9,           # error
                  [0.7, 0.7] + [0.0] * 9], np.float32)  # empate con 1 negativo: 1/2
    m, correct, _ = rank_metrics(C)
    assert np.allclose(correct, [1 / 11, 1.0, 0.0, 0.5])
    assert math.isclose(m["mrr"], (np.mean(1 / np.arange(1, 12)) + 1 + 0.5 + 0.75) / 4, rel_tol=1e-6)


def test_sigreg_matches_lejepa_eppspulley_weights():
    """Misma cuenta que lejepa.univariate.EppsPulley (pesos 2·dt, dt en los extremos, ×φ, ×N)."""
    torch.manual_seed(0)
    z = torch.randn(256, 8) * 1.3 + 0.2
    t_max, n = 3.0, 17
    g = torch.Generator(device="cpu").manual_seed(0 * 1_000_003 + 5)
    A = torch.randn(8, 16, generator=g)
    A = A / A.norm(dim=0, keepdim=True)
    h = z @ A
    t = torch.linspace(0, t_max, n)
    phi = torch.exp(-0.5 * t ** 2)
    dt = t_max / (n - 1)
    w = torch.full((n,), 2 * dt)
    w[[0, -1]] = dt
    err = ((h[..., None] * t).cos().mean(0) - phi) ** 2 + (h[..., None] * t).sin().mean(0) ** 2
    ref = ((err @ (w * phi)) * z.shape[0]).mean()
    ours = sigreg(z, 5, n_slices=16, t_max=t_max, n_points=n, seed=0)
    assert torch.allclose(ours, ref, rtol=1e-5), (float(ours), float(ref))


def test_triplet_fallback_uses_easiest_negative():
    from clm_losses.losses import Triplet
    heads = Heads(hidden=H, width=24, depth=3, proj=D)
    m = Triplet(heads, {"margin": 0.2})
    d = torch.tensor([[0.9, 0.1, 0.5], [0.2, 0.1, 0.8], [0.3, 0.4, 0.2]])   # fila 0: sin semi-hard
    loss, pos = m._dir(d, None, 0.2)
    assert math.isclose(float(loss[0]), max(0.0, 0.2 + 0.9 - 0.5), rel_tol=1e-6)
