"""Shared display-8 harness (GAUSS8-51/BILAT8-51 common brief).

Ports the measurement functions used to answer one question for an approximate
per-pixel path: does it change what the artist sees? "Sees" is fixed as the ACES
1.x RRT + sRGB ODT (Hill's fitted form) rounded to 8 bits — the same transform any
approximate stdlib builtin is judged against in this wave. Torch-only, no new
dependency.

Two synthetic scene-linear plates (`plate_day`, `plate_night`) stand in for a
comp's own content: one broad-daylight-ish exterior with hard highlights, one
near-black interior with a handful of small, very bright practical lights (the
harder case for a highlight-heavy tone curve). `aces_srgb8` maps a plate through
the fitted transform and rounds to an int16 8-bit code image; `code_diff_stats`
summarizes a pixel-wise code difference between two such images (an approximate
path's output vs. an exact reference), both overall and restricted to the centre
half of the frame (excluding a border margin edge effects can dominate).
"""
from __future__ import annotations

import torch

# -- ACES fitted RRT+ODT sRGB (Hill), input = scene-linear Rec.709 ------------
_ACES_IN = torch.tensor(
    [[0.59719, 0.35458, 0.04823],
     [0.07600, 0.90834, 0.01566],
     [0.02840, 0.13383, 0.83777]], dtype=torch.float64,
)
_ACES_OUT = torch.tensor(
    [[1.60475, -0.53108, -0.07367],
     [-0.10208, 1.10813, -0.00605],
     [-0.00327, -0.07276, 1.07602]], dtype=torch.float64,
)


def _srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=0) ** (1 / 2.4) - 0.055)


def aces_srgb8(bchw: torch.Tensor) -> torch.Tensor:
    """Map a [1, 3, H, W] scene-linear Rec.709 image through the ACES RRT + sRGB
    ODT (Hill's fitted form) and round to 8-bit display codes.

    Returns an int16 [H, W, 3] tensor (int16, not uint8, so a signed difference
    between two such images never wraps)."""
    x = bchw[0].permute(1, 2, 0).double().cpu()  # HWC, float64
    v = x @ _ACES_IN.T
    v = (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081)
    v = (v @ _ACES_OUT.T).clamp(0, 1)
    return torch.round(_srgb_oetf(v) * 255).to(torch.int16)


def _band_noise(H: int, W: int, device: torch.device, g: torch.Generator,
                 scales=(2, 8, 32)) -> torch.Tensor:
    n = torch.zeros(1, 1, H, W, device=device)
    for s in scales:
        z = torch.randn(1, 1, H // s + 2, W // s + 2, device=device, generator=g)
        n += torch.nn.functional.interpolate(z, size=(H, W), mode="bicubic", align_corners=False) / len(scales)
    return n


def plate_day(H: int, W: int, device: torch.device, seed: int = 1) -> torch.Tensor:
    """A broad-daylight-ish exterior plate: a gradient base, a few hard-edged
    rectangles (a shadowed patch, a highlight, a sky window, a saturated red
    accent) and band-limited multiplicative texture noise. Scene-linear
    Rec.709, [1, 3, H, W]."""
    g = torch.Generator(device=device).manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing="ij")
    base = torch.stack([0.18 + 0.12 * torch.sin(6 * xx) * yy, 0.16 + 0.1 * xx, 0.12 + 0.14 * (1 - yy)])[None]
    base = base * torch.exp(0.35 * _band_noise(H, W, device, g))
    rects = [(0.1, 0.1, 0.3, 0.25, (0.02, 0.02, 0.025)), (0.55, 0.05, 0.9, 0.3, (0.6, 0.55, 0.45)),
             (0.2, 0.6, 0.45, 0.95, (0.04, 0.09, 0.03)), (0.6, 0.55, 0.75, 0.7, (4.0, 3.6, 3.0)),  # window
             (0.8, 0.75, 0.84, 0.8, (14.0, 11.0, 7.0)), (0.35, 0.35, 0.5, 0.5, (0.9, 0.2, 0.1))]
    for x0, y0, x1, y1, c in rects:
        sl = (slice(None), slice(None), slice(int(y0 * H), int(y1 * H)), slice(int(x0 * W), int(x1 * W)))
        base[sl] = torch.tensor(c, device=device).view(1, 3, 1, 1) * torch.exp(0.15 * _band_noise(H, W, device, g)[sl])
    return base.contiguous()


def plate_night(H: int, W: int, device: torch.device, seed: int = 2) -> torch.Tensor:
    """A near-black interior plate with a blue-shifted ambient and forty small,
    very bright practical lights scattered at random — the harder case for a
    highlight-compressing tone curve under a large blur. Scene-linear
    Rec.709, [1, 3, H, W]."""
    g = torch.Generator(device=device).manual_seed(seed)
    base = torch.full((1, 3, H, W), 0.012, device=device) * torch.exp(0.6 * _band_noise(H, W, device, g)).expand(1, 3, H, W)
    base[:, 2] *= 1.6
    pos = torch.rand(40, 2, device=device, generator=g)
    for i in range(40):
        y, x = int(pos[i, 0] * (H - 8)), int(pos[i, 1] * (W - 8))
        base[:, :, y:y + 6, x:x + 6] = torch.tensor((20.0, 12.0, 5.0), device=device).view(1, 3, 1, 1)
    return base.contiguous()


def code_diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Summarize a pixel-wise 8-bit-code difference between two `aces_srgb8`
    outputs (worst channel per pixel), both overall and restricted to the
    centre half of the frame (`centre_*`, excluding a border margin where
    boundary handling can dominate). `a`/`b` are [H, W, 3] int16. On frames
    with H or W < 4 the centre half is empty, so the centre stats fall back
    to the full frame (every pixel there is border anyway)."""
    sd = (a - b).double()
    d = sd.abs().amax(dim=-1)
    n = d.numel()
    h, w = d.shape
    c = d[h // 4:3 * h // 4, w // 4:3 * w // 4]
    if c.numel() == 0:
        c = d
    return {
        "changed": float((d >= 1).sum()) / n,
        "ge2": float((d >= 2).sum()) / n,
        "ge4": float((d >= 4).sum()) / n,
        "max": int(d.max()),
        "mean_signed": float(sd.mean()),
        "centre_ge2": float((c >= 2).sum()) / c.numel(),
        "centre_max": int(c.max()),
    }
