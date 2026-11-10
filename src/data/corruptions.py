from typing import Callable, Dict

import torch
import torch.nn.functional as F

SEVERITIES = (1, 2, 3, 4, 5)


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _separable(img: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    c = img.shape[1]
    img = F.conv2d(F.pad(img, (kx.numel() // 2,) * 2, mode="reflect"), kx.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(img, (0, 0) + (ky.numel() // 2,) * 2, mode="reflect"), ky.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def gaussian_noise(img, s, gen):
    sigma = (0.04, 0.06, 0.08, 0.10, 0.13)[s - 1]
    return img + sigma * torch.randn(img.shape, generator=gen, device="cpu").to(img.device)


def gaussian_blur(img, s, gen):
    k = _gaussian_kernel((1.0, 1.5, 2.0, 3.0, 4.0)[s - 1], img.device)
    return _separable(img, k, k)


def motion_blur(img, s, gen):
    length = (5, 9, 13, 17, 23)[s - 1]
    k = torch.full((length,), 1.0 / length, device=img.device)
    return _separable(img, k, torch.ones(1, device=img.device))


def fog(img, s, gen):
    strength = (0.25, 0.35, 0.45, 0.55, 0.65)[s - 1]
    b, _, h, w = img.shape
    haze = torch.rand((b, 1, max(h // 64, 2), max(w // 64, 2)), generator=gen, device="cpu").to(img.device)
    haze = F.interpolate(haze, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
    alpha = strength * (0.6 + 0.4 * haze)
    return img * (1 - alpha) + 0.8 * alpha


def low_light(img, s, gen):
    gamma, gain = ((1.4, 0.75), (1.8, 0.6), (2.2, 0.5), (2.6, 0.4), (3.0, 0.3))[s - 1]
    dark = gain * img.clamp(0, 1) ** gamma
    return dark + 0.02 * s * torch.randn(img.shape, generator=gen, device="cpu").to(img.device) * dark.sqrt()


def contrast(img, s, gen):
    factor = (0.6, 0.5, 0.4, 0.3, 0.2)[s - 1]
    mean = img.mean(dim=(1, 2, 3), keepdim=True)
    return (img - mean) * factor + mean


CORRUPTIONS: Dict[str, Callable] = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "fog": fog,
    "low_light": low_light,
    "contrast": contrast,
}


def corrupt(images: torch.Tensor, name: str, severity: int, seed: int) -> torch.Tensor:
    """Applies a corruption to a [B, 3, H, W] batch in [0, 1], deterministically for a given seed."""
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; choose from {sorted(CORRUPTIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be 1-5, got {severity}")
    gen = torch.Generator().manual_seed(seed)
    return CORRUPTIONS[name](images.float(), severity, gen).clamp(0.0, 1.0)
