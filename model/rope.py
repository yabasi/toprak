# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Rotary Position Embedding (RoPE)
Pozisyon bilgisini attention skorlarına doğrudan enjekte eder.
Learned positional embedding'e göre avantajları:
- Ekstra parametre yok
- Relative position kodlama
- Context uzunluğu dışına extrapolation kabiliyeti

Referans: https://arxiv.org/abs/2104.09864 (RoFormer)
"""

import math
from typing import Optional

import torch


def _yarn_find_correction_dim(num_rotations, dim, base, original_max_seq_len):
    """Belirli sayıda tam dönüş yapan frekans boyutunu bul (YaRN)."""
    return (dim * math.log(original_max_seq_len / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_mscale(factor: float, mscale: float = 1.0) -> float:
    """YaRN attention sıcaklık düzeltmesi: 0.1 * ln(s) + 1."""
    if factor <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(factor) + 1.0


def scaled_inv_freqs(
    dim: int,
    theta: float,
    rope_scaling: Optional[dict],
    device: torch.device = None,
) -> tuple:
    """
    RoPE ters frekanslarını ve attention ölçeğini (mscale) hesapla.

    Desteklenen ölçekleme yöntemleri:
    - linear: pozisyonlar factor'e bölünür (Position Interpolation)
    - ntk:    base frekansı NTK-aware biçimde büyütülür
    - yarn:   yüksek frekanslar korunur, düşük frekanslar interpolasyonla
              uzatılır, aradaki bant rampa ile karıştırılır + mscale

    Returns:
        (inv_freq, mscale)
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    if not rope_scaling:
        return inv_freq, 1.0

    kind = rope_scaling.get("type", "yarn")
    factor = float(rope_scaling.get("factor", 1.0))
    if factor <= 1.0:
        return inv_freq, 1.0

    if kind == "linear":
        return inv_freq / factor, 1.0

    if kind == "ntk":
        new_theta = theta * factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (new_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        return inv_freq, 1.0

    if kind == "yarn":
        original = int(rope_scaling.get("original_max_seq_len", 2048))
        beta_fast = float(rope_scaling.get("beta_fast", 32.0))
        beta_slow = float(rope_scaling.get("beta_slow", 1.0))
        low = math.floor(_yarn_find_correction_dim(beta_fast, dim, theta, original))
        high = math.ceil(_yarn_find_correction_dim(beta_slow, dim, theta, original))
        low, high = max(low, 0), min(high, dim // 2 - 1)
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim // 2, device=device).float() - low) / (high - low)
        ramp = ramp.clamp(0, 1)
        # ramp=0 → yüksek frekans (ekstrapolasyon), ramp=1 → düşük frekans (interpolasyon)
        interpolation = inv_freq / factor
        inv_freq = inv_freq * (1 - ramp) + interpolation * ramp
        mscale = yarn_mscale(factor, float(rope_scaling.get("mscale", 1.0)))
        return inv_freq, mscale

    raise ValueError(f"Bilinmeyen rope_scaling tipi: {kind!r} (linear, ntk, yarn)")


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
    rope_scaling: Optional[dict] = None,
) -> torch.Tensor:
    """
    RoPE frekans tablosunu önceden hesapla.

    Args:
        dim: Head boyutu (head_dim) — çift sayı olmalı
        max_seq_len: Maksimum sequence uzunluğu
        theta: Base frekans (varsayılan: 10000, uzun context için 500000)
        device: Hesaplama cihazı
        rope_scaling: Uzun bağlam ölçeklemesi (bkz. scaled_inv_freqs)

    Returns:
        freqs_cis: (max_seq_len, dim // 2) — complex tensor
    """
    # Frekansları hesapla: theta_i = 1 / (theta^(2i/dim)), gerekirse ölçekle
    freqs, mscale = scaled_inv_freqs(dim, theta, rope_scaling, device)
    # (dim // 2,)

    # Pozisyon indeksleri
    t = torch.arange(max_seq_len, device=device).float()
    # (max_seq_len,)

    # Dış çarpım: her pozisyon × her frekans
    freqs = torch.outer(t, freqs)
    # (max_seq_len, dim // 2)

    # Complex forma dönüştür: e^(i * theta) = cos(theta) + i * sin(theta)
    # YaRN: genliği mscale ile çarp → q·k skorları mscale² ile ölçeklenir
    freqs_cis = torch.polar(torch.full_like(freqs, mscale), freqs)
    # (max_seq_len, dim // 2) — complex64

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis'i x ile broadcast edebilecek şekle getir.

    Args:
        freqs_cis: (seq_len, head_dim // 2)
        x: (batch, num_heads, seq_len, head_dim // 2)

    Returns:
        (1, 1, seq_len, head_dim // 2)
    """
    ndim = x.ndim
    assert ndim >= 2
    # (seq_len, head_dim//2) → (1, 1, seq_len, head_dim//2)
    shape = [1] * (ndim - 2) + list(freqs_cis.shape)
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple:
    """
    Q ve K tensorlerine Rotary Position Embedding uygula.

    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_kv_heads, seq_len, head_dim)
        freqs_cis: (seq_len, head_dim // 2) — complex tensor

    Returns:
        (q_rotated, k_rotated) — aynı boyutlarda
    """
    # Real tensor'ı complex'e dönüştür:
    # (B, H, T, D) → (B, H, T, D//2, 2) → complex (B, H, T, D//2)
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # freqs_cis'i broadcast için reshape et
    freqs_cis = reshape_for_broadcast(freqs_cis, q_complex)

    # Rotary embedding uygula (complex çarpım)
    q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(-2)

    return q_rotated.type_as(q), k_rotated.type_as(k)
