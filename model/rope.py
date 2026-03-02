"""
Toprak — Rotary Position Embedding (RoPE)
Pozisyon bilgisini attention skorlarına doğrudan enjekte eder.
Learned positional embedding'e göre avantajları:
- Ekstra parametre yok
- Relative position kodlama
- Context uzunluğu dışına extrapolation kabiliyeti

Referans: https://arxiv.org/abs/2104.09864 (RoFormer)
"""

import torch


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    RoPE frekans tablosunu önceden hesapla.

    Args:
        dim: Head boyutu (head_dim) — çift sayı olmalı
        max_seq_len: Maksimum sequence uzunluğu
        theta: Base frekans (varsayılan: 10000, uzun context için 500000)
        device: Hesaplama cihazı

    Returns:
        freqs_cis: (max_seq_len, dim // 2) — complex tensor
    """
    # Frekansları hesapla: theta_i = 1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # (dim // 2,)

    # Pozisyon indeksleri
    t = torch.arange(max_seq_len, device=device).float()
    # (max_seq_len,)

    # Dış çarpım: her pozisyon × her frekans
    freqs = torch.outer(t, freqs)
    # (max_seq_len, dim // 2)

    # Complex forma dönüştür: e^(i * theta) = cos(theta) + i * sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
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
