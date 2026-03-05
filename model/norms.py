# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — RMSNorm
Root Mean Square Layer Normalization — bias'sız, daha hızlı normalizasyon.
Modern decoder-only mimarilerde kullanılan standart normalizasyon tekniği.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    LayerNorm'dan farkı:
    - Mean çıkarma yok (re-centering yok)
    - Bias yok
    - Sadece scale (weight) parametresi
    - ~%5-8 daha hızlı

    Referans: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """RMS normalizasyonu uygula."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim) — herhangi bir şekilde, son boyut dim olmalı
        Returns:
            Normalize edilmiş tensor, aynı şekilde
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
