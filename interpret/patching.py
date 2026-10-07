# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Nedensel Müdahale: Yön/Özellik Silme ve Aktivasyon Yamalama

Sondalar bir bilginin aktivasyonlardan OKUNABİLDİĞİNİ gösterir; modelin o
bilgiyi KULLANDIĞINI göstermez. Nedensel kanıt için temsile müdahale edip
çıktının nasıl değiştiğine bakılır:

- ``ablate_feature``: bir katmanın çıktısından bir yönü (ör. sonda ağırlığı,
  ortalama farkı) izdüşümle silmek ya da bir SAE latentinin katkısını çıkarmak.
- ``patch_activations``: "temiz" bir çalıştırmanın aktivasyonunu "bozuk" bir
  çalıştırmaya yamamak (activation patching / causal tracing).
- ``next_token_logprob``: bir token kümesine (ör. ince ünlülü ekler) verilen
  toplam log-olasılığı ölçmek; müdahale öncesi/sonrası farkı nedensel etkidir.

Hepsi bağlam yöneticisidir; çıkışta kancalar kaldırılır.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def _site_module(model, layer: int, site: str):
    block = model.blocks[layer]
    if site == "resid_post":
        return block
    if site == "attn_out":
        return block.attn
    if site == "ffn_out":
        return block.ffn
    raise ValueError(f"Desteklenmeyen müdahale noktası: {site}")


def _replace_first(output, fn):
    if isinstance(output, tuple):
        return (fn(output[0]),) + tuple(output[1:])
    if isinstance(output, list):
        return [fn(output[0])] + list(output[1:])
    return fn(output)


@contextmanager
def edit_activation(model, layer: int, fn, site: str = "resid_post"):
    """Genel müdahale: ``fn(tensor) -> tensor`` katman çıktısına uygulanır."""
    module = _site_module(model, layer, site)
    handle = module.register_forward_hook(lambda m, a, out: _replace_first(out, fn))
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_feature(
    model,
    layer: int,
    direction: Optional[torch.Tensor] = None,
    sae=None,
    feature: Optional[int] = None,
    site: str = "resid_post",
    scale: float = 0.0,
):
    """
    Bir yönü veya SAE latentini katman çıktısından sil (ya da ölçekle).

    Kullanım (ikisinden biri):
        direction: (d_model,) yön — çıktının bu yöndeki bileşeni ``scale`` ile
            çarpılır (0 = tamamen sil, 2 = iki katına çıkar)
        sae + feature: TopKSAE ve latent indeksi — latentin yeniden kurulumdaki
            katkısı (z_f · W_dec[:, f]) ``(1 - scale)`` oranında çıkarılır

    Örnek:
        >>> with ablate_feature(model, 3, direction=probe_dir):
        ...     logits, _, _ = model(ids)
    """
    if direction is not None:
        u = direction.detach().float()
        u = u / u.norm().clamp_min(1e-8)

        def fn(x):
            uu = u.to(x.device, x.dtype)
            coef = x @ uu
            return x + (scale - 1.0) * coef.unsqueeze(-1) * uu
    elif sae is not None and feature is not None:
        def fn(x):
            flat = x.reshape(-1, x.size(-1))
            z = sae.encode(flat)[:, feature]
            col = sae.W_dec[:, feature].detach() * sae.input_scale
            delta = (1.0 - scale) * z.unsqueeze(-1) * col
            return (flat.float() - delta).to(x.dtype).view_as(x)
    else:
        raise ValueError("direction ya da (sae, feature) verilmeli")
    with edit_activation(model, layer, fn, site=site):
        yield


@contextmanager
def patch_activations(
    model,
    layer: int,
    source: torch.Tensor,
    positions: Optional[Sequence[int]] = None,
    site: str = "resid_post",
):
    """
    Aktivasyon yamalama: katman çıktısının ``positions`` konumlarını
    ``source`` (B, T, d) tensöründeki değerlerle değiştir.

    ``source`` tipik olarak ``ActivationRecorder`` ile başka (temiz) bir
    girdiden kaydedilir. ``positions`` None ise tüm konumlar yamalanır.
    """
    def fn(x):
        x = x.clone()
        src = source.to(x.device, x.dtype)
        if positions is None:
            T = min(x.size(1), src.size(1))
            x[:, :T] = src[:, :T]
        else:
            idx = torch.as_tensor(list(positions), dtype=torch.long, device=x.device)
            x[:, idx] = src[:, idx]
        return x

    with edit_activation(model, layer, fn, site=site):
        yield


@torch.no_grad()
def next_token_logprob(model, input_ids: torch.Tensor, token_set: Sequence[int]) -> torch.Tensor:
    """
    Her konumda bir sonraki token'ın ``token_set`` içinde olma log-olasılığı.

    Returns:
        (B, T) — log Σ_{t∈set} p(t | önek)
    """
    logits = model(input_ids)[0].float()
    logp = F.log_softmax(logits, dim=-1)
    idx = torch.as_tensor(list(token_set), dtype=torch.long, device=logp.device)
    return torch.logsumexp(logp.index_select(-1, idx), dim=-1)
