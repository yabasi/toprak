# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — TopK Seyrek Otokodlayıcı (Sparse Autoencoder, SAE)

Bir katmanın artık akışını (residual stream) çok sayıda seyrek, yorumlanabilir
"özelliğe" (latent) ayrıştırır. Gao ve ark. (2024) TopK SAE tarifi:

    z    = TopK( W_enc (x − b_pre) + b_enc )      # yalnız en büyük k latent, ReLU
    x̂   = W_dec z + b_pre

- Kod çözücü (decoder) sütunları birim normda tutulur (her adımdan sonra
  yeniden normlanır; gradyanın sütuna paralel bileşeni atılır).
- b_pre, verinin ortalamasıyla başlatılır; W_enc, W_decᵀ ile başlatılır.
- Girdi, ortalama kare normu d_in olacak biçimde ölçeklenir (ölçek SAE içinde
  saklanır; çağıran ham aktivasyonu verir).
- Ölü latentler (son epokta hiç ateşlenmeyen) izlenir; isteğe bağlı AuxK
  kaybı ölü latentlerle artığı (residual) modelleyerek onları canlandırır.
- Kalite: normalize MSE = E‖x − x̂‖² / E‖x − x̄‖², açıklanan varyans
  oranı (FVE) = 1 − NMSE.

Özellik bulma:
    ``feature_top_tokens``      — her latent için en çok ateşlendiği bağlamlar
    ``feature_label_association`` — bir etiketle (ör. "beklenen_uyum = ince")
        en güçlü ilişkili latentler (nokta-çift serili / point-biserial
        korelasyon ve ortalama farkı). "Ünlü uyumu özelliği" böyle aranır.

Referans: Gao et al., "Scaling and evaluating sparse autoencoders", 2024.
https://arxiv.org/abs/2406.04093
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from interpret.probes import few_threads


class TopKSAE(nn.Module):
    """TopK seyrek otokodlayıcı."""

    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        if not 1 <= k <= d_hidden:
            raise ValueError("k, 1 ile d_hidden arasında olmalı")
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_hidden))
        self.register_buffer("input_scale", torch.ones(()))
        nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)
        self.normalize_decoder()
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.t())

    @torch.no_grad()
    def normalize_decoder(self):
        """Kod çözücü sütunlarını birim norma getir."""
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def remove_parallel_grad(self):
        """Gradyanın decoder sütunlarına paralel bileşenini at (norm sabit kalsın)."""
        if self.W_dec.grad is None:
            return
        proj = (self.W_dec.grad * self.W_dec).sum(dim=0, keepdim=True)
        self.W_dec.grad.sub_(proj * self.W_dec)

    def pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / self.input_scale
        return (x - self.b_pre) @ self.W_enc.t() + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(N, d_in) ham aktivasyon → (N, d_hidden) seyrek latentler (en çok k sıfır-dışı)."""
        pre = self.pre_activation(x)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, torch.relu(vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latentler → ham ölçekte yeniden kurulum."""
        return (z @ self.W_dec.t() + self.b_pre) * self.input_scale

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


def normalized_mse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """E‖x − x̂‖² / E‖x − x̄‖²."""
    x = x.float()
    num = (x - x_hat.float()).pow(2).sum(dim=-1).mean()
    den = (x - x.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean().clamp_min(1e-12)
    return float(num / den)


@dataclass
class SAETrainResult:
    """``train_sae`` çıktısı."""

    sae: TopKSAE
    history: List[float]                 # epok sonu NMSE (tüm veri)
    nmse: float
    fve: float                           # açıklanan varyans oranı
    dead_fraction: float                 # son epokta hiç ateşlenmeyen latent oranı
    fire_counts: torch.Tensor = field(repr=False)   # (d_hidden,) son değerlendirmede ateşlenme sayıları
    initial_nmse: float = float("nan")

    def summary(self) -> dict:
        return {
            "d_in": self.sae.d_in,
            "d_hidden": self.sae.d_hidden,
            "k": self.sae.k,
            "initial_nmse": self.initial_nmse,
            "nmse": self.nmse,
            "fve": self.fve,
            "dead_fraction": self.dead_fraction,
            "history": self.history,
        }


@torch.no_grad()
def evaluate_sae(sae: TopKSAE, acts: torch.Tensor, batch_size: int = 4096):
    """(NMSE, ateşlenme sayıları) döndür."""
    recon, fires = [], torch.zeros(sae.d_hidden)
    for i in range(0, len(acts), batch_size):
        x_hat, z = sae(acts[i:i + batch_size])
        recon.append(x_hat)
        fires += (z > 0).sum(dim=0).float()
    return normalized_mse(acts, torch.cat(recon)), fires


def train_sae(
    acts: torch.Tensor,
    d_hidden: int,
    k: int,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 0,
    batch_size: int = 256,
    aux_k: Optional[int] = None,
    aux_coef: float = 1.0 / 32,
) -> SAETrainResult:
    """
    TopK SAE eğit.

    Args:
        acts: (N, d_in) aktivasyonlar (ör. ``CollectedActivations.acts["resid_post"][L]``)
        d_hidden: latent sayısı (tipik: 4–32 × d_in)
        k: token başına aktif latent sayısı
        epochs, lr, batch_size: Adam eğitimi
        seed: başlatma ve karıştırma tohumu
        aux_k: AuxK için ölü latent sayısı (None → min(2k, d_hidden/2); 0 = kapalı)
        aux_coef: AuxK kayıp katsayısı

    Returns:
        SAETrainResult (NMSE geçmişi, FVE, ölü latent oranı)
    """
    torch.manual_seed(seed)
    acts = acts.detach().float().cpu()
    N, d_in = acts.shape
    sae = TopKSAE(d_in, d_hidden, k)
    with torch.no_grad():
        scale = (acts.pow(2).sum(dim=-1).mean() / d_in).sqrt().clamp_min(1e-8)
        sae.input_scale.fill_(float(scale))
        sae.b_pre.copy_(acts.mean(dim=0) / scale)
    if aux_k is None:
        aux_k = min(2 * k, max(1, d_hidden // 2))

    initial_nmse, _ = evaluate_sae(sae, acts)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    steps_since_fire = torch.zeros(d_hidden)
    dead_threshold = max(1, N)  # bir epok boyunca hiç ateşlenmeyen = ölü
    history: List[float] = []

    with torch.enable_grad(), few_threads(batch_size * max(d_in, d_hidden)):
        for _ in range(epochs):
            perm = torch.randperm(N, generator=gen)
            for i in range(0, N, batch_size):
                xb = acts[perm[i:i + batch_size]]
                x_scaled = xb / sae.input_scale
                pre = sae.pre_activation(xb)
                vals, idx = pre.topk(k, dim=-1)
                z = torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))
                recon = z @ sae.W_dec.t() + sae.b_pre
                err = x_scaled - recon
                loss = err.pow(2).sum(-1).mean() / x_scaled.var(dim=0, unbiased=False).sum().clamp_min(1e-8)

                fired = (z > 0).any(dim=0)
                steps_since_fire += len(xb)
                steps_since_fire[fired] = 0
                dead = steps_since_fire >= dead_threshold
                if aux_k and aux_coef > 0 and dead.any():
                    ka = min(aux_k, int(dead.sum()))
                    dead_pre = pre.masked_fill(~dead, float("-inf"))
                    avals, aidx = dead_pre.topk(ka, dim=-1)
                    za = torch.zeros_like(pre).scatter(-1, aidx, torch.relu(avals))
                    aux_recon = za @ sae.W_dec.t()
                    aux_loss = (err.detach() - aux_recon).pow(2).sum(-1).mean() / err.detach().pow(2).sum(-1).mean().clamp_min(1e-8)
                    loss = loss + aux_coef * aux_loss

                opt.zero_grad()
                loss.backward()
                sae.remove_parallel_grad()
                opt.step()
                sae.normalize_decoder()
            nmse, _ = evaluate_sae(sae, acts)
            history.append(nmse)

    nmse, fires = evaluate_sae(sae, acts)
    return SAETrainResult(
        sae=sae,
        history=history,
        nmse=nmse,
        fve=1.0 - nmse,
        dead_fraction=float((fires == 0).float().mean()),
        fire_counts=fires,
        initial_nmse=initial_nmse,
    )


@torch.no_grad()
def encode_all(sae: TopKSAE, acts: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    """Tüm aktivasyonları latentlere çevir (N, d_hidden)."""
    return torch.cat([sae.encode(acts[i:i + batch_size].float()) for i in range(0, len(acts), batch_size)])


def _context(token_strings: Sequence[str], i: int, window: int, sentence_ids: Optional[torch.Tensor]) -> dict:
    lo, hi = max(0, i - window), min(len(token_strings), i + 2)
    if sentence_ids is not None:
        s = sentence_ids[i]
        while lo < i and sentence_ids[lo] != s:
            lo += 1
        while hi > i + 1 and sentence_ids[hi - 1] != s:
            hi -= 1
    return {
        "left": "".join(token_strings[lo:i]).replace("▁", " "),
        "token": token_strings[i],
        "right": "".join(token_strings[i + 1:hi]).replace("▁", " "),
    }


@torch.no_grad()
def feature_top_tokens(
    sae: TopKSAE,
    acts: torch.Tensor,
    token_strings: Sequence[str],
    n: int = 8,
    features: Optional[Sequence[int]] = None,
    window: int = 6,
    sentence_ids: Optional[torch.Tensor] = None,
    latents: Optional[torch.Tensor] = None,
) -> Dict[int, List[dict]]:
    """
    Her latent için en yüksek aktivasyonlu ``n`` token bağlamı.

    Args:
        token_strings: düzleştirilmiş token metinleri (acts satırlarıyla hizalı)
        features: incelenecek latent indeksleri (None = hepsi)
        window: soldaki bağlam token sayısı
        sentence_ids: (N,) verilirse bağlam cümle sınırında kesilir

    Returns:
        {latent: [{"activation", "index", "left", "token", "right"}, ...]}
    """
    z = encode_all(sae, acts) if latents is None else latents
    feats = range(sae.d_hidden) if features is None else features
    out: Dict[int, List[dict]] = {}
    for f in feats:
        col = z[:, f]
        m = min(n, int((col > 0).sum()))
        if m == 0:
            out[int(f)] = []
            continue
        vals, idx = col.topk(m)
        out[int(f)] = [
            {"activation": float(v), "index": int(i), **_context(token_strings, int(i), window, sentence_ids)}
            for v, i in zip(vals, idx)
        ]
    return out


@torch.no_grad()
def feature_label_association(
    sae: TopKSAE,
    acts: torch.Tensor,
    labels: torch.Tensor,
    positive_class: int = 1,
    top: int = 10,
    latents: Optional[torch.Tensor] = None,
    min_fires: int = 1,
) -> List[dict]:
    """
    Latentleri bir etiketle ilişkisine göre sırala.

    Etiket ikili hale getirilir (``labels == positive_class`` → 1, diğer geçerli
    etiketler → 0, ``labels < 0`` atılır). Her latent için nokta-çift serili
    korelasyon r, sınıf ortalamaları farkı ve ateşlenme oranları hesaplanır;
    r'ye göre azalan sırada (pozitif ilişki) ilk ``top`` latent döner.

    Returns:
        [{"feature", "r", "mean_pos", "mean_neg", "mean_diff",
          "fire_rate_pos", "fire_rate_neg"}, ...]
    """
    z = encode_all(sae, acts) if latents is None else latents
    keep = labels >= 0
    z = z[keep]
    yb = (labels[keep] == positive_class).float()
    n_pos, n = yb.sum(), float(len(yb))
    if n_pos == 0 or n_pos == n:
        return []
    p = n_pos / n
    mean_pos = (z * yb[:, None]).sum(0) / n_pos
    mean_neg = (z * (1 - yb)[:, None]).sum(0) / (n - n_pos)
    std = z.std(dim=0, unbiased=False)
    r = (mean_pos - mean_neg) * torch.sqrt(p * (1 - p)) / std.clamp_min(1e-8)
    r = torch.where(std > 0, r, torch.zeros_like(r))
    fires = z > 0
    fire_pos = (fires & (yb[:, None] > 0)).sum(0).float() / n_pos
    fire_neg = (fires & (yb[:, None] == 0)).sum(0).float() / (n - n_pos)
    valid = fires.sum(0) >= min_fires
    r_rank = torch.where(valid, r, torch.full_like(r, -2.0))
    order = r_rank.argsort(descending=True)[:top]
    return [
        {
            "feature": int(f),
            "r": float(r[f]),
            "mean_pos": float(mean_pos[f]),
            "mean_neg": float(mean_neg[f]),
            "mean_diff": float(mean_pos[f] - mean_neg[f]),
            "fire_rate_pos": float(fire_pos[f]),
            "fire_rate_neg": float(fire_neg[f]),
        }
        for f in order.tolist()
        if valid[f]
    ]
