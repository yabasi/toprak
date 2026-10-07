# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Morfoloji Yönlendirmeli Uzman Karışımı (Morph-Routed MoE)

Yoğun SwiGLU FFN yerine N adet küçük SwiGLU uzmanı kullanılır; her token için
yönlendirici (router) en uygun top-k uzmanı seçer. Toplam parametre artar,
token başına aktif hesap ise yaklaşık sabit kalır.

Türkçe'ye özgü fark — morfolojik yönlendirme ipucu:
    Yönlendiricinin girdisine, giriş token'ının morfolojik sınıfının
    (0=kök, 1=ek, 2=özel) öğrenilebilir bir gömmesi eklenir. Böylece
    uzmanlar kendiliğinden "kök uzmanı", "hal eki uzmanı" gibi
    özelleşebilir. Sınıf giriş token'ından bilindiği için çıkarımda da
    ek maliyet yoktur.

Yük dengeleme:
    Switch Transformer tarzı yardımcı kayıp: N * Σ_i f_i * P_i
    (f_i = uzmana giden token oranı, P_i = ortalama yönlendirme olasılığı).

Görselleştirme:
    Her forward'da `last_routing` (B, T, top_k) uzman indeksleri saklanır;
    `interpret/` araçları veya `routing_by_morph_class()` ile hangi uzmanın
    hangi morfolojik sınıfa baktığı incelenebilir.

Referanslar:
- Switch Transformer: https://arxiv.org/abs/2101.03961
- Mixtral: https://arxiv.org/abs/2401.04088
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_MORPH_CLASSES = 3


class ExpertFFN(nn.Module):
    """Tek bir SwiGLU uzmanı (bias yok)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MorphRoutedMoE(nn.Module):
    """
    Top-k uzman karışımı, opsiyonel morfolojik yönlendirme ipucu ile.

    forward(x, morph_classes=None) → (out, aux_loss)
    """

    def __init__(self, config):
        super().__init__()
        assert config.num_experts >= 2, "MoE için en az 2 uzman gerekir"
        assert 1 <= config.experts_top_k <= config.num_experts
        self.num_experts = config.num_experts
        self.top_k = config.experts_top_k
        self.use_morph_routing = config.moe_morph_routing

        expert_d_ff = config.moe_d_ff or max(8, config.d_ff // config.experts_top_k)
        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)
        if self.use_morph_routing:
            self.morph_route_emb = nn.Embedding(NUM_MORPH_CLASSES, config.d_model)
            nn.init.zeros_(self.morph_route_emb.weight)
        self.experts = nn.ModuleList([
            ExpertFFN(config.d_model, expert_d_ff) for _ in range(config.num_experts)
        ])

        self.last_routing = None   # (B, T, top_k) — analiz için
        self.last_aux_loss = 0.0

    def forward(self, x, morph_classes=None):
        B, T, C = x.shape
        router_in = x
        if self.use_morph_routing and morph_classes is not None:
            router_in = x + self.morph_route_emb(morph_classes)

        router_logits = self.router(router_in).float()          # (B, T, E)
        probs = F.softmax(router_logits, dim=-1)
        top_p, top_idx = probs.topk(self.top_k, dim=-1)          # (B, T, k)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)
        top_p = top_p.to(x.dtype)

        flat_x = x.reshape(-1, C)
        flat_idx = top_idx.reshape(-1, self.top_k)
        flat_w = top_p.reshape(-1, self.top_k)
        out = torch.zeros_like(flat_x)

        for expert_id, expert in enumerate(self.experts):
            token_pos, slot = (flat_idx == expert_id).nonzero(as_tuple=True)
            if token_pos.numel() == 0:
                continue
            expert_out = expert(flat_x[token_pos])
            out.index_add_(0, token_pos, expert_out * flat_w[token_pos, slot].unsqueeze(-1))

        # Yük dengeleme kaybı (Switch): N * Σ f_i * P_i
        one_hot = F.one_hot(top_idx, self.num_experts).float().sum(dim=2)  # (B, T, E)
        fraction = one_hot.reshape(-1, self.num_experts).mean(dim=0) / self.top_k
        mean_prob = probs.reshape(-1, self.num_experts).mean(dim=0)
        aux_loss = self.num_experts * (fraction * mean_prob).sum()

        self.last_routing = top_idx.detach()
        self.last_aux_loss = float(aux_loss.detach())
        return out.view(B, T, C), aux_loss


def routing_by_morph_class(routing: torch.Tensor, morph_classes: torch.Tensor, num_experts: int):
    """
    Uzman × morfolojik sınıf sayım tablosu döndür.

    Args:
        routing: (B, T, k) uzman indeksleri (MorphRoutedMoE.last_routing)
        morph_classes: (B, T) giriş token sınıfları (0=kök, 1=ek, 2=özel)

    Returns:
        (num_experts, 3) LongTensor — [uzman, sınıf] sayımları
    """
    table = torch.zeros(num_experts, NUM_MORPH_CLASSES, dtype=torch.long)
    k = routing.size(-1)
    experts = routing.reshape(-1).cpu()
    classes = morph_classes.unsqueeze(-1).expand(-1, -1, k).reshape(-1).cpu()
    table.index_put_((experts, classes), torch.ones_like(experts), accumulate=True)
    return table
