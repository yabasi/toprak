# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Doğrusal Sondalar (Linear Probes) ve Kontrol Görevleri

Bir katmanın aktivasyonlarından bir dilbilimsel etiketi (ör. "sıradaki ek
ince ünlülü olmalı") doğrusal bir sınıflandırıcıyla ne kadar iyi okuyabildiğimizi
ölçer. Saf torch ile çok sınıflı lojistik regresyon (tam-batch Adam + L2).

Aşırı iddiadan kaçınmak için:
- Çoğunluk sınıfı tabanı (majority baseline) ve makro-F1 raporlanır.
- Hewitt & Liang (2019) kontrol görevi: her token TÜRÜNE (ID) gerçek etiket
  dağılımından rastgele ama sabit bir etiket atanır. Aynı sonda bu anlamsız
  görevi de ezberleyebiliyorsa yüksek doğruluk "temsil" değil "kapasite"
  göstergesidir. Seçicilik = doğruluk − kontrol doğruluğu.
- Doğrulama bölmesi tercihen cümle düzeyindedir (``groups`` / ``val_mask``),
  böylece aynı cümlenin token'ları hem eğitimde hem testte bulunmaz.

Referans: Hewitt & Liang, "Designing and Interpreting Probes with Control
Tasks", EMNLP 2019. https://arxiv.org/abs/1909.03368
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def few_threads(numel: int, limit: int = 4_000_000):
    """
    Küçük problemlerde tek iş parçacığı kullan: tam-batch küçük matris
    işlemlerinde çok iş parçacığı senkronizasyonu hesaptan pahalıdır
    (özellikle yüklü makinelerde 10-50× yavaşlama görülebilir).
    """
    prev = torch.get_num_threads()
    if numel <= limit and prev > 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if torch.get_num_threads() != prev:
            torch.set_num_threads(prev)


@dataclass
class ProbeResult:
    """Tek bir sondanın sonucu."""

    accuracy: float
    macro_f1: float
    majority_baseline: float
    train_accuracy: float
    num_train: int
    num_val: int
    num_classes: int
    class_counts: List[int]
    probe: nn.Linear
    mean: torch.Tensor
    std: torch.Tensor

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """(N, d) aktivasyon → (N, C) sınıf olasılıkları."""
        z = (X.float() - self.mean) / self.std
        return F.softmax(self.probe(z), dim=-1)

    def summary(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "majority_baseline": self.majority_baseline,
            "train_accuracy": self.train_accuracy,
            "num_train": self.num_train,
            "num_val": self.num_val,
            "class_counts": self.class_counts,
        }


def macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Doğrulamada görülen sınıflar üzerinden makro-F1."""
    scores = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        if tp + fn == 0:
            continue
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def group_val_mask(groups: torch.Tensor, val_split: float, seed: int) -> torch.Tensor:
    """Grupları (ör. cümle indeksi) karıştırıp ``val_split`` oranını doğrulamaya ayır."""
    uniq = torch.unique(groups)
    gen = torch.Generator().manual_seed(seed)
    perm = uniq[torch.randperm(len(uniq), generator=gen)]
    n_val = max(1, int(round(len(uniq) * val_split))) if len(uniq) > 1 else 0
    val_groups = perm[:n_val]
    return torch.isin(groups, val_groups)


def train_linear_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: Optional[int] = None,
    epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    val_split: float = 0.25,
    seed: int = 0,
    class_balanced: bool = True,
    groups: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
) -> ProbeResult:
    """
    Doğrusal sonda (çok sınıflı lojistik regresyon) eğit ve değerlendir.

    Args:
        X: (N, d) aktivasyonlar
        y: (N,) etiketler; ``y < 0`` satırlar atılır
        num_classes: sınıf sayısı (None → max(y)+1)
        epochs: tam-batch Adam adımı sayısı
        lr, weight_decay: Adam öğrenme oranı ve L2 (decoupled değil, kayba eklenir)
        val_split: doğrulama oranı (``val_mask`` verilmemişse)
        seed: bölme ve başlatma tohumu
        class_balanced: sınıf frekansının tersiyle ağırlıklı kayıp
        groups: (N,) grup kimlikleri — verilirse bölme grup düzeyinde yapılır
        val_mask: (N,) bool — açık doğrulama maskesi (öncelikli)

    Returns:
        ProbeResult (doğrulama doğruluğu, makro-F1, çoğunluk tabanı, sonda)
    """
    X = X.detach().float().cpu()
    y = y.detach().long().cpu()
    keep = y >= 0
    if val_mask is None:
        if groups is not None:
            val_mask = group_val_mask(groups.cpu(), val_split, seed)
        else:
            gen = torch.Generator().manual_seed(seed)
            val_mask = torch.zeros(len(y), dtype=torch.bool)
            idx = torch.randperm(len(y), generator=gen)[: int(round(len(y) * val_split))]
            val_mask[idx] = True
    val_mask = val_mask.cpu().bool()
    tr = keep & ~val_mask
    va = keep & val_mask
    if num_classes is None:
        num_classes = int(y[keep].max().item()) + 1 if keep.any() else 1
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    if len(ytr) == 0 or len(yva) == 0:
        raise ValueError("Sonda için yeterli etiketli eğitim/doğrulama örneği yok")

    counts = torch.bincount(ytr, minlength=num_classes).float()
    mean = Xtr.mean(dim=0)
    std = Xtr.std(dim=0, unbiased=False).clamp_min(1e-6)
    Ztr = (Xtr - mean) / std
    Zva = (Xva - mean) / std

    torch.manual_seed(seed)
    probe = nn.Linear(X.size(1), num_classes)
    nn.init.zeros_(probe.weight)
    nn.init.zeros_(probe.bias)
    weight = None
    if class_balanced:
        weight = torch.where(counts > 0, counts.sum() / (num_classes * counts.clamp_min(1)), torch.zeros_like(counts))
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    with torch.enable_grad(), few_threads(Ztr.numel()):
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(probe(Ztr), ytr, weight=weight)
            loss = loss + weight_decay * probe.weight.pow(2).sum()
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred_va = probe(Zva).argmax(-1)
        pred_tr = probe(Ztr).argmax(-1)
    val_counts = torch.bincount(yva, minlength=num_classes)
    majority_class = int(counts.argmax().item())
    return ProbeResult(
        accuracy=float((pred_va == yva).float().mean()),
        macro_f1=macro_f1(pred_va, yva, num_classes),
        majority_baseline=float((yva == majority_class).float().mean()),
        train_accuracy=float((pred_tr == ytr).float().mean()),
        num_train=int(len(ytr)),
        num_val=int(len(yva)),
        num_classes=num_classes,
        class_counts=[int(c) for c in (counts + val_counts.float()).tolist()],
        probe=probe,
        mean=mean,
        std=std,
    )


def control_labels(token_ids: torch.Tensor, labels: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    Hewitt & Liang kontrol görevi: her token türüne (ID) gerçek etiketlerin
    marjinal dağılımından örneklenmiş SABİT bir rastgele etiket ata.
    ``labels < 0`` konumlar -1 kalır.
    """
    token_ids = token_ids.cpu().long()
    labels = labels.cpu().long()
    keep = labels >= 0
    out = torch.full_like(labels, -1)
    if not keep.any():
        return out
    probs = torch.bincount(labels[keep]).float()
    probs = probs / probs.sum()
    gen = torch.Generator().manual_seed(seed + 7919)
    uniq, inverse = torch.unique(token_ids[keep], return_inverse=True)
    type_labels = torch.multinomial(probs, len(uniq), replacement=True, generator=gen)
    out[keep] = type_labels[inverse]
    return out


def probe_all_layers(
    acts_by_layer: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
    token_ids: Optional[torch.Tensor] = None,
    control: bool = True,
    groups: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
    seeds: Sequence[int] = (0,),
    **probe_kwargs,
) -> Dict[str, object]:
    """
    Her katman için sonda eğit; katman × doğruluk / seçicilik tablosu döndür.

    Args:
        acts_by_layer: sıralı {katman_etiketi: (N, d)}
        labels: (N,) etiketler (-1 = yok say)
        token_ids: (N,) — kontrol görevi için gerekli
        control: kontrol görevi ve seçicilik hesaplansın mı
        groups / val_mask: bölme denetimi (bkz. ``train_linear_probe``)
        seeds: birden çok tohum → ortalama ve standart sapma
        probe_kwargs: ``train_linear_probe``'a aktarılır

    Returns:
        {"rows": [ {layer, accuracy, accuracy_std, macro_f1, majority_baseline,
                    control_accuracy, selectivity, num_train, num_val}, ... ],
         "probes": {katman: ProbeResult (ilk tohum)}}
    """
    if num_classes is None:
        num_classes = int(labels[labels >= 0].max().item()) + 1
    if control and token_ids is None:
        raise ValueError("Kontrol görevi için token_ids gerekli")
    rows: List[dict] = []
    probes: Dict[str, ProbeResult] = {}
    for layer, X in acts_by_layer.items():
        accs, f1s, ctrl = [], [], []
        first = None
        for seed in seeds:
            vm = val_mask
            if vm is None and groups is not None:
                vm = group_val_mask(groups, probe_kwargs.get("val_split", 0.25), seed)
            res = train_linear_probe(X, labels, num_classes=num_classes, seed=seed, val_mask=vm, **probe_kwargs)
            accs.append(res.accuracy)
            f1s.append(res.macro_f1)
            first = first or res
            if control:
                cy = control_labels(token_ids, labels, seed=seed)
                cres = train_linear_probe(X, cy, num_classes=num_classes, seed=seed, val_mask=vm, **probe_kwargs)
                ctrl.append(cres.accuracy)
        probes[layer] = first
        acc_t = torch.tensor(accs)
        row = {
            "layer": layer,
            "accuracy": float(acc_t.mean()),
            "accuracy_std": float(acc_t.std(unbiased=False)) if len(accs) > 1 else 0.0,
            "macro_f1": float(sum(f1s) / len(f1s)),
            "majority_baseline": first.majority_baseline,
            "num_train": first.num_train,
            "num_val": first.num_val,
        }
        if control:
            row["control_accuracy"] = float(sum(ctrl) / len(ctrl))
            row["selectivity"] = row["accuracy"] - row["control_accuracy"]
        rows.append(row)
    return {"rows": rows, "probes": probes}
