# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Aktivasyon Kaydedici

ToprakLM bloklarına ileri besleme kancaları (forward hook) takarak her
katmandaki ara temsilleri yakalar:

- ``resid_pre``  : bloğa giren artık akış (residual stream); 0. bloğun girişi
                   gömme (embedding) katmanıdır
- ``resid_post`` : bloktan çıkan artık akış (x + attn + ffn)
- ``attn_out``   : dikkat alt katmanının artık akışa eklediği çıktı
- ``ffn_out``    : FFN (SwiGLU veya MoE) alt katmanının artık akışa eklediği çıktı

MoE bloklarında ``MorphRoutedMoE.forward`` ``(out, aux_loss)`` demeti döndürür;
kanca demetin ilk elemanını alır ve modülün ``last_routing`` (B, T, k) uzman
indekslerini ayrıca kaydeder. Kaydedici bir bağlam yöneticisidir; ``with``
bloğundan çıkıldığında tüm kancalar kaldırılır.

Örnek:
    >>> with ActivationRecorder(model, sites=("resid_post",)) as rec:
    ...     model(input_ids)
    >>> rec.get("resid_post", 3).shape   # (B, T, d_model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch

VALID_SITES = ("resid_pre", "resid_post", "attn_out", "ffn_out")


def _first(output):
    """Demet döndüren modüllerde (dikkat, MoE) tensör çıktıyı al."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class ActivationRecorder:
    """
    ToprakLM blokları için aktivasyon kaydedici (bağlam yöneticisi).

    Args:
        model: ToprakLM (``model.blocks`` listesi olan herhangi bir model)
        sites: kaydedilecek noktalar (bkz. ``VALID_SITES``)
        layers: kaydedilecek blok indeksleri (None = hepsi)
        record_routing: MoE bloklarında ``last_routing`` kaydedilsin mi
        to_cpu: kayıtlar CPU'ya ve float32'ye taşınsın mı

    Her ileri geçişte (forward) her (site, katman) için bir tensör eklenir;
    ``get`` aynı noktadaki kayıtları birleştirir (dim=0, batch ekseni).
    """

    def __init__(
        self,
        model,
        sites: Sequence[str] = ("resid_post", "attn_out", "ffn_out"),
        layers: Optional[Iterable[int]] = None,
        record_routing: bool = True,
        to_cpu: bool = True,
    ):
        unknown = [s for s in sites if s not in VALID_SITES]
        if unknown:
            raise ValueError(f"Bilinmeyen kayıt noktası: {unknown}; geçerli: {VALID_SITES}")
        self.model = model
        self.sites = tuple(sites)
        num_layers = len(model.blocks)
        self.layers = sorted(set(range(num_layers) if layers is None else layers))
        for i in self.layers:
            if not 0 <= i < num_layers:
                raise ValueError(f"Katman indeksi aralık dışında: {i} (0..{num_layers - 1})")
        self.record_routing = record_routing
        self.to_cpu = to_cpu
        self.records: Dict[str, Dict[int, List[torch.Tensor]]] = {
            s: {i: [] for i in self.layers} for s in self.sites
        }
        self.routing: Dict[int, List[torch.Tensor]] = {}
        self._handles: list = []

    # ── kanca yönetimi ──────────────────────────────────────────
    def _store(self, site: str, layer: int, tensor: torch.Tensor):
        t = tensor.detach()
        if self.to_cpu:
            t = t.float().cpu()
        self.records[site][layer].append(t)

    def attach(self):
        """Kancaları tak (bağlam yöneticisi bunu otomatik çağırır)."""
        if self._handles:
            return self
        for i in self.layers:
            block = self.model.blocks[i]
            if "resid_pre" in self.sites:
                def pre_hook(module, args, kwargs=None, _i=i):
                    x = args[0] if args else kwargs["x"]
                    self._store("resid_pre", _i, x)
                self._handles.append(block.register_forward_pre_hook(pre_hook))
            if "resid_post" in self.sites:
                def post_hook(module, args, output, _i=i):
                    self._store("resid_post", _i, _first(output))
                self._handles.append(block.register_forward_hook(post_hook))
            if "attn_out" in self.sites:
                def attn_hook(module, args, output, _i=i):
                    self._store("attn_out", _i, _first(output))
                self._handles.append(block.attn.register_forward_hook(attn_hook))
            if "ffn_out" in self.sites or (self.record_routing and getattr(block, "use_moe", False)):
                def ffn_hook(module, args, output, _i=i):
                    if "ffn_out" in self.sites:
                        self._store("ffn_out", _i, _first(output))
                    routing = getattr(module, "last_routing", None)
                    if self.record_routing and routing is not None:
                        self.routing.setdefault(_i, []).append(routing.detach().cpu())
                self._handles.append(block.ffn.register_forward_hook(ffn_hook))
        return self

    def remove(self):
        """Tüm kancaları kaldır."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self):
        """Kayıtları boşalt (kancalar takılı kalır)."""
        for s in self.records:
            for i in self.records[s]:
                self.records[s][i] = []
        self.routing = {}

    def __enter__(self):
        return self.attach()

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False

    # ── erişim ──────────────────────────────────────────────────
    def get(self, site: str, layer: int) -> torch.Tensor:
        """(site, katman) kayıtlarını batch ekseninde birleştirip döndür."""
        chunks = self.records[site][layer]
        if not chunks:
            raise KeyError(f"Kayıt yok: site={site}, layer={layer}")
        return torch.cat(chunks, dim=0)

    def get_routing(self, layer: int) -> Optional[torch.Tensor]:
        chunks = self.routing.get(layer)
        return torch.cat(chunks, dim=0) if chunks else None


@dataclass
class CollectedActivations:
    """
    ``collect`` çıktısı — tüm cümlelerin token'ları düzleştirilmiş (N satır).

    Alanlar:
        acts: {site: {katman: (N, d_model)}}; ``acts["embed"][0]`` gömme çıktısı
        token_ids: cümle başına token ID listeleri
        token_strings: cümle başına token metinleri (SentencePiece parçaları)
        sentence_index: (N,) her satırın cümle indeksi
        position: (N,) cümle içi konum
        flat_token_ids: (N,) düzleştirilmiş token ID'leri
        morph_classes: (N,) modelin kök/ek/özel tablosundan sınıf
        routing: {katman: (N, k)} MoE uzman indeksleri
        texts: girdi cümleleri
    """

    acts: Dict[str, Dict[int, torch.Tensor]]
    token_ids: List[List[int]]
    token_strings: List[List[str]]
    sentence_index: torch.Tensor
    position: torch.Tensor
    flat_token_ids: torch.Tensor
    morph_classes: torch.Tensor
    routing: Dict[int, torch.Tensor] = field(default_factory=dict)
    texts: List[str] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return int(self.flat_token_ids.numel())

    def flat_token_strings(self) -> List[str]:
        return [t for sent in self.token_strings for t in sent]

    def layer_dict(self, site: str = "resid_post", include_embedding: bool = True) -> Dict[str, torch.Tensor]:
        """
        Sondalar için sıralı {etiket: (N, d)} sözlüğü: "emb", "L1", ..., "LN".
        "Lk", k. bloğun (1 tabanlı) çıktısıdır.
        """
        out: Dict[str, torch.Tensor] = {}
        if include_embedding and "embed" in self.acts and 0 in self.acts["embed"]:
            out["emb"] = self.acts["embed"][0]
        for i in sorted(self.acts.get(site, {})):
            out[f"L{i + 1}"] = self.acts[site][i]
        return out


def _encode(tokenizer, text: str, add_bos: bool) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_bos=add_bos, add_eos=False))
    except TypeError:
        return list(tokenizer.encode(text))


@torch.no_grad()
def collect(
    model,
    tokenizer,
    texts: Sequence[str],
    max_len: int = 128,
    sites: Sequence[str] = ("resid_post", "attn_out", "ffn_out"),
    layers: Optional[Iterable[int]] = None,
    add_bos: bool = True,
    device: Optional[str] = None,
) -> CollectedActivations:
    """
    Metinleri modelden geçirip aktivasyonları token'larla hizalı topla.

    Cümleler tek tek işlenir (dolgu/padding yok); böylece her satır gerçek bir
    token'a karşılık gelir. Gömme çıktısı ``acts["embed"][0]`` altında
    her zaman kaydedilir (0. bloğun ``resid_pre``'si).

    Args:
        model: ToprakLM
        tokenizer: ``encode(text, add_bos=..., add_eos=...)`` ve
            ``id_to_token(id)`` sağlayan tokenizer
        texts: cümleler
        max_len: cümle başına en çok token (model bağlamını da aşmaz)
        sites: kaydedilecek noktalar
        layers: kaydedilecek bloklar (None = hepsi)
        add_bos: başa <s> eklensin mi
        device: None ise modelin cihazı

    Returns:
        CollectedActivations
    """
    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    limit = min(max_len, getattr(model.config, "max_seq_len", max_len))

    rec_sites = tuple(dict.fromkeys(tuple(sites) + ("resid_pre",)))
    layer_list = sorted(set(range(len(model.blocks)) if layers is None else layers))
    rec_layers = sorted(set(layer_list) | {0})

    all_ids: List[List[int]] = []
    all_strs: List[List[str]] = []
    sent_idx: List[int] = []
    pos: List[int] = []
    store: Dict[str, Dict[int, List[torch.Tensor]]] = {}
    routing: Dict[int, List[torch.Tensor]] = {}
    kept_texts: List[str] = []

    from interpret.probes import few_threads

    n_params = sum(p.numel() for p in model.parameters())
    with ActivationRecorder(model, sites=rec_sites, layers=rec_layers) as rec, few_threads(n_params, limit=5_000_000):
        for s, text in enumerate(texts):
            ids = _encode(tokenizer, text, add_bos)[:limit]
            if not ids:
                continue
            rec.clear()
            model(torch.tensor([ids], dtype=torch.long, device=device))
            for site in rec_sites:
                for i in rec_layers:
                    if site == "resid_pre":
                        if i == 0:
                            store.setdefault("embed", {}).setdefault(0, []).append(rec.get(site, 0)[0])
                        if site not in sites or i not in layer_list:
                            continue
                    elif i not in layer_list:
                        continue
                    store.setdefault(site, {}).setdefault(i, []).append(rec.get(site, i)[0])
            for i in rec_layers:
                r = rec.get_routing(i)
                if r is not None and i in layer_list:
                    routing.setdefault(i, []).append(r[0])
            all_ids.append(ids)
            kept_texts.append(text)
            all_strs.append([tokenizer.id_to_token(t) for t in ids])
            sent_idx.extend([len(all_ids) - 1] * len(ids))
            pos.extend(range(len(ids)))

    if was_training:
        model.train()

    acts = {site: {i: torch.cat(v, dim=0) for i, v in d.items()} for site, d in store.items()}
    flat_ids = torch.tensor([t for ids in all_ids for t in ids], dtype=torch.long)
    morph = model.token_morph_classes.detach().cpu()[flat_ids] if flat_ids.numel() else flat_ids
    return CollectedActivations(
        acts=acts,
        token_ids=all_ids,
        token_strings=all_strs,
        sentence_index=torch.tensor(sent_idx, dtype=torch.long),
        position=torch.tensor(pos, dtype=torch.long),
        flat_token_ids=flat_ids,
        morph_classes=morph,
        routing={i: torch.cat(v, dim=0) for i, v in routing.items()},
        texts=kept_texts,
    )
