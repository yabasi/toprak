# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Saf PyTorch Ağırlık-Yalnız Kuantizasyon
Yerel (CPU / dizüstü / telefon prototipi) çıkarım için int8 ve int4.

- int8: çıkış kanalı (satır) başına simetrik ölçek, q ∈ [-127, 127]
- int4: giriş boyutunda grup başına ölçek (group_size 32/64/128), iki nibble
  bir bayta paketlenir; simetrik (q ∈ [-8, 7]) veya sıfır noktalı
  (asimetrik, q ∈ [0, 15]).
- İleri geçişte ağırlık anında float'a açılır (dequantize-on-the-fly).

ÖNEMLİ: Bu Python çekirdeği HIZ için değil, BOYUT ve KALİTE ölçümü içindir.
Her ileri geçişte ağırlık yeniden açıldığı için fp32'den yavaştır. Gerçek
hızlı int4 çıkarımı için GGUF (llama.cpp, Q4_K_M) veya MLX (-q) kullanın
(bkz. EDGE.md).

Bağlı embedding (tok_emb = lm_head) kuantize edilmez ve bağlı kalır.

Kullanım:
    python -m export.quantize --checkpoint checkpoints/toprak_best.pt \\
        --bits 4 --group-size 64 --out checkpoints/toprak_int4.pt
"""

import argparse
import math
import os
import sys
from typing import Iterable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANT_FORMAT = "toprak-quant-v1"
DEFAULT_SKIP = ("lm_head", "tok_emb", "router")


# ── int4 paketleme ────────────────────────────────────────

def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """
    [0, 15] aralığındaki değerleri son boyutta ikişer ikişer bir bayta paketle.
    Düşük nibble = çift indeks, yüksek nibble = tek indeks.

    Args:
        q: (..., 2N) tamsayı tensör, değerler 0..15
    Returns:
        (..., N) uint8
    """
    if q.shape[-1] % 2:
        raise ValueError("pack_int4: son boyut çift olmalı")
    q = q.to(torch.uint8)
    if q.numel() and int(q.max()) > 15:
        raise ValueError("pack_int4: değerler 0..15 aralığında olmalı")
    return q[..., 0::2] | (q[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """pack_int4'ün tersi: (..., N) uint8 → (..., 2N) uint8 (0..15)."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


# ── Kuantize lineer katman ─────────────────────────────────

class QuantLinear(nn.Module):
    """
    Ağırlık-yalnız kuantize nn.Linear (bias desteklenir).

    Buffer'lar (state_dict'e yazılır):
        int8: qweight int8 (out, in), scales (out, 1)
        int4: qweight uint8 (out, in_pad/2), scales (out, n_groups),
              [zeros uint8 (out, n_groups) — zero_point=True ise]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 64,
        zero_point: bool = False,
        bias: bool = False,
        scale_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"bits 4 veya 8 olmalı, {bits} verildi")
        if bits == 4 and (group_size <= 0 or group_size % 2):
            raise ValueError("int4 için group_size pozitif ve çift olmalı (32/64/128)")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        if bits == 4:
            # Giriş boyutu gruptan küçükse tek grup (gereksiz dolguyu önle)
            group_size = min(group_size, in_features + in_features % 2)
        self.group_size = group_size if bits == 4 else in_features
        self.zero_point = bool(zero_point) and bits == 4

        if bits == 8:
            self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer("scales", torch.zeros(out_features, 1, dtype=scale_dtype))
        else:
            n_groups = math.ceil(in_features / self.group_size)
            padded = n_groups * self.group_size
            self.register_buffer("qweight", torch.zeros(out_features, padded // 2, dtype=torch.uint8))
            self.register_buffer("scales", torch.zeros(out_features, n_groups, dtype=scale_dtype))
            if self.zero_point:
                self.register_buffer("zeros", torch.zeros(out_features, n_groups, dtype=torch.uint8))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=scale_dtype))
        else:
            self.bias = None

    @property
    def n_groups(self) -> int:
        return self.scales.shape[1]

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 64,
        zero_point: bool = False,
        scale_dtype: torch.dtype = torch.float16,
    ) -> "QuantLinear":
        """Var olan nn.Linear'dan kuantize katman üret."""
        q = cls(
            linear.in_features, linear.out_features, bits=bits, group_size=group_size,
            zero_point=zero_point, bias=linear.bias is not None, scale_dtype=scale_dtype,
        ).to(linear.weight.device)
        q.quantize_(linear.weight.detach())
        if linear.bias is not None:
            q.bias.copy_(linear.bias.detach().to(q.bias.dtype))
        return q

    @torch.no_grad()
    def quantize_(self, weight: torch.Tensor) -> None:
        """Float ağırlığı kuantize edip buffer'lara yaz."""
        w = weight.float()
        if w.shape != (self.out_features, self.in_features):
            raise ValueError(f"Ağırlık şekli uyumsuz: {tuple(w.shape)}")
        sdtype = self.scales.dtype

        if self.bits == 8:
            amax = w.abs().amax(dim=1, keepdim=True)
            scale = (amax / 127.0).clamp(min=1e-12).to(sdtype)
            q = torch.round(w / scale.float()).clamp(-127, 127)
            self.qweight.copy_(q.to(torch.int8))
            self.scales.copy_(scale)
            return

        G, gs = self.n_groups, self.group_size
        pad = G * gs - self.in_features
        if pad:
            w = F.pad(w, (0, pad))
        wg = w.view(self.out_features, G, gs)
        if self.zero_point:
            wmin = wg.amin(dim=-1).clamp(max=0)
            wmax = wg.amax(dim=-1).clamp(min=0)
            scale = ((wmax - wmin) / 15.0).clamp(min=1e-12).to(sdtype)
            s = scale.float()
            zero = torch.round(-wmin / s).clamp(0, 15)
            q = torch.round(wg / s.unsqueeze(-1)) + zero.unsqueeze(-1)
            q = q.clamp(0, 15)
            self.zeros.copy_(zero.to(torch.uint8))
        else:
            amax = wg.abs().amax(dim=-1)
            scale = (amax / 7.0).clamp(min=1e-12).to(sdtype)
            q = torch.round(wg / scale.float().unsqueeze(-1)).clamp(-8, 7) + 8
        self.scales.copy_(scale)
        self.qweight.copy_(pack_int4(q.view(self.out_features, -1).to(torch.uint8)))

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Paketli ağırlığı (out, in) float tensöre aç."""
        if self.bits == 8:
            return (self.qweight.float() * self.scales.float()).to(dtype)
        G, gs = self.n_groups, self.group_size
        q = unpack_int4(self.qweight).view(self.out_features, G, gs).float()
        if self.zero_point:
            q = q - self.zeros.float().unsqueeze(-1)
        else:
            q = q - 8.0
        w = (q * self.scales.float().unsqueeze(-1)).view(self.out_features, G * gs)
        return w[:, : self.in_features].to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        extra = f", group_size={self.group_size}, zero_point={self.zero_point}" if self.bits == 4 else ""
        return f"in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}{extra}"


# ── Model düzeyi ──────────────────────────────────────────

def _should_quantize(name: str, module: nn.Module, skip: Iterable[str]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if not name.startswith("blocks."):
        return False  # yalnız attention/FFN (mtp_heads, morph_head, lm_head hariç)
    parts = name.split(".")
    return not any(s in parts or name.startswith(s) for s in skip)


def _set_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    skip: Iterable[str] = DEFAULT_SKIP,
    zero_point: bool = False,
    scale_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Transformer bloklarındaki (attention + FFN; MoE uzmanları dahil) nn.Linear
    katmanlarını QuantLinear ile yerinde değiştir.

    `skip` içindeki adlar (ör. "lm_head", "tok_emb", "router") atlanır.
    Bağlı embedding/lm_head her zaman float kalır.

    Returns:
        Aynı model nesnesi; `model.quantization` meta sözlüğü eklenir.
    """
    skip = tuple(skip)
    targets = [(n, m) for n, m in model.named_modules() if _should_quantize(n, m, skip)]
    for name, linear in targets:
        _set_submodule(
            model, name,
            QuantLinear.from_linear(linear, bits=bits, group_size=group_size,
                                    zero_point=zero_point, scale_dtype=scale_dtype),
        )
    model.quantization = {
        "format": QUANT_FORMAT,
        "bits": bits,
        "group_size": group_size if bits == 4 else None,
        "zero_point": bool(zero_point) and bits == 4,
        "scale_dtype": str(scale_dtype).replace("torch.", ""),
        "skip": list(skip),
        "modules": [n for n, _ in targets],
    }
    return model


def model_size_bytes(model: nn.Module, include_buffers: bool = True) -> int:
    """
    Modelin bellekteki ağırlık boyutu (bayt). Bağlı/paylaşılan tensörler
    bir kez sayılır; kalıcı olmayan buffer'lar (RoPE tablosu) sayılmaz.
    """
    seen = set()
    total = 0

    def add(t):
        nonlocal total
        if t is None:
            return
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), t.numel(), t.dtype)
        if key in seen:
            return
        seen.add(key)
        total += t.numel() * t.element_size()

    for p in model.parameters():
        add(p)
    if include_buffers:
        for module in model.modules():
            for name, buf in module._buffers.items():
                if name in module._non_persistent_buffers_set:
                    continue
                add(buf)
    return total


def size_report(model: nn.Module) -> dict:
    """Boyut kırılımı: kuantize katmanlar, float parametreler, toplam (MB)."""
    quant = sum(
        sum(b.numel() * b.element_size() for b in m.buffers())
        for m in model.modules() if isinstance(m, QuantLinear)
    )
    total = model_size_bytes(model)
    return {
        "total_bytes": total,
        "total_mb": total / 2**20,
        "quantized_linear_bytes": quant,
        "other_bytes": total - quant,
    }


@torch.no_grad()
def quantization_error_report(
    model_fp: nn.Module,
    model_q: nn.Module,
    sample_ids: torch.Tensor,
    pad_token_id: Optional[int] = None,
) -> dict:
    """
    Tam hassasiyetli ve kuantize model arasındaki fark.

    Args:
        sample_ids: (B, T) token ID'leri — perplexity bir sonraki token
            tahmini üzerinden (ids[:, 1:]) hesaplanır.

    Returns:
        {"logit_mse", "logit_max_abs", "top1_agreement", "kl_fp_q",
         "ppl_fp", "ppl_q", "ppl_delta", "ppl_delta_pct"}
    """
    model_fp.eval()
    model_q.eval()
    lf = model_fp(sample_ids)[0].float()
    lq = model_q(sample_ids)[0].float()
    V = lf.size(-1)

    def ppl(logits):
        if sample_ids.size(1) < 2:
            return float("nan")
        ignore = -100 if pad_token_id is None else pad_token_id
        ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, V), sample_ids[:, 1:].reshape(-1), ignore_index=ignore
        )
        return math.exp(ce.item())

    kl = F.kl_div(
        F.log_softmax(lq, -1).view(-1, V), F.log_softmax(lf, -1).view(-1, V),
        log_target=True, reduction="batchmean",
    ).item()
    ppl_fp, ppl_q = ppl(lf), ppl(lq)
    return {
        "logit_mse": F.mse_loss(lq, lf).item(),
        "logit_max_abs": (lq - lf).abs().max().item(),
        "top1_agreement": (lq.argmax(-1) == lf.argmax(-1)).float().mean().item(),
        "kl_fp_q": kl,
        "ppl_fp": ppl_fp,
        "ppl_q": ppl_q,
        "ppl_delta": ppl_q - ppl_fp,
        "ppl_delta_pct": 100.0 * (ppl_q - ppl_fp) / ppl_fp if ppl_fp == ppl_fp else float("nan"),
    }


# ── Kaydet / yükle ────────────────────────────────────────

def save_quantized(model: nn.Module, path: str, extra: Optional[dict] = None) -> None:
    """
    Kuantize checkpoint kaydet: paketli state_dict + config + kuantizasyon meta.
    Bağlı embedding tek tensör olarak saklanır (torch.save paylaşımı korur).
    """
    meta = getattr(model, "quantization", None)
    if not meta:
        raise ValueError("Model kuantize edilmemiş; önce quantize_model çağırın")
    checkpoint = {
        "format": QUANT_FORMAT,
        "model_state_dict": model.state_dict(),
        "config": model.config.architecture_dict(),
        "quantization": dict(meta),
    }
    if extra:
        checkpoint.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(checkpoint, path)


def load_quantized(path: str, device: str = "cpu", tokenizer=None) -> nn.Module:
    """
    save_quantized ile kaydedilmiş checkpoint'ten çıkarıma hazır ToprakLM kur.
    Ağırlıklar yeniden kuantize edilmez; paketli buffer'lar doğrudan yüklenir.
    """
    from export.hf_llama import config_from_dict
    from model.transformer import ToprakLM

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta = checkpoint.get("quantization")
    if not meta or checkpoint.get("format", meta.get("format")) != QUANT_FORMAT:
        raise ValueError(f"Kuantize Toprak checkpoint'i değil: {path}")

    config = config_from_dict(checkpoint["config"])
    model = ToprakLM(config, tokenizer=tokenizer)
    scale_dtype = getattr(torch, meta.get("scale_dtype", "float16"))
    for name in meta["modules"]:
        linear = model.get_submodule(name)
        _set_submodule(model, name, QuantLinear(
            linear.in_features, linear.out_features, bits=meta["bits"],
            group_size=meta.get("group_size") or linear.in_features,
            zero_point=meta.get("zero_point", False),
            bias=linear.bias is not None, scale_dtype=scale_dtype,
        ))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.quantization = dict(meta)
    model.config.device = device
    return model.to(device).eval()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="🌱 Toprak — ağırlık-yalnız int8/int4 kuantizasyon (boyut/kalite ölçümü)"
    )
    parser.add_argument("--checkpoint", required=True, help="Tam hassasiyetli Toprak checkpoint'i")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=64, choices=[32, 64, 128],
                        help="int4 grup boyutu (varsayılan: 64)")
    parser.add_argument("--zero-point", action="store_true", help="int4 için asimetrik (sıfır noktalı)")
    parser.add_argument("--out", required=True, help="Kuantize checkpoint çıktısı (.pt)")
    parser.add_argument("--tokenizer", default="toprak_tokenizer.model",
                        help="Kalite raporu için tokenizer")
    parser.add_argument("--eval-text", default=None,
                        help="Kalite raporu için düz metin dosyası (yoksa rapor atlanır)")
    parser.add_argument("--eval-tokens", type=int, default=512, help="Raporda kullanılacak token sayısı")
    args = parser.parse_args(argv)

    import copy
    from inference.generate import load_model

    model_fp, config = load_model(args.checkpoint, device="cpu")
    size_fp = model_size_bytes(model_fp)
    model_q = quantize_model(copy.deepcopy(model_fp), bits=args.bits,
                             group_size=args.group_size, zero_point=args.zero_point)
    size_q = model_size_bytes(model_q)
    save_quantized(model_q, args.out)

    print("🌱 Toprak kuantizasyon")
    print(f"  Mod        : int{args.bits}" + (f" (grup={args.group_size})" if args.bits == 4 else ""))
    print(f"  Katman     : {len(model_q.quantization['modules'])} lineer katman kuantize edildi")
    print(f"  Boyut      : {size_fp / 2**20:.1f} MB → {size_q / 2**20:.1f} MB "
          f"(×{size_fp / size_q:.2f} küçülme)")
    print(f"  Kaydedildi : {args.out}")

    if args.eval_text and os.path.exists(args.eval_text) and os.path.exists(args.tokenizer):
        from model.tokenizer import ToprakTokenizer
        tok = ToprakTokenizer(args.tokenizer)
        with open(args.eval_text, encoding="utf-8") as f:
            ids = tok.encode(f.read())[: min(args.eval_tokens, config.max_seq_len)]
        report = quantization_error_report(model_fp, model_q, torch.tensor([ids]),
                                           pad_token_id=config.pad_token_id)
        print(f"  Top-1 uyum : {report['top1_agreement']:.3f}")
        print(f"  Logit MSE  : {report['logit_mse']:.5f}")
        print(f"  Perplexity : {report['ppl_fp']:.2f} → {report['ppl_q']:.2f} "
              f"({report['ppl_delta_pct']:+.2f}%)")


if __name__ == "__main__":
    main()
