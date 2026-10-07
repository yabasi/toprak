# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak → HuggingFace Llama Dönüştürücü
Toprak checkpoint'ini standart `LlamaForCausalLM` klasörüne çevirir:

    out_dir/
        config.json               (LlamaForCausalLM, bağlı embedding)
        model.safetensors         (bağlı ağırlık tek kez: model.embed_tokens.weight)
        generation_config.json
        tokenizer.model           (SentencePiece — llama.cpp bunu okur)
        tokenizer.json            (tokenizers — transformers / MLX bunu okur)
        tokenizer_config.json
        special_tokens_map.json

Böylece model; transformers, llama.cpp (GGUF), MLX, Ollama gibi Llama
mimarisini tanıyan tüm araçlarla telefonda/dizüstünde çalıştırılabilir.

Eşleme ayrıntıları:
- RoPE: Toprak ardışık çiftleri döndürür (view_as_complex; (0,1), (2,3), ...).
  HF Llama `rotate_half` kullanır (i, i + d/2). Bu yüzden q_proj ve k_proj
  satırları her head içinde permüte edilir (Meta → HF permütasyonu). Skorlar
  q·k iç çarpımı olduğu için iki tarafın aynı permütasyonu sonucu değiştirmez.
- RMSNorm: Toprak `weight * x / rms(x)` (float32'de normalize, sonra giriş
  dtype'ına dönüş) — HF LlamaRMSNorm ile aynı; eps = config.norm_eps.
- rope_scaling: linear → {"rope_type": "linear"}, ntk (statik taban
  değişimi) → rope_theta = theta * factor^(d/(d-2)) ve ölçekleme yok,
  yarn → {"rope_type": "yarn", "factor", "original_max_position_embeddings",
  "beta_fast", "beta_slow", "attention_factor"}.
- MoE (num_experts > 0) desteklenmez (Mixtral eşlemesi ileride).
- MTP başlıkları ve morfoloji başlığı atılır (çıkarımda gerekmez).

Kullanım:
    python -m export.hf_llama --checkpoint checkpoints/toprak_best.pt \\
        --out exports/toprak-hf --dtype float16
"""

import argparse
import dataclasses
import json
import math
import os
import shutil
import sys
import warnings
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.config import ModelConfig
from model.rope import scaled_inv_freqs, yarn_mscale

DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

DEFAULT_TOKENIZER = "toprak_tokenizer.model"


# ── Kaynak yükleme ─────────────────────────────────────────

def config_from_dict(cfg_dict: dict) -> ModelConfig:
    """Checkpoint config sözlüğünden ModelConfig kur (bilinmeyen alanları yok say)."""
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    kwargs = {k: v for k, v in (cfg_dict or {}).items() if k in known}
    kwargs["device"] = "cpu"
    return ModelConfig(**kwargs)


def _clean_state_dict(state_dict: dict) -> dict:
    """torch.compile önekini (`_orig_mod.`) temizle."""
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        cleaned[key] = value
    return cleaned


def load_source(model_or_checkpoint) -> tuple:
    """
    ToprakLM örneği, checkpoint yolu veya checkpoint sözlüğünü
    (state_dict, ModelConfig) çiftine çevir.
    """
    if isinstance(model_or_checkpoint, torch.nn.Module):
        model = getattr(model_or_checkpoint, "_orig_mod", model_or_checkpoint)
        return _clean_state_dict(model.state_dict()), model.config

    if isinstance(model_or_checkpoint, (str, os.PathLike)):
        checkpoint = torch.load(model_or_checkpoint, map_location="cpu", weights_only=False)
    elif isinstance(model_or_checkpoint, dict):
        checkpoint = model_or_checkpoint
    else:
        raise TypeError(
            "model_or_checkpoint bir ToprakLM, checkpoint yolu veya checkpoint sözlüğü olmalı; "
            f"{type(model_or_checkpoint).__name__} verildi"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint 'model_state_dict' anahtarı içermiyor")
    if "quantization" in checkpoint:
        raise ValueError(
            "Kuantize edilmiş checkpoint HF'ye çevrilemez; önce tam hassasiyetli "
            "(fp32/bf16) checkpoint'i dönüştürün, kuantizasyonu GGUF/MLX tarafında yapın."
        )
    config = config_from_dict(checkpoint.get("config", {}))
    return _clean_state_dict(checkpoint["model_state_dict"]), config


# ── Ağırlık eşleme ─────────────────────────────────────────

def permute_rotary(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """
    Ardışık-çift (interleaved) RoPE düzenindeki q/k projeksiyon satırlarını
    HF `rotate_half` (yarım-yarım) düzenine çevir.

    Head içinde satır sırası: [0, 2, 4, ..., d-2, 1, 3, ..., d-1]
    """
    in_dim = weight.shape[1]
    return (
        weight.view(n_heads, head_dim // 2, 2, in_dim)
        .transpose(1, 2)
        .reshape(n_heads * head_dim, in_dim)
    )


def unpermute_rotary(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """permute_rotary'nin tersi (HF → Toprak)."""
    in_dim = weight.shape[1]
    return (
        weight.view(n_heads, 2, head_dim // 2, in_dim)
        .transpose(1, 2)
        .reshape(n_heads * head_dim, in_dim)
    )


def check_exportable(config: ModelConfig, state_dict: dict) -> None:
    """Llama biçimine çevrilemeyen yapılandırmalarda açık hata ver."""
    has_moe_keys = any(".ffn.experts." in k or ".ffn.router." in k for k in state_dict)
    if config.num_experts > 0 or has_moe_keys:
        raise NotImplementedError(
            "MoE (num_experts > 0) Toprak modelleri Llama biçimine çevrilemez: Llama "
            "yoğun (dense) FFN bekler. Gelecek çalışma: uzmanları Mixtral eşlemesine "
            "(block_sparse_moe.experts.{i}.w1/w2/w3 + gate) aktarmak; morfolojik "
            "yönlendirme ipucu Mixtral'de karşılıksızdır."
        )


def toprak_to_llama_state_dict(
    state_dict: dict,
    config: ModelConfig,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Toprak state_dict → HF Llama state_dict.

    Bağlı ağırlık (tok_emb = lm_head) yalnız `model.embed_tokens.weight`
    olarak yazılır; `tie_word_embeddings=True` ile transformers lm_head'i bağlar.
    MTP ve morph başlıkları atlanır (uyarı ile).
    """
    check_exportable(config, state_dict)
    H, KV, D = config.num_heads, config.num_kv_heads, config.head_dim

    dropped_mtp = sorted({k.split(".")[1] for k in state_dict if k.startswith("mtp_heads.")})
    if dropped_mtp:
        warnings.warn(
            f"{len(dropped_mtp)} MTP başlığı dışa aktarımda atıldı "
            "(Llama biçiminde karşılığı yok; standart tek-token üretim etkilenmez).",
            UserWarning,
            stacklevel=2,
        )

    def get(name):
        if name not in state_dict:
            raise KeyError(f"Checkpoint'te beklenen ağırlık yok: {name}")
        return state_dict[name].detach().to(torch.float32)

    out = {"model.embed_tokens.weight": get("tok_emb.weight")}
    if "lm_head.weight" in state_dict and not torch.equal(
        state_dict["lm_head.weight"].float(), out["model.embed_tokens.weight"]
    ):
        raise ValueError("lm_head ve tok_emb ağırlıkları bağlı değil; Toprak her zaman bağlı olmalı")

    for i in range(config.num_layers):
        src, dst = f"blocks.{i}", f"model.layers.{i}"
        out[f"{dst}.input_layernorm.weight"] = get(f"{src}.ln1.weight")
        out[f"{dst}.post_attention_layernorm.weight"] = get(f"{src}.ln2.weight")
        out[f"{dst}.self_attn.q_proj.weight"] = permute_rotary(get(f"{src}.attn.q_proj.weight"), H, D)
        out[f"{dst}.self_attn.k_proj.weight"] = permute_rotary(get(f"{src}.attn.k_proj.weight"), KV, D)
        out[f"{dst}.self_attn.v_proj.weight"] = get(f"{src}.attn.v_proj.weight")
        out[f"{dst}.self_attn.o_proj.weight"] = get(f"{src}.attn.out_proj.weight")
        out[f"{dst}.mlp.gate_proj.weight"] = get(f"{src}.ffn.gate_proj.weight")
        out[f"{dst}.mlp.up_proj.weight"] = get(f"{src}.ffn.up_proj.weight")
        out[f"{dst}.mlp.down_proj.weight"] = get(f"{src}.ffn.down_proj.weight")
    out["model.norm.weight"] = get("ln_f.weight")

    return {k: v.to(dtype).contiguous() for k, v in out.items()}


# ── RoPE / config ──────────────────────────────────────────

def _hf_yarn_inv_freq(dim, base, factor, original, beta_fast, beta_slow):
    """transformers `_compute_yarn_parameters` ters frekans formülünün kopyası."""
    def corr(rot):
        return (dim * math.log(original / (rot * 2 * math.pi))) / (2 * math.log(base))
    low = max(math.floor(corr(beta_fast)), 0)
    high = min(math.ceil(corr(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    pos = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extra = 1.0 / pos
    inter = 1.0 / (factor * pos)
    keep = 1 - ramp
    return inter * (1 - keep) + extra * keep


def rope_config_for_hf(config: ModelConfig) -> tuple:
    """
    Toprak RoPE ayarını HF Llama alanlarına çevir.

    Returns:
        (rope_theta, rope_scaling_or_None)
    """
    theta = float(config.rope_theta)
    scaling = config.rope_scaling
    if not scaling:
        return theta, None
    kind = scaling.get("type", "yarn")
    factor = float(scaling.get("factor", 1.0))
    if factor <= 1.0:
        return theta, None
    dim = config.head_dim

    if kind == "linear":
        return theta, {"rope_type": "linear", "type": "linear", "factor": factor}

    if kind == "ntk":
        # Toprak'ın "ntk"si statik taban değişimidir → sadece theta değişir.
        return theta * factor ** (dim / (dim - 2)), None

    if kind == "yarn":
        original = int(scaling.get("original_max_seq_len", 2048))
        beta_fast = float(scaling.get("beta_fast", 32.0))
        beta_slow = float(scaling.get("beta_slow", 1.0))
        attention_factor = yarn_mscale(factor, float(scaling.get("mscale", 1.0)))
        ours, _ = scaled_inv_freqs(dim, theta, scaling)
        theirs = _hf_yarn_inv_freq(dim, theta, factor, original, beta_fast, beta_slow)
        if not torch.allclose(ours, theirs, rtol=1e-5, atol=1e-7):
            warnings.warn(
                "YaRN ters frekansları transformers formülüyle birebir örtüşmüyor "
                "(Toprak düzeltme aralığını dim//2-1 ile, transformers dim-1 ile kırpar). "
                "Uzun bağlam davranışı hafifçe farklı olabilir.",
                UserWarning,
                stacklevel=2,
            )
        return theta, {
            "rope_type": "yarn",
            "type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "attention_factor": attention_factor,
        }

    raise ValueError(f"Bilinmeyen rope_scaling tipi: {kind!r}")


def build_llama_config(config: ModelConfig, dtype_name: str = "float16") -> dict:
    """HF `config.json` içeriği (LlamaForCausalLM)."""
    rope_theta, rope_scaling = rope_config_for_hf(config)
    rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
    if rope_scaling:
        rope_parameters = {k: v for k, v in rope_scaling.items() if k != "type"}
        rope_parameters["rope_theta"] = rope_theta
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "intermediate_size": config.d_ff,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_heads,
        "num_key_value_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": config.norm_eps,
        # Eski araçlar (llama.cpp, mlx_lm, transformers 4.x) bu iki alanı okur;
        # transformers 5.x `rope_parameters` alanını kullanır.
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling,
        "rope_parameters": rope_parameters,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "pretraining_tp": 1,
        "use_cache": True,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "torch_dtype": dtype_name,
        "dtype": dtype_name,
    }


def build_generation_config(config: ModelConfig) -> dict:
    """HF `generation_config.json` — inference/generate.py CLI varsayılanlarıyla."""
    return {
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "do_sample": True,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.3,  # inference/generate.py varsayılanı
    }


# ── Tokenizer ──────────────────────────────────────────────

def _sp_bpe_merges(proto) -> list:
    """SentencePiece BPE parçalarından HF BPE merge listesi üret (sıra = parça id)."""
    vocab = {p.piece: i for i, p in enumerate(proto.pieces)}
    normal = {i for i, p in enumerate(proto.pieces) if p.type == 1}
    merges = []
    for piece, idx in vocab.items():
        if idx not in normal:
            continue
        for cut in range(1, len(piece)):
            left, right = piece[:cut], piece[cut:]
            li, ri = vocab.get(left), vocab.get(right)
            if li in normal and ri in normal:
                merges.append((idx, li, ri, left, right))
    merges.sort()
    return [(left, right) for *_, left, right in merges]


def build_hf_tokenizer(sp_model_path: str):
    """
    SentencePiece (BPE, byte_fallback, nfkc) modelinden `tokenizers.Tokenizer` kur.

    Eşdeğerlik: düz metinde SentencePiece ile birebir aynı ID'ler (testlerle
    doğrulandı: NFKC, baştaki/sondaki/çoklu boşluk, satır sonu, rakam ayırma,
    byte fallback). Bilinen fark: kullanıcı tanımlı semboller (<sep>, <cls>,
    <mask>, sohbet tokenları) metin içinde geçerse, hemen yanlarındaki boşluk
    SentencePiece'te ayrı bir "▁" tokenı olurken burada yutulur/öne eklenir.
    Ayrıca <s>, </s>, <pad>, <unk> metin içinde yazılırsa burada özel token
    olarak eşlenir, SentencePiece'te ise harf harf kodlanır.
    """
    from sentencepiece import sentencepiece_model_pb2
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, processors
    from tokenizers.models import BPE

    proto = sentencepiece_model_pb2.ModelProto()
    with open(sp_model_path, "rb") as f:
        proto.ParseFromString(f.read())
    if proto.trainer_spec.model_type != 2:
        raise ValueError("Yalnız SentencePiece BPE modelleri desteklenir")

    vocab = {p.piece: i for i, p in enumerate(proto.pieces)}
    unk_piece = proto.trainer_spec.unk_piece or "<unk>"
    tokenizer = Tokenizer(
        BPE(
            vocab=vocab,
            merges=_sp_bpe_merges(proto),
            unk_token=unk_piece,
            fuse_unk=True,
            byte_fallback=proto.trainer_spec.byte_fallback,
        )
    )

    spec = proto.normalizer_spec
    norm = []
    if spec.precompiled_charsmap:
        norm.append(normalizers.Precompiled(spec.precompiled_charsmap))
    if spec.remove_extra_whitespaces:
        # SentencePiece yalnız ASCII boşluğu kırpar/sıkıştırır (\n, \t korunur)
        norm.append(normalizers.Replace(Regex(r"\A +| +\z"), ""))
        norm.append(normalizers.Replace(Regex(" {2,}"), " "))
    norm.append(normalizers.Replace(" ", "▁"))
    if spec.add_dummy_prefix:
        norm.append(normalizers.Prepend("▁"))
    tokenizer.normalizer = normalizers.Sequence(norm)
    tokenizer.pre_tokenizer = None

    dec = [decoders.Replace("▁", " ")]
    if proto.trainer_spec.byte_fallback:
        dec += [decoders.ByteFallback()]
    dec += [decoders.Fuse()]
    if spec.add_dummy_prefix:
        dec += [decoders.Strip(content=" ", left=1)]
    tokenizer.decoder = decoders.Sequence(dec)

    # Kontrol tokenları (pad/unk/bos/eos) özel; kullanıcı tanımlılar normal eklenmiş token
    tokenizer.add_special_tokens([
        AddedToken(p.piece, normalized=False, special=True)
        for p in proto.pieces if p.type in (2, 3)
    ])
    tokenizer.add_tokens([
        AddedToken(p.piece, normalized=False, special=False)
        for p in proto.pieces if p.type == 4
    ])

    bos = proto.pieces[proto.trainer_spec.bos_id].piece if proto.trainer_spec.bos_id >= 0 else None
    if bos is not None:
        bos_id = proto.trainer_spec.bos_id
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{bos} $A",
            pair=f"{bos} $A {bos} $B",
            special_tokens=[(bos, bos_id)],
        )
    return tokenizer


def export_tokenizer(sp_model_path: str, out_dir: str, model_max_length: int = 2048) -> list:
    """
    Tokenizer dosyalarını yaz: tokenizer.model (kopya), tokenizer.json,
    tokenizer_config.json, special_tokens_map.json.

    `AutoTokenizer.from_pretrained(out_dir)` metni ToprakTokenizer ile aynı
    ID'lere kodlar (add_special_tokens=True → başta BOS, sonda EOS yok;
    ToprakTokenizer.encode(text, add_bos=True, add_eos=False) ile aynı).
    """
    import sentencepiece as spm

    os.makedirs(out_dir, exist_ok=True)
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    def piece(idx):
        return sp.id_to_piece(idx) if 0 <= idx < sp.get_piece_size() else None

    bos, eos, unk, pad = piece(sp.bos_id()), piece(sp.eos_id()), piece(sp.unk_id()), piece(sp.pad_id())

    shutil.copyfile(sp_model_path, os.path.join(out_dir, "tokenizer.model"))
    tokenizer = build_hf_tokenizer(sp_model_path)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))

    special = {k: v for k, v in {
        "bos_token": bos, "eos_token": eos, "unk_token": unk, "pad_token": pad,
    }.items() if v is not None}
    tok_config = {
        # Llama sınıfı yerine genel hızlı tokenizer: Llama sınıfı kendi
        # normalizer'ını kurup tokenizer.json'daki NFKC/boşluk kurallarını ezebilir.
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": model_max_length,
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "padding_side": "left",
        **special,
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special, f, ensure_ascii=False, indent=2)
    return ["tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


# ── Ana dönüştürücü ────────────────────────────────────────

def _resolve_tokenizer_path(tokenizer_path: Optional[str]) -> Optional[str]:
    if tokenizer_path:
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer bulunamadı: {tokenizer_path}")
        return tokenizer_path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (DEFAULT_TOKENIZER, os.path.join(root, DEFAULT_TOKENIZER)):
        if os.path.exists(candidate):
            return candidate
    return None


def convert_to_hf_llama(
    model_or_checkpoint: Union[str, dict, torch.nn.Module],
    out_dir: str,
    tokenizer_path: Optional[str] = None,
    dtype: str = "float16",
    include_tokenizer: bool = True,
) -> dict:
    """
    Toprak modelini HF LlamaForCausalLM klasörüne dönüştür.

    Args:
        model_or_checkpoint: ToprakLM, checkpoint yolu (.pt) veya checkpoint sözlüğü
        out_dir: Çıktı dizini
        tokenizer_path: SentencePiece .model (None → proje kökündeki
            toprak_tokenizer.model aranır)
        dtype: "float16" | "bfloat16" | "float32"
        include_tokenizer: False ise tokenizer dosyaları yazılmaz

    Returns:
        Özet sözlük: {"out_dir", "files", "num_parameters", "dtype", "dropped"}
    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype {list(DTYPES)} içinden olmalı, {dtype!r} verildi")
    torch_dtype = DTYPES[dtype]
    dtype_name = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}[torch_dtype]

    state_dict, config = load_source(model_or_checkpoint)
    hf_state = toprak_to_llama_state_dict(state_dict, config, dtype=torch_dtype)

    os.makedirs(out_dir, exist_ok=True)
    from safetensors.torch import save_file
    save_file(hf_state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(build_llama_config(config, dtype_name), f, indent=2)
    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(build_generation_config(config), f, indent=2)

    files = ["config.json", "model.safetensors", "generation_config.json"]
    if include_tokenizer:
        sp_path = _resolve_tokenizer_path(tokenizer_path)
        if sp_path is None:
            warnings.warn("Tokenizer bulunamadı; yalnız model ağırlıkları yazıldı.", UserWarning)
        else:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(sp_path)
            if sp.get_piece_size() != config.vocab_size:
                warnings.warn(
                    f"Tokenizer vocab ({sp.get_piece_size()}) model vocab'ı "
                    f"({config.vocab_size}) ile uyuşmuyor.",
                    UserWarning,
                )
            files += export_tokenizer(sp_path, out_dir, model_max_length=config.max_seq_len)

    dropped = sorted({k.split(".")[0] for k in state_dict if k.startswith(("mtp_heads.", "morph_head."))})
    num_params = sum(t.numel() for t in hf_state.values())
    return {
        "out_dir": out_dir,
        "files": files,
        "num_parameters": num_params,
        "dtype": dtype_name,
        "dropped": dropped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="🌱 Toprak → HuggingFace Llama dönüştürücü (GGUF / MLX için ilk adım)"
    )
    parser.add_argument("--checkpoint", required=True, help="Toprak checkpoint dosyası (.pt)")
    parser.add_argument("--out", required=True, help="Çıktı dizini")
    parser.add_argument("--tokenizer", default=None,
                        help="SentencePiece .model (varsayılan: toprak_tokenizer.model)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Ağırlık tipi (varsayılan: float16)")
    parser.add_argument("--no-tokenizer", action="store_true", help="Tokenizer dosyalarını yazma")
    args = parser.parse_args(argv)

    summary = convert_to_hf_llama(
        args.checkpoint, args.out,
        tokenizer_path=args.tokenizer, dtype=args.dtype,
        include_tokenizer=not args.no_tokenizer,
    )
    print("🌱 Toprak → HF Llama dönüşümü tamamlandı")
    print(f"  Dizin      : {summary['out_dir']}")
    print(f"  Parametre  : {summary['num_parameters'] / 1e6:.1f}M ({summary['dtype']})")
    print(f"  Dosyalar   : {', '.join(summary['files'])}")
    if summary["dropped"]:
        print(f"  Atılanlar  : {', '.join(summary['dropped'])}")
    print("\nSonraki adım (GGUF): python llama.cpp/convert_hf_to_gguf.py "
          f"{summary['out_dir']} --outtype f16  (ayrıntılar: EDGE.md)")


if __name__ == "__main__":
    main()
