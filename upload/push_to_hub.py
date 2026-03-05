# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — HuggingFace Hub Entegrasyonu
Model ve tokenizer'ı HuggingFace'e yükle.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import PreTrainedModel, PretrainedConfig

from model.config import ModelConfig, TOPRAK_SMALL
from model.transformer import ToprakLM


class ToprakConfig(PretrainedConfig):
    """HuggingFace uyumlu Toprak konfigürasyonu."""

    model_type = "toprak_lm"

    def __init__(
        self,
        vocab_size=32000,
        d_model=512,
        num_heads=8,
        num_kv_heads=2,
        num_layers=12,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.0,
        rope_theta=10000.0,
        norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def to_model_config(self) -> ModelConfig:
        """ModelConfig'e dönüştür."""
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            rope_theta=self.rope_theta,
            norm_eps=self.norm_eps,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )


class ToprakForCausalLM(PreTrainedModel):
    """HuggingFace uyumlu Toprak CausalLM wrapper."""

    config_class = ToprakConfig

    def __init__(self, config: ToprakConfig):
        super().__init__(config)
        model_config = config.to_model_config()
        self.model = ToprakLM(model_config)

    def forward(self, input_ids, labels=None, **kwargs):
        logits, loss, _ = self.model(input_ids, targets=labels)

        # HuggingFace formatına uygun output
        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    def generate(self, input_ids, max_new_tokens=100, **kwargs):
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


def push_to_hub(
    checkpoint_path: str,
    repo_name: str,
    tokenizer_model_path: str = "toprak_tokenizer.model",
    device: str = "cpu",
):
    """
    Model ve tokenizer'ı HuggingFace Hub'a yükle.

    Args:
        checkpoint_path: PyTorch checkpoint dosyası
        repo_name: HuggingFace repo adı (ör: 'kullanici/toprak-v1')
        tokenizer_model_path: SentencePiece model dosyası
        device: Yükleme cihazı
    """
    from huggingface_hub import HfApi

    print("🌱 Toprak — HuggingFace'e Yükleme")
    print("=" * 50)

    # 1. Checkpoint yükle
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = checkpoint.get("config", {})

    # 2. HuggingFace config oluştur
    hf_config = ToprakConfig(**cfg_dict)

    # 3. Model oluştur ve ağırlıkları yükle
    model = ToprakForCausalLM(hf_config)
    model.model.load_state_dict(checkpoint["model_state_dict"])

    step = checkpoint.get("global_step", "?")
    params = model.model.count_parameters()
    print(f"  Checkpoint: step {step}")
    print(f"  Parametreler: {params/1e6:.1f}M")

    # 4. HuggingFace'e push
    print(f"\n  Yükleniyor: {repo_name}")

    model.push_to_hub(
        repo_name,
        commit_message=f"Toprak v1 — step {step}, {params/1e6:.0f}M params",
    )

    # 5. Tokenizer dosyasını da yükle
    api = HfApi()
    api.upload_file(
        path_or_fileobj=tokenizer_model_path,
        path_in_repo="toprak_tokenizer.model",
        repo_id=repo_name,
        commit_message="Toprak BPE tokenizer",
    )

    print(f"\n✅ Başarıyla yüklendi!")
    print(f"   https://huggingface.co/{repo_name}")


# Model kartı şablonu
MODEL_CARD_TEMPLATE = """---
language:
  - tr
license: mit
tags:
  - turkish
  - transformer
  - text-generation
  - from-scratch
pipeline_tag: text-generation
---

# 🌱 Toprak — Türkçe Dil Modeli

Sıfırdan eğitilmiş, tamamen özgün bir Türkçe dil modeli.

## Model Bilgileri

| Özellik | Değer |
|---|---|
| Mimari | Decoder-only Transformer |
| Parametreler | {param_count} |
| Vocab Size | {vocab_size} |
| Context Length | {max_seq_len} |
| Eğitim Verisi | Türkçe web verisi |

## Kullanım

```python
from model.transformer import ToprakLM
from model.tokenizer import ToprakTokenizer

tokenizer = ToprakTokenizer("toprak_tokenizer.model")
model = ToprakLM.from_pretrained("repo-adi/toprak-v1")
```

## Limitasyonlar

- Bu bir araştırma projesidir, ticari kullanım için tasarlanmamıştır.
- Model hallüsinasyon yapabilir ve hatalı bilgi üretebilir.
- Eğitim verisi sınırlıdır, daha fazla veri ile performans artacaktır.

## Lisans

MIT License
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toprak — HuggingFace'e Yükle")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace repo adı (ör: kullanici/toprak-v1)")
    parser.add_argument("--tokenizer", type=str, default="toprak_tokenizer.model")

    args = parser.parse_args()
    push_to_hub(args.checkpoint, args.repo, args.tokenizer)
