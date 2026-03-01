"""
Toprak Model Configuration
Sıfırdan Türkçe Dil Modeli — Model Konfigürasyonu
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Toprak LM model konfigürasyonu."""

    # Model mimarisi
    vocab_size: int = 32_000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1

    # Özel tokenler
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Cihaz — Apple Silicon GPU
    device: str = "mps"

    # Eğitim parametreleri
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_steps: int = 100_000
    warmup_steps: int = 5000
    batch_size: int = 16
    grad_accum_steps: int = 4
    gradient_clip: float = 0.5

    # Checkpoint
    save_every: int = 5000
    keep_last_n: int = 3

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) num_heads'e ({self.num_heads}) tam bölünmeli"
        return self.d_model // self.num_heads


# Hazır konfigürasyonlar (~85M parametre)
TOPRAK_SMALL = ModelConfig(
    vocab_size=32_000,
    d_model=640,
    num_heads=10,
    num_layers=14,
    d_ff=2560,
    max_seq_len=512,
)

# (~125M parametre)
TOPRAK_MEDIUM = ModelConfig(
    vocab_size=32_000,
    d_model=768,
    num_heads=12,
    num_layers=16,
    d_ff=3072,
    max_seq_len=1024,
    learning_rate=1e-4,
    batch_size=8,
    grad_accum_steps=8,
    warmup_steps=4000,
    max_steps=200_000,
)
