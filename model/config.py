# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak Model Configuration
Sıfırdan Türkçe Dil Modeli — Model Konfigürasyonu

Modern mimari: RMSNorm, SwiGLU, RoPE, GQA
Multi-device: MPS (Apple Silicon) / CUDA (NVIDIA) / CPU
"""

from dataclasses import dataclass
import torch


def detect_device() -> str:
    """Mevcut en iyi cihazı otomatik algıla."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelConfig:
    """Toprak LM model konfigürasyonu."""

    # Model mimarisi
    vocab_size: int = 32_000
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: int = 2          # GQA: KV head sayısı (< num_heads)
    num_layers: int = 12
    d_ff: int = 2048               # SwiGLU için: int(2/3 * 4 * d_model) → 8'in katı
    max_seq_len: int = 512
    dropout: float = 0.0           # Modern modellerde dropout kullanılmıyor

    # RoPE
    rope_theta: float = 10000.0    # RoPE base frekansı

    # RMSNorm
    norm_eps: float = 1e-6         # RMSNorm epsilon

    # Özel tokenler
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Cihaz — otomatik algılama
    device: str = "auto"

    # Eğitim parametreleri
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_steps: int = 100_000
    warmup_steps: int = 5000
    batch_size: int = 16
    grad_accum_steps: int = 4
    gradient_clip: float = 1.0

    # Checkpoint
    save_every: int = 5000
    keep_last_n: int = 3

    def __post_init__(self):
        """Cihaz otomatik algılama."""
        if self.device == "auto":
            self.device = detect_device()

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) num_heads'e ({self.num_heads}) tam bölünmeli"
        return self.d_model // self.num_heads

    @property
    def device_type(self) -> str:
        """autocast ve GradScaler için device tipi."""
        if self.device.startswith("cuda"):
            return "cuda"
        return self.device  # "mps" veya "cpu"


def _round_to_multiple(x: int, multiple: int = 8) -> int:
    """x'i verilen katına yuvarla (SwiGLU FFN boyutu için)."""
    return multiple * ((x + multiple - 1) // multiple)


# ── Hazır Konfigürasyonlar ──────────────────────────────────

# ~80M parametre — M4 Pro'da hızlı test
TOPRAK_SMALL = ModelConfig(
    vocab_size=32_000,
    d_model=640,
    num_heads=10,
    num_kv_heads=2,
    num_layers=14,
    d_ff=_round_to_multiple(int(2 / 3 * 4 * 640)),
    max_seq_len=512,
)

# ~125M parametre — M4 Pro'da orta seviye
TOPRAK_MEDIUM = ModelConfig(
    vocab_size=32_000,
    d_model=768,
    num_heads=12,
    num_kv_heads=4,
    num_layers=16,
    d_ff=_round_to_multiple(int(2 / 3 * 4 * 768)),
    max_seq_len=512,             # 1024→512: M4 Pro'da 2-3x hız artışı
    learning_rate=1e-4,
    batch_size=8,
    grad_accum_steps=4,          # 8→4: efektif batch 64→32, M4 Pro için dengeli
    warmup_steps=2000,           # 4000→2000: toplam step'e orantılı
    max_steps=100_000,           # 200K→100K: ilk iterasyon için yeterli
)

# ~350M parametre — M4 Pro'da eğitilebilir, RTX 4090'da hızlı
TOPRAK_LARGE = ModelConfig(
    vocab_size=32_000,
    d_model=1024,
    num_heads=16,
    num_kv_heads=4,
    num_layers=28,
    d_ff=_round_to_multiple(int(2 / 3 * 4 * 1024)),
    max_seq_len=2048,
    learning_rate=3e-4,
    batch_size=4,           # M4 Pro: 4, RTX 4090: 16-32
    grad_accum_steps=16,    # Efektif batch = 64
    warmup_steps=2000,
    max_steps=300_000,
)

# ~1B parametre — RTX 4090 için (M4 Pro'da sadece test/inference)
TOPRAK_XL = ModelConfig(
    vocab_size=32_000,
    d_model=1536,
    num_heads=16,
    num_kv_heads=4,
    num_layers=36,
    d_ff=_round_to_multiple(int(2 / 3 * 4 * 1536)),
    max_seq_len=2048,
    rope_theta=500000.0,    # Uzun context için yüksek theta
    learning_rate=3e-4,
    batch_size=2,           # M4 Pro: 1-2, RTX 4090: 8-16
    grad_accum_steps=32,    # Efektif batch = 64
    warmup_steps=2000,
    max_steps=500_000,
)

# Preset sözlüğü — kolay erişim
CONFIGS = {
    "small": TOPRAK_SMALL,
    "medium": TOPRAK_MEDIUM,
    "large": TOPRAK_LARGE,
    "xl": TOPRAK_XL,
}
