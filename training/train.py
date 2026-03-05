# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Ana Eğitim Scripti
Komut satırından model eğitimi başlatmak için.
"""

import argparse
import sys
import os

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.config import ModelConfig, CONFIGS, detect_device
from model.transformer import ToprakLM
from model.tokenizer import ToprakTokenizer
from data.dataset import ToprakDataset, create_dataloader
from utils.validation import (
    validate_tokenizer, validate_dir_has_data,
    validate_checkpoint, validate_dataset_size,
    setup_error_handler, ToprakError,
)
from training.trainer import ToprakTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="🌱 Toprak — Türkçe Dil Modeli Eğitimi"
    )

    # Model
    parser.add_argument(
        "--model-size", type=str, default="small",
        choices=["small", "medium", "large", "xl"],
        help="Model boyutu: small (~80M), medium (~125M), large (~342M), xl (~941M)"
    )

    # Veri
    parser.add_argument(
        "--data-dir", type=str, default="data_cache/clean/train",
        help="Eğitim verisi dizini (varsayılan: data_cache/clean/train)"
    )
    parser.add_argument(
        "--eval-data-dir", type=str, default=None,
        help="Eval verisi dizini (opsiyonel)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="toprak_tokenizer.model",
        help="Tokenizer model dosyası"
    )

    # Eğitim
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)

    # Checkpoint
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Eğitime devam etmek için checkpoint dosyası"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Checkpoint kayıt dizini"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["mps", "cpu", "cuda"],
        help="Eğitim cihazı (varsayılan: otomatik algılama)"
    )

    # Optimizasyonlar
    parser.add_argument(
        "--no-compile", action="store_true",
        help="torch.compile() devre dışı bırak"
    )
    parser.add_argument(
        "--no-grad-checkpoint", action="store_true",
        help="Gradient checkpointing devre dışı bırak"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="TensorBoard log dizini"
    )

    # Ünlü Uyumu Loss
    parser.add_argument(
        "--vowel-harmony", action="store_true",
        help="Ünlü uyumu auxiliary loss aktifleştir (Türkçe dilbilgisi kaybı)"
    )
    parser.add_argument(
        "--vh-lambda", type=float, default=0.1,
        help="Ünlü uyumu loss ağırlığı (varsayılan: 0.1)"
    )
    parser.add_argument(
        "--vh-warmup-steps", type=int, default=1000,
        help="Ünlü uyumu loss warmup adım sayısı (varsayılan: 1000)"
    )

    # Morfolojik Ağırlıklı Kayıp
    parser.add_argument(
        "--morph-weight", action="store_true",
        help="Morfolojik ağırlıklı kayıp fonksiyonunu aktifleştir (ek tokenlerine yüksek ağırlık)"
    )
    parser.add_argument(
        "--morph-suffix-weight", type=float, default=1.3,
        help="Ek token'ları için kayıp ağırlığı (varsayılan: 1.3)"
    )
    parser.add_argument(
        "--morph-warmup-steps", type=int, default=500,
        help="Morfolojik ağırlık warmup adım sayısı (varsayılan: 500)"
    )

    return parser.parse_args()


def main():
    setup_error_handler()
    args = parse_args()

    print("🌱 Toprak — Türkçe Dil Modeli")
    print("=" * 50)

    # ─────────────────────────────────────────────
    # 1. Konfigürasyon
    # ─────────────────────────────────────────────
    config = CONFIGS[args.model_size]

    # CLI argümanları ile override
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.warmup_steps:
        config.warmup_steps = args.warmup_steps
    if args.grad_accum:
        config.grad_accum_steps = args.grad_accum
    if args.save_every:
        config.save_every = args.save_every
    if args.device:
        config.device = args.device
    else:
        config.device = detect_device()

    # Cihaz kontrolü
    if config.device == "mps" and not torch.backends.mps.is_available():
        print("⚠ MPS kullanılamıyor, CPU'ya geçiliyor...")
        config.device = "cpu"
    elif config.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA kullanılamıyor, CPU'ya geçiliyor...")
        config.device = "cpu"

    if config.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🚀 GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    print(f"\n📋 Konfigürasyon:")
    print(f"  Model:     Toprak {args.model_size.upper()} ({config.d_model}d, {config.num_layers}L, {config.num_heads}H/{config.num_kv_heads}KV)")
    print(f"  Vocab:     {config.vocab_size:,}")
    print(f"  Max Seq:   {config.max_seq_len}")
    print(f"  Device:    {config.device}")

    # ─────────────────────────────────────────────
    # 2. Tokenizer — dosya kontrolü
    # ─────────────────────────────────────────────
    validate_tokenizer(args.tokenizer)
    print(f"\n📝 Tokenizer yükleniyor: {args.tokenizer}")
    tokenizer = ToprakTokenizer(args.tokenizer)
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size: {config.vocab_size:,}")

    # ─────────────────────────────────────────────
    # 3. Dataset — veri kontrolü
    # ─────────────────────────────────────────────
    validate_dir_has_data(args.data_dir, description="Eğitim verisi dizini")
    print(f"\n📦 Veri yükleniyor: {args.data_dir}")
    train_dataset = ToprakDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        split="train",
    )
    validate_dataset_size(train_dataset, min_blocks=1, description="Eğitim verisi")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    eval_loader = None
    if args.eval_data_dir:
        validate_dir_has_data(args.eval_data_dir, description="Eval verisi dizini")
        eval_dataset = ToprakDataset(
            data_dir=args.eval_data_dir,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            split="eval",
            shuffle_docs=False,
        )
        eval_loader = create_dataloader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

    # ─────────────────────────────────────────────
    # 4. Resume checkpoint kontrolü
    # ─────────────────────────────────────────────
    if args.resume:
        validate_checkpoint(args.resume)

    # ─────────────────────────────────────────────
    # 5. Model
    # ─────────────────────────────────────────────
    model = ToprakLM(config)
    param_count = model.count_parameters()
    print(f"\n🧠 Model oluşturuldu: {param_count/1e6:.1f}M parametre")

    # ─────────────────────────────────────────────
    # 6. Ünlü Uyumu Loss (opsiyonel)
    # ─────────────────────────────────────────────
    vh_loss = None
    if args.vowel_harmony:
        from model.vowel_harmony import VowelHarmonyLoss
        vh_loss = VowelHarmonyLoss(
            tokenizer=tokenizer,
            lambda_weight=args.vh_lambda,
            warmup_steps=args.vh_warmup_steps,
        )

    # ─────────────────────────────────────────────
    # 6b. Morfolojik Ağırlıklı Kayıp (opsiyonel)
    # ─────────────────────────────────────────────
    morph_loss = None
    if args.morph_weight:
        from model.morph_weighting import MorphWeightedCELoss
        morph_loss = MorphWeightedCELoss(
            tokenizer=tokenizer,
            suffix_weight=args.morph_suffix_weight,
            warmup_steps=args.morph_warmup_steps,
        )

    # ─────────────────────────────────────────────
    # 7. Eğitim
    # ─────────────────────────────────────────────
    trainer = ToprakTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        checkpoint_dir=args.checkpoint_dir,
        use_compile=not args.no_compile,
        use_gradient_checkpointing=not args.no_grad_checkpoint,
        log_dir=args.log_dir,
        vowel_harmony_loss=vh_loss,
        morph_weight_loss=morph_loss,
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
