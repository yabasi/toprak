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

from model.config import ModelConfig, TOPRAK_SMALL, TOPRAK_MEDIUM
from model.transformer import ToprakGPT
from model.tokenizer import ToprakTokenizer
from data.dataset import ToprakDataset, create_dataloader
from training.trainer import ToprakTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="🌱 Toprak — Türkçe Dil Modeli Eğitimi"
    )

    # Model
    parser.add_argument(
        "--model-size", type=str, default="small",
        choices=["small", "medium"],
        help="Model boyutu: small (85M) veya medium (125M)"
    )

    # Veri
    parser.add_argument(
        "--data-dir", type=str, default="data_cache/clean",
        help="Eğitim verisi dizini"
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
        help="Eğitim cihazı"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("🌱 Toprak — Türkçe Dil Modeli")
    print("=" * 50)

    # ─────────────────────────────────────────────
    # 1. Konfigürasyon
    # ─────────────────────────────────────────────
    if args.model_size == "small":
        config = TOPRAK_SMALL
    else:
        config = TOPRAK_MEDIUM

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

    # MPS kontrolü
    if config.device == "mps" and not torch.backends.mps.is_available():
        print("⚠ MPS kullanılamıyor, CPU'ya geçiliyor...")
        config.device = "cpu"

    print(f"\n📋 Konfigürasyon:")
    print(f"  Model:     Toprak {args.model_size} ({config.d_model}d, {config.num_layers}L)")
    print(f"  Vocab:     {config.vocab_size:,}")
    print(f"  Max Seq:   {config.max_seq_len}")
    print(f"  Device:    {config.device}")

    # ─────────────────────────────────────────────
    # 2. Tokenizer
    # ─────────────────────────────────────────────
    print(f"\n📝 Tokenizer yükleniyor: {args.tokenizer}")
    tokenizer = ToprakTokenizer(args.tokenizer)
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size: {config.vocab_size:,}")

    # ─────────────────────────────────────────────
    # 3. Dataset
    # ─────────────────────────────────────────────
    print(f"\n📦 Veri yükleniyor: {args.data_dir}")
    train_dataset = ToprakDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        split="train",
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    eval_loader = None
    if args.eval_data_dir:
        eval_dataset = ToprakDataset(
            data_dir=args.eval_data_dir,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            split="eval",
        )
        eval_loader = create_dataloader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

    # ─────────────────────────────────────────────
    # 4. Model
    # ─────────────────────────────────────────────
    model = ToprakGPT(config)
    param_count = model.count_parameters()
    print(f"\n🧠 Model oluşturuldu: {param_count/1e6:.1f}M parametre")

    # ─────────────────────────────────────────────
    # 5. Eğitim
    # ─────────────────────────────────────────────
    trainer = ToprakTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
