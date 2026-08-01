# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

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
from data.dataset import ToprakDataset, ToprakShardDataset, create_dataloader
from utils.validation import (
    validate_tokenizer, validate_dir_has_data,
    validate_checkpoint, validate_dataset_size,
    setup_error_handler, ToprakError,
)
from training.trainer import ToprakTrainer
from utils.reproducibility import (
    build_experiment_manifest,
    seed_everything,
    write_manifest,
)


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
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Model başlangıcı ve veri sırası için deney seed'i (varsayılan: 42)"
    )
    parser.add_argument(
        "--experiment-name", type=str, default="default",
        help="Checkpoint ve ablation raporlarında saklanan deney kimliği"
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Deterministik PyTorch algoritmalarını zorunlu kıl"
    )
    parser.add_argument(
        "--data-fingerprint", choices=["auto", "manifest", "full", "metadata", "off"],
        default="auto",
        help="Deney manifestindeki veri parmak izi yöntemi (varsayılan: auto)"
    )

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
    parser.add_argument(
        "--bin-mode", action="store_true",
        help="Pre-tokenize edilmiş .bin shard'larını kullan (data-dir manifest.json içermeli)"
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="CUDA üzerinde bfloat16 mixed precision kullan (A100/H100 önerilir; fp16 yerine)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader worker sayısı (bin-mode'da 2-4 önerilir)"
    )
    parser.add_argument(
        "--verify-data-hashes", action="store_true",
        help="Bin shard SHA-256 değerlerini eğitimden önce doğrula"
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

    # Ünsüz Benzeşmesi Loss
    parser.add_argument(
        "--consonant-harmony", action="store_true",
        help="Ünsüz benzeşmesi auxiliary loss aktifleştir (Türkçe dilbilgisi kaybı)"
    )
    parser.add_argument(
        "--ch-lambda", type=float, default=0.1,
        help="Ünsüz benzeşmesi loss ağırlığı (varsayılan: 0.1)"
    )
    parser.add_argument(
        "--ch-warmup-steps", type=int, default=1000,
        help="Ünsüz benzeşmesi loss warmup adım sayısı (varsayılan: 1000)"
    )

    # Morfolojik Başlık (Auxiliary POS/Boundary Head)
    parser.add_argument(
        "--morph-head", action="store_true",
        help="Morfolojik sınır ve POS çoklu görev başlığını aktifleştir"
    )
    parser.add_argument(
        "--mh-lambda", type=float, default=0.2,
        help="Morfolojik başlık loss ağırlığı (varsayılan: 0.2)"
    )

    # Hece ve Kafiye Kaybı (Syllable & Rhyme Loss)
    parser.add_argument(
        "--syllable-rhyme", action="store_true",
        help="Hece ve kafiye auxiliary loss aktifleştir (Türkçe şiirsel kısıtlar)"
    )
    parser.add_argument(
        "--sr-lambda-syllable", type=float, default=0.1,
        help="Hece ölçüsü loss ağırlığı (varsayılan: 0.1)"
    )
    parser.add_argument(
        "--sr-lambda-rhyme", type=float, default=0.1,
        help="Kafiye loss ağırlığı (varsayılan: 0.1)"
    )
    parser.add_argument(
        "--sr-warmup-steps", type=int, default=1000,
        help="Hece ve kafiye loss warmup adım sayısı (varsayılan: 1000)"
    )

    return parser.parse_args()


def build_training_recipe(args, config) -> dict:
    """Checkpoint'e yazılacak, ablation karşılaştırmasına uygun eğitim tarifi."""
    return {
        "experiment_name": args.experiment_name,
        "model_size": args.model_size,
        "data_dir": os.path.abspath(args.data_dir),
        "eval_data_dir": (
            os.path.abspath(args.eval_data_dir) if args.eval_data_dir else None
        ),
        "bin_mode": args.bin_mode,
        "seed": args.seed,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "max_steps": config.max_steps,
        "warmup_steps": config.warmup_steps,
        "grad_accum_steps": config.grad_accum_steps,
        "save_every": config.save_every,
        "device": config.device,
        "num_workers": args.num_workers,
        "bf16": args.bf16,
        "compile": not args.no_compile,
        "gradient_checkpointing": not args.no_grad_checkpoint,
        "deterministic": args.deterministic,
        "data_fingerprint_mode": args.data_fingerprint,
        "verify_data_hashes": args.verify_data_hashes,
        "auxiliary_losses": {
            "vowel_harmony": {
                "enabled": args.vowel_harmony,
                "lambda": args.vh_lambda,
                "warmup_steps": args.vh_warmup_steps,
            },
            "morph_weight": {
                "enabled": args.morph_weight,
                "suffix_weight": args.morph_suffix_weight,
                "warmup_steps": args.morph_warmup_steps,
            },
            "consonant_harmony": {
                "enabled": args.consonant_harmony,
                "lambda": args.ch_lambda,
                "warmup_steps": args.ch_warmup_steps,
            },
            "morph_head": {
                "enabled": args.morph_head,
                "lambda": args.mh_lambda,
            },
            "syllable_rhyme": {
                "enabled": args.syllable_rhyme,
                "lambda_syllable": args.sr_lambda_syllable,
                "lambda_rhyme": args.sr_lambda_rhyme,
                "warmup_steps": args.sr_warmup_steps,
            },
        },
    }


def main():
    setup_error_handler()
    args = parse_args()

    if not args.experiment_name.strip():
        raise ValueError("--experiment-name boş olamaz")
    seed_everything(args.seed, deterministic=args.deterministic)

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
    print(f"\n📦 Veri yükleniyor: {args.data_dir}")

    if args.bin_mode:
        # Pre-tokenized shard mode (önerilir, büyük korpuslar için)
        if not os.path.exists(os.path.join(args.data_dir, "manifest.json")):
            raise FileNotFoundError(
                f"--bin-mode için {args.data_dir}/manifest.json gerekli. "
                f"Önce: python scripts/pretokenize.py --input-dir <jsonl_dir> "
                f"--tokenizer {args.tokenizer} --output-dir {args.data_dir}"
            )
        train_dataset = ToprakShardDataset(
            bin_dir=args.data_dir,
            split="train",
            max_seq_len=config.max_seq_len,
            shuffle_shards=False,  # curriculum'u koru
            expected_vocab_size=config.vocab_size,
            seed=args.seed,
            verify_hashes=args.verify_data_hashes,
        )
        # Eval split aynı bin_dir altında manifest'te tanımlı
        try:
            eval_dataset = ToprakShardDataset(
                bin_dir=args.data_dir,
                split="eval",
                max_seq_len=config.max_seq_len,
                expected_vocab_size=config.vocab_size,
                verify_hashes=args.verify_data_hashes,
            )
        except RuntimeError as e:
            print(f"  ⚠ Eval shard'ları yüklenemedi: {e}")
            eval_dataset = None

        # Curriculum manifestleri kalite sırasını blok seviyesinde taşır.
        # Global shuffle bu sırayı tamamen bozacağından yalnız normal
        # shard setlerinde karıştırma yapılır.
        shuffle_train = not train_dataset.curriculum
        if train_dataset.curriculum:
            print("  ✓ Curriculum sırası korunuyor (DataLoader shuffle kapalı)")
        train_loader = create_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=shuffle_train,
            num_workers=args.num_workers,
            pin_memory=(config.device == "cuda"),
            seed=args.seed,
        )
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = create_dataloader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=max(1, args.num_workers // 2),
                pin_memory=(config.device == "cuda"),
                drop_last=False,
                seed=args.seed,
            )
    else:
        # JSONL mode (geriye uyumlu)
        validate_dir_has_data(args.data_dir, description="Eğitim verisi dizini")
        train_dataset = ToprakDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            split="train",
            seed=args.seed,
        )
        validate_dataset_size(train_dataset, min_blocks=1, description="Eğitim verisi")

        train_loader = create_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            seed=args.seed,
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
                drop_last=False,
                seed=args.seed,
            )

    # ─────────────────────────────────────────────
    # 4. Resume checkpoint kontrolü
    # ─────────────────────────────────────────────
    if args.resume:
        validate_checkpoint(args.resume)

    # ─────────────────────────────────────────────
    # 5. Model
    # ─────────────────────────────────────────────
    model = ToprakLM(config, tokenizer=tokenizer)
    if args.morph_head:
        model.use_morph_head = True
        model.morph_lambda = args.mh_lambda
        print(f"  ✓ Morfolojik Başlık aktif (λ={args.mh_lambda})")

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
    # 6c. Ünsüz Benzeşmesi Loss (opsiyonel)
    # ─────────────────────────────────────────────
    ch_loss = None
    if args.consonant_harmony:
        from model.consonant_harmony import ConsonantHarmonyLoss
        ch_loss = ConsonantHarmonyLoss(
            tokenizer=tokenizer,
            lambda_weight=args.ch_lambda,
            warmup_steps=args.ch_warmup_steps,
        )

    # ─── 6d. Hece ve Kafiye Loss (opsiyonel) ───
    sr_loss = None
    if args.syllable_rhyme:
        from model.syllable_rhyme import SyllableRhymeLoss
        sr_loss = SyllableRhymeLoss(
            tokenizer=tokenizer,
            lambda_syllable=args.sr_lambda_syllable,
            lambda_rhyme=args.sr_lambda_rhyme,
            warmup_steps=args.sr_warmup_steps,
        )

    # ─────────────────────────────────────────────
    # 7. Eğitim
    # ─────────────────────────────────────────────
    training_recipe = build_training_recipe(args, config)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n🔐 Deney parmak izi hazırlanıyor ({args.data_fingerprint})...")
    experiment_manifest = build_experiment_manifest(
        project_root=project_root,
        training_recipe=training_recipe,
        tokenizer_path=args.tokenizer,
        data_path=args.data_dir,
        data_fingerprint_mode=args.data_fingerprint,
        argv=sys.argv,
    )
    manifest_paths = write_manifest(
        experiment_manifest, args.checkpoint_dir, args.log_dir
    )
    print(f"  ✓ Deney manifesti: {manifest_paths[0]}")

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
        consonant_harmony_loss=ch_loss,
        syllable_rhyme_loss=sr_loss,
        use_bf16=args.bf16,
        training_recipe=training_recipe,
        experiment_manifest=experiment_manifest,
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
