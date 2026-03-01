"""
Toprak — Değerlendirme
Perplexity hesaplama ve model değerlendirmesi.
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from model.config import ModelConfig, TOPRAK_SMALL
from model.transformer import ToprakLM
from model.tokenizer import ToprakTokenizer
from data.dataset import ToprakDataset, create_dataloader


@torch.no_grad()
def compute_perplexity(
    model: ToprakLM,
    dataloader,
    device: str = "mps",
    max_batches: int = 200,
) -> float:
    """
    Perplexity hesapla.

    Perplexity = exp(average cross-entropy loss)
    Düşük perplexity = daha iyi model.

    Args:
        model: ToprakLM modeli
        dataloader: Eval DataLoader
        device: Hesaplama cihazı
        max_batches: Maksimum batch sayısı

    Returns:
        Perplexity değeri
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Eval", total=max_batches):
        if num_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits, loss = model(input_ids, targets=labels)

        # Pad tokenlerini sayma
        non_pad = (labels != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
        num_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    return perplexity


def evaluate_model(
    checkpoint_path: str,
    eval_data_dir: str,
    tokenizer_path: str,
    config: ModelConfig = None,
    device: str = "mps",
):
    """
    Bir checkpoint'i değerlendir.

    Args:
        checkpoint_path: Checkpoint dosyası
        eval_data_dir: Eval verisi dizini
        tokenizer_path: Tokenizer model dosyası
        config: Model konfigürasyonu
        device: Cihaz
    """
    print("🌱 Toprak — Model Değerlendirmesi")
    print("=" * 50)

    # Tokenizer
    tokenizer = ToprakTokenizer(tokenizer_path)

    # Config
    if config is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg_dict = checkpoint.get("config", {})
        config = ModelConfig(**cfg_dict) if cfg_dict else TOPRAK_SMALL

    config.device = device

    # Model
    model = ToprakLM(config).to(device)

    # Checkpoint yükle
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    step = checkpoint.get("global_step", "?")
    print(f"  Checkpoint: step {step}")
    print(f"  Parametreler: {model.count_parameters()/1e6:.1f}M")

    # Dataset
    eval_dataset = ToprakDataset(
        data_dir=eval_data_dir,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        split="eval",
    )
    eval_loader = create_dataloader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Perplexity
    ppl = compute_perplexity(model, eval_loader, device)

    print(f"\n{'='*50}")
    print(f"  📊 Perplexity: {ppl:.2f}")
    if ppl < 50:
        print(f"  ✅ Hedef perplexity (<50) başarıldı!")
    elif ppl < 100:
        print(f"  🟡 İyi yoldasın, devam et.")
    else:
        print(f"  🔴 Daha fazla veri ve eğitim gerekli.")
    print(f"{'='*50}")

    return ppl


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toprak — Model Değerlendirmesi")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="toprak_tokenizer.model")
    parser.add_argument("--device", type=str, default="mps")

    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.eval_data, args.tokenizer, device=args.device)
