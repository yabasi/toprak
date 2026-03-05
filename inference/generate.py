# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Metin Üretimi
Top-k, top-p, temperature sampling ile autoregressive text generation.
KV Cache destekli hızlı inference.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from model.config import ModelConfig, TOPRAK_SMALL, detect_device
from model.transformer import ToprakLM
from model.tokenizer import ToprakTokenizer
from utils.validation import validate_checkpoint, validate_tokenizer, setup_error_handler


def generate_text(
    model: ToprakLM,
    tokenizer: ToprakTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 4,
    device: str = "mps",
) -> str:
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated = list(input_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    past_kvs = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if past_kvs is None:
                # Prefill: tüm prompt'u işle
                idx_input = input_tensor
            else:
                # Decode: sadece son token
                idx_input = input_tensor[:, -1:]

            # Forward pass — KV cache ile
            output = model(idx_input, past_kvs=past_kvs)
            logits, _, past_kvs = output
            logits = logits[:, -1, :]

            # Repetition penalty uygula
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # No-repeat ngram engelle
            if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size:
                for ngram_size in range(1, no_repeat_ngram_size + 1):
                    ngram = tuple(generated[-(ngram_size-1):]) if ngram_size > 1 else ()
                    for i in range(len(generated) - ngram_size + 1):
                        if tuple(generated[i:i+ngram_size-1]) == ngram:
                            banned_token = generated[i + ngram_size - 1]
                            logits[0, banned_token] = float('-inf')

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float('-inf')

            # Top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_logits[cumulative_probs > top_p] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            # Sample
            probs = torch.softmax(logits, dim=-1)
            probs = probs.clamp(min=0)  # negatif değerleri sıfırla
            if torch.isnan(probs).any() or (probs.sum() < 1e-8):
                probs = torch.ones_like(probs) / probs.size(-1)
            else:
                probs = probs / probs.sum()  # yeniden normalize et

            next_token = torch.multinomial(probs, num_samples=1)

            # EOS kontrolü
            if next_token.item() == tokenizer.eos_token_id:
                break

            generated.append(next_token.item())
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

    return tokenizer.decode(generated)


def load_model(
    checkpoint_path: str,
    device: str = "mps",
    config: ModelConfig = None,
) -> tuple:
    """
    Checkpoint'ten model yükle.

    Returns:
        (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if config is None:
        cfg_dict = checkpoint.get("config", {})
        config = ModelConfig(**cfg_dict) if cfg_dict else TOPRAK_SMALL

    config.device = device
    model = ToprakLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def main():
    setup_error_handler()
    import argparse

    parser = argparse.ArgumentParser(description="🌱 Toprak — Metin Üretimi")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint dosyası")
    parser.add_argument("--tokenizer", type=str, default="toprak_tokenizer.model",
                        help="Tokenizer model dosyası")
    parser.add_argument("--prompt", type=str, default="Türkiye'nin başkenti",
                        help="Başlangıç metni")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Üretilecek maks token sayısı")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None,
                        help="Cihaz (varsayılan: otomatik)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Kaç farklı çıktı üretilecek")

    args = parser.parse_args()

    # Dosya kontrolleri
    validate_checkpoint(args.checkpoint)
    validate_tokenizer(args.tokenizer)

    device = args.device or detect_device()

    print("🌱 Toprak — Metin Üretimi")
    print("=" * 50)

    # Model yükle
    model, config = load_model(args.checkpoint, device)
    tokenizer = ToprakTokenizer(args.tokenizer)

    print(f"  Model: {model.count_parameters()/1e6:.1f}M parametre")
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}, Top-p: {args.top_p}")
    print("=" * 50)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- Örnek {i+1} ---")

        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print(f"\n{text}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
