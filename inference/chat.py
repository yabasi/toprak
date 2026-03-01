"""
Toprak — Sohbet Arayüzü
Terminal tabanlı interaktif chat.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.config import ModelConfig, TOPRAK_SMALL
from model.transformer import ToprakLM
from model.tokenizer import ToprakTokenizer
from inference.generate import load_model, generate_text


def chat(
    model: ToprakLM,
    tokenizer: ToprakTokenizer,
    device: str = "mps",
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty=1.3,
    no_repeat_ngram_size=4,
):
    """
    İnteraktif sohbet arayüzü.

    Kullanım:
        - Mesajınızı yazın ve Enter'a basın
        - 'çık' veya 'exit' yazarak çıkın
        - 'ayar' yazarak parametreleri değiştirin
        - 'temizle' yazarak geçmişi temizleyin
    """
    print("\n" + "=" * 60)
    print("  🌱 Toprak — Türkçe Sohbet")
    print("  Sıfırdan eğitilmiş Türkçe dil modeli")
    print("=" * 60)
    print("  Komutlar:")
    print("    çık / exit    — Çıkış")
    print("    ayar          — Parametreleri değiştir")
    print("    temizle       — Geçmişi temizle")
    print("=" * 60 + "\n")

    conversation_history = ""

    while True:
        try:
            user_input = input("🧑 Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Görüşmek üzere!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("çık", "exit", "quit", "q"):
            print("\n👋 Görüşmek üzere!")
            break

        if user_input.lower() == "temizle":
            conversation_history = ""
            print("✓ Sohbet geçmişi temizlendi.\n")
            continue

        if user_input.lower() == "ayar":
            try:
                print(f"  Mevcut: temp={temperature}, top_k={top_k}, top_p={top_p}, max={max_new_tokens}")
                t = input("  Temperature (Enter=aynı): ").strip()
                if t:
                    temperature = float(t)
                k = input("  Top-k (Enter=aynı): ").strip()
                if k:
                    top_k = int(k)
                p = input("  Top-p (Enter=aynı): ").strip()
                if p:
                    top_p = float(p)
                m = input("  Max tokens (Enter=aynı): ").strip()
                if m:
                    max_new_tokens = int(m)
                print(f"  ✓ Güncellendi: temp={temperature}, top_k={top_k}, top_p={top_p}, max={max_new_tokens}\n")
            except ValueError:
                print("  ⚠ Geçersiz değer.\n")
            continue

        # Prompt oluştur
        prompt = conversation_history + user_input

        # Metin üret
        print("🌱 Toprak: ", end="", flush=True)

        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device,
        )

        # Prompt'u response'tan çıkar
        if response.startswith(prompt):
            response = response[len(prompt):]

        response = response.strip()
        print(response)

        # Geçmişe ekle
        conversation_history = prompt + " " + response + " "

        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="🌱 Toprak — Sohbet Arayüzü")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint dosyası")
    parser.add_argument("--tokenizer", type=str, default="toprak_tokenizer.model",
                        help="Tokenizer model dosyası")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--no-repeat-ngram", type=int, default=4)

    args = parser.parse_args()

    # Model yükle
    print("Model yükleniyor...")
    model, config = load_model(args.checkpoint, args.device)
    tokenizer = ToprakTokenizer(args.tokenizer)
    print(f"✓ Model hazır: {model.count_parameters()/1e6:.1f}M parametre")

    # Sohbet başlat
    chat(
    model=model,
    tokenizer=tokenizer,
    device=args.device,
    max_new_tokens=args.max_tokens,
    temperature=args.temperature,
    top_k=args.top_k,
    top_p=args.top_p,
    repetition_penalty=args.repetition_penalty,
    no_repeat_ngram_size=args.no_repeat_ngram,
)


if __name__ == "__main__":
    main()
