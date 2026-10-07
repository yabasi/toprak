# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Sohbet Arayüzü
Terminal tabanlı interaktif chat; SFT/DPO ile aynı sohbet şablonunu
(model/chat_template.py) kullanır ve geçmişi bağlam sınırına göre kırpar.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from model.chat_template import ChatTemplate
from model.config import ModelConfig, TOPRAK_SMALL, detect_device
from model.transformer import ToprakLM
from model.tokenizer import ToprakTokenizer
from inference.generate import load_model, generate_text
from utils.validation import validate_checkpoint, validate_tokenizer, setup_error_handler

# Prompt'a en az bu kadar konum bırak (çok büyük max_new_tokens'a karşı)
MIN_PROMPT_TOKENS = 16


def fit_history(
    template: ChatTemplate,
    messages: Sequence[Dict[str, str]],
    max_positions: int,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, str]], List[int], int]:
    """
    Sohbet geçmişini modelin pozisyon sınırına sığdır.

    Bütçe = max_positions - max_new_tokens. En eski turlar
    `template.truncate_history` ile atılır (sistem mesajı ve son kullanıcı
    mesajı korunur). Tek başına son mesaj bile sığmıyorsa prompt'un sonu
    (asistan rol öneki dahil) tutulup başı kırpılır; böylece RoPE tablosu
    hiçbir koşulda aşılmaz.

    Returns:
        (kırpılmış mesajlar, prompt token ID'leri, kullanılabilir max_new_tokens)
    """
    if max_positions <= 1:
        raise ValueError("max_positions en az 2 olmalı")
    max_new_tokens = max(1, min(max_new_tokens, max_positions - min(MIN_PROMPT_TOKENS, max_positions - 1)))
    budget = max_positions - max_new_tokens
    kept = template.truncate_history(messages, budget)
    prompt_ids = template.encode_prompt(kept)
    if len(prompt_ids) > budget:
        prompt_ids = [template.bos_id] + prompt_ids[-(budget - 1):] if budget > 1 else prompt_ids[-budget:]
    return kept, prompt_ids, max_new_tokens


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
    system_prompt: Optional[str] = None,
    logits_processors: Optional[List[Callable]] = None,
    input_fn: Callable[[str], str] = input,
):
    """
    İnteraktif sohbet arayüzü (sohbet şablonu ile).

    Geçmiş mesaj listesi olarak tutulur ve her turda modelin pozisyon
    sınırına (max pozisyon − max_new_tokens) sığacak şekilde en eski
    turlardan kırpılır.

    Kullanım:
        - Mesajınızı yazın ve Enter'a basın
        - 'çık' veya 'exit' yazarak çıkın
        - 'ayar' yazarak parametreleri değiştirin
        - 'temizle' yazarak geçmişi temizleyin

    Args:
        system_prompt: Sistem mesajı (ör. Toprak Anayasası özeti)
        logits_processors: generate_text'e aynen iletilen logit işlemcileri
        input_fn: Kullanıcı girdisi fonksiyonu (test için değiştirilebilir)
    """
    template = ChatTemplate(tokenizer, system_prompt=system_prompt)
    max_positions = model.freqs_cis.size(0)

    print("\n" + "=" * 60)
    print("  🌱 Toprak — Türkçe Sohbet")
    print("  Sıfırdan eğitilmiş Türkçe dil modeli")
    print("=" * 60)
    print("  Komutlar:")
    print("    çık / exit    — Çıkış")
    print("    ayar          — Parametreleri değiştir")
    print("    temizle       — Geçmişi temizle")
    if system_prompt:
        print(f"  Sistem: {system_prompt[:50]}{'…' if len(system_prompt) > 50 else ''}")
    print("=" * 60 + "\n")

    messages: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input_fn("🧑 Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Görüşmek üzere!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("çık", "exit", "quit", "q"):
            print("\n👋 Görüşmek üzere!")
            break

        if user_input.lower() == "temizle":
            messages = []
            print("✓ Sohbet geçmişi temizlendi.\n")
            continue

        if user_input.lower() == "ayar":
            try:
                print(f"  Mevcut: temp={temperature}, top_k={top_k}, top_p={top_p}, max={max_new_tokens}")
                t = input_fn("  Temperature (Enter=aynı): ").strip()
                if t:
                    temperature = float(t)
                k = input_fn("  Top-k (Enter=aynı): ").strip()
                if k:
                    top_k = int(k)
                p = input_fn("  Top-p (Enter=aynı): ").strip()
                if p:
                    top_p = float(p)
                m = input_fn("  Max tokens (Enter=aynı): ").strip()
                if m:
                    max_new_tokens = int(m)
                print(f"  ✓ Güncellendi: temp={temperature}, top_k={top_k}, top_p={top_p}, max={max_new_tokens}\n")
            except ValueError:
                print("  ⚠ Geçersiz değer.\n")
            continue

        messages.append({"role": "user", "content": user_input})
        kept, prompt_ids, new_tokens = fit_history(template, messages, max_positions, max_new_tokens)
        # Sistem mesajını geçmişte tutma; şablon her turda yeniden ekler
        messages = [m for m in kept if m["role"] != "system"] if system_prompt else kept

        print("🌱 Toprak: ", end="", flush=True)
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="",
            max_new_tokens=new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device,
            logits_processors=logits_processors,
            stop_ids=template.stop_ids,
            prompt_ids=prompt_ids,
            return_new_only=True,
        ).strip()
        print(response)
        messages.append({"role": "assistant", "content": response})
        print()


def main():
    setup_error_handler()
    import argparse

    parser = argparse.ArgumentParser(description="🌱 Toprak — Sohbet Arayüzü")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint dosyası")
    parser.add_argument("--tokenizer", type=str, default="toprak_tokenizer.model",
                        help="Tokenizer model dosyası")
    parser.add_argument("--device", type=str, default=None,
                        help="Cihaz (varsayılan: otomatik)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--no-repeat-ngram", type=int, default=4)
    parser.add_argument("--system", type=str, default=None,
                        help="Sistem mesajı (asistanın rolü ve kuralları)")
    parser.add_argument("--grammar-guard", type=str, default="off",
                        choices=["off", "mask", "penalty"],
                        help="Ünlü uyumu / ünsüz benzeşmesi korumalı üretim "
                             "(inference/grammar_guard.py)")
    parser.add_argument("--guard-penalty", type=float, default=5.0,
                        help="--grammar-guard penalty modunda logit cezası")

    args = parser.parse_args()

    # Dosya kontrolleri
    validate_checkpoint(args.checkpoint)
    validate_tokenizer(args.tokenizer)

    device = args.device or detect_device()

    # Model yükle
    print("Model yükleniyor...")
    model, config = load_model(args.checkpoint, device)
    tokenizer = ToprakTokenizer(args.tokenizer)
    print(f"✓ Model hazır: {model.count_parameters()/1e6:.1f}M parametre")

    logits_processors = None
    if args.grammar_guard != "off":
        from inference.grammar_guard import build_grammar_guard
        logits_processors = [build_grammar_guard(
            tokenizer, mode=args.grammar_guard, penalty=args.guard_penalty
        )]
        print(f"✓ Dilbilgisi koruması: {args.grammar_guard}")

    # Sohbet başlat
    chat(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        system_prompt=args.system,
        logits_processors=logits_processors,
    )


if __name__ == "__main__":
    main()
