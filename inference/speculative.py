# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Kendi Kendine Spekülatif Çözümleme (Self-Speculative Decoding)

Ayrı bir taslak (draft) modele gerek yoktur. Taslak tokenlar iki kaynaktan
gelebilir:

1. "mtp"   — Modelin Çoklu Token Tahmini başlıkları (num_mtp_heads > 0).
             Türkçe'de bir kelime çoğu zaman 3-6 tokendır; MTP başlıkları
             kelimenin geri kalan eklerini tek adımda önerir.
2. "ngram" — Prompt lookup: dizinin sonundaki n-gram daha önce geçtiyse,
             ardından gelen tokenlar taslak olarak önerilir. MTP'siz
             checkpoint'lerde de çalışır (RAG ve özetlemede çok etkilidir).

Doğrulama: [kesin token + K taslak] tek bir ileri geçişte KV cache ile
işlenir; ana modelin greedy tahminleriyle örtüşen en uzun önek kabul edilir,
KV cache reddedilen kısımdan kırpılır.

Garanti: greedy (temperature=0) modda çıktı, standart greedy çözümlemeyle
token token aynıdır — yalnız daha az ileri geçiş yapılır.
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


@dataclass
class SpeculativeStats:
    """Spekülatif çözümleme istatistikleri."""
    generated_tokens: int = 0
    forward_passes: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    elapsed_sec: float = 0.0
    accepted_per_round: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0

    @property
    def tokens_per_forward(self) -> float:
        return self.generated_tokens / self.forward_passes if self.forward_passes else 0.0

    def as_dict(self) -> dict:
        return {
            "generated_tokens": self.generated_tokens,
            "forward_passes": self.forward_passes,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "tokens_per_forward": round(self.tokens_per_forward, 4),
            "elapsed_sec": round(self.elapsed_sec, 4),
        }


def truncate_kv_cache(past_kvs, length: int):
    """KV cache'i ilk `length` pozisyona kırp."""
    return [(k[:, :, :length], v[:, :, :length]) for k, v in past_kvs]


def ngram_draft(tokens: Sequence[int], num_draft: int, max_ngram: int = 4, min_ngram: int = 1) -> List[int]:
    """
    Prompt lookup taslağı: dizinin sonundaki n-gram'ın önceki son geçişini
    bul ve onu izleyen tokenları öner (uzun n-gram'lar önce denenir).
    """
    n_tokens = len(tokens)
    for n in range(min(max_ngram, n_tokens - 1), min_ngram - 1, -1):
        suffix = list(tokens[-n:])
        for start in range(n_tokens - n - 1, -1, -1):
            if list(tokens[start:start + n]) == suffix:
                follow = list(tokens[start + n:start + n + num_draft])
                if follow:
                    return follow
    return []


@torch.no_grad()
def greedy_generate(model, input_ids: List[int], max_new_tokens: int, stop_ids: Sequence[int] = ()) -> List[int]:
    """Referans greedy çözümleme (KV cache ile, token başına bir ileri geçiş)."""
    device = model.freqs_cis.device
    model.eval()
    tokens = list(input_ids)
    out = []
    past = None
    step_input = torch.tensor([tokens], device=device)
    for _ in range(max_new_tokens):
        logits, _, past = model(step_input, past_kvs=past)
        nxt = int(logits[0, -1].argmax())
        if nxt in stop_ids:
            break
        out.append(nxt)
        step_input = torch.tensor([[nxt]], device=device)
    return out


@torch.no_grad()
def speculative_generate(
    model,
    input_ids: List[int],
    max_new_tokens: int = 200,
    num_draft: Optional[int] = None,
    draft_source: str = "auto",
    stop_ids: Sequence[int] = (),
    max_ngram: int = 4,
) -> tuple:
    """
    Greedy spekülatif çözümleme.

    Args:
        model: ToprakLM
        input_ids: prompt token ID'leri (BOS dahil)
        max_new_tokens: üretilecek maksimum token
        num_draft: tur başına taslak token sayısı (MTP'de en fazla num_mtp_heads)
        draft_source: "mtp", "ngram" veya "auto" (MTP varsa mtp, yoksa ngram)
        stop_ids: üretimi durduran token ID'leri (EOS, tur sonu)

    Returns:
        (yeni_tokenlar, SpeculativeStats)
    """
    model.eval()
    device = model.freqs_cis.device
    num_heads = len(model.mtp_heads)
    if draft_source == "auto":
        draft_source = "mtp" if num_heads > 0 else "ngram"
    if draft_source == "mtp" and num_heads == 0:
        raise ValueError("draft_source='mtp' için modelin num_mtp_heads > 0 olmalı")
    if draft_source not in ("mtp", "ngram"):
        raise ValueError(f"Bilinmeyen draft_source: {draft_source!r}")
    if num_draft is None:
        num_draft = num_heads if draft_source == "mtp" else 4
    if draft_source == "mtp":
        num_draft = min(num_draft, num_heads)

    max_positions = model.freqs_cis.size(0)
    stats = SpeculativeStats()
    start = time.time()
    stop_ids = set(stop_ids)

    tokens = list(input_ids)
    generated: List[int] = []

    # Prefill
    logits, _, past, hidden = model(torch.tensor([tokens], device=device), return_hidden=True)
    stats.forward_passes += 1
    next_token = int(logits[0, -1].argmax())
    last_hidden = hidden[:, -1:]

    while len(generated) < max_new_tokens:
        if next_token in stop_ids:
            break

        # Taslak üret
        if draft_source == "mtp":
            drafts = [int(l[0, -1].argmax()) for l in model.mtp_logits(last_hidden)[:num_draft]]
        else:
            drafts = ngram_draft(tokens + [next_token], num_draft, max_ngram=max_ngram)

        budget = max_new_tokens - len(generated) - 1
        room = max_positions - len(tokens) - 1
        drafts = drafts[:max(0, min(budget, room))]
        candidate = [next_token] + drafts
        if len(tokens) + len(candidate) > max_positions:
            break

        cache_len = len(tokens)
        logits, _, past, hidden = model(
            torch.tensor([candidate], device=device), past_kvs=past, return_hidden=True,
        )
        stats.forward_passes += 1
        predictions = logits[0].argmax(dim=-1).tolist()   # predictions[i] → candidate[i]'den sonra

        accepted = 0
        for i, draft in enumerate(drafts):
            if predictions[i] != draft or candidate[i] in stop_ids:
                break
            accepted += 1

        stats.drafted_tokens += len(drafts)
        stats.accepted_tokens += accepted
        stats.accepted_per_round.append(accepted)

        new_tokens = candidate[:accepted + 1]
        finished = False
        for tok in new_tokens:
            if tok in stop_ids:
                finished = True
                break
            generated.append(tok)
            tokens.append(tok)
            if len(generated) >= max_new_tokens:
                finished = True
                break
        if finished:
            break

        past = truncate_kv_cache(past, cache_len + accepted + 1)
        next_token = predictions[accepted]
        last_hidden = hidden[:, accepted:accepted + 1]

    stats.generated_tokens = len(generated)
    stats.elapsed_sec = time.time() - start
    return generated, stats


def main():
    import argparse
    from inference.generate import load_model
    from model.config import detect_device
    from model.tokenizer import ToprakTokenizer
    from utils.validation import validate_checkpoint, validate_tokenizer, setup_error_handler

    setup_error_handler()
    parser = argparse.ArgumentParser(description="🌱 Toprak — Spekülatif Çözümleme")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/toprak_last.pt")
    parser.add_argument("--tokenizer", type=str, default="toprak_tokenizer.model")
    parser.add_argument("--prompt", type=str, default="Türkiye'nin başkenti")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--num-draft", type=int, default=None)
    parser.add_argument("--draft-source", choices=["auto", "mtp", "ngram"], default="auto")
    parser.add_argument("--compare", action="store_true",
                        help="Standart greedy ile hız ve çıktı eşitliğini karşılaştır")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    validate_checkpoint(args.checkpoint)
    validate_tokenizer(args.tokenizer)
    device = args.device or detect_device()
    model, _ = load_model(args.checkpoint, device)
    tokenizer = ToprakTokenizer(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)

    out, stats = speculative_generate(
        model, prompt_ids, args.max_tokens, args.num_draft, args.draft_source,
        stop_ids=(tokenizer.eos_token_id,),
    )
    print(tokenizer.decode(prompt_ids + out))
    print("\n📊", stats.as_dict())

    if args.compare:
        t0 = time.time()
        ref = greedy_generate(model, prompt_ids, args.max_tokens, stop_ids=(tokenizer.eos_token_id,))
        ref_time = time.time() - t0
        print(f"  Greedy referans: {ref_time:.3f}s, spekülatif: {stats.elapsed_sec:.3f}s, "
              f"hızlanma: {ref_time / max(stats.elapsed_sec, 1e-9):.2f}x, aynı çıktı: {ref == out}")


if __name__ == "__main__":
    main()
