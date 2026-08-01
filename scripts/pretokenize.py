# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Pre-tokenization Pipeline

Temizlenmiş JSONL dosyalarını sabit boyutlu .bin shard'larına dönüştürür.
Eğitim sırasında JSON parse + tokenize maliyeti ortadan kalkar → I/O hızı +%20-40.

Format:
    - uint16 numpy memmap (vocab <= 65535 olduğu için yeterli)
    - Her shard: train_00000.bin, train_00001.bin, ... + eval_00000.bin
    - Manifest: data_bin/manifest.json (shard listesi + token sayıları)

Curriculum:
    --curriculum açıksa, Wikipedia dokümanları son shard'lara yerleştirilir
    (annealing — eğitim sonunda yüksek kaliteli veriye geçiş için).

Kullanım:
    python scripts/pretokenize.py \
        --input-dir data_raw/clean \
        --tokenizer toprak_tokenizer.model \
        --output-dir data_bin \
        --shard-size 100000000 \
        --eval-ratio 0.005 \
        --curriculum
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from model.tokenizer import ToprakTokenizer
from utils.reproducibility import file_sha256, fingerprint_data


DTYPE = np.uint16  # vocab 32000 << 65535


class ShardWriter:
    """Belirlenen boyuta ulaştığında yeni shard'a geçen yazıcı."""

    def __init__(self, out_dir: str, prefix: str, shard_size: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_size = shard_size  # token cinsinden
        self.shard_idx = 0
        self.buffer = np.empty(shard_size, dtype=DTYPE)
        self.cursor = 0
        self.total_tokens = 0
        self.shards: List[Tuple[str, int]] = []  # (path, token_count)
        os.makedirs(out_dir, exist_ok=True)

    def _flush(self):
        if self.cursor == 0:
            return
        path = os.path.join(self.out_dir, f"{self.prefix}_{self.shard_idx:05d}.bin")
        self.buffer[: self.cursor].tofile(path)
        self.shards.append((path, int(self.cursor)))
        self.shard_idx += 1
        self.cursor = 0

    def write(self, tokens: np.ndarray):
        """Tokenları (np.uint16 array) buffer'a yaz, gerekirse shard'ı kapat."""
        n = len(tokens)
        self.total_tokens += n
        pos = 0
        while pos < n:
            free = self.shard_size - self.cursor
            take = min(free, n - pos)
            self.buffer[self.cursor : self.cursor + take] = tokens[pos : pos + take]
            self.cursor += take
            pos += take
            if self.cursor >= self.shard_size:
                self._flush()

    def close(self):
        self._flush()
        return self.shards, self.total_tokens


def _iter_jsonl_docs(input_dir: str, sources_filter: Optional[set] = None):
    """Bir dizindeki tüm JSONL dosyalarındaki dokümanları sırayla yield et."""
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".jsonl") and not f.startswith(".")
    )
    for fname in files:
        path = os.path.join(input_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = doc.get("source", "")
                if sources_filter is not None and src not in sources_filter:
                    continue
                text = doc.get("text", "")
                if text:
                    yield src, text


def tokenize_corpus(
    input_dir: str,
    tokenizer: ToprakTokenizer,
    out_dir: str,
    shard_size: int,
    eval_ratio: float = 0.005,
    seed: int = 42,
    curriculum: bool = False,
    high_quality_sources: Tuple[str, ...] = ("wiki",),
    tokenizer_path: Optional[str] = None,
):
    """
    Temizlenmiş JSONL'leri tokenize edip shard'lara yaz.

    Args:
        input_dir: temizlenmiş JSONL'lerin dizini
        tokenizer: ToprakTokenizer
        out_dir: çıktı dizini
        shard_size: shard başına token sayısı
        eval_ratio: doc-level eval split oranı
        seed: rng seed
        curriculum: True ise high_quality_sources (örn. wiki) son shard'lara yazılır
        high_quality_sources: curriculum'da öne çekilmek yerine sona bırakılacak kaynaklar
        tokenizer_path: Manifestte SHA-256'sı saklanacak tokenizer yolu
    """
    rng = random.Random(seed)
    input_fingerprint = fingerprint_data(input_dir, mode="full")
    # Kaynak dizinin makineye özel mutlak yolu manifest hash'ini değiştirmesin.
    input_fingerprint.pop("path", None)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    train_writer = ShardWriter(out_dir, "train", shard_size)
    eval_writer = ShardWriter(out_dir, "eval", max(shard_size // 10, 1_000_000))

    # Curriculum stratejisi: high-quality kaynakları ayrı listeye topla,
    # düşük öncelik olanları önce yaz, yüksek kalitelileri sona bırak.
    print(f"\n📑 Doküman taraması başlıyor: {input_dir}")
    print(f"   Curriculum: {'AÇIK (HQ sona)' if curriculum else 'KAPALI (karışık)'}")
    print(f"   Yüksek kalite kaynaklar: {high_quality_sources}")

    # Tüm dokümanları belleğe almıyoruz — yalnızca path/index referansı yok.
    # İki geçişli yaklaşım: önce non-HQ, sonra HQ. JSONL'i iki kez okur.
    if curriculum:
        passes = [
            ("low",  lambda src: src not in high_quality_sources),
            ("high", lambda src: src in high_quality_sources),
        ]
    else:
        passes = [("all", lambda src: True)]

    total_docs = 0
    eval_docs = 0
    t0 = time.time()

    for pass_name, src_filter_fn in passes:
        print(f"\n▶ Pass: {pass_name}")
        pbar = tqdm(_iter_jsonl_docs(input_dir), desc=f"tokenize:{pass_name}", unit="doc")
        for src, text in pbar:
            if not src_filter_fn(src):
                continue
            try:
                ids = tokenizer.encode(text, add_bos=False, add_eos=False)
            except Exception:
                continue
            if not ids:
                continue
            # BOS/EOS sarmalama
            arr = np.empty(len(ids) + 2, dtype=DTYPE)
            arr[0] = bos
            arr[1:-1] = np.asarray(ids, dtype=DTYPE)
            arr[-1] = eos

            # Eval split (doc-level)
            if rng.random() < eval_ratio:
                eval_writer.write(arr)
                eval_docs += 1
            else:
                train_writer.write(arr)
            total_docs += 1

            if total_docs % 5000 == 0:
                pbar.set_postfix(
                    train_tok=f"{train_writer.total_tokens/1e6:.1f}M",
                    eval_tok=f"{eval_writer.total_tokens/1e6:.1f}M",
                )

    train_shards, train_total = train_writer.close()
    eval_shards, eval_total = eval_writer.close()

    def shard_entries(shards):
        return [
            {
                "path": os.path.basename(path),
                "tokens": tokens,
                "sha256": file_sha256(path),
            }
            for path, tokens in shards
        ]

    manifest = {
        "format_version": "toprak-shards-v2",
        "seed": seed,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": file_sha256(tokenizer_path) if tokenizer_path else None,
        "input_fingerprint": input_fingerprint,
        "dtype": "uint16",
        "shard_size_tokens": shard_size,
        "eval_ratio": eval_ratio,
        "curriculum": curriculum,
        "high_quality_sources": list(high_quality_sources),
        "train": {
            "shards": shard_entries(train_shards),
            "total_tokens": int(train_total),
        },
        "eval": {
            "shards": shard_entries(eval_shards),
            "total_tokens": int(eval_total),
        },
        "total_docs": total_docs,
        "eval_docs": eval_docs,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ Pre-tokenization tamam ({dt/60:.1f} dk)")
    print(f"   Toplam doküman:  {total_docs:,}")
    print(f"   Train token:     {train_total:,}  ({len(train_shards)} shard)")
    print(f"   Eval  token:     {eval_total:,}  ({len(eval_shards)} shard)")
    print(f"   Manifest:        {manifest_path}")
    print(f"   Boyut:           {train_total*2/1e9:.2f} GB train + {eval_total*2/1e9:.2f} GB eval")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="🌱 Toprak — Pre-tokenization")
    parser.add_argument("--input-dir", required=True,
                        help="Temizlenmiş JSONL'lerin bulunduğu dizin")
    parser.add_argument("--tokenizer", required=True,
                        help="Tokenizer .model dosyası")
    parser.add_argument("--output-dir", default="data_bin",
                        help="Shard çıktı dizini")
    parser.add_argument("--shard-size", type=int, default=100_000_000,
                        help="Shard başına token sayısı (varsayılan 100M ≈ 200MB)")
    parser.add_argument("--eval-ratio", type=float, default=0.005,
                        help="Eval split oranı (varsayılan 0.005 = %%0.5)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Wikipedia gibi yüksek kalite kaynakları son shard'lara yerleştir")
    parser.add_argument("--hq-sources", nargs="*", default=["wiki"],
                        help="Yüksek kalite kaynak etiketleri (source alanı)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"🌱 Toprak — Pre-tokenize")
    print(f"   Tokenizer:  {args.tokenizer}")
    print(f"   Input:      {args.input_dir}")
    print(f"   Output:     {args.output_dir}")
    print(f"   Shard size: {args.shard_size:,} token")

    tokenizer = ToprakTokenizer(args.tokenizer)
    print(f"   Vocab:      {tokenizer.get_vocab_size():,}")

    tokenize_corpus(
        input_dir=args.input_dir,
        tokenizer=tokenizer,
        out_dir=args.output_dir,
        shard_size=args.shard_size,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        curriculum=args.curriculum,
        high_quality_sources=tuple(args.hq_sources),
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()
