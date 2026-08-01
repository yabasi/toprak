# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Birleşik Türkçe Korpus İndirici

3 Türkçe veri kaynağını streaming ile indirip ortak JSONL formatına yazar:
  - Wikipedia TR  (wikimedia/wikipedia 20231101.tr)
  - FineWeb-2 TR  (HuggingFaceFW/fineweb-2, config=tur_Latn)
  - CulturaX TR   (uonlp/CulturaX, config=tr)

Streaming kullanılır → diskte sadece filtrelenmiş JSONL kalır.
Boyut hedefi (gigabyte) aşılınca otomatik durur.

Kullanım (CPU pod'da çalıştırın — A100 saati harcamayın):
    python scripts/download_all_corpora.py \
        --output-dir /workspace/data_raw \
        --wikipedia --fineweb2 --culturax \
        --fineweb-target-gb 30 --culturax-target-gb 15

Ortak JSONL satır şeması:
    {"text": str, "source": "wiki|fineweb2|culturax", "word_count": int}
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm


MIN_WORDS = 30   # Bu kelime sayısının altındaki dokümanları indirme aşamasında at
MAX_WORDS = 100_000


def _write_doc(fout, text: str, source: str) -> int:
    """Tek bir dokümanı JSONL satırı olarak yaz. Yazılan byte sayısı döner (0 = atlandı)."""
    if not text:
        return 0
    text = text.strip()
    if not text:
        return 0
    words = text.split()
    n = len(words)
    if n < MIN_WORDS or n > MAX_WORDS:
        return 0
    line = json.dumps(
        {"text": text, "source": source, "word_count": n},
        ensure_ascii=False,
    ) + "\n"
    fout.write(line)
    return len(line.encode("utf-8"))


def download_wikipedia(output_dir: str, max_articles: Optional[int] = None) -> str:
    """Türkçe Wikipedia (HuggingFace mirror)."""
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "wikipedia_tr.jsonl")

    print(f"\n📥 Wikipedia TR (wikimedia/wikipedia 20231101.tr) → {out_file}")
    ds = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)

    count = 0
    bytes_written = 0
    t0 = time.time()
    with open(out_file, "w", encoding="utf-8") as fout:
        pbar = tqdm(ds, desc="Wikipedia", unit="doc")
        for article in pbar:
            if max_articles and count >= max_articles:
                break
            written = _write_doc(fout, article.get("text", ""), "wiki")
            if written:
                count += 1
                bytes_written += written
                if count % 1000 == 0:
                    pbar.set_postfix(saved=count, gb=f"{bytes_written/1e9:.2f}")

    dt = time.time() - t0
    print(f"  ✓ {count:,} doc, {bytes_written/1e9:.2f} GB, {dt/60:.1f} dk")
    return out_file


def download_streaming_hf(
    repo: str,
    config: str,
    output_file: str,
    source_tag: str,
    target_gb: float,
    text_field: str = "text",
    split: str = "train",
) -> str:
    """
    Generic HF streaming downloader (FineWeb-2, CulturaX vs.).

    Args:
        repo: HF dataset repo (örn. 'HuggingFaceFW/fineweb-2')
        config: dataset config (örn. 'tur_Latn', 'tr')
        output_file: hedef JSONL
        source_tag: 'fineweb2' veya 'culturax'
        target_gb: yazılan JSONL bu boyuta ulaşınca dur
        text_field: dokümandaki metin alanı adı
    """
    from datasets import load_dataset

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    target_bytes = int(target_gb * 1e9)

    print(f"\n📥 {repo} ({config}) → {output_file}")
    print(f"   Hedef boyut: {target_gb:.1f} GB (streaming, kesim oto)")

    try:
        ds = load_dataset(repo, config, split=split, streaming=True)
    except Exception as e:
        print(f"   ❌ Dataset yüklenemedi: {e}")
        print(f"   HF Hub erişimi / login durumunu kontrol edin: huggingface-cli login")
        return output_file

    count = 0
    bytes_written = 0
    skipped = 0
    t0 = time.time()

    with open(output_file, "w", encoding="utf-8") as fout:
        pbar = tqdm(ds, desc=source_tag, unit="doc")
        for ex in pbar:
            text = ex.get(text_field, "")
            written = _write_doc(fout, text, source_tag)
            if written:
                count += 1
                bytes_written += written
                if count % 5000 == 0:
                    pbar.set_postfix(
                        saved=count,
                        gb=f"{bytes_written/1e9:.2f}/{target_gb:.0f}",
                        skip=skipped,
                    )
            else:
                skipped += 1

            if bytes_written >= target_bytes:
                pbar.close()
                print(f"   ✓ Hedef boyuta ulaşıldı, durduruluyor.")
                break

    dt = time.time() - t0
    print(f"  ✓ {count:,} doc, {bytes_written/1e9:.2f} GB, atlanan {skipped:,}, {dt/60:.1f} dk")
    return output_file


def download_fineweb2(output_dir: str, target_gb: float) -> str:
    """FineWeb-2 Türkçe (tur_Latn)."""
    out = os.path.join(output_dir, "fineweb2_tr.jsonl")
    return download_streaming_hf(
        repo="HuggingFaceFW/fineweb-2",
        config="tur_Latn",
        output_file=out,
        source_tag="fineweb2",
        target_gb=target_gb,
        text_field="text",
    )


def download_culturax(output_dir: str, target_gb: float) -> str:
    """CulturaX Türkçe."""
    out = os.path.join(output_dir, "culturax_tr.jsonl")
    return download_streaming_hf(
        repo="uonlp/CulturaX",
        config="tr",
        output_file=out,
        source_tag="culturax",
        target_gb=target_gb,
        text_field="text",
    )


def main():
    parser = argparse.ArgumentParser(description="🌱 Toprak — Birleşik Korpus İndirici")
    parser.add_argument("--output-dir", default="data_raw",
                        help="JSONL'lerin yazılacağı dizin")

    # Kaynak seçimi
    parser.add_argument("--wikipedia", action="store_true", help="Wikipedia TR indir")
    parser.add_argument("--fineweb2", action="store_true", help="FineWeb-2 tur_Latn indir")
    parser.add_argument("--culturax", action="store_true", help="CulturaX TR indir")
    parser.add_argument("--all", action="store_true",
                        help="3 kaynağı da indir (= --wikipedia --fineweb2 --culturax)")

    # Boyut hedefleri
    parser.add_argument("--wiki-max-articles", type=int, default=None,
                        help="Wikipedia maksimum makale (None = tümü)")
    parser.add_argument("--fineweb-target-gb", type=float, default=30.0,
                        help="FineWeb-2 için hedef ham JSONL boyutu (GB)")
    parser.add_argument("--culturax-target-gb", type=float, default=15.0,
                        help="CulturaX için hedef ham JSONL boyutu (GB)")

    args = parser.parse_args()

    if args.all:
        args.wikipedia = args.fineweb2 = args.culturax = True

    if not (args.wikipedia or args.fineweb2 or args.culturax):
        print("⚠ Hiç kaynak seçilmedi. --wikipedia / --fineweb2 / --culturax / --all kullanın.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🌱 Toprak — Birleşik Korpus İndirici")
    print(f"   Çıktı dizini: {args.output_dir}")

    t0 = time.time()
    written = []

    if args.wikipedia:
        written.append(download_wikipedia(args.output_dir, args.wiki_max_articles))
    if args.fineweb2:
        written.append(download_fineweb2(args.output_dir, args.fineweb_target_gb))
    if args.culturax:
        written.append(download_culturax(args.output_dir, args.culturax_target_gb))

    total_size = sum(os.path.getsize(f) for f in written if os.path.exists(f))
    print(f"\n{'='*60}")
    print(f"✅ Tüm indirmeler tamam.")
    print(f"   Toplam boyut: {total_size/1e9:.2f} GB")
    print(f"   Süre: {(time.time()-t0)/60:.1f} dk")
    print(f"   Dosyalar:")
    for f in written:
        if os.path.exists(f):
            print(f"     - {f}  ({os.path.getsize(f)/1e9:.2f} GB)")
    print(f"\n📋 Sonraki adım:")
    print(f"   python scripts/prepare_data.py --step tokenizer --data-dir {args.output_dir}")
    print(f"   python scripts/pretokenize.py --input-dir {args.output_dir} \\")
    print(f"       --tokenizer toprak_tokenizer.model --output-dir data_bin")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
