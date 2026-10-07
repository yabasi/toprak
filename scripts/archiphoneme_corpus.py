# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Arkifonemik Korpus Dönüştürücü

Düz metin korpusunu (satır satır) yerleşik SEZGİSEL segmenterle arkifonemik
soyut biçime çevirir ("kitaplarda" → "kitap+lAr+DA") ve istatistik raporlar.
Her kelime gidiş-dönüş doğrulanır; çözülemeyen kelimeler aynen kalır.

Kullanım:
    python scripts/archiphoneme_corpus.py --input data/corpus.txt \\
        --output data/corpus_arch.txt [--lexicon kokler.txt] [--verify]

Ardından tokenizer:
    from model.tokenizer import train_tokenizer
    from model.chat_template import CHAT_SPECIAL_TOKENS
    from model.archiphoneme import archiphoneme_user_symbols
    train_tokenizer("data/corpus_arch.txt",
                    extra_symbols=list(CHAT_SPECIAL_TOKENS) + archiphoneme_user_symbols())
"""

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.archiphoneme import (
    ArchiphonemeCodec,
    RuleBasedSegmenter,
    transform_corpus_line,
)


def main():
    parser = argparse.ArgumentParser(
        description="🌱 Toprak — Korpusu arkifonemik biçime çevir (sezgisel segmenter)"
    )
    parser.add_argument("--input", required=True, help="Girdi metin dosyası (UTF-8)")
    parser.add_argument("--output", required=True, help="Çıktı dosyası")
    parser.add_argument("--lexicon", default=None,
                        help="İsteğe bağlı kök sözlüğü (satır başına bir kök)")
    parser.add_argument("--min-stem-len", type=int, default=3,
                        help="Sözlük dışı köklerde en az kök uzunluğu")
    parser.add_argument("--max-lines", type=int, default=0,
                        help="En fazla işlenecek satır (0 = tümü)")
    parser.add_argument("--verify", action="store_true",
                        help="Her satırda decode(çıktı) == girdi kontrolü yap")
    parser.add_argument("--top", type=int, default=15,
                        help="En sık kaç soyut ek raporlansın")
    args = parser.parse_args()

    lexicon = None
    if args.lexicon:
        with open(args.lexicon, encoding="utf-8") as f:
            lexicon = [ln.strip() for ln in f if ln.strip()]

    codec = ArchiphonemeCodec()
    segmenter = RuleBasedSegmenter(lexicon=lexicon, min_stem_len=args.min_stem_len, codec=codec)
    stats = {"words": 0, "segmented": 0, "rejected": 0}
    suffix_counts = Counter()
    lines = mismatches = 0
    t0 = time.time()

    with open(args.input, encoding="utf-8") as fin, \
            open(args.output, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.rstrip("\n")
            out = transform_corpus_line(line, segmenter, codec, stats)
            for tok in out.split():
                if "+" in tok:
                    for piece in tok.split("+")[1:]:
                        suffix_counts["+" + piece.rstrip(".,;:!?)\"'»…")] += 1
            if args.verify and codec.decode(out) != line:
                mismatches += 1
            fout.write(out + "\n")
            lines += 1
            if args.max_lines and lines >= args.max_lines:
                break

    words = max(stats["words"], 1)
    print("🌱 Arkifonemik dönüşüm tamamlandı")
    print(f"  Satır:            {lines:,}")
    print(f"  Kelime:           {stats['words']:,}")
    print(f"  Segmentlenen:     {stats['segmented']:,} ({stats['segmented'] / words:.1%})")
    print(f"  Reddedilen:       {stats['rejected']:,} (gidiş-dönüş tutmadı)")
    if args.verify:
        print(f"  Doğrulama hatası: {mismatches:,} satır")
    print(f"  Süre:             {time.time() - t0:.1f} sn")
    if suffix_counts:
        print(f"  En sık {args.top} ek:")
        for suf, n in suffix_counts.most_common(args.top):
            print(f"    {suf:10s} {n:,}")


if __name__ == "__main__":
    main()
