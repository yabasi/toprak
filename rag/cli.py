# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak RAG — Komut Satırı

    # 1) İndeks kur (klasör .txt/.md ya da JSONL)
    python -m rag.cli index --docs rag/examples --out rag_index.json

    # 2) Ara
    python -m rag.cli search --index rag_index.json --query "ödünç süresi kaç gün"

    # 3) Kaynaklı cevap üret + atıf doğrulama raporu
    python -m rag.cli ask --index rag_index.json \\
        --checkpoint checkpoints/toprak_best.pt \\
        --tokenizer toprak_tokenizer.model \\
        --query "Kütüphaneden en fazla kaç kitap ödünç alınabilir?" --speculative
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.citations import render_report, verify_citations
from rag.index import BM25Index
from rag.prompt import DEFAULT_SYSTEM_PROMPT, build_rag_messages


def cmd_index(args) -> int:
    chunk_kwargs = {"max_chars": args.max_chars, "overlap_sentences": args.overlap}
    index = BM25Index.from_path(args.docs, chunk_kwargs=chunk_kwargs, k1=args.k1, b=args.b)
    index.save(args.out)
    n_docs = len({c.doc_id for c in index.chunks})
    print(f"İndeks kaydedildi: {args.out}  ({n_docs} belge, {len(index)} parça, "
          f"{len(index.idf)} terim)")
    return 0


def _print_results(results, show_text: bool = True):
    for rank, r in enumerate(results, start=1):
        c = r.chunk
        where = f"{c.title}" + (f" — {c.article}" if c.article else "")
        boosts = ", ".join(f"{k}+{v:.1f}" for k, v in r.boosts.items())
        print(f"{rank}. [{r.score:.3f}] {where}  ({c.id}{'; ' + boosts if boosts else ''})")
        if show_text:
            body = " ".join(c.text.split())
            print(f"   {body[:300]}{'…' if len(body) > 300 else ''}")
        meta = [x for x in (c.url, c.license, c.date) if x]
        if meta:
            print(f"   kaynak: {' | '.join(meta)}")


def cmd_search(args) -> int:
    index = BM25Index.load(args.index)
    results = index.search(args.query, k=args.k, boost=not args.no_boost)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    elif not results:
        print("Sonuç bulunamadı.")
    else:
        _print_results(results)
    return 0


def cmd_ask(args) -> int:
    import torch

    from inference.generate import generate_text, load_model
    from model.chat_template import ChatTemplate
    from model.config import detect_device
    from model.tokenizer import ToprakTokenizer

    device = args.device or detect_device()
    index = BM25Index.load(args.index)
    results = index.search(args.query, k=args.k)

    tokenizer = ToprakTokenizer(args.tokenizer)
    model, config = load_model(args.checkpoint, device)
    template = ChatTemplate(tokenizer)

    context = args.max_context or config.max_seq_len
    budget = max(64, context - args.max_new_tokens)
    rag = build_rag_messages(
        args.query, results, tokenizer=tokenizer, max_prompt_tokens=budget,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT, template=template,
    )
    prompt_ids = template.encode_prompt(rag.messages)

    torch.manual_seed(args.seed)
    stats = None
    if args.speculative:
        from inference.speculative import speculative_generate
        new_ids, stats = speculative_generate(
            model, prompt_ids, max_new_tokens=args.max_new_tokens,
            draft_source="ngram", stop_ids=template.stop_ids,
        )
        answer = tokenizer.decode(new_ids)
    else:
        answer = generate_text(
            model, tokenizer, prompt="", max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=0,
            device=device, stop_ids=template.stop_ids, prompt_ids=prompt_ids,
            return_new_only=True,
        )
    answer = answer.strip()
    report = verify_citations(answer, rag.sources, threshold=args.support_threshold)

    if args.json:
        print(json.dumps({
            "query": args.query,
            "answer": answer,
            "prompt_tokens": len(prompt_ids),
            "sources": [s.to_dict() for s in rag.sources],
            "dropped_sources": len(rag.dropped),
            "citations": report.to_dict(),
            "speculative": stats.as_dict() if stats else None,
        }, ensure_ascii=False, indent=2))
        return 0

    print(f"Soru: {args.query}")
    print("=" * 60)
    print(answer or "(boş cevap)")
    print("=" * 60)
    print(f"Kaynaklar ({len(rag.sources)} kullanıldı, {len(rag.dropped)} bütçe nedeniyle "
          f"düşürüldü; prompt {len(prompt_ids)}/{context} token):")
    for s in rag.sources:
        c = s.chunk
        meta = " | ".join(x for x in (c.url, c.license, c.date) if x)
        trunc = " (kırpıldı)" if s.truncated else ""
        print(f"  [{s.number}] {c.title}" + (f" — {c.article}" if c.article else "")
              + f"{trunc}  skor={s.score:.2f}" + (f"\n      {meta}" if meta else ""))
    print()
    print(render_report(report))
    if stats is not None:
        st = stats.as_dict()
        print(f"\nSpekülatif (n-gram): kabul oranı {st['acceptance_rate']:.2f}, "
              f"ileri geçiş başına {st['tokens_per_forward']:.2f} token")
    print("\nNot: Bu çıktı hukuki danışmanlık değildir; resmi metni kaynağından doğrulayın.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.cli",
        description="🌱 Toprak RAG — kaynaklı Türkçe soru-cevap (mevzuat odaklı)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="Belgelerden BM25 indeksi kur")
    p.add_argument("--docs", required=True, help="Belge klasörü (.txt/.md) veya JSONL dosyası")
    p.add_argument("--out", required=True, help="Çıkış indeks dosyası (JSON)")
    p.add_argument("--max-chars", type=int, default=1200, help="Parça başına karakter bütçesi")
    p.add_argument("--overlap", type=int, default=1, help="Parçalar arası örtüşen cümle sayısı")
    p.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parametresi")
    p.add_argument("--b", type=float, default=0.75, help="BM25 b parametresi")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="İndekste ara")
    p.add_argument("--index", required=True, help="İndeks dosyası")
    p.add_argument("--query", required=True, help="Sorgu metni")
    p.add_argument("--k", type=int, default=5, help="Döndürülecek sonuç sayısı")
    p.add_argument("--no-boost", action="store_true", help="Madde/sayı/ifade ek puanlarını kapat")
    p.add_argument("--json", action="store_true", help="JSON çıktı")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="Kaynaklı cevap üret ve atıfları doğrula")
    p.add_argument("--index", required=True, help="İndeks dosyası")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint dosyası")
    p.add_argument("--tokenizer", default="toprak_tokenizer.model", help="Tokenizer model dosyası")
    p.add_argument("--query", required=True, help="Soru")
    p.add_argument("--k", type=int, default=8, help="Aday kaynak sayısı (bütçeye sığanlar kullanılır)")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Üretilecek en fazla token")
    p.add_argument("--max-context", type=int, default=None,
                   help="Bağlam uzunluğu (varsayılan: checkpoint config.max_seq_len)")
    p.add_argument("--speculative", action="store_true",
                   help="Greedy n-gram spekülatif çözümleme (RAG'da kaynaktan kopyalamayı hızlandırır)")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05,
                   help="Kaynaktan alıntıyı cezalandırmamak için düşük tutun")
    p.add_argument("--support-threshold", type=float, default=0.5,
                   help="Cümlenin desteklenmiş sayılması için örtüşme eşiği")
    p.add_argument("--system-prompt", default=None, help="Özel sistem mesajı")
    p.add_argument("--device", default=None, help="Cihaz (varsayılan: otomatik)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="JSON çıktı")
    p.set_defaults(func=cmd_ask)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
