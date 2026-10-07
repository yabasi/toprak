# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak RAG — Kaynaklı Türkçe Soru-Cevap (mevzuat / yönetmelik odaklı)

Bileşenler:
- rag.text       Türkçe normalizasyon, tokenizasyon, sezgisel kök bulucu,
                 durak kelimeler, kısaltma duyarlı cümle bölücü
- rag.chunker    Belge yükleme (klasör / JSONL) ve madde yapısını koruyan
                 örtüşmeli parçalama
- rag.index      BM25 ters indeks + madde/sayı/ifade ek puanları, JSON kayıt
- rag.prompt     Numaralı kaynaklı, token bütçeli ChatTemplate mesajları
- rag.citations  Atıf doğrulama ve Türkçe rapor
- rag.cli        index / search / ask komutları

Ayrıntılar için LONG_CONTEXT_RAG.md dosyasına bakın.
"""

from rag.chunker import Chunk, Document, chunk_document, chunk_documents, load_documents
from rag.citations import CitationReport, render_report, verify_citations
from rag.index import BM25Index, SearchResult
from rag.prompt import DEFAULT_SYSTEM_PROMPT, NO_ANSWER, RagPrompt, build_rag_messages
from rag.text import analyze, normalize, split_sentences, stem, tokenize, turkish_lower

__all__ = [
    "BM25Index", "Chunk", "CitationReport", "DEFAULT_SYSTEM_PROMPT", "Document",
    "NO_ANSWER", "RagPrompt", "SearchResult", "analyze", "build_rag_messages",
    "chunk_document", "chunk_documents", "load_documents", "normalize",
    "render_report", "split_sentences", "stem", "tokenize", "turkish_lower",
    "verify_citations",
]
