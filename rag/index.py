# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak RAG — BM25 Arama İndeksi

Köklenmiş terimler üzerinde Okapi BM25 ters indeks:

    skor(q, d) = Σ_t idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
    idf(t)     = ln(1 + (N − df + 0.5) / (df + 0.5))

Hukuk metinlerinde kanun numaraları ("5237 sayılı") ve madde numaraları
("Madde 12") sorgunun en ayırt edici kısmıdır; bu yüzden BM25 skoruna
açıklanabilir ek puanlar eklenir:

- article_boost: Sorgudaki madde atfı ("madde 12", "geçici madde 1") parçanın
  kendi madde etiketiyle aynıysa.
- number_boost: Sorgudaki sayı parçada geçiyorsa (4+ haneli sayılar — kanun
  numaraları — iki kat).
- phrase_boost: Sorgudaki "tırnak içindeki ifade" parçada aynen geçiyorsa.

İndeks JSON olarak kaydedilir/yüklenir; dış bağımlılık yoktur.
"""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from rag.chunker import Chunk, Document, chunk_documents, load_documents
from rag.text import analyze, article_key, is_article_term, is_number_term, normalize

INDEX_FORMAT = "toprak-rag-bm25-v1"


@dataclass
class SearchResult:
    """Arama sonucu: parça, toplam skor ve skor dökümü."""
    chunk: Chunk
    score: float
    bm25: float = 0.0
    boosts: Dict[str, float] = field(default_factory=dict)
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "bm25": round(self.bm25, 4),
            "boosts": {k: round(v, 4) for k, v in self.boosts.items()},
            "matched_terms": self.matched_terms,
            "chunk": self.chunk.to_dict(),
        }


class BM25Index:
    """Okapi BM25 + madde/sayı/ifade ek puanlı ters indeks."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        article_boost: float = 4.0,
        number_boost: float = 1.5,
        phrase_boost: float = 3.0,
        use_stemming: bool = True,
    ):
        self.k1 = k1
        self.b = b
        self.article_boost = article_boost
        self.number_boost = number_boost
        self.phrase_boost = phrase_boost
        self.use_stemming = use_stemming
        self.chunks: List[Chunk] = []
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_lens: List[int] = []
        self._rebuild()

    # ── kurulum ──────────────────────────────────────────

    def _analyze(self, text: str) -> List[str]:
        return analyze(text, use_stemming=self.use_stemming)

    def _chunk_terms(self, chunk: Chunk) -> List[str]:
        terms = self._analyze(chunk.text)
        terms += self._analyze(chunk.title)
        key = article_key(chunk.article)
        if key and key not in terms:
            terms.append(key)
        return terms

    def add_chunks(self, chunks: Iterable[Chunk]) -> "BM25Index":
        for chunk in chunks:
            terms = self._chunk_terms(chunk)
            self.chunks.append(chunk)
            self.term_freqs.append(dict(Counter(terms)))
            self.doc_lens.append(len(terms))
        self._rebuild()
        return self

    def _rebuild(self):
        self.postings: Dict[str, List[int]] = defaultdict(list)
        for i, tf in enumerate(self.term_freqs):
            for term in tf:
                self.postings[term].append(i)
        n = len(self.chunks)
        self.avgdl = (sum(self.doc_lens) / n) if n else 0.0
        self.idf = {
            term: math.log(1.0 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, ids in self.postings.items()
        }

    @classmethod
    def from_documents(cls, docs: Iterable[Document], chunk_kwargs: Optional[dict] = None, **params) -> "BM25Index":
        index = cls(**params)
        index.add_chunks(chunk_documents(docs, **(chunk_kwargs or {})))
        return index

    @classmethod
    def from_path(cls, path: str, chunk_kwargs: Optional[dict] = None, **params) -> "BM25Index":
        """Klasör (.txt/.md) veya JSONL'den indeks kur."""
        return cls.from_documents(load_documents(path), chunk_kwargs=chunk_kwargs, **params)

    def __len__(self) -> int:
        return len(self.chunks)

    # ── arama ────────────────────────────────────────────

    def _bm25(self, term: str, i: int) -> float:
        tf = self.term_freqs[i].get(term, 0)
        if not tf:
            return 0.0
        denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / (self.avgdl or 1.0))
        return self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom

    def search(self, query: str, k: int = 5, boost: bool = True) -> List[SearchResult]:
        """
        Sorguya en uygun `k` parçayı skor sırasıyla döndür.

        Skor = BM25 + (boost=True ise) madde/sayı/ifade ek puanları.
        Eşit skorlar parça sırasına göre çözülür (deterministik).
        """
        if not self.chunks:
            return []
        q_terms = self._analyze(query)
        unique_terms = list(dict.fromkeys(q_terms))
        q_counts = Counter(q_terms)
        candidates = sorted({i for t in unique_terms for i in self.postings.get(t, ())})

        article_refs = [t for t in unique_terms if is_article_term(t) and "/" not in t]
        numbers = [t for t in unique_terms if is_number_term(t)]
        phrases = [normalize(p) for p in re.findall(r'"([^"]{3,})"', normalize(query)) if p.strip()]
        if boost and phrases:
            candidates = sorted(set(candidates) | {
                i for i, c in enumerate(self.chunks) if any(p in normalize(c.text) for p in phrases)
            })

        results = []
        for i in candidates:
            chunk = self.chunks[i]
            matched = [t for t in unique_terms if t in self.term_freqs[i]]
            bm25 = sum(self._bm25(t, i) * q_counts[t] for t in matched)
            boosts: Dict[str, float] = {}
            if boost:
                key = article_key(chunk.article)
                if key and key in article_refs:
                    boosts["article"] = self.article_boost
                num_bonus = sum(
                    self.number_boost * (2.0 if len(re.sub(r"\D", "", t)) >= 4 else 1.0)
                    for t in numbers if t in self.term_freqs[i]
                )
                if num_bonus:
                    boosts["number"] = num_bonus
                if phrases:
                    norm_text = normalize(chunk.text)
                    hits = sum(1 for p in phrases if p in norm_text)
                    if hits:
                        boosts["phrase"] = self.phrase_boost * hits
            score = bm25 + sum(boosts.values())
            if score > 0:
                results.append(SearchResult(chunk, score, bm25, boosts, matched))
        results.sort(key=lambda r: -r.score)
        return results[:k]

    # ── kalıcılık ────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "format": INDEX_FORMAT,
            "params": {
                "k1": self.k1, "b": self.b,
                "article_boost": self.article_boost,
                "number_boost": self.number_boost,
                "phrase_boost": self.phrase_boost,
                "use_stemming": self.use_stemming,
            },
            "chunks": [c.to_dict() for c in self.chunks],
            "term_freqs": self.term_freqs,
            "doc_lens": self.doc_lens,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        if data.get("format") != INDEX_FORMAT:
            raise ValueError(f"Desteklenmeyen indeks biçimi: {data.get('format')!r}")
        index = cls(**data.get("params", {}))
        index.chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        index.term_freqs = [dict(tf) for tf in data["term_freqs"]]
        index.doc_lens = list(data["doc_lens"])
        index._rebuild()
        return index

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
