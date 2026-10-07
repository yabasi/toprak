# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak RAG — Belge Yükleme ve Parçalama (Chunking)

Belgeler cümle sınırlarında, örtüşmeli (overlap) ve karakter ya da token
bütçeli parçalara bölünür. Hukuk/mevzuat metinlerinin yapısı korunur:

- "MADDE 5 –", "Madde 12-", "Geçici Madde 1 –" gibi bir satır her zaman
  yeni bir parça başlatır ve parçaya `article` metaverisi yazılır.
- Madde başlığından hemen önceki kısa başlık satırları ("Amaç",
  "BİRİNCİ BÖLÜM") o maddenin parçasına dahil edilir.
- Uzun maddeler birden çok parçaya bölünür; hepsi aynı `article` etiketini
  taşır. Örtüşme bir madde sınırını aşmaz.

Her parça kaynak belgenin provenance alanlarını (url, lisans, tarih)
taşır; böylece cevaptaki her atıf kaynağına kadar izlenebilir
(bkz. DATA_GOVERNANCE.md, data/governance.py).

Belge biçimleri:
- Klasör: .txt / .md dosyaları. Dosyanın başında isteğe bağlı basit
  front matter bulunabilir:
      ---
      title: Örnek Kütüphane Yönetmeliği
      url: https://example.org/yonetmelik
      license: CC0-1.0
      date: 2026-01-01
      ---
- JSONL: her satır {"id","title","text","url","license","date"}. Toprak
  korpus şemasındaki (toprak-document-v1) `source_url`, `licenses`,
  `downloaded_at` alanları da tanınır.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from rag.text import ARTICLE_HEADER_RE, Sentence, _is_heading_line, parse_article_header, split_sentences


@dataclass
class Document:
    """Kaynak belge ve provenance metaverisi."""
    id: str
    title: str
    text: str
    url: Optional[str] = None
    license: Optional[str] = None
    date: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Aranabilir metin parçası (text == belge.text[start:end])."""
    id: str
    doc_id: str
    title: str
    article: Optional[str]
    text: str
    start: int
    end: int
    url: Optional[str] = None
    license: Optional[str] = None
    date: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        known.setdefault("metadata", {})
        return cls(**known)


# ── Yükleme ──────────────────────────────────────────────

_PROVENANCE_KEYS = (
    "source", "dataset_id", "license_status", "license_url", "source_record_id",
    "content_sha256", "schema_version", "fictional",
)


def _first_license(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else None
    return str(value)


def document_from_record(record: dict, fallback_id: str) -> Document:
    """JSONL kaydından Document üret (toprak-document-v1 alanları dahil)."""
    prov = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    merged = {**prov, **record}
    text = merged.get("text") or merged.get("content") or ""
    doc_id = str(merged.get("id") or merged.get("source_record_id") or fallback_id)
    title = merged.get("title") or (text.strip().splitlines()[0][:120] if text.strip() else doc_id)
    url = merged.get("url") or merged.get("source_url")
    license_ = _first_license(merged.get("license") or merged.get("licenses"))
    date = merged.get("date") or merged.get("published_at") or merged.get("downloaded_at")
    metadata = {k: merged[k] for k in _PROVENANCE_KEYS if merged.get(k) is not None}
    return Document(doc_id, str(title), text, url, license_, date, metadata)


def _parse_front_matter(raw: str) -> Tuple[Dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    lines = raw.splitlines(keepends=True)
    meta: Dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "".join(lines[i + 1:]).lstrip("\n")
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return {}, raw


def load_text_file(path: str, doc_id: Optional[str] = None) -> Document:
    """Tek bir .txt/.md dosyasını Document olarak yükle."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    meta, text = _parse_front_matter(raw)
    if "title" not in meta:
        for line in text.splitlines():
            if line.strip():
                meta["title"] = line.strip().lstrip("#").strip()
                break
    doc_id = meta.pop("id", None) or doc_id or os.path.splitext(os.path.basename(path))[0]
    record = {"id": doc_id, "text": text, **meta}
    return document_from_record(record, doc_id)


def load_documents(path: str) -> List[Document]:
    """
    Belgeleri yükle: klasör (.txt/.md, alt klasörler dahil, sıralı) veya
    JSONL dosyası.
    """
    if os.path.isdir(path):
        docs = []
        for root, _, files in sorted(os.walk(path)):
            for name in sorted(files):
                if name.lower().endswith((".txt", ".md")) and name.lower() != "readme.md":
                    full = os.path.join(root, name)
                    rel = os.path.splitext(os.path.relpath(full, path))[0].replace(os.sep, "/")
                    docs.append(load_text_file(full, doc_id=rel))
        return docs
    if path.endswith((".jsonl", ".json")):
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    docs.append(document_from_record(json.loads(line), f"doc{i}"))
        return docs
    return [load_text_file(path)]


# ── Parçalama ────────────────────────────────────────────

def find_sections(text: str) -> List[Tuple[int, int, Optional[str]]]:
    """
    Metni madde sınırlarında bölümlere ayır.

    Returns:
        [(start, end, article_label_or_None), ...]
    """
    headers = []
    for m in ARTICLE_HEADER_RE.finditer(text):
        line_end = text.find("\n", m.start())
        line = text[m.start(): line_end if line_end != -1 else len(text)]
        label = parse_article_header(line)
        if label is None:
            continue
        start = m.start()
        # Önceki kısa başlık satırlarını (en fazla 3) maddeye dahil et
        for _ in range(3):
            prev_end = text.rfind("\n", 0, start)
            if prev_end <= 0:
                break
            prev_start = text.rfind("\n", 0, prev_end) + 1
            prev_line = text[prev_start:prev_end]
            if not _is_heading_line(prev_line) or prev_line.strip().startswith("#") \
                    or parse_article_header(prev_line):
                break
            start = prev_start
        headers.append((start, label))

    sections = []
    cursor = 0
    current_label = None
    for start, label in headers:
        if start > cursor and text[cursor:start].strip():
            sections.append((cursor, start, current_label))
        cursor, current_label = max(start, cursor), label
    if text[cursor:].strip():
        sections.append((cursor, len(text), current_label))
    return sections


def _split_long(text: str, sent: Sentence, measure: Callable[[str], int], budget: int) -> List[Sentence]:
    """Bütçeyi aşan tek bir cümleyi kelime sınırlarında böl."""
    if measure(sent.text) <= budget:
        return [sent]
    pieces, piece_start, last_end = [], None, None
    pos = sent.start
    for word in sent.text.split():
        w_start = text.index(word, pos)
        w_end = w_start + len(word)
        pos = w_end
        if piece_start is None:
            piece_start = w_start
        elif measure(text[piece_start:w_end]) > budget:
            pieces.append(Sentence(text[piece_start:last_end], piece_start, last_end))
            piece_start = w_start
        last_end = w_end
    if piece_start is not None:
        pieces.append(Sentence(text[piece_start:last_end], piece_start, last_end))
    return pieces


def chunk_document(
    doc: Document,
    max_chars: int = 1200,
    overlap_sentences: int = 1,
    token_counter: Optional[Callable[[str], int]] = None,
    max_tokens: Optional[int] = None,
) -> List[Chunk]:
    """
    Belgeyi örtüşmeli parçalara böl.

    Args:
        max_chars: Parça başına karakter bütçesi (token_counter yoksa)
        overlap_sentences: Ardışık parçalar arasında tekrar eden cümle sayısı
            (aynı madde içinde)
        token_counter / max_tokens: Verilirse bütçe token cinsinden ölçülür
            (ör. lambda s: len(tokenizer.encode(s, False, False)))
    """
    if token_counter is not None and max_tokens is not None:
        measure, budget = token_counter, max_tokens
    else:
        measure, budget = len, max_chars

    text = doc.text
    chunks: List[Chunk] = []

    def emit(sents: List[Sentence], label: Optional[str]):
        start, end = sents[0].start, sents[-1].end
        chunks.append(Chunk(
            id=f"{doc.id}#{len(chunks)}",
            doc_id=doc.id,
            title=doc.title,
            article=label,
            text=text[start:end],
            start=start,
            end=end,
            url=doc.url,
            license=doc.license,
            date=doc.date,
            metadata=dict(doc.metadata),
        ))

    for sec_start, sec_end, label in find_sections(text):
        sents: List[Sentence] = []
        for s in split_sentences(text[sec_start:sec_end], offset=sec_start):
            sents.extend(_split_long(text, s, measure, budget))
        current: List[Sentence] = []
        new_in_current = 0
        for sent in sents:
            if current and new_in_current > 0 and measure(text[current[0].start:sent.end]) > budget:
                emit(current, label)
                keep = current[-overlap_sentences:] if overlap_sentences > 0 else []
                # Örtüşme bütçeyi tek başına doldurmasın
                while keep and measure(text[keep[0].start:sent.end]) > budget:
                    keep = keep[1:]
                current, new_in_current = list(keep), 0
            current.append(sent)
            new_in_current += 1
        if current and new_in_current > 0:
            emit(current, label)
    return chunks


def chunk_documents(docs: Iterable[Document], **kwargs) -> List[Chunk]:
    out: List[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, **kwargs))
    return out
