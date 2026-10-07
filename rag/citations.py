# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak RAG — Atıf Doğrulama

Modelin cevabındaki [n] atıflarını kaynaklara karşı sözcüksel olarak denetler.
Her cümle için:

- cited_ids:     cümledeki atıf numaraları ([1], [1, 2], [1-3], [1][2])
- fabricated:    kaynak listesinde olmayan (uydurma) numaralar
- support:       cümlenin köklenmiş içerik terimlerinin, atıf yapılan
                 kaynak(lar)da geçme oranı (0–1)
- missing_numbers: cümlede geçip atıf yapılan kaynakta geçmeyen sayılar /
                 madde numaraları (mevzuatta en tehlikeli hata türü)
- uncited_factual: atıfsız ama olgu içeren cümle (sayı, tarih, madde atfı,
                 ay adı veya cümle ortasında büyük harfli özel ad)
- suggested_id:  atıfsız cümleyi en iyi destekleyen kaynak (varsa)

Genel ölçütler:
- citation_precision = desteklenen geçerli atıf sayısı / toplam atıf sayısı
- supported_ratio    = desteklenen iddia cümleleri / tüm iddia cümleleri

Bu yöntem sözcükseldir: eş anlamlı yeniden ifade etmeyi kaçırabilir ve
olumsuzlamayı ("verilir" / "verilmez") ayırt edemez. Bir doğal dil
çıkarımı (NLI) modelinin yerini tutmaz; hızlı ve açıklanabilir bir ilk
filtre olarak tasarlanmıştır.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set

from rag.prompt import NO_ANSWER
from rag.text import analyze, is_article_term, is_number_term, normalize, split_sentences

CITATION_RE = re.compile(r"\[(\s*\d+(?:\s*(?:[,;]|[-–])\s*\d+)*\s*)\]")

_MONTHS = (
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos",
    "eylül", "ekim", "kasım", "aralık",
)


@dataclass
class SentenceCheck:
    """Tek bir cevap cümlesinin atıf denetimi."""
    text: str
    cited_ids: List[int]
    valid_ids: List[int]
    fabricated_ids: List[int]
    support: float
    per_source_support: Dict[int, float]
    missing_numbers: List[str]
    is_claim: bool
    is_factual: bool
    abstention: bool
    supported: bool
    uncited_factual: bool
    suggested_id: Optional[int] = None


@dataclass
class CitationReport:
    """Cevabın tamamı için atıf raporu."""
    sentences: List[SentenceCheck]
    citation_precision: Optional[float]
    supported_ratio: Optional[float]
    total_citations: int
    fabricated_ids: List[int]
    uncited_factual_count: int
    abstained: bool
    threshold: float = 0.5
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_citation_ids(text: str) -> List[int]:
    """Metindeki tüm atıf numaralarını sırayla (tekrarsız) döndür."""
    ids: List[int] = []
    for m in CITATION_RE.finditer(text):
        for part in re.split(r"[,;]", m.group(1)):
            part = part.strip()
            rng = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
            if rng:
                a, b = int(rng.group(1)), int(rng.group(2))
                if 0 < b - a <= 20:
                    values = range(a, b + 1)
                else:
                    values = [a, b]
            else:
                values = [int(part)] if part.isdigit() else []
            for v in values:
                if v not in ids:
                    ids.append(v)
    return ids


def strip_citations(text: str) -> str:
    return CITATION_RE.sub(" ", text)


def _source_texts(sources) -> Dict[int, str]:
    """
    Kaynakları {numara: metin} sözlüğüne çevir. Kabul edilenler:
    - {numara: str | Chunk | NumberedSource | dict}
    - NumberedSource listesi (number alanı kullanılır)
    - str / Chunk listesi (1'den numaralanır)
    """
    def text_of(obj) -> str:
        if isinstance(obj, str):
            return obj
        if hasattr(obj, "chunk"):
            obj = obj.chunk
        if isinstance(obj, dict):
            parts = [obj.get("title", ""), obj.get("article") or "", obj.get("text", "")]
            return " ".join(p for p in parts if p)
        parts = [getattr(obj, "title", ""), getattr(obj, "article", None) or "", getattr(obj, "text", "")]
        return " ".join(p for p in parts if p)

    if isinstance(sources, Mapping):
        return {int(k): text_of(v) for k, v in sources.items()}
    out = {}
    for i, src in enumerate(sources, start=1):
        number = getattr(src, "number", None)
        out[int(number) if number is not None else i] = text_of(src)
    return out


def _is_factual(sentence: str, terms: Sequence[str]) -> bool:
    if any(is_number_term(t) or is_article_term(t) for t in terms):
        return True
    low = normalize(sentence)
    if any(re.search(rf"(?<!\w){m}\w*", low) for m in _MONTHS):
        return True
    words = re.findall(r"[^\W\d_]+", strip_citations(sentence))
    # Cümle ortasında büyük harfle başlayan kelime → özel ad sezgisi
    return any(w[0].isupper() for w in words[1:])


def _overlap(terms: Set[str], source_terms: Set[str]) -> float:
    if not terms:
        return 0.0
    return len(terms & source_terms) / len(terms)


def verify_citations(answer: str, sources, threshold: float = 0.5) -> CitationReport:
    """
    Cevaptaki atıfları kaynaklara karşı doğrula.

    Args:
        answer: Modelin cevabı
        sources: Prompt'taki numaralı kaynaklar (bkz. _source_texts)
        threshold: Bir cümlenin "destekleniyor" sayılması için gereken en
            düşük sözcüksel örtüşme oranı
    """
    source_texts = _source_texts(sources)
    source_terms = {n: set(analyze(t)) for n, t in source_texts.items()}

    checks: List[SentenceCheck] = []
    total_citations = 0
    good_citations = 0
    all_fabricated: List[int] = []
    abstained = normalize(NO_ANSWER).rstrip(".") in normalize(answer)

    # "… alabilir. [1]" → atıf yalnız kalmışsa önceki cümleye bağla
    sentence_list: List[str] = []
    for sent in split_sentences(answer):
        if sentence_list and not strip_citations(sent.text).strip(" .;,"):
            sentence_list[-1] = sentence_list[-1] + " " + sent.text
        else:
            sentence_list.append(sent.text)

    for text in sentence_list:
        cited = parse_citation_ids(text)
        content = strip_citations(text)
        terms = set(analyze(content))
        abstention = normalize(NO_ANSWER).rstrip(".") in normalize(content)
        valid = [i for i in cited if i in source_texts]
        fabricated = [i for i in cited if i not in source_texts]
        per_source = {i: round(_overlap(terms, source_terms[i]), 4) for i in valid}
        union_terms = set().union(*(source_terms[i] for i in valid)) if valid else set()
        support = _overlap(terms, union_terms) if valid else 0.0
        claim_numbers = sorted(t for t in terms if is_number_term(t) or is_article_term(t))
        missing = [t for t in claim_numbers if valid and t not in union_terms]
        is_claim = bool(terms) and not abstention
        factual = is_claim and _is_factual(content, list(terms))
        supported = bool(valid) and not fabricated and support >= threshold and not missing

        total_citations += len(cited)
        good_citations += sum(1 for i in valid if per_source[i] >= threshold)
        for i in fabricated:
            if i not in all_fabricated:
                all_fabricated.append(i)

        suggested = None
        if not cited and is_claim and source_terms:
            best = max(source_terms, key=lambda n: (_overlap(terms, source_terms[n]), -n))
            if _overlap(terms, source_terms[best]) >= threshold:
                suggested = best

        checks.append(SentenceCheck(
            text=text,
            cited_ids=cited,
            valid_ids=valid,
            fabricated_ids=fabricated,
            support=round(support, 4),
            per_source_support=per_source,
            missing_numbers=missing,
            is_claim=is_claim,
            is_factual=factual,
            abstention=abstention,
            supported=supported,
            uncited_factual=factual and not cited,
            suggested_id=suggested,
        ))

    claims = [c for c in checks if c.is_claim]
    flags = []
    if all_fabricated:
        flags.append("fabricated_ids")
    if any(c.uncited_factual for c in checks):
        flags.append("uncited_factual")
    if any(c.missing_numbers for c in checks):
        flags.append("unsupported_numbers")
    if any(c.cited_ids and not c.supported and not c.fabricated_ids for c in checks):
        flags.append("weak_support")

    return CitationReport(
        sentences=checks,
        citation_precision=(good_citations / total_citations) if total_citations else None,
        supported_ratio=(sum(c.supported for c in claims) / len(claims)) if claims else None,
        total_citations=total_citations,
        fabricated_ids=all_fabricated,
        uncited_factual_count=sum(c.uncited_factual for c in checks),
        abstained=abstained,
        threshold=threshold,
        flags=flags,
    )


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"%{value * 100:.0f}"


def render_report(report: CitationReport) -> str:
    """Atıf raporunu Türkçe, okunabilir metin olarak yaz."""
    lines = ["Atıf Doğrulama Raporu", "=" * 40]
    lines.append(f"Atıf kesinliği (citation precision): {_pct(report.citation_precision)}"
                 f"  ({report.total_citations} atıf)")
    lines.append(f"Desteklenen cümle oranı: {_pct(report.supported_ratio)}"
                 f"  (eşik: {report.threshold:.2f})")
    if report.abstained:
        lines.append(f"Model cevap vermekten kaçındı: \"{NO_ANSWER}\"")
    if report.fabricated_ids:
        lines.append("UYARI: Uydurma kaynak numaraları: "
                     + ", ".join(f"[{i}]" for i in report.fabricated_ids))
    if report.uncited_factual_count:
        lines.append(f"UYARI: Atıfsız olgu cümlesi sayısı: {report.uncited_factual_count}")
    lines.append("-" * 40)
    for i, c in enumerate(report.sentences, start=1):
        if c.abstention:
            status = "KAÇINMA"
        elif not c.is_claim:
            status = "—"
        elif c.fabricated_ids:
            status = "UYDURMA ATIF"
        elif c.supported:
            status = "DESTEKLENİYOR"
        elif c.cited_ids:
            status = "ZAYIF DESTEK"
        elif c.uncited_factual:
            status = "ATIFSIZ OLGU"
        else:
            status = "ATIFSIZ"
        cites = ", ".join(f"[{n}]" for n in c.cited_ids) or "yok"
        snippet = c.text if len(c.text) <= 100 else c.text[:97] + "..."
        lines.append(f"{i}. {status} | atıf: {cites} | destek: {c.support:.2f}")
        lines.append(f"   \"{snippet}\"")
        if c.missing_numbers:
            lines.append("   Kaynakta bulunmayan sayı/madde: " + ", ".join(c.missing_numbers))
        if c.suggested_id is not None:
            lines.append(f"   Öneri: bu cümle [{c.suggested_id}] ile desteklenebilir.")
    return "\n".join(lines)
