# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak RAG — Kaynaklı Prompt Oluşturma

Arama sonuçlarını numaralı kaynaklar halinde ChatTemplate'e hazır bir mesaj
listesine dönüştürür:

    [sistem]    Yalnız numaralı kaynaklardan cevap ver, [1] biçiminde atıf
                yap, bilgi yoksa "Kaynaklarda bu bilgi yok." de.
    [kullanıcı] Kaynaklar:
                [1] <başlık> — Madde 5: <metin>
                [2] ...
                Soru: <soru>

Kaynaklar token bütçesine sığdırılır: önce en yüksek skorlu kaynaklar
alınır, sığmayan en düşük skorlular düşürülür; tek kaynak bile sığmıyorsa
metni token düzeyinde kırpılır. Aynı kod 2K bağlamlı modelde 2–3 kaynakla,
YaRN ile 32K'ya genişletilmiş modelde onlarca kaynakla çalışır.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from rag.chunker import Chunk

NO_ANSWER = "Kaynaklarda bu bilgi yok."

DEFAULT_SYSTEM_PROMPT = (
    "Sen Toprak'sın; mevzuat ve yönetmelik metinleri üzerine çalışan bir "
    "Türkçe asistansın. Soruyu YALNIZCA aşağıda numaralandırılmış kaynaklara "
    "dayanarak cevapla. Her bilgiden sonra dayandığın kaynağın numarasını "
    "köşeli parantez içinde yaz, örneğin [1] veya [2]. Kaynaklarda olmayan "
    "bilgi ekleme, madde ya da kanun numarası uydurma. Cevap kaynaklarda "
    f"yoksa yalnızca \"{NO_ANSWER}\" yaz. Bu bir hukuki danışmanlık değildir."
)


@dataclass
class NumberedSource:
    """Prompt'a giren numaralı kaynak."""
    number: int
    chunk: Chunk
    score: float
    rendered: str
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "score": round(self.score, 4),
            "truncated": self.truncated,
            "chunk": self.chunk.to_dict(),
        }


@dataclass
class RagPrompt:
    """Oluşturulan prompt: mesajlar, kullanılan ve düşürülen kaynaklar."""
    messages: List[Dict[str, str]]
    sources: List[NumberedSource]
    dropped: list = field(default_factory=list)
    prompt_tokens: Optional[int] = None


def source_header(number: int, chunk: Chunk) -> str:
    head = f"[{number}] {chunk.title}"
    if chunk.article:
        head += f" — {chunk.article}"
    return head


def render_source(number: int, chunk: Chunk, text: Optional[str] = None) -> str:
    """Kaynağı "[n] <başlık> — Madde X: metin" biçiminde yaz."""
    body = " ".join((chunk.text if text is None else text).split())
    return f"{source_header(number, chunk)}: {body}"


def render_user_message(question: str, rendered_sources: Sequence[str]) -> str:
    if rendered_sources:
        block = "\n\n".join(rendered_sources)
    else:
        block = "(Kaynak bulunamadı.)"
    return (
        f"Kaynaklar:\n\n{block}\n\n"
        f"Soru: {question.strip()}\n"
        "Cevabı yalnız bu kaynaklara dayanarak, atıflarıyla birlikte ver."
    )


def make_token_counter(tokenizer=None) -> Callable[[str], int]:
    """Tokenizer varsa gerçek token sayısı, yoksa ~3.5 karakter/token tahmini."""
    if tokenizer is None:
        return lambda text: max(1, (len(text) + 3) // 4) if text else 0
    return lambda text: len(tokenizer.encode(text, add_bos=False, add_eos=False))


def _result_parts(result):
    """SearchResult veya (chunk, score) / Chunk kabul et."""
    if hasattr(result, "chunk"):
        return result.chunk, float(getattr(result, "score", 0.0))
    if isinstance(result, Chunk):
        return result, 0.0
    chunk, score = result
    return chunk, float(score)


def _truncate_to_tokens(text: str, budget: int, tokenizer, counter) -> str:
    if budget <= 0:
        return ""
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        if len(ids) <= budget:
            return text
        return tokenizer.decode(ids[:budget]).rstrip() + " …"
    # Tokenizer yoksa kelime bazında kırp
    words, out = text.split(), []
    for w in words:
        if counter(" ".join(out + [w])) > budget:
            break
        out.append(w)
    return " ".join(out) + " …"


def build_rag_messages(
    question: str,
    results: Sequence,
    tokenizer=None,
    max_prompt_tokens: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    template=None,
    min_source_tokens: int = 32,
) -> RagPrompt:
    """
    ChatTemplate'e hazır, kaynaklı mesaj listesi oluştur.

    Args:
        question: Kullanıcı sorusu
        results: SearchResult listesi (veya (Chunk, skor) çiftleri)
        tokenizer: Token sayımı için (encode(text, add_bos, add_eos))
        max_prompt_tokens: Prompt için toplam token bütçesi (ör. bağlam
            uzunluğu − üretilecek token sayısı). None ise sınır yok.
        template: Verilirse (ChatTemplate) bütçe şablonun gerçek
            kodlamasıyla kesin olarak doğrulanır.
        min_source_tokens: Tek kaynak kırpılacaksa en az bu kadar token kalmalı

    Returns:
        RagPrompt — kaynaklar skor sırasıyla [1], [2], … numaralanır.
    """
    counter = make_token_counter(tokenizer)
    ranked = sorted((_result_parts(r) for r in results), key=lambda cs: -cs[1])
    ranked_raw = sorted(results, key=lambda r: -_result_parts(r)[1])

    def messages_for(rendered: List[str]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": render_user_message(question, rendered)},
        ]

    def measure(rendered: List[str]) -> int:
        msgs = messages_for(rendered)
        if template is not None:
            return len(template.encode_prompt(msgs))
        # Rol işaretleri + BOS + tur sonları için küçük pay
        return sum(counter(m["content"]) + 4 for m in msgs) + 2

    selected: List[NumberedSource] = []
    if max_prompt_tokens is None:
        for i, (chunk, score) in enumerate(ranked, start=1):
            selected.append(NumberedSource(i, chunk, score, render_source(i, chunk)))
        dropped = []
    else:
        overhead = measure([])
        used = overhead
        dropped = []
        for idx, (chunk, score) in enumerate(ranked):
            number = len(selected) + 1
            rendered = render_source(number, chunk)
            cost = counter(rendered) + 2
            if used + cost <= max_prompt_tokens:
                selected.append(NumberedSource(number, chunk, score, rendered))
                used += cost
                continue
            if not selected:
                room = max_prompt_tokens - used - counter(source_header(number, chunk) + ": ") - 4
                if room >= min_source_tokens:
                    text = _truncate_to_tokens(" ".join(chunk.text.split()), room, tokenizer, counter)
                    rendered = render_source(number, chunk, text)
                    selected.append(NumberedSource(number, chunk, score, rendered, truncated=True))
                    used += counter(rendered) + 2
                    continue
            # Skor sırasıyla eklendiği için kalan her şey daha düşük skorlu
            dropped = list(ranked_raw[idx:])
            break
        # Kesin doğrulama: tahmini bütçe aşıldıysa en düşük skorluyu düşür
        while selected and measure([s.rendered for s in selected]) > max_prompt_tokens:
            dropped.insert(0, ranked_raw[len(selected) - 1])
            selected.pop()

    rendered = [s.rendered for s in selected]
    return RagPrompt(
        messages=messages_for(rendered),
        sources=selected,
        dropped=dropped,
        prompt_tokens=measure(rendered) if (template is not None or tokenizer is not None) else None,
    )
