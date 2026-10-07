# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak RAG — Türkçe Metin İşleme

Geri getirme (retrieval) için Türkçe'ye duyarlı yardımcılar:

- `turkish_lower` / `turkish_upper`: İ→i, I→ı (ve tersi) dönüşümlerini doğru
  yapan büyük/küçük harf çevirimi. Python'un `str.lower()` fonksiyonu "İ"
  harfini "i̇" (i + birleşik nokta) yapar ve "I" harfini "i" yapar; ikisi de
  Türkçe için yanlıştır.
- `normalize`: Unicode NFC, Türkçe küçük harf, tırnak/tire birleştirme,
  boşluk sadeleştirme. Aksan (ç, ğ, ı, ö, ş, ü) silme isteğe bağlıdır ve
  varsayılan olarak KAPALIDIR (Türkçe'de "kar"/"kâr", "sık"/"sik" gibi
  ayrımlar anlam taşır).
- `tokenize`: kelimeler, sayılar ("5237", "01.01.2020", "12/3") ve madde
  atıfları ("Madde 12", "m. 12/3", "12 nci madde", "Geçici Madde 1") için
  özel terimler ("madde:12", "madde:12/3", "geçici_madde:1").
- `stem`: Hafif, sezgisel (heuristic) Türkçe ek budayıcı. Gerçek bir
  morfolojik çözümleyici DEĞİLDİR; yaygın çekim eklerini en uzun eşleşme
  ilkesiyle, en az 3 harflik kök bırakacak şekilde siler. Sorgu ve belgeye
  aynı işlem uygulandığı için tutarlılık, dilbilimsel doğruluktan daha
  önemlidir. Hukuk metinlerinde sık geçen kökler için küçük bir koruma
  sözlüğü vardır ("kanun", "madde", "yönetmelik" …).
- `split_sentences`: Kısaltmalara (Md., vb., Dr., T.C., s., No.) ve
  numaralı maddelere/fıkralara duyarlı cümle bölücü; karakter ofsetleri
  korunur.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# ── Büyük/küçük harf ─────────────────────────────────────

_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})
_UPPER_MAP = str.maketrans({"i": "İ", "ı": "I"})
_DIACRITIC_MAP = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "â": "a", "î": "i", "û": "u",
})


def turkish_lower(text: str) -> str:
    """Türkçe kurallarıyla küçük harfe çevir (İ→i, I→ı)."""
    return text.translate(_LOWER_MAP).lower()


def turkish_upper(text: str) -> str:
    """Türkçe kurallarıyla büyük harfe çevir (i→İ, ı→I)."""
    return text.translate(_UPPER_MAP).upper()


def strip_diacritics(text: str) -> str:
    """Türkçe ve şapkalı harfleri ASCII karşılıklarına indir (küçük harf metin için)."""
    return text.translate(_DIACRITIC_MAP)


_QUOTE_MAP = str.maketrans({
    "’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'",
    "“": '"', "”": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "−": "-",
})


def normalize(text: str, strip_accents: bool = False) -> str:
    """
    Arama için metni normalize et.

    Args:
        text: Ham metin
        strip_accents: True ise ç/ğ/ı/ö/ş/ü ASCII'ye indirilir (varsayılan: hayır)
    """
    text = unicodedata.normalize("NFC", text).translate(_QUOTE_MAP)
    text = turkish_lower(text)
    if strip_accents:
        text = strip_diacritics(text)
    return re.sub(r"\s+", " ", text).strip()


# ── Durak kelimeler ──────────────────────────────────────

STOPWORDS = frozenset("""
acaba ama ancak artık aslında az bana bazen bazı bazıları belki ben benden beni benim
beri bile bir birçok biri birkaç birşey biz bizden bize bizi bizim bu buna bunda bundan
bunlar bunları bunların bunu bunun burada böyle böylece çok çünkü da daha dahi de defa
değil diğer diye dolayı dolayısıyla eğer en fakat gibi göre hâlâ hangi hatta hem hep
hepsi her herhangi hiç hiçbir için ile ilgili ise işte kadar karşın kendi kendine
kez ki kim kimse mi mı mu mü na nasıl ne neden nedenle nerede nereye niçin niye o olan
olarak oldu olduğu olduğunu olmak olması olup olur ona ondan onlar onları onların onu
onun öyle ötürü sadece sanki siz sizden size sizi sizin şayet şey şimdi şöyle şu şuna
şunda şundan şunlar şunu şunun tarafından tüm üzere var veya ve ya yani yine yoksa
zaten nedir midir mıdır hangisi kaç
""".split())


# ── Tokenizasyon ─────────────────────────────────────────

_ORDINAL = r"(?:'?\s*(?:nci|ncı|ncu|ncü|inci|ıncı|uncu|üncü)|\.)"

_TOKEN_RE = re.compile(
    r"(?P<art>(?<!\w)(?:(?P<art_kind>geçici|ek)\s+)?(?:madde|md\.|mad\.|m\.)\s*"
    r"(?P<art_no>\d+)(?:\s*/\s*(?P<art_par>\d+))?)"
    r"|(?P<oart>(?<!\w)(?P<oart_no>\d+)\s*" + _ORDINAL + r"\s*madde\w*)"
    r"|(?P<num>\d+(?:[.,/]\d+)*)"
    r"|(?P<word>[^\W\d_]+(?:'[^\W\d_]+)?)"
)


def article_term(number, kind: str = "", paragraph=None) -> str:
    """Madde atfı için arama terimi: "madde:12", "madde:12/3", "geçici_madde:1"."""
    kind = turkish_lower(kind or "").strip()
    prefix = f"{kind}_madde" if kind in ("geçici", "ek") else "madde"
    term = f"{prefix}:{int(number)}"
    if paragraph is not None:
        term += f"/{int(paragraph)}"
    return term


@dataclass
class Token:
    """Ham arama tokenı."""
    text: str
    kind: str   # "word" | "number" | "article"


def tokenize(text: str, strip_accents: bool = False) -> List[Token]:
    """
    Metni arama tokenlarına ayır (normalize edilmiş, küçük harf).

    Madde atıfları tek bir terim olur; "Madde 12/3" hem "madde:12" hem
    "madde:12/3" üretir. Kesme işaretinden sonraki ek atılır ("Kanun'un" →
    "kanun").
    """
    norm = normalize(text, strip_accents=strip_accents)
    tokens: List[Token] = []
    for m in _TOKEN_RE.finditer(norm):
        if m.group("art"):
            kind = m.group("art_kind") or ""
            tokens.append(Token(article_term(m.group("art_no"), kind), "article"))
            if m.group("art_par"):
                tokens.append(Token(article_term(m.group("art_no"), kind, m.group("art_par")), "article"))
        elif m.group("oart"):
            tokens.append(Token(article_term(m.group("oart_no")), "article"))
        elif m.group("num"):
            tokens.append(Token(m.group("num").rstrip(".,/"), "number"))
        else:
            word = m.group("word").split("'")[0]
            if word:
                tokens.append(Token(word, "word"))
    return tokens


# ── Kök bulucu (sezgisel) ────────────────────────────────

# Yaygın çekim ekleri (ünlü uyumuna göre tüm biçimler). Yapım ekleri
# ("-lık", "-cı" …) bilinçli olarak dahil edilmedi: anlamı değiştirirler.
_SUFFIXES = sorted(set("""
lar ler
ları leri larını lerini larının lerinin larına lerine larında lerinde larından lerinden
larla lerle
ın in un ün nın nin nun nün yın yin yun yün
ı i u ü yı yi yu yü nı ni nu nü
a e ya ye na ne
da de ta te nda nde
dan den tan ten ndan nden
la le yla yle
sı si su sü
ım im um üm ımız imiz umuz ümüz ınız iniz unuz ünüz
dır dir dur dür tır tir tur tür
ki
ca ce ça çe
mak mek ması mesi masına mesine
""".split()), key=len, reverse=True)

_VOWELS = set("aeıioöuüâîû")
_BUFFER = set("nys")

# Hukuk metinlerinde sık geçen ve budayıcının bozabileceği kökler.
# Kelime bu öneklerden biriyle başlıyorsa önek kök olarak döner.
PROTECTED_STEMS = (
    "kanun", "madde", "yönetmelik", "yönerge", "tüzük", "karar", "fıkra", "bent",
    "hüküm", "hükm", "mevzuat", "kurul", "kurum", "üye", "süre", "ceza", "dava",
    "mahkeme", "başvuru", "ödünç", "kütüphane", "belediye", "bakanlık", "sözleşme",
)
_PROTECTED_CANON = {"hükm": "hüküm"}

_SOFTEN = {"b": "p", "c": "ç", "ğ": "k"}

MIN_STEM = 3


def stem(word: str, min_stem: int = MIN_STEM) -> str:
    """
    Sezgisel Türkçe ek budama (en uzun eşleşme önce).

    Kurallar:
      - Ünlüyle başlayan ek yalnız ünsüzle biten köke, kaynaştırma ünsüzüyle
        (n/y/s) başlayan ek yalnız ünlüyle biten köke uygulanır.
      - Kalan kök en az `min_stem` harf olmalıdır.
      - En fazla 3 tur ek silinir; sonra b→p, c→ç, ğ→k yumuşama geri alınır.
      - Korunan kökler (PROTECTED_STEMS) doğrudan döndürülür.

    Bu bir morfolojik çözümleyici değildir; arama için tutarlı bir
    eşleştirme anahtarı üretir.
    """
    if not word or not word.isalpha():
        return word
    for prefix in PROTECTED_STEMS:
        if word.startswith(prefix):
            return _PROTECTED_CANON.get(prefix, prefix)
    if len(word) <= min_stem + 1:
        return word

    current = word
    stripped = False
    for _ in range(3):
        removed = False
        for suffix in _SUFFIXES:
            if not current.endswith(suffix):
                continue
            base = current[: -len(suffix)]
            if len(base) < min_stem:
                continue
            last = base[-1]
            if suffix[0] in _VOWELS and last in _VOWELS:
                continue
            if suffix[0] in _BUFFER and len(suffix) > 1 and suffix[1] in _VOWELS and last not in _VOWELS:
                continue
            current = base
            removed = stripped = True
            break
        if not removed:
            break
    if stripped and current[-1] in _SOFTEN:
        current = current[:-1] + _SOFTEN[current[-1]]
    return current


def analyze(
    text: str,
    use_stemming: bool = True,
    remove_stopwords: bool = True,
    strip_accents: bool = False,
) -> List[str]:
    """
    Metni BM25 terimlerine dönüştür: kelimeler köklenir, durak kelimeler
    atılır; sayılar ve madde atıfları olduğu gibi korunur.
    """
    terms: List[str] = []
    for tok in tokenize(text, strip_accents=strip_accents):
        if tok.kind != "word":
            terms.append(tok.text)
            continue
        if remove_stopwords and tok.text in STOPWORDS:
            continue
        if len(tok.text) < 2:
            continue
        terms.append(stem(tok.text) if use_stemming else tok.text)
    return terms


def is_number_term(term: str) -> bool:
    return bool(term) and term[0].isdigit()


def is_article_term(term: str) -> bool:
    return "madde:" in term


# ── Madde başlıkları ─────────────────────────────────────

ARTICLE_HEADER_RE = re.compile(
    r"^[ \t]*(?:(?P<kind>GEÇİCİ|Geçici|GEÇICI|EK|Ek)[ \t]+)?(?:MADDE|Madde)[ \t]+"
    r"(?P<no>\d+)(?:[ \t]*/[ \t]*(?P<sub>[A-Za-z]))?[ \t]*(?:[-–—:.]|$)",
    re.MULTILINE,
)


def parse_article_header(line: str) -> Optional[str]:
    """
    Satır bir madde başlığıysa ("MADDE 5 –", "Madde 12-", "Geçici Madde 1 –")
    okunabilir etiketini döndür ("Madde 5", "Geçici Madde 1"), değilse None.
    """
    m = ARTICLE_HEADER_RE.match(line)
    if not m:
        return None
    kind = turkish_lower(m.group("kind") or "")
    prefix = {"geçici": "Geçici Madde", "ek": "Ek Madde"}.get(kind, "Madde")
    label = f"{prefix} {int(m.group('no'))}"
    if m.group("sub"):
        label += f"/{m.group('sub').upper()}"
    return label


def article_key(label: Optional[str]) -> Optional[str]:
    """Madde etiketinin arama terimi ("Geçici Madde 1" → "geçici_madde:1")."""
    if not label:
        return None
    terms = [t.text for t in tokenize(label) if t.kind == "article"]
    return terms[0] if terms else None


# ── Cümle bölücü ─────────────────────────────────────────

ABBREVIATIONS = frozenset("""
md. mad. vb. vs. vd. dr. prof. doç. av. t.c. s. sy. no. nu. bkz. örn. yy. y. m. fık. c.
sk. cad. mah. apt. tel. st. sn. say. r.g. rg. ltd. şti. a.ş. bl. ks. yön. kan. hk. bşk.
gn. gnl. alb. yrd. öğr. gör. uzm. hz. ord. ar. arş. dk. sa. sf. ss. tic. san. müd.
""".split())

_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:\(\d+\)|\d+[.)]|[a-zçğıöşü]\)|[-•*])[ \t]+")
_SENT_END_RE = re.compile(r"[.!?…]+[\"')\]]*")


@dataclass
class Sentence:
    """Ofsetleriyle birlikte bir cümle (text == kaynak[start:end])."""
    text: str
    start: int
    end: int


def _is_heading_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("#"):
        return True
    return len(s) <= 80 and s[-1] not in ".;:,!?" and (s[0].isupper() or s[0].isdigit())


def _line_spans(text: str) -> List[Tuple[int, int]]:
    spans, pos = [], 0
    for line in text.splitlines(keepends=True):
        spans.append((pos, pos + len(line.rstrip("\r\n"))))
        pos += len(line)
    return spans


def _blocks(text: str) -> List[Tuple[int, int]]:
    """Metni satır yapısına göre bloklara ayır (boş satır, madde başlığı, liste öğesi, başlık)."""
    blocks: List[Tuple[int, int]] = []
    cur: Optional[List[int]] = None
    lines = _line_spans(text)
    for i, (s, e) in enumerate(lines):
        line = text[s:e]
        if not line.strip():
            if cur:
                blocks.append((cur[0], cur[1]))
            cur = None
            continue
        next_line = text[lines[i + 1][0]:lines[i + 1][1]] if i + 1 < len(lines) else ""
        is_header = parse_article_header(line) is not None
        is_heading = _is_heading_line(line) and (
            line.strip().startswith("#")
            or parse_article_header(next_line) is not None
            or (line.strip().isupper() and len(line.strip()) > 3)
        )
        starts_new = is_header or is_heading or bool(_LIST_ITEM_RE.match(line))
        if cur is None or starts_new:
            if cur:
                blocks.append((cur[0], cur[1]))
            cur = [s, e]
        else:
            cur[1] = e
        if is_heading:
            blocks.append((cur[0], cur[1]))
            cur = None
    if cur:
        blocks.append((cur[0], cur[1]))
    return blocks


def _trimmed(text: str, start: int, end: int) -> Optional[Sentence]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return Sentence(text[start:end], start, end)


def _is_boundary(block: str, punct_start: int, punct_end: int) -> bool:
    punct = block[punct_start:punct_end]
    if punct_end < len(block) and not block[punct_end].isspace():
        return False
    rest = block[punct_end:].lstrip()
    if not rest:
        return True
    nxt = rest[0]
    if punct.startswith(".") and len(punct.rstrip("\"')]")) == 1:
        word_start = punct_start
        while word_start > 0 and not block[word_start - 1].isspace():
            word_start -= 1
        word = turkish_lower(block[word_start:punct_start + 1]).lstrip("(\"'")
        if word in ABBREVIATIONS:
            return False
        bare = word[:-1]
        if len(bare) == 1 and bare.isalpha():
            return False                     # baş harf: "A. Yılmaz"
        if bare.isdigit() and (nxt.islower() or nxt.isdigit()):
            return False                     # sıra sayısı: "15. madde"
        if nxt.islower():
            return False
    return True


def split_sentences(text: str, offset: int = 0) -> List[Sentence]:
    """
    Kısaltma ve madde yapısına duyarlı cümle bölücü.

    Madde başlıkları, liste öğeleri ("(1)", "a)") ve boş satırlar her zaman
    yeni cümle başlatır. Dönen ofsetler `offset` eklenerek verilir.
    """
    sentences: List[Sentence] = []
    for b_start, b_end in _blocks(text):
        block = text[b_start:b_end]
        last = 0
        for m in _SENT_END_RE.finditer(block):
            if _is_boundary(block, m.start(), m.end()):
                sent = _trimmed(text, b_start + last, b_start + m.end())
                if sent:
                    sentences.append(sent)
                last = m.end()
        sent = _trimmed(text, b_start + last, b_end)
        if sent:
            sentences.append(sent)
    if offset:
        sentences = [Sentence(s.text, s.start + offset, s.end + offset) for s in sentences]
    return sentences


def sentence_texts(text: str) -> List[str]:
    return [s.text for s in split_sentences(text)]


def unique(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
