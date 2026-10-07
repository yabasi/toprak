# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Arkifonemik Türkçe Biçimbilgisi (Morfofonoloji)

Türkçe ekler soyut biçimde, *arkifonemlerle* yazılır; yüzey biçimi kök ve
önceki eklerden kurallarla türetilir:

    A  → a / e            (iki yönlü / büyük ünlü uyumu)
    H  → ı / i / u / ü    (dört yönlü / küçük ünlü uyumu)
    D  → d / t            (sert ünsüzden sonra t; "fıstıkçı şahap")
    C  → c / ç            (sert ünsüzden sonra ç)
    G  → g / k            (sert ünsüzden sonra k; ör. sev+GH → sevgi)
    (y), (n), (s), (ş)    kaynaştırma ünsüzü: yalnız ünlüyle biten biçimden sonra
    (H), (A)              koşullu ünlü: yalnız ünsüzle biten biçimden sonra

Örnekler:  kitap+lAr+DA → kitaplarda,  ev+(y)A → eve,  kitap+(s)H → kitabı,
           oku+Hyor → okuyor,  bekle+Hyor → bekliyor.

Bu modül:
  - `realize(stem, suffixes)`: deterministik yüzey gerçekleştirme
    (ünlü uyumu, ünsüz benzeşmesi, kaynaştırma, süreksiz ünsüz yumuşaması,
    -Hyor öncesi ünlü daralması, alıntı kelime uyum istisnaları, Türkçe
    büyük/küçük harf).
  - `abstract_suffix(surface)`: yüzey ek → arkifonemik biçim (belgelenmiş tablo).
  - `ArchiphonemeCodec`: tokenizer eğitimi için soyut metin biçimi
    ("kitap+lAr+DA") üretir ve geri çözer (`decode`).
  - `transform_corpus_line(line, segmenter)`: korpus satırını soyut biçime
    çevirir; segmenter takılabilir (Zemberek / TRmorph çıktısı).
  - `RuleBasedSegmenter`: küçük, SEZGİSEL (heuristic) yerleşik segmenter (demo).
  - `archiphoneme_user_symbols()`: SentencePiece `user_defined_symbols` listesi.

Soyut metin biçimi (tokenizer eğitimi için kesin tanım):
  * Kelimeler boşlukla ayrılır (orijinal boşluklar aynen korunur).
  * Çekimli kelime = kök + her ek için "+" + arkifonemik ek, ARADA BOŞLUK YOK:
        "kitaplarda"   → "kitap+lAr+DA"
        "Türkiye'nin"  → "Türkiye'+(n)Hn"   (kesme işareti kökte kalır)
        "evlerde."     → "ev+lAr+DA."       (sondaki noktalama en sona eklenir)
  * Boşluksuz biçim seçildi çünkü SentencePiece `user_defined_symbols` ile her
    "+ek" tek bir token olur ve "▁" kelime sınırı bilgisi bozulmaz.
  * Çözümde yalnız tanınan ek dizileri ("+" + bilinen arkifonemik ek veya
    arkifonem harfi / parantez içeren ek) birleştirilir; "a+b", "C++" gibi
    düz metinler olduğu gibi kalır.

Sınırlamalar: düzensiz fiiller (git→gidiyor, et→eder; de/ye+Hyor desteklenir),
sözlüksel ünlü düşmesi (ağız→ağzı, burun→burnu), ikizleşme (hak→hakkı) ve
sayılardan sonraki ekler desteklenmez. Yerleşik segmenter sözlüksüzdür ve
hatalı kök bulabilir; ama dönüşüm her kelime için gidiş-dönüşü doğrular,
bu yüzden `decode(transform(line)) == line` korunur.
"""

import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ── Ses Sınıfları ──────────────────────────────────────────

BACK_VOWELS = set("aıou")
FRONT_VOWELS = set("eiöü")
VOWELS = BACK_VOWELS | FRONT_VOWELS | set("âîû")
ROUNDED = set("ouöüû")
VOICELESS = set("fstkçşhp")          # fıstıkçı şahap
ARCHIPHONEMES = set("AHDCG")
APOSTROPHES = ("'", "’")

# Şapkalı ünlülerin uyum açısından karşılıkları
_CIRCUMFLEX = {"â": "a", "î": "i", "û": "u"}
# Kalın ünlü → ince karşılığı (alıntı kelime uyum istisnaları için)
_BACK_TO_FRONT = {"a": "e", "ı": "i", "o": "ö", "u": "ü"}
_SOFTEN = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}


# ── İstisna Listeleri ──────────────────────────────────────

# Son ünlüsü kalın olduğu halde ekleri İNCE alan alıntı kelimeler
# (çoğu ince "l" ya da Arapça "-at"; ör. saat+lAr → saatler, rol+(y)H → rolü).
DEFAULT_HARMONY_EXCEPTIONS: Set[str] = {
    "saat", "kalp", "harf", "hal", "rol", "gol", "alkol", "petrol",
    "istikbal", "istiklal", "ihtimal", "ihmal", "ikbal", "hayal", "misal",
    "meal", "kemal", "cemal", "celal", "hilal", "kabul", "usul", "mahsul",
    "meşgul", "kontrol", "sembol", "protokol", "festival", "terminal",
    "metal", "mineral", "liberal", "general", "moral", "sinyal", "dikkat",
    "itaat", "cemaat", "kabahat", "ziraat", "şefkat", "seyahat",
    "hakikat", "inkılap", "infilak", "idrak", "ittifak", "iştirak",
    "emsal", "evrak", "santral", "mentol", "monopol", "vokal", "global",
}

# Ünlüyle başlayan ekten önce YUMUŞAMAYAN kelimeler. Tek heceliler zaten
# varsayılan olarak yumuşamaz; en yaygınları belge amaçlı listelenmiştir.
DEFAULT_NON_SOFTENING: Set[str] = {
    # tek heceli
    "at", "et", "ot", "it", "süt", "saç", "top", "ip", "ok", "kek", "tek",
    "yok", "park", "tank", "kat", "sap", "küp", "hap", "koç", "suç", "göç",
    "haç", "ait",
    # çok heceli (alıntılar, -t ile bitenler belge amaçlı)
    "hukuk", "ahlak", "merak", "sepet", "devlet", "millet", "hayat",
    "sanat", "cumhuriyet", "saat", "adet", "ibadet", "dikkat", "sıfat",
    "fiyat", "kuvvet", "hürriyet", "medeniyet", "şirket", "hareket",
    "cesaret", "ticaret", "ziyaret", "rahat", "zanaat", "tabiat", "ihtiyat",
    "evrak", "idrak", "ittifak", "iştirak", "infilak", "tebrik",
    # ünsüzle biten çok heceli fiiller (fiil kökleri genelde yumuşamaz)
    "bırak", "kapat", "uzat", "anlat", "yaklaş", "kopar",
}

# Tek heceli olduğu (veya -t ile bittiği) halde YUMUŞAYAN kelimeler
# (nk→ng ayrıca otomatik: renk → rengi).
DEFAULT_SOFTENING: Set[str] = {
    "kalp", "dip", "uç", "kap", "harp", "yurt", "kurt", "dört", "çok",
    "genç", "renk", "denk", "kanat", "umut", "armut", "dert", "yoğurt",
}


# ── Türkçe Büyük/Küçük Harf ────────────────────────────────

def tr_lower(text: str) -> str:
    """Türkçe'ye duyarlı küçük harf (I→ı, İ→i)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(text: str) -> str:
    """Türkçe'ye duyarlı büyük harf (i→İ, ı→I)."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def _norm_key(word: str) -> str:
    """İstisna listesi araması için anahtar: küçük harf, şapkasız, yalnız harfler."""
    low = tr_lower(word)
    low = "".join(_CIRCUMFLEX.get(ch, ch) for ch in low)
    return "".join(ch for ch in low if ch.isalpha())


def _is_vowel(ch: str) -> bool:
    return ch in VOWELS


def _last_letter(form: str) -> str:
    for ch in reversed(form):
        if ch.isalpha():
            return ch
    return ""


def _vowel_count(word: str) -> int:
    return sum(1 for ch in word if _is_vowel(ch))


# ── Ek Ayrıştırma ──────────────────────────────────────────

_SEGMENT_RE = re.compile(r"\(([^()]+)\)|(.)")


def _parse_suffix(suffix: str) -> List[Tuple[str, str]]:
    """
    Arkifonemik eki (tür, karakter) parçalarına ayır.

    Türler: 'lit' (sabit harf), 'arch' (A/H/D/C/G), 'buf_c' (parantezli
    ünsüz: ünlüden sonra çıkar), 'buf_v' (parantezli ünlü/arkifonem:
    ünsüzden sonra çıkar).
    """
    out: List[Tuple[str, str]] = []
    for m in _SEGMENT_RE.finditer(suffix):
        if m.group(1) is not None:
            inner = m.group(1)
            for ch in inner:
                is_v = ch in ("A", "H") or _is_vowel(ch)
                out.append(("buf_v" if is_v else "buf_c", ch))
        else:
            ch = m.group(2)
            out.append(("arch" if ch in ARCHIPHONEMES else "lit", ch))
    return out


# ── Gerçekleştirme (realize) ───────────────────────────────

def _harmony_vowel(form: str, stem_end: int, stem_exception: bool) -> str:
    """Biçimdeki son ünlüyü döndür (istisna kökte inceye çevrilmiş)."""
    for i in range(len(form) - 1, -1, -1):
        ch = _CIRCUMFLEX.get(form[i], form[i])
        if ch in BACK_VOWELS or ch in FRONT_VOWELS:
            if stem_exception and i < stem_end and ch in BACK_VOWELS:
                return _BACK_TO_FRONT[ch]
            return ch
    return "e"  # ünlüsüz biçim (kısaltma vb.): ince varsayılır


def _realize_arch(ch: str, form: str, stem_end: int, stem_exception: bool) -> str:
    if ch == "A":
        v = _harmony_vowel(form, stem_end, stem_exception)
        return "a" if v in BACK_VOWELS else "e"
    if ch == "H":
        v = _harmony_vowel(form, stem_end, stem_exception)
        if v in BACK_VOWELS:
            return "u" if v in ROUNDED else "ı"
        return "ü" if v in ROUNDED else "i"
    prev = _last_letter(form)
    voiceless = prev in VOICELESS
    if ch == "D":
        return "t" if voiceless else "d"
    if ch == "C":
        return "ç" if voiceless else "c"
    if ch == "G":
        return "k" if voiceless else "g"
    raise ValueError(f"Bilinmeyen arkifonem: {ch}")


def _stem_softens(
    stem_key: str,
    non_softening: Set[str],
    softening: Set[str],
) -> bool:
    """Kökün ünlüyle başlayan ekten önce süreksiz ünsüz yumuşaması geçirip geçirmediği."""
    if not stem_key or stem_key[-1] not in _SOFTEN:
        return False
    if stem_key in non_softening:
        return False
    if stem_key in softening:
        return True
    if stem_key.endswith("nk"):
        return True                      # renk → rengi, ahenk → ahengi
    if _vowel_count(stem_key) <= 1:
        return False                     # tek heceliler çoğunlukla yumuşamaz
    if stem_key[-1] == "t":
        return False                     # çok heceli -t çoğunlukla alıntı/fiil
    return True                          # kitap → kitabı, köpek → köpeği


def _soften_form(form: str) -> str:
    """Biçimin son harfini yumuşat (nk → ng)."""
    idx = len(form) - 1
    while idx >= 0 and not form[idx].isalpha():
        idx -= 1
    if idx < 0:
        return form
    last = form[idx]
    if last == "k" and idx > 0 and form[idx - 1] == "n":
        new = "g"
    else:
        new = _SOFTEN.get(last, last)
    return form[:idx] + new + form[idx + 1:]


def realize(
    stem: str,
    suffixes: Sequence[str],
    non_softening: Optional[Iterable[str]] = None,
    harmony_exceptions: Optional[Iterable[str]] = None,
    softening_words: Optional[Iterable[str]] = None,
    soften: Optional[bool] = None,
) -> str:
    """
    Kök + arkifonemik ekler → Türkçe yüzey biçimi.

    Args:
        stem: Kök (sözlük biçimi). Sonunda kesme işareti varsa ("İstanbul'")
            özel isim kabul edilir: yazımda yumuşama yapılmaz, kesme korunur.
        suffixes: Arkifonemik ekler, ör. ["lAr", "DA"]; baştaki "+" isteğe bağlı.
        non_softening: Yumuşamayan kelimeler (varsayılan DEFAULT_NON_SOFTENING).
        harmony_exceptions: Ekleri ince alan alıntılar (DEFAULT_HARMONY_EXCEPTIONS).
        softening_words: Tek heceli olup yumuşayanlar (DEFAULT_SOFTENING).
        soften: Kök yumuşaması — None: kurallara göre, False: hiç, True: daima.

    Returns:
        Yüzey biçimi; kökün büyük/küçük harf düzeni korunur (tümü büyükse ekler
        de Türkçe kurallarıyla büyütülür).
    """
    non_soft = set(DEFAULT_NON_SOFTENING if non_softening is None else non_softening)
    exc = set(DEFAULT_HARMONY_EXCEPTIONS if harmony_exceptions is None else harmony_exceptions)
    soft_words = set(DEFAULT_SOFTENING if softening_words is None else softening_words)

    proper = stem.endswith(APOSTROPHES)
    form = tr_lower(stem)
    stem_end = len(form)
    key = _norm_key(stem)
    stem_exception = key in exc

    prev_is_stem = True
    prev_suffix = ""
    for raw in suffixes:
        suf = raw[1:] if raw.startswith("+") else raw
        if not suf:
            continue
        parts = _parse_suffix(suf)
        last = _last_letter(form)
        ends_vowel = _is_vowel(last)

        segs = []
        for kind, ch in parts:
            if kind == "buf_c" and not ends_vowel:
                continue
            if kind == "buf_v" and ends_vowel:
                continue
            segs.append(ch)

        # -Hyor: ünlüyle biten biçimde H düşer; son a/e ise daralır (ara→arı)
        if suf.startswith("Hyor") and ends_vowel and segs and segs[0] == "H":
            segs = segs[1:]
            idx = len(form) - 1
            while idx >= 0 and not form[idx].isalpha():
                idx -= 1
            if idx >= 0 and form[idx] in ("a", "e") and not proper:
                head = form[:idx]
                if any(_is_vowel(c) for c in head):
                    narrowed = _realize_arch("H", head, min(stem_end, len(head)), stem_exception)
                else:
                    narrowed = "ı" if form[idx] == "a" else "i"
                form = head + narrowed + form[idx + 1:]

        # Süreksiz ünsüz yumuşaması: ünlüyle başlayan ekten önce
        if segs:
            first = segs[0]
            starts_vowel = first in ("A", "H") or _is_vowel(first)
            if starts_vowel and last in _SOFTEN:
                if prev_is_stem:
                    do_soft = (not proper) and (
                        soften if soften is not None
                        else _stem_softens(key, non_soft, soft_words)
                    )
                else:
                    # Ek sonu k (-lHk, -(y)AcAk, -DHk) daima yumuşar
                    do_soft = prev_suffix.endswith("k") and prev_suffix not in ("k", "mAk")
                if do_soft:
                    form = _soften_form(form)

        for ch in segs:
            if ch in ARCHIPHONEMES:
                form += _realize_arch(ch, form, stem_end, stem_exception)
            else:
                form += ch
        prev_is_stem = False
        prev_suffix = suf

    return _restore_case(stem, form)


def _restore_case(stem: str, form: str) -> str:
    letters = [c for c in stem if c.isalpha()]
    all_upper = len(letters) >= 2 and all(c == tr_upper(c) for c in letters)
    out = []
    for i, ch in enumerate(form):
        if i < len(stem):
            out.append(tr_upper(ch) if stem[i] != tr_lower(stem[i]) else ch)
        else:
            out.append(tr_upper(ch) if all_upper else ch)
    return "".join(out)


# ── Ek Tablosu ─────────────────────────────────────────────

# (arkifonemik biçim, açıklama). Sıra önceliği belirler: aynı yüzey biçimi
# birden çok soyut eke karşılık gelirse ilk sıradaki seçilir (ör. "ı" →
# belirtme "(y)H"; iyelik "(s)H" ünsüzden sonra aynı yüzeyi verir, bu yüzden
# gidiş-dönüş bozulmaz).
SUFFIX_TABLE: List[Tuple[str, str]] = [
    ("lAr", "çoğul / 3ç kişi"),
    ("DA", "bulunma"),
    ("DAn", "ayrılma"),
    ("(y)H", "belirtme"),
    ("(y)A", "yönelme / istek"),
    ("(n)Hn", "ilgi"),
    ("(y)lA", "vasıta"),
    ("CA", "eşitlik / görelik"),
    ("(H)m", "iyelik 1t / kişi 1t"),
    ("(H)n", "iyelik 2t"),
    ("(s)H", "iyelik 3t"),
    ("(H)mHz", "iyelik 1ç"),
    ("(H)nHz", "iyelik 2ç"),
    ("lArH", "iyelik 3ç"),
    ("nDA", "zamir n'li bulunma (evi+nDA)"),
    ("nDAn", "zamir n'li ayrılma"),
    ("nA", "zamir n'li yönelme"),
    ("nH", "zamir n'li belirtme"),
    ("DH", "görülen geçmiş"),
    ("(y)DH", "ek-fiil hikâye"),
    ("mHş", "öğrenilen geçmiş"),
    ("(y)mHş", "ek-fiil rivayet"),
    ("(y)AcAk", "gelecek / sıfat-fiil"),
    ("Hyor", "şimdiki zaman (yor değişmez)"),
    ("sA", "dilek-şart"),
    ("(y)sA", "ek-fiil şart"),
    ("mAlH", "gereklilik"),
    ("(y)Hp", "zarf-fiil -ip"),
    ("(y)ken", "zarf-fiil -ken (değişmez)"),
    ("ki", "ilgi -ki (değişmez)"),
    ("DHr", "ek-fiil genişletme"),
    ("(y)Hm", "kişi 1t"),
    ("sHn", "kişi 2t"),
    ("(y)Hz", "kişi 1ç"),
    ("sHnHz", "kişi 2ç"),
    ("m", "kişi 1t (geçmiş/şart sonrası)"),
    ("n", "kişi 2t (geçmiş/şart sonrası)"),
    ("k", "kişi 1ç (geçmiş/şart sonrası)"),
    ("nHz", "kişi 2ç (geçmiş/şart sonrası)"),
    ("(H)r", "geniş zaman -Hr"),
    ("(A)r", "geniş zaman -Ar"),
    ("mA", "olumsuzluk / ad-fiil"),
    ("mAz", "olumsuz geniş zaman"),
    ("mAk", "mastar"),
    ("DHk", "sıfat-fiil -DHk"),
    ("(y)An", "sıfat-fiil -(y)An"),
    ("(y)Abil", "yeterlik"),
    ("lHk", "yapım -lHk"),
    ("CH", "yapım -CH"),
    ("lH", "yapım -lH"),
    ("sHz", "yapım -sHz"),
    ("GH", "yapım -GH"),
]

ABSTRACT_SUFFIXES: List[str] = [a for a, _ in SUFFIX_TABLE]
_ABSTRACT_SET = set(ABSTRACT_SUFFIXES)

_ARCH_CHOICES = {"A": "ae", "H": "ıiuü", "D": "dt", "C": "cç", "G": "gk"}


def _expand(abstract: str) -> Set[str]:
    """Bir soyut ekin olası tüm yüzey biçimleri (bağlamdan bağımsız)."""
    results = {""}
    for kind, ch in _parse_suffix(abstract):
        if kind in ("buf_c", "buf_v"):
            opts = {""} | set(_ARCH_CHOICES.get(ch, ch))
        elif kind == "arch":
            opts = set(_ARCH_CHOICES[ch])
        else:
            opts = {ch}
        results = {r + o for r in results for o in opts}
    if abstract.startswith("Hyor"):
        results |= {"yor"}
    softened = {r[:-1] + "ğ" for r in results if r.endswith("k") and abstract not in ("k", "mAk")}
    return {r for r in results | softened if r}


def _build_reverse() -> Dict[str, str]:
    rev: Dict[str, str] = {}
    for abstract in ABSTRACT_SUFFIXES:
        for surf in sorted(_expand(abstract)):
            rev.setdefault(surf, abstract)
    return rev


SURFACE_TO_ABSTRACT: Dict[str, str] = _build_reverse()


def abstract_suffix(surface: str) -> str:
    """
    Yüzey ek biçimini arkifonemik biçime çevir.

        lar/ler → lAr, da/de/ta/te → DA, dan/den/tan/ten → DAn,
        ı/i/u/ü → (y)H, yı/yi/yu/yü → (y)H, sı/si → (s)H, ıyor/iyor/yor → Hyor,
        acak/ecek/yacak/yecek → (y)AcAk, ken/yken → (y)ken, ki → ki ...

    Zaten soyut bir ek verilirse aynen döner. Tam tablo: SUFFIX_TABLE.

    Raises:
        KeyError: Ek tabloda yoksa.
    """
    s = surface[1:] if surface.startswith("+") else surface
    if s in _ABSTRACT_SET and (s != tr_lower(s) or "(" in s):
        return s
    low = tr_lower(s)
    if low in SURFACE_TO_ABSTRACT:
        return SURFACE_TO_ABSTRACT[low]
    if s in _ABSTRACT_SET:
        return s
    raise KeyError(f"Bilinmeyen yüzey eki: {surface!r}")


def is_abstract_suffix(text: str) -> bool:
    """Metnin geçerli (çözülebilir) bir arkifonemik ek olup olmadığı."""
    if text in _ABSTRACT_SET:
        return True
    if not _VALID_SUFFIX_RE.fullmatch(text):
        return False
    return any(ch in ARCHIPHONEMES or ch == "(" for ch in text)


def archiphoneme_user_symbols() -> List[str]:
    """
    SentencePiece `user_defined_symbols` için soyut ek parçaları ("+lAr", "+DA", ...).

    Kullanım:
        train_tokenizer(corpus, extra_symbols=CHAT_SPECIAL_TOKENS + archiphoneme_user_symbols())
    """
    return ["+" + a for a in ABSTRACT_SUFFIXES]


# ── Codec ──────────────────────────────────────────────────

_SUF_UNIT = r"(?:\([a-zçğıöşüAH]+\)|[a-zçğıöşüâîûAHDCG])"
_VALID_SUFFIX_RE = re.compile(rf"{_SUF_UNIT}+")
_WORD_RE = re.compile(
    rf"^(?P<stem>.*?[^\s+])(?P<sufs>(?:\+{_SUF_UNIT}+)+)(?P<tail>[^\w+]*)$"
)


class ArchiphonemeCodec:
    """
    Arkifonemik soyut metin ↔ yüzey Türkçe dönüştürücü.

    Biçim (bkz. modül belgesi): "kök+ek1+ek2", ör. "kitap+lAr+DA".
    """

    def __init__(
        self,
        non_softening: Optional[Iterable[str]] = None,
        harmony_exceptions: Optional[Iterable[str]] = None,
        softening_words: Optional[Iterable[str]] = None,
    ):
        self.kw = dict(
            non_softening=non_softening,
            harmony_exceptions=harmony_exceptions,
            softening_words=softening_words,
        )

    def encode_segmented(self, stem: str, surface_suffixes: Sequence[str]) -> str:
        """Kök + yüzey ekleri → "kök+Ek1+Ek2" (ekler `abstract_suffix` ile soyutlanır)."""
        return stem + "".join("+" + abstract_suffix(s) for s in surface_suffixes)

    def realize(self, stem: str, suffixes: Sequence[str]) -> str:
        return realize(stem, suffixes, **self.kw)

    def decode_word(self, token: str) -> str:
        """Tek bir boşluksuz soyut kelimeyi yüzey biçimine çevir."""
        m = _WORD_RE.match(token)
        if not m:
            return token
        sufs = m.group("sufs").split("+")[1:]
        if not all(is_abstract_suffix(s) for s in sufs):
            return token
        return self.realize(m.group("stem"), sufs) + m.group("tail")

    def decode(self, text: str) -> str:
        """Soyut metni yüzey Türkçe'ye çevir (boşluklar korunur)."""
        parts = re.split(r"(\s+)", text)
        return "".join(p if (not p or p.isspace()) else self.decode_word(p) for p in parts)


# ── Segmentasyon ve Korpus Dönüşümü ────────────────────────

Segmenter = Callable[[str], Optional[Tuple[str, List[str]]]]

_LEAD_RE = re.compile(r"^[^\w]*")
_TAIL_RE = re.compile(r"[^\w]*$")


def _safe_surface_set() -> Set[str]:
    """Yerleşik segmenterin kullandığı 'güvenli' allomorflar."""
    unsafe_abstract = {"sA", "mA", "(H)r", "(A)r", "(y)An", "CA", "lH", "GH",
                       "mAk", "m", "n", "k", "(y)Abil", "nA", "nH", "ki"}
    safe = set()
    for surf, abstract in SURFACE_TO_ABSTRACT.items():
        if abstract in unsafe_abstract or len(surf) < 2:
            continue
        if _is_vowel(surf[0]) and len(surf) < 3:
            continue      # "ın", "ım", "ar" gibi kısa ünlü başlı ekler çok belirsiz
        if surf in ("la", "le"):
            continue      # mahalle, kale, lale ...
        safe.add(surf)
    return safe


class RuleBasedSegmenter:
    """
    SEZGİSEL (heuristic) yerleşik segmenter — yalnız demo ve hızlı deneyler için.

    Sağdan sola bilinen ek allomorflarını soyar (en fazla `max_suffixes`),
    her analizi `realize` ile gidiş-dönüş doğrular ve şunu seçer:
      1. kökü (veya yumuşamamış hali) `lexicon` içindeyse o analiz (en uzun kök),
      2. aksi halde `min_stem_len` koşulunu sağlayan EN KISA (yüzey) kök.
    Ünlü başlı tek harfli ekler yalnız yumuşamış kökten sonra denenir ve
    kök sözlük biçimine geri çevrilir (kitab+ı → kitap+(y)H).
    Kesme işaretli kelimelerde (Türkiye'nin) kök kesmeye kadar alınır.

    Sözlük olmadan yanlış kök bulabilir (ör. "ağacı" → "ağa"+"cı", "gidiyor"
    → "gid"+"iyor"); gidiş-dönüş yine de korunur.
    üretim kalitesi için Zemberek / TRmorph çıktısını segmenter olarak takın.
    """

    DEMO_LEXICON = {"ev", "at", "et", "su", "göz", "oku", "ara", "bekle", "de", "ye"}

    def __init__(
        self,
        lexicon: Optional[Iterable[str]] = None,
        min_stem_len: int = 3,
        max_suffixes: int = 4,
        codec: Optional[ArchiphonemeCodec] = None,
    ):
        self.lexicon = set(self.DEMO_LEXICON) | {tr_lower(w) for w in (lexicon or ())}
        self.min_stem_len = min_stem_len
        self.max_suffixes = max_suffixes
        self.codec = codec or ArchiphonemeCodec()
        self.safe = _safe_surface_set()
        self.all_surfaces = set(SURFACE_TO_ABSTRACT)
        self._max_len = max(len(s) for s in self.all_surfaces)

    def _split_all(self, text: str, allowed: Set[str]) -> Optional[List[str]]:
        """Metni tümüyle allomorf dizisine böl (en uzun önce, geri izlemeli)."""
        if not text:
            return []
        for ln in range(min(self._max_len, len(text)), 0, -1):
            head = text[:ln]
            if head in allowed:
                rest = self._split_all(text[ln:], allowed)
                if rest is not None:
                    return [head] + rest
        return None

    def _candidates(self, word: str) -> List[Tuple[str, List[str]]]:
        out = []

        def rec(rest: str, sufs: List[str]):
            if sufs:
                out.append((rest, list(sufs)))
            if len(sufs) >= self.max_suffixes:
                return
            for ln in range(min(self._max_len, len(rest) - 2), 0, -1):
                tail = rest[-ln:]
                if tail in self.safe:
                    rec(rest[:-ln], [tail] + sufs)
                elif (tail in self.all_surfaces and _is_vowel(tail[0])
                      and rest[:-ln][-1:] in ("b", "c", "d", "ğ", "g")):
                    # Yumuşamış kök + ünlü başlı ek (kitab+ı, reng+i): yalnız
                    # yumuşamamış kök doğrulanırsa kabul edilir (bkz. __call__).
                    rec(rest[:-ln], [tail] + sufs)

        rec(word, [])
        return out

    def _verify(self, stem: str, sufs: List[str], word: str) -> List[str]:
        """Gidiş-dönüşü tutan kök adayları (tercih sırasıyla)."""
        try:
            abstracts = [abstract_suffix(s) for s in sufs]
        except KeyError:
            return []
        options = []
        low = tr_lower(stem)
        rev = {"b": "p", "c": "ç", "d": "t", "ğ": "k"}
        if low and low[-1] in rev:
            options.append(stem[:-1] + rev[low[-1]])      # kitab → kitap
        if low.endswith("ng"):
            options.append(stem[:-1] + "k")               # reng → renk
        options.append(stem)
        if sufs and abstracts[0] == "Hyor" and low and low[-1] in "ıiuü":
            options.append(stem[:-1] + ("a" if low[-1] in "ıu" else "e"))  # bekli → bekle
        return [c for c in options if self.codec.realize(c, abstracts) == word]

    def __call__(self, word: str) -> Optional[Tuple[str, List[str]]]:
        if not word or not any(ch.isalpha() for ch in word):
            return None
        for ap in APOSTROPHES:
            if ap in word:
                stem, _, rest = word.partition(ap)
                if not stem or not rest:
                    return None
                sufs = self._split_all(tr_lower(rest), self.all_surfaces)
                if not sufs:
                    return None
                stem = stem + ap
                if stem in self._verify(stem, sufs, word):
                    return stem, sufs
                return None
        if not word.isalpha():
            return None
        best = None
        best_lex = None
        for stem, sufs in self._candidates(word):
            if not any(_is_vowel(c) for c in tr_lower(stem)):
                continue
            verified = self._verify(stem, sufs, word)
            softened_path = sufs[0] not in self.safe
            if softened_path:
                verified = [v for v in verified if v != stem]   # yalnız kitab→kitap
            if not verified:
                continue
            in_lex = [v for v in verified if tr_lower(v) in self.lexicon]
            if in_lex:
                if best_lex is None or len(in_lex[0]) > len(best_lex[0]):
                    best_lex = (in_lex[0], sufs)
            elif len(stem) >= self.min_stem_len:
                if best is None or len(stem) < best[0]:
                    best = (len(stem), verified[0], sufs)
        if best_lex:
            return best_lex
        return (best[1], best[2]) if best else None


def transform_corpus_line(
    line: str,
    segmenter: Optional[Segmenter] = None,
    codec: Optional[ArchiphonemeCodec] = None,
    stats: Optional[Dict[str, int]] = None,
) -> str:
    """
    Bir korpus satırını arkifonemik soyut biçime çevir.

    Args:
        line: Ham Türkçe satır.
        segmenter: kelime → (kök, [yüzey ekleri]) veya None. Zemberek/TRmorph
            çıktısını bu imzaya saran bir fonksiyon takılabilir. Varsayılan:
            RuleBasedSegmenter (sezgisel).
        codec: ArchiphonemeCodec (istisna listeleri için).
        stats: Verilirse 'words', 'segmented', 'rejected' sayaçları güncellenir.

    Her kelime için `decode(encode(kelime)) == kelime` doğrulanır; tutmazsa
    kelime değiştirilmeden bırakılır. Böylece `codec.decode(sonuç) == line`.
    """
    codec = codec or ArchiphonemeCodec()
    segmenter = segmenter or RuleBasedSegmenter(codec=codec)
    parts = re.split(r"(\s+)", line)
    out = []
    for p in parts:
        if not p or p.isspace():
            out.append(p)
            continue
        if stats is not None:
            stats["words"] = stats.get("words", 0) + 1
        lead = _LEAD_RE.match(p).group(0)
        rest = p[len(lead):]
        tail = _TAIL_RE.search(rest).group(0) if rest else ""
        core = rest[: len(rest) - len(tail)] if tail else rest
        # Sondaki kesme işareti noktalama değil kök parçasıdır; yalnız çekirdek
        result = None
        if core and "+" not in p:
            try:
                seg = segmenter(core)
            except Exception:
                seg = None
            if seg:
                stem, sufs = seg
                try:
                    enc = lead + codec.encode_segmented(stem, sufs) + tail
                except KeyError:
                    enc = None
                if enc is not None and codec.decode_word(enc) == p:
                    result = enc
                elif stats is not None:
                    stats["rejected"] = stats.get("rejected", 0) + 1
        if result is not None:
            if stats is not None:
                stats["segmented"] = stats.get("segmented", 0) + 1
            out.append(result)
        else:
            out.append(p)
    return "".join(out)
