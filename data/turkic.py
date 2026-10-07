# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Türk Dünyası (çok dilli Türk dilleri) yardımcıları.

Bu modül Toprak'ı Türkiye Türkçesinin yanında akraba Türk dilleriyle de
eğitebilmek için gereken saf-Python araçları içerir:

* ``TURKIC_LANGUAGES``: dil kayıt tablosu (ad, ISO kodu, yazı sistemleri,
  dil etiketi tokenı ``<dil:xx>``, ek harf notları, veri kümesi kodları);
* ``detect_script``: Latin/Kiril/Arap harf payları ve baskın yazı;
* ``guess_turkic_language``: ayırt edici harf ve kelimelere dayalı **sezgisel**
  dil tahmini. Üretim ortamında fastText ``lid.176`` veya GlotLID gibi eğitilmiş
  bir dil tanıma modeli kullanılmalıdır; bu fonksiyon yalnız hızlı ön eleme,
  hata ayıklama ve küçük örneklemler içindir;
* deterministik harf çevirisi (transliterasyon): Kazakça (2021 resmî Latin
  alfabesi), Kırgızca (Ortak Türk Alfabesi temelli yaygın şema), Özbekçe
  (resmî 1995 Latin alfabesi), Tatarca (Zamanälif'e yakın şema) ve Uygurca
  (UEY Arap yazısı → ULY Latin yazısı);
* Osmanlıca için **sahte transliterasyon yapılmaz** (abjad yazımı belirsizdir);
  bunun yerine paralel veri biçimi (``ParallelExample``) ve
  ``make_translation_example`` sunulur, ayrıca Osmanlıca metin normalizasyonu;
* ``tag_text`` / ``language_tag_tokens`` / ``turkic_tokenizer_symbols``:
  tokenizer ``extra_symbols`` listesine eklenecek dil etiketleri ve
  ``<çeviri>`` tokenı;
* ``normalize_turkic``: dile özgü Unicode ve kesme işareti normalizasyonu.

Tüm fonksiyonlar deterministiktir ve harici bağımlılık gerektirmez.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Dil kayıt tablosu
# ---------------------------------------------------------------------------

LATIN = "latin"
CYRILLIC = "cyrillic"
ARABIC = "arabic"
OTHER = "other"
SCRIPTS = (LATIN, CYRILLIC, ARABIC)

LANGUAGE_TAG_FORMAT = "<dil:{}>"
TRANSLATION_TOKEN = "<çeviri>"


@dataclass(frozen=True)
class TurkicLanguage:
    """Tek bir Türk dilinin kayıt bilgisi."""

    code: str
    name: str
    iso639_3: str
    branch: str
    scripts: Tuple[str, ...]
    primary_script: str
    extra_letters: str
    notes: str = ""
    iso639_1: Optional[str] = None
    # Veri kümesi kodları ipucudur; kullanmadan önce veri kartından doğrulayın.
    fineweb2_configs: Tuple[str, ...] = field(default_factory=tuple)
    culturax_code: Optional[str] = None
    wikipedia_code: Optional[str] = None

    @property
    def tag(self) -> str:
        return LANGUAGE_TAG_FORMAT.format(self.code)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scripts"] = list(self.scripts)
        payload["fineweb2_configs"] = list(self.fineweb2_configs)
        payload["tag"] = self.tag
        return payload


TURKIC_LANGUAGES: Dict[str, TurkicLanguage] = {
    "tr": TurkicLanguage(
        code="tr", iso639_1="tr", iso639_3="tur", name="Türkiye Türkçesi",
        branch="Oğuz", scripts=(LATIN,), primary_script=LATIN,
        extra_letters="ç ğ ı İ ö ş ü (â î û düzeltme işaretli)",
        notes="Toprak'ın ana dili; karışımda baskın tutulur.",
        fineweb2_configs=("tur_Latn",), culturax_code="tr", wikipedia_code="tr",
    ),
    "az": TurkicLanguage(
        code="az", iso639_1="az", iso639_3="aze", name="Azerbaycan Türkçesi",
        branch="Oğuz", scripts=(LATIN, CYRILLIC, ARABIC), primary_script=LATIN,
        extra_letters="ə q x (Türkçeye ek olarak); Kiril dönemi: ә ғ ҹ ј һ",
        notes=(
            "Kuzey Azerbaycan Latin yazısı kullanır; Güney Azerbaycan (İran) "
            "Arap yazısı kullanır ve ayrı kaynak olarak ele alınmalıdır."
        ),
        fineweb2_configs=("azj_Latn", "azb_Arab"), culturax_code="az",
        wikipedia_code="az",
    ),
    "tk": TurkicLanguage(
        code="tk", iso639_1="tk", iso639_3="tuk", name="Türkmen Türkçesi",
        branch="Oğuz", scripts=(LATIN, CYRILLIC), primary_script=LATIN,
        extra_letters="ä ň ý ž (ç ş ö ü ile birlikte); y = ı sesi",
        notes="1993'ten beri Latin; eski metinler Kiril olabilir.",
        fineweb2_configs=("tuk_Latn",), culturax_code="tk", wikipedia_code="tk",
    ),
    "uz": TurkicLanguage(
        code="uz", iso639_1="uz", iso639_3="uzb", name="Özbek Türkçesi",
        branch="Karluk", scripts=(LATIN, CYRILLIC), primary_script=LATIN,
        extra_letters="oʻ gʻ sh ch ng q x; tutuq belgisi ʼ (Kiril: ў ғ қ ҳ)",
        notes=(
            "Latin (1995) resmî; Kiril hâlâ çok yaygın. oʻ/gʻ için doğru "
            "işaret U+02BB'dir; webde ' ` ‘ ’ varyantları görülür."
        ),
        fineweb2_configs=("uzn_Latn", "uzn_Cyrl"), culturax_code="uz",
        wikipedia_code="uz",
    ),
    "kk": TurkicLanguage(
        code="kk", iso639_1="kk", iso639_3="kaz", name="Kazak Türkçesi",
        branch="Kıpçak", scripts=(CYRILLIC, LATIN, ARABIC), primary_script=CYRILLIC,
        extra_letters="ә ғ қ ң ө ұ ү һ і (Kiril); 2021 Latin: ä ğ q ñ ö ū ü ı",
        notes="Latin alfabesine geçiş sürüyor; Çin'de Arap yazısı kullanılır.",
        fineweb2_configs=("kaz_Cyrl",), culturax_code="kk", wikipedia_code="kk",
    ),
    "ky": TurkicLanguage(
        code="ky", iso639_1="ky", iso639_3="kir", name="Kırgız Türkçesi",
        branch="Kıpçak", scripts=(CYRILLIC, ARABIC), primary_script=CYRILLIC,
        extra_letters="ң ө ү (Kiril)",
        notes="Kazakçadan farklı olarak қ ғ ә і ұ harflerini kullanmaz.",
        fineweb2_configs=("kir_Cyrl",), culturax_code="ky", wikipedia_code="ky",
    ),
    "tt": TurkicLanguage(
        code="tt", iso639_1="tt", iso639_3="tat", name="Tatar Türkçesi",
        branch="Kıpçak", scripts=(CYRILLIC, LATIN), primary_script=CYRILLIC,
        extra_letters="ә ө ү җ ң һ (Kiril); Zamanälif: ä ö ü c ñ ğ q ı",
        notes="Resmî yazı Kiril; Latin (Zamanälif) diasporada ve bazı sitelerde.",
        fineweb2_configs=("tat_Cyrl",), culturax_code="tt", wikipedia_code="tt",
    ),
    "ba": TurkicLanguage(
        code="ba", iso639_1="ba", iso639_3="bak", name="Başkurt Türkçesi",
        branch="Kıpçak", scripts=(CYRILLIC,), primary_script=CYRILLIC,
        extra_letters="ә ө ү ғ ҡ ң ҙ ҫ һ",
        notes="ҙ ҫ ҡ harfleri Tatarcadan ayırt edici; transliterasyon sağlanmaz.",
        fineweb2_configs=("bak_Cyrl",), culturax_code="ba", wikipedia_code="ba",
    ),
    "ug": TurkicLanguage(
        code="ug", iso639_1="ug", iso639_3="uig", name="Uygur Türkçesi",
        branch="Karluk", scripts=(ARABIC, LATIN, CYRILLIC), primary_script=ARABIC,
        extra_letters=(
            "UEY Arap yazısı, ünlüler yazılır: ئا ئە ئو ئۇ ئۆ ئۈ ئې ئى; "
            "ULY Latin: ë ö ü gh ng zh sh ch"
        ),
        notes="UEY Arap yazısı (Uyghur Ereb Yéziqi) ana yazıdır.",
        fineweb2_configs=("uig_Arab",), culturax_code="ug", wikipedia_code="ug",
    ),
    "crh": TurkicLanguage(
        code="crh", iso639_3="crh", name="Kırım Tatar Türkçesi",
        branch="Kıpçak (Oğuz etkili)", scripts=(LATIN, CYRILLIC), primary_script=LATIN,
        extra_letters="â ñ q (Türkçe harflere ek); Kiril: къ гъ нъ",
        notes="Latin yazımı Türkiye Türkçesine çok yakındır.",
        fineweb2_configs=("crh_Latn",), culturax_code=None, wikipedia_code="crh",
    ),
    "gag": TurkicLanguage(
        code="gag", iso639_3="gag", name="Gagavuz Türkçesi",
        branch="Oğuz", scripts=(LATIN, CYRILLIC), primary_script=LATIN,
        extra_letters="ä ê ţ (Türkçe harflere ek); eski Kiril: ӓ ӧ ӱ",
        notes="Çok düşük kaynaklı; Türkiye Türkçesine en yakın dillerden.",
        fineweb2_configs=("gag_Latn",), culturax_code=None, wikipedia_code="gag",
    ),
    "ota": TurkicLanguage(
        code="ota", iso639_3="ota", name="Osmanlı Türkçesi",
        branch="Oğuz (tarihî)", scripts=(ARABIC,), primary_script=ARABIC,
        extra_letters="Arap-Fars harfleri: پ چ ژ گ ڭ (sağır kef) ve Arapça harfler",
        notes=(
            "Abjad yazımı ünlüleri göstermez; otomatik transliterasyon "
            "belirsizdir. Paralel (Osmanlıca ↔ Türkçe) veri ile öğretilir."
        ),
        fineweb2_configs=(), culturax_code=None, wikipedia_code=None,
    ),
}


def get_language(code: str) -> TurkicLanguage:
    """Dil kodundan kaydı döndürür; bilinmeyen kodlarda ValueError."""
    try:
        return TURKIC_LANGUAGES[code]
    except KeyError as exc:
        known = ", ".join(sorted(TURKIC_LANGUAGES))
        raise ValueError(f"Bilinmeyen Türk dili kodu: {code!r} (bilinenler: {known})") from exc


def language_tag(code: str) -> str:
    return get_language(code).tag


def language_tag_tokens() -> List[str]:
    """Kayıt sırasıyla tüm ``<dil:xx>`` tokenları."""
    return [language.tag for language in TURKIC_LANGUAGES.values()]


def turkic_tokenizer_symbols() -> List[str]:
    """Tokenizer ``extra_symbols`` listesine eklenecek Türk dünyası tokenları.

    ``model.chat_template.CHAT_SPECIAL_TOKENS`` ile birlikte verilmelidir::

        extra_symbols = list(CHAT_SPECIAL_TOKENS) + turkic_tokenizer_symbols()
    """
    return language_tag_tokens() + [TRANSLATION_TOKEN]


def tag_text(text: str, lang: str) -> str:
    """Metnin başına dil etiketi ekler: ``<dil:kk> Сәлем``."""
    return f"{language_tag(lang)} {text}"


# ---------------------------------------------------------------------------
# Yazı sistemi tespiti
# ---------------------------------------------------------------------------

_LATIN_MODIFIERS = {"ʻ", "ʼ"}  # ʻ ʼ (Özbekçe oʻ / tutuq belgisi)
_SCRIPT_CACHE: Dict[str, str] = {}


def _char_script(char: str) -> str:
    cached = _SCRIPT_CACHE.get(char)
    if cached is not None:
        return cached
    if char in _LATIN_MODIFIERS:
        script = LATIN
    else:
        name = unicodedata.name(char, "")
        if name.startswith("LATIN"):
            script = LATIN
        elif name.startswith("CYRILLIC"):
            script = CYRILLIC
        elif name.startswith("ARABIC"):
            script = ARABIC
        else:
            script = OTHER
    _SCRIPT_CACHE[char] = script
    return script


def detect_script(text: str) -> dict:
    """Harflerin Latin/Kiril/Arap/diğer paylarını ve baskın yazıyı döndürür.

    Dönüş: ``{"latin": .., "cyrillic": .., "arabic": .., "other": ..,
    "letters": n, "dominant": "latin" | "cyrillic" | "arabic" | "other" | None}``.
    Yalnız harfler sayılır (rakam, noktalama ve harekeler hariç).
    """
    counts = {LATIN: 0, CYRILLIC: 0, ARABIC: 0, OTHER: 0}
    for char in text:
        if char.isalpha() or char in _LATIN_MODIFIERS:
            counts[_char_script(char)] += 1
    total = sum(counts.values())
    shares = {name: (count / total if total else 0.0) for name, count in counts.items()}
    dominant = max(counts, key=lambda name: counts[name]) if total else None
    return {**shares, "letters": total, "dominant": dominant}


# ---------------------------------------------------------------------------
# Normalizasyon
# ---------------------------------------------------------------------------

UZBEK_OKINA = "ʻ"  # ʻ  — oʻ, gʻ
UZBEK_TUTUQ = "ʼ"  # ʼ  — tutuq belgisi (ayırma işareti)
_APOSTROPHES = "'`‘’ʻʼ´"
_UZ_OKINA_RE = re.compile(rf"(?<=[oOgG])[{_APOSTROPHES}]")
_UZ_TUTUQ_RE = re.compile(
    rf"(?<=[^\W\d_])(?<![oOgG])[{_APOSTROPHES.replace(UZBEK_TUTUQ, '')}](?=[^\W\d_])"
)

_ARABIC_HARAKAT_RE = re.compile("[ً-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_PUNCT = {"،": ",", "؛": ";", "؟": "?", "٪": "%"}
_ARABIC_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}
_ARABIC_DIGITS.update({chr(0x06F0 + i): str(i) for i in range(10)})

# Kiril metinde sık görülen Latin "benzer harf" hataları (kk/ba/tt/ky)
_KK_LATIN_LOOKALIKES = {"i": "і", "I": "І", "ə": "ә", "Ə": "Ә", "h": "һ", "H": "Һ"}
_CYRL_SCHWA_FIX = {"ə": "ә", "Ə": "Ә"}


def normalize_ottoman(
    text: str,
    remove_harakat: bool = False,
    remove_tatweel: bool = True,
    unify_letters: bool = True,
) -> str:
    """Osmanlıca (Arap yazılı) metin normalizasyonu.

    * NFKC: Arap sunum biçimlerini (ﻻ, ﺑ …) temel harflere indirir;
    * ``remove_tatweel``: keşide (ـ) kaldırılır;
    * ``remove_harakat``: hareke/tenvin/şedde gibi işaretler kaldırılır
      (varsayılan kapalı; harekeli metin değerli bilgi taşır);
    * ``unify_letters``: Arapça ك/ي/ى biçimleri Osmanlı matbaa geleneğindeki
      Farsça ک/ی biçimlerine birleştirilir.

    Bu fonksiyon harf çevirisi **yapmaz**.
    """
    text = unicodedata.normalize("NFKC", text)
    if remove_tatweel:
        text = text.replace(_TATWEEL, "")
    if remove_harakat:
        text = _ARABIC_HARAKAT_RE.sub("", text)
    if unify_letters:
        text = text.replace("ك", "ک")  # ك → ک
        text = text.replace("ي", "ی")  # ي → ی
        text = text.replace("ى", "ی")  # ى → ی
    return re.sub(r"[ \t]+", " ", text).strip()


def _normalize_uzbek_apostrophes(text: str) -> str:
    text = _UZ_OKINA_RE.sub(UZBEK_OKINA, text)
    return _UZ_TUTUQ_RE.sub(UZBEK_TUTUQ, text)


def normalize_turkic(text: str, lang: str) -> str:
    """Dile özgü güvenli normalizasyon (NFC + bilinen yazım varyantları).

    * tümü: NFC, sıfır genişlikli boşluk/BOM temizliği;
    * ``uz``: oʻ/gʻ sonrası ' ` ‘ ’ ʼ → ʻ (U+02BB), diğer kelime içi
      kesmeler → ʼ (U+02BC);
    * ``tr``/``az``/``tk``/``crh``/``gag``: ’ ‘ → ' (özel ad kesmesi);
      ``az`` için Kiril ә (U+04D9) yanlış kullanımı → ə (U+0259);
    * ``kk``/``ky``/``tt``/``ba``: Kiril harf içeren kelimelerde Latin ə → ә;
      ``kk``'da Latin i/I/h → і/І/һ (yaygın klavye hatası);
    * ``ug``: Arap noktalama/rakamları korunur, keşide kaldırılır, ک → ك;
    * ``ota``: :func:`normalize_ottoman` (hareke korunur).
    """
    get_language(lang)
    text = unicodedata.normalize("NFC", text).replace("﻿", "").replace("​", "")
    if lang == "ota":
        return normalize_ottoman(text)
    if lang == "uz":
        return _normalize_uzbek_apostrophes(text)
    if lang in ("tr", "az", "tk", "crh", "gag"):
        text = text.replace("’", "'").replace("‘", "'")
        if lang == "az":
            text = text.replace("ә", "ə").replace("Ә", "Ə")
        return text
    if lang in ("kk", "ky", "tt", "ba"):
        table = _KK_LATIN_LOOKALIKES if lang == "kk" else _CYRL_SCHWA_FIX
        return _replace_lookalikes_in_cyrillic_words(text, table)
    if lang == "ug":
        text = text.replace(_TATWEEL, "").replace("ک", "ك")
        return text
    return text


def _replace_lookalikes_in_cyrillic_words(text: str, table: Dict[str, str]) -> str:
    """Yalnız Kiril harf içeren kelimelerdeki Latin benzerlerini değiştirir."""

    def fix(match):
        word = match.group(0)
        if any(_char_script(ch) == CYRILLIC for ch in word if ch.isalpha()):
            return "".join(table.get(ch, ch) for ch in word)
        return word

    return re.sub(r"[^\W\d_]+", fix, text)


# ---------------------------------------------------------------------------
# Transliterasyon
# ---------------------------------------------------------------------------

def _tr_upper(text: str) -> str:
    """Türk usulü büyük harf: i → İ, ı → I."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def _is_letter(char: str) -> bool:
    return char.isalpha()


def _all_caps_context(text: str, index: int) -> bool:
    """Büyük harfin bir TAMAMI BÜYÜK kelimenin parçası olup olmadığı."""
    nxt = text[index + 1] if index + 1 < len(text) else ""
    if nxt and _is_letter(nxt):
        return nxt.isupper()
    prev = text[index - 1] if index > 0 else ""
    return bool(prev) and _is_letter(prev) and prev.isupper()


def _apply_case(latin: str, all_caps: bool, turkish_case: bool) -> str:
    if not latin:
        return latin
    upper = _tr_upper if turkish_case else str.upper
    if all_caps:
        return upper(latin)
    return upper(latin[0]) + latin[1:]


def _word_bounds(text: str, index: int) -> Tuple[int, int]:
    start = index
    while start > 0 and _is_letter(text[start - 1]):
        start -= 1
    end = index + 1
    while end < len(text) and _is_letter(text[end]):
        end += 1
    return start, end


ContextRule = Callable[[str, int, str], Optional[str]]


def _transliterate(
    text: str,
    table: Dict[str, str],
    turkish_case: bool,
    context: Optional[ContextRule] = None,
) -> str:
    text = unicodedata.normalize("NFC", text)
    out = []
    for index, char in enumerate(text):
        lower = char.lower()
        if lower not in table:
            out.append(char)
            continue
        latin = context(text, index, lower) if context else None
        if latin is None:
            latin = table[lower]
        if char != lower:
            latin = _apply_case(latin, _all_caps_context(text, index), turkish_case)
        out.append(latin)
    return "".join(out)


# Kazakça — 2021 resmî Latin alfabesi (31 harf: A Ä B D E F G Ğ H I İ J K L M
# N Ñ O Ö P Q R S Ş T U Ū Ü V Y Z; Ç ve C alıntılar için).
KAZAKH_CYRILLIC_TO_LATIN = {
    "а": "a", "ә": "ä", "б": "b", "в": "v", "г": "g", "ғ": "ğ", "д": "d",
    "е": "e", "ё": "io", "ж": "j", "з": "z",
    # Yaklaşım: и (ıy/iy) ve й 2021 alfabesinde tek harfe (İ i) iner.
    "и": "i", "й": "i",
    "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "ñ", "о": "o",
    "ө": "ö", "п": "p", "р": "r", "с": "s", "т": "t",
    # Yaklaşım: у (u/uw/üw) bağlamdan bağımsız olarak U u'ya iner.
    "у": "u", "ұ": "ū", "ү": "ü", "ф": "f", "х": "h", "һ": "h", "ц": "ts",
    "ч": "ç", "ш": "ş", "щ": "şş", "ъ": "", "ы": "y", "і": "ı", "ь": "",
    "э": "e", "ю": "iu", "я": "ia",
}

# Kırgızca — Ortak Türk Alfabesi (1993) temelli, Türkiye Türkçesi okuruna
# yakın yaygın şema: ы → ı, ң → ñ, ж → j, ч → ç, ш → ş, й → y.
KYRGYZ_CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "ң": "ñ", "о": "o", "ө": "ö", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ү": "ü", "ф": "f", "х": "h", "ц": "ts", "ч": "ç",
    "ш": "ş", "щ": "şç", "ъ": "", "ы": "ı", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}

# Özbekçe — resmî 1995 Latin alfabesi (O‘zbek lotin alifbosi).
UZBEK_CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
    "ъ": UZBEK_TUTUQ, "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o" + UZBEK_OKINA, "қ": "q", "ғ": "g" + UZBEK_OKINA, "ҳ": "h",
}
_UZBEK_CYRILLIC_VOWELS = set("аеёиоуэюяў")

# Tatarca — Zamanälif'e yakın şema; к/г ve я/ю ünlü uyumuna göre seçilir.
TATAR_CYRILLIC_TO_LATIN = {
    "а": "a", "ә": "ä", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
    "ё": "yo", "ж": "j", "җ": "c", "з": "z", "и": "i", "й": "y", "к": "k",
    "л": "l", "м": "m", "н": "n", "ң": "ñ", "о": "o", "ө": "ö", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ү": "ü", "ф": "f", "х": "x",
    "һ": "h", "ц": "ts", "ч": "ç", "ш": "ş", "щ": "şç", "ъ": "", "ы": "ı",
    "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
_TATAR_FRONT = set("әеиөүэь")
_TATAR_VOWELS = set("аәеёиоөуүыэюя")


def _uzbek_context(text: str, index: int, lower: str) -> Optional[str]:
    prev = text[index - 1].lower() if index > 0 else ""
    prev_is_letter = bool(prev) and _is_letter(text[index - 1])
    if lower == "е":
        if not prev_is_letter or prev in _UZBEK_CYRILLIC_VOWELS or prev in "ъь":
            return "ye"
        return "e"
    if lower == "ц":
        return "ts" if prev in _UZBEK_CYRILLIC_VOWELS else "s"
    return None


def _tatar_context(text: str, index: int, lower: str) -> Optional[str]:
    if lower not in "кгяюв":
        return None
    if lower == "в":
        prev = text[index - 1].lower() if index > 0 else ""
        return "w" if prev in _TATAR_VOWELS else "v"
    start, end = _word_bounds(text, index)
    front = any(ch in _TATAR_FRONT for ch in text[start:end].lower())
    return {
        "к": "k" if front else "q",
        "г": "g" if front else "ğ",
        "я": "yä" if front else "ya",
        "ю": "yü" if front else "yu",
    }[lower]


def kazakh_cyrillic_to_latin(text: str) -> str:
    """Kazak Kirilinden 2021 resmî Latin alfabesine çeviri.

    Örnek: ``Қазақстан`` → ``Qazaqstan``, ``Алматы`` → ``Almaty``,
    ``Әліпби`` → ``Älıpbi``. Belgelenmiş yaklaşımlar: и/й → i (İ), у → u,
    ю → iu, я → ia, ё → io, ц → ts, щ → şş; ъ/ь düşer. Büyük/küçük harf
    korunur (і → ı/I, и → i/İ).
    """
    return _transliterate(text, KAZAKH_CYRILLIC_TO_LATIN, turkish_case=True)


def kyrgyz_cyrillic_to_latin(text: str) -> str:
    """Kırgız Kirilinden Ortak Türk Alfabesi temelli Latin şemaya çeviri.

    Resmî bir Kırgız Latin alfabesi yürürlükte değildir; bu şema Türkiye
    Türkçesi okuruna en okunur biçimi hedefler: ы → ı, ң → ñ, ж → j,
    ч → ç, ш → ş, й → y, ё/ю/я → yo/yu/ya, х → h, ц → ts.
    Örnek: ``Кыргызстан`` → ``Kırgızstan``, ``Бишкек`` → ``Bişkek``.
    """
    return _transliterate(text, KYRGYZ_CYRILLIC_TO_LATIN, turkish_case=True)


def uzbek_cyrillic_to_latin(text: str) -> str:
    """Özbek Kirilinden resmî 1995 Latin alfabesine çeviri.

    ш → sh, ч → ch, ў → oʻ, ғ → gʻ, қ → q, ҳ → h, х → x, ё/ю/я → yo/yu/ya,
    ъ → ʼ. Bağlam kuralları: е kelime başında, ünlüden veya ъ/ь'den sonra
    ``ye``, aksi hâlde ``e``; ц ünlüden sonra ``ts``, aksi hâlde ``s``.
    Örnek: ``Ўзбекистон`` → ``Oʻzbekiston``, ``шаҳар`` → ``shahar``.
    """
    return _transliterate(text, UZBEK_CYRILLIC_TO_LATIN, turkish_case=False,
                          context=_uzbek_context)


def tatar_cyrillic_to_latin(text: str) -> str:
    """Tatar Kirilinden Zamanälif'e yakın Latin yazıya çeviri.

    Zamanälif'te к/г ve я/ю ünlü uyumuna göre yazılır; burada kelimede ön
    ünlü (ә е и ө ү э) veya ь varsa ``k/g/yä/yü``, yoksa ``q/ğ/ya/yu``
    seçilir. в ünlüden sonra ``w``, aksi hâlde ``v``. җ → c, ң → ñ, х → x,
    һ → h. Rusça alıntılarda bu kurallar yaklaşık sonuç verir.
    Örnek: ``Казан`` → ``Qazan``, ``китап`` → ``kitap``.
    """
    return _transliterate(text, TATAR_CYRILLIC_TO_LATIN, turkish_case=True,
                          context=_tatar_context)


# Uygurca — UEY (Arap) → ULY (Latin)
UYGHUR_VOWELS = {
    "ا": "a",   # ا
    "ە": "e",   # ە
    "و": "o",   # و
    "ۇ": "u",   # ۇ
    "ۆ": "ö",   # ۆ
    "ۈ": "ü",   # ۈ
    "ې": "ë",   # ې
    "ى": "i",   # ى
}
UYGHUR_CONSONANTS = {
    "ب": "b",   # ب
    "پ": "p",   # پ
    "ت": "t",   # ت
    "ج": "j",   # ج
    "چ": "ch",  # چ
    "خ": "x",   # خ
    "د": "d",   # د
    "ر": "r",   # ر
    "ز": "z",   # ز
    "ژ": "zh",  # ژ
    "س": "s",   # س
    "ش": "sh",  # ش
    "غ": "gh",  # غ
    "ف": "f",   # ف
    "ق": "q",   # ق
    "ك": "k",   # ك
    "ک": "k",   # ک (Farsça biçim)
    "گ": "g",   # گ
    "ڭ": "ng",  # ڭ
    "ل": "l",   # ل
    "م": "m",   # م
    "ن": "n",   # ن
    "ھ": "h",   # ھ
    "ه": "h",   # ه (yanlış kodlanmış ھ)
    "ۋ": "w",   # ۋ
    "ي": "y",   # ي
    "ی": "y",   # ی (Farsça biçim)
}
UYGHUR_HAMZA = "ئ"  # ئ
# Ardışık iki harfin bir ULY digrafı gibi okunmasını engellemek için ' eklenir.
_ULY_SEPARATE = {("n", "g"), ("s", "h"), ("c", "h"), ("z", "h"), ("g", "h")}
_SENTENCE_END = set(".!?")


def uyghur_arabic_to_latin(text: str, capitalize_sentences: bool = False) -> str:
    """Uygur Arap yazısından (UEY) Uygur Latin yazısına (ULY) çeviri.

    Ünlü harfler (ا ە و ۇ ۆ ۈ ې ى) doğrudan karşılanır; hemze taşıyıcısı
    ئ kelime başında sessizdir (ئۇ → u), kelime içinde ULY ayırma işareti
    ``'`` olur (مەسئۇل → mes'ul). ن+گ, س+ھ gibi diziler digrafla
    karışmasın diye ``n'g``, ``s'h`` yazılır. Arap noktalama/rakamları
    Latin karşılıklarına çevrilir. Arap yazısında büyük harf olmadığından
    çıktı küçük harftir; ``capitalize_sentences=True`` cümle başlarını
    büyütür. Örnek: ``ئۇيغۇر`` → ``uyghur``, ``تىل`` → ``til``.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_TATWEEL, "").replace("‌", "")
    text = _ARABIC_HARAKAT_RE.sub("", text)
    out: List[str] = []
    previous_was_letter = False
    for index, char in enumerate(text):
        if char == UYGHUR_HAMZA:
            nxt = text[index + 1] if index + 1 < len(text) else ""
            if not previous_was_letter and nxt in UYGHUR_VOWELS:
                previous_was_letter = True
                continue
            out.append("'")
            previous_was_letter = True
            continue
        latin = UYGHUR_VOWELS.get(char) or UYGHUR_CONSONANTS.get(char)
        if latin is None:
            latin = _ARABIC_PUNCT.get(char) or _ARABIC_DIGITS.get(char) or char
            out.append(latin)
            previous_was_letter = char.isalpha()
            continue
        if previous_was_letter and out and out[-1] and (out[-1][-1], latin[0]) in _ULY_SEPARATE:
            out.append("'")
        out.append(latin)
        previous_was_letter = True
    result = "".join(out)
    if capitalize_sentences:
        result = _capitalize_sentences(result)
    return result


def _capitalize_sentences(text: str) -> str:
    chars = list(text)
    capitalize = True
    for index, char in enumerate(chars):
        if capitalize and char.isalpha():
            chars[index] = char.upper()
            capitalize = False
        elif char in _SENTENCE_END:
            capitalize = True
    return "".join(chars)


TRANSLITERATORS: Dict[str, Callable[[str], str]] = {
    "kk": kazakh_cyrillic_to_latin,
    "ky": kyrgyz_cyrillic_to_latin,
    "uz": uzbek_cyrillic_to_latin,
    "tt": tatar_cyrillic_to_latin,
    "ug": uyghur_arabic_to_latin,
}


def can_transliterate(lang: str) -> bool:
    """Dil için Latin'e deterministik çeviri tanımlı mı (veya zaten Latin mi)."""
    language = get_language(lang)
    return lang in TRANSLITERATORS or language.primary_script == LATIN


def transliterate_to_latin(text: str, lang: str) -> str:
    """Metni ilgili dilin Latin yazısına çevirir.

    Ana yazısı Latin olan dillerde (tr, az, tk, crh, gag) metin değişmez;
    Özbekçe gibi iki yazılı dillerde yalnız Kiril harfler çevrilir.
    Osmanlıca ve Başkurtça için ValueError yükseltilir (Osmanlıca için
    paralel veri kullanın: :func:`make_translation_example`).
    """
    language = get_language(lang)
    if lang in TRANSLITERATORS:
        return TRANSLITERATORS[lang](text)
    if language.primary_script == LATIN:
        return text
    if lang == "ota":
        raise ValueError(
            "Osmanlıca için otomatik transliterasyon yapılmaz (abjad belirsiz); "
            "make_translation_example ile paralel veri kullanın."
        )
    raise ValueError(f"{lang} için Latin transliterasyonu tanımlı değil")


# ---------------------------------------------------------------------------
# Paralel / çeviri verisi
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParallelExample:
    """İki dilli hizalı örnek (ör. Osmanlıca ↔ Türkiye Türkçesi).

    ``source`` alanı mixture config'teki kaynak adıyla eşleşmelidir
    (ör. ``parallel_ota_tr``); ``license`` veri kaynağının lisansıdır.
    """

    src_text: str
    src_lang: str
    tgt_text: str
    tgt_lang: str
    source: str = "parallel"
    license: Optional[str] = None
    transcription: Optional[str] = None

    def __post_init__(self):
        get_language(self.src_lang)
        get_language(self.tgt_lang)
        if not self.src_text.strip() or not self.tgt_text.strip():
            raise ValueError("Paralel örnekte kaynak ve hedef metin boş olamaz")

    def to_text(self) -> str:
        return make_translation_example(
            self.src_text, self.src_lang, self.tgt_text, self.tgt_lang
        )

    def to_record(self) -> dict:
        """Temizleme/pretokenize hattına uygun JSONL kaydı."""
        record = {
            "text": self.to_text(),
            "source": self.source,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
        }
        if self.license:
            record["license"] = self.license
        if self.transcription:
            record["transcription"] = self.transcription
        return record


def make_translation_example(src_text: str, src_lang: str, tgt_text: str, tgt_lang: str) -> str:
    """Çeviri eğitim metni: ``<dil:ota> … <çeviri> <dil:tr> …``.

    Kaynak metin dile göre normalize edilir (Osmanlıca için
    :func:`normalize_ottoman`). Ters yön için argümanları yer değiştirin;
    iki yönü de eğitim verisine koymak genelde faydalıdır.
    """
    src = normalize_turkic(src_text, src_lang).strip()
    tgt = normalize_turkic(tgt_text, tgt_lang).strip()
    if not src or not tgt:
        raise ValueError("Çeviri örneğinde kaynak ve hedef metin boş olamaz")
    return f"{tag_text(src, src_lang)} {TRANSLATION_TOKEN} {tag_text(tgt, tgt_lang)}"


# ---------------------------------------------------------------------------
# Sezgisel dil tahmini
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[^\W\d_]+(?:[ʻʼ'’‘`][^\W\d_]+)*")

# Her dil için: (harf ağırlıkları, kelime kümesi, düzenli ifade ağırlıkları,
# ceza harfleri). Harf ağırlıkları ilk 3 tekrar için sayılır.
_FEATURES = {
    LATIN: {
        "tr": (
            {},
            {"ve", "için", "değil", "merhaba", "çok", "gibi", "ile", "ama", "daha",
             "ben", "nasıl", "var", "yok", "olarak", "türkiye", "birlikte", "evet",
             "teşekkür", "ederim", "hoşça", "kal", "bugün", "güzel"},
            {r"\w+(?:ıyor|iyor|uyor|üyor|yor)(?:um|sun|uz|sunuz|lar|du|muş)?\b": 2.5,
             r"\w+(?:lık|lik|luk|lük)\b": 0.3},
            {"q": 1.5, "x": 1.5, "w": 1.5, "ə": 3, "ñ": 2, "ň": 3, "ý": 3, "ä": 2,
             "ê": 3, "ţ": 3, "ū": 3, "ë": 3, "ʻ": 3},
        ),
        "az": (
            {"ə": 3, "x": 0.6, "q": 0.6},
            {"və", "üçün", "deyil", "mən", "sən", "necəsən", "salam", "çox", "olan",
             "ilə", "həm", "bakı", "bakıda", "axşam", "nə", "yaxşı", "gün"},
            {r"\w+(?:ıram|irəm|uram|ürəm|yıram|yirəm)\b": 2.0},
            {"ň": 3, "ý": 3, "ä": 2, "ʻ": 3, "ñ": 2},
        ),
        "tk": (
            {"ň": 3, "ý": 3, "ž": 3, "ä": 1.2},
            {"we", "bilen", "üçin", "salam", "nähili", "howa", "örän", "gowy", "şu",
             "men", "biz", "ýaly", "hem"},
            {},
            {"ə": 3, "ʻ": 3, "x": 1, "q": 1},
        ),
        "uz": (
            {"ʻ": 3, "x": 0.6, "q": 0.6},
            {"va", "bilan", "uchun", "emas", "salom", "assalomu", "alaykum", "men",
             "biz", "juda", "yaxshi", "havo", "bugun", "qalay", "rahmat", "yurt"},
            {r"[ogOG][ʻ'’‘`]": 3.0, r"sh|ch": 0.4},
            {"ə": 3, "ı": 2, "ş": 2, "ç": 2, "ğ": 2, "ü": 2, "ö": 2, "ä": 2},
        ),
        "crh": (
            {"ñ": 2, "â": 0.8, "q": 1.0},
            {"selâm", "men", "biz", "çoq", "yahşı", "içün", "qırım", "aqşam", "bar",
             "pek", "ava", "edem", "sağ", "oluñız", "yurt"},
            {r"\w*(?:ñ|q)\w*": 0.5},
            {"ə": 3, "ň": 3, "ý": 3, "ä": 2, "x": 1, "ʻ": 3},
        ),
        "gag": (
            {"ê": 3, "ţ": 3, "ä": 1.5},
            {"bän", "sän", "hem", "gözäl", "ii", "günnär", "gagauziya", "dil",
             "içeriz", "bizdä"},
            {r"\w+êêr\w*": 2.0},
            {"ə": 3, "ň": 3, "ý": 3, "q": 1.5, "x": 1.5, "ʻ": 3},
        ),
        "kk": (
            {"ū": 3, "ñ": 0.8, "ä": 0.5, "q": 0.5},
            {"jäne", "bız", "üşın", "sälem", "qazaqstan"},
            {},
            {"ə": 3, "ʻ": 3},
        ),
        "tt": (
            {"ñ": 0.5, "ä": 0.5, "w": 0.5},
            {"häm", "belän", "öçen", "isänmesez", "tatarstan", "qazan"},
            {},
            {"ə": 3, "ʻ": 3},
        ),
        "ug": (
            {"ë": 3},
            {"bilen", "üchün", "yaxshimusiz", "uyghur", "rehmet"},
            {r"gh|ch|sh|zh": 0.3},
            {"ə": 3, "ı": 2, "ş": 2, "ç": 2, "ğ": 2},
        ),
    },
    CYRILLIC: {
        "kk": (
            {"ұ": 3, "і": 3, "қ": 1.5, "ғ": 1, "ә": 1, "ң": 0.5, "ө": 0.3, "ү": 0.3,
             "һ": 0.3},
            {"және", "бұл", "үшін", "мен", "жақсы", "сәлеметсіз", "қалайсыз",
             "қалай", "бе", "біз"},
            {},
            {"ҙ": 3, "ҫ": 3, "ҡ": 3, "ў": 3, "җ": 3, "ҳ": 2},
        ),
        "ky": (
            {"ң": 1, "ө": 1, "ү": 1},
            {"жана", "менен", "үчүн", "бул", "абдан", "жакшы", "кандай", "кандайсыз",
             "саламатсызбы", "турам", "мен", "биз", "аба", "ырайы", "кооз"},
            {r"\w+(?:сызбы|сыңар|быз|биз)\b": 0.5},
            {"қ": 3, "ғ": 3, "ә": 3, "і": 3, "ұ": 3, "һ": 3, "җ": 3, "ҙ": 3, "ҫ": 3,
             "ҡ": 3, "ў": 3, "ҳ": 3},
        ),
        "tt": (
            {"җ": 3, "ә": 1, "ө": 0.5, "ү": 0.5, "ң": 0.5, "һ": 0.5},
            {"һәм", "белән", "өчен", "бу", "сәлам", "исәнмесез", "ничек", "бик",
             "әйбәт", "без", "мин", "түгел", "рәхмәт", "хәлләр"},
            {},
            {"қ": 3, "ұ": 3, "і": 3, "ҙ": 3, "ҫ": 3, "ҡ": 3, "ғ": 2, "ў": 3, "ҳ": 3},
        ),
        "ba": (
            {"ҙ": 3, "ҫ": 3, "ҡ": 3, "ғ": 1, "ә": 0.5, "ө": 0.5, "ү": 0.5, "һ": 0.5,
             "ң": 0.5},
            {"менән", "өсөн", "был", "беҙ", "мин", "һәм", "бик", "һаумыһығыҙ"},
            {},
            {"қ": 3, "ұ": 3, "і": 3, "җ": 3, "ў": 3, "ҳ": 2},
        ),
        "uz": (
            {"ў": 3, "ҳ": 2, "қ": 1, "ғ": 1},
            {"ва", "билан", "учун", "эмас", "бу", "салом", "ассалому", "жуда", "яхши"},
            {},
            {"ә": 3, "ө": 3, "ү": 3, "ң": 3, "і": 3, "ұ": 3, "җ": 3, "ҙ": 3},
        ),
        "az": (
            {"ҹ": 3, "ј": 3, "ә": 0.5},
            {"вә", "үчүн", "мән"},
            {},
            {"ұ": 3, "і": 3, "қ": 2, "ң": 2},
        ),
        "tk": (
            {"җ": 0.5, "ә": 0.3, "ң": 0.3},
            {"билен", "үчин"},
            {},
            {"қ": 3, "ұ": 3, "і": 3, "ҙ": 3, "ў": 3},
        ),
        "crh": ({}, {"селям"}, {r"къ|гъ|нъ": 1.5}, {"ә": 2, "ң": 2, "қ": 2}),
        "gag": ({"ӓ": 3, "ӧ": 3, "ӱ": 3}, set(), {}, {}),
    },
    ARABIC: {
        "ug": (
            {"ې": 3, "ۆ": 3, "ۈ": 3, "ۇ": 2, "ە": 2, "ۋ": 2, "ڭ": 0.5, "ھ": 1, "ئ": 1},
            {"بۇ", "ۋە", "بىلەن", "ئۈچۈن", "ياخشى", "مەن"},
            {},
            {"ث": 2, "ذ": 2, "ص": 2, "ض": 2, "ط": 2, "ظ": 2, "ع": 2, "ة": 3},
        ),
        "ota": (
            {"ث": 1, "ذ": 1, "ص": 1, "ض": 1, "ط": 1, "ظ": 1, "ع": 1, "ة": 2,
             "ه": 0.5, "پ": 0.3, "گ": 0.3, "ك": 0.3, "ک": 0.3, "ی": 0.2},
            {"و", "بر", "بو", "ایچون", "ایچین", "دگل", "ایله", "بن", "بز", "پك", "پک"},
            {},
            {"ې": 3, "ۆ": 3, "ۈ": 3, "ۇ": 3, "ە": 3, "ۋ": 3},
        ),
        "az": ({}, set(), {}, {"ې": 3, "ۆ": 3, "ۈ": 3, "ۇ": 3, "ە": 3}),
        "kk": ({}, set(), {}, {}),
        "ky": ({}, set(), {}, {}),
    },
}

_PRIORS = {
    LATIN: {"tr": 1.0},
    CYRILLIC: {"kk": 0.3, "ky": 0.25, "tt": 0.25, "ba": 0.2, "uz": 0.2},
    ARABIC: {"ug": 0.3, "ota": 0.3},
}
_DEFAULT_PRIMARY_PRIOR = 0.2
_DEFAULT_SECONDARY_PRIOR = 0.05


def _detection_lower(text: str) -> str:
    return text.replace("İ", "i").lower()


def guess_turkic_language(text: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """Türk dili için **sezgisel** tahmin; puanı azalan (kod, olasılık) listesi.

    Yöntem: önce baskın yazı belirlenir, yalnız o yazıyı kullanan diller aday
    olur. Her aday için ayırt edici harfler (ör. ə → az; ň/ý/ž → tk;
    oʻ/gʻ → uz; қ/ұ/і → kk; ң/ө/ү ama қ/ғ/ә/і yok → ky; җ → tt; ҙ/ҫ/ҡ → ba;
    ې/ۆ/ۈ → ug), sık işlev kelimeleri ve birkaç ek kalıbı puanlanır;
    başka dile ait ayırt edici harfler ceza verir. Puanlar toplamı 1 olacak
    biçimde normalize edilir.

    Bu fonksiyon eğitilmiş bir dil tanıma modeli değildir: Türkçe dışı
    (Rusça, Farsça, Arapça, İngilizce) metinleri de en yakın Türk diline
    atar. Üretimde fastText ``lid.176`` veya GlotLID ile birlikte kullanın.
    """
    script_info = detect_script(text)
    dominant = script_info["dominant"]
    if dominant is None or dominant == OTHER:
        return []
    lowered = _detection_lower(unicodedata.normalize("NFC", text))
    lowered = _UZ_OKINA_RE.sub(UZBEK_OKINA, lowered)
    words = [w.strip("'’‘`ʼ") for w in _WORD_RE.findall(lowered)]
    char_counts: Dict[str, int] = {}
    for char in lowered:
        if char.isalpha() or char in _LATIN_MODIFIERS:
            char_counts[char] = char_counts.get(char, 0) + 1
    word_count = max(len(words), 1)

    scores: Dict[str, float] = {}
    features = _FEATURES.get(dominant, {})
    for code, language in TURKIC_LANGUAGES.items():
        if dominant not in language.scripts:
            continue
        prior = _PRIORS.get(dominant, {}).get(code)
        if prior is None:
            prior = (
                _DEFAULT_PRIMARY_PRIOR if language.primary_script == dominant
                else _DEFAULT_SECONDARY_PRIOR
            )
        score = prior
        char_weights, lexicon, patterns, penalties = features.get(code, ({}, set(), {}, {}))
        for char, weight in char_weights.items():
            score += weight * min(char_counts.get(char, 0), 3)
        hits = sum(word in lexicon for word in words)
        score += 1.5 * hits * min(1.0, 6.0 / word_count) if hits else 0.0
        for pattern, weight in patterns.items():
            score += weight * min(len(re.findall(pattern, lowered)), 3)
        for char, weight in penalties.items():
            if char_counts.get(char):
                score -= weight * min(char_counts[char], 3)
        scores[code] = max(score, 0.0)

    total = sum(scores.values())
    if total <= 0:
        return []
    ranked = sorted(
        ((code, value / total) for code, value in scores.items() if value > 0),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:top_k] if top_k else ranked


__all__ = [
    "ARABIC", "CYRILLIC", "LATIN", "OTHER", "SCRIPTS",
    "LANGUAGE_TAG_FORMAT", "TRANSLATION_TOKEN",
    "TurkicLanguage", "TURKIC_LANGUAGES", "get_language", "language_tag",
    "language_tag_tokens", "turkic_tokenizer_symbols", "tag_text",
    "detect_script", "guess_turkic_language",
    "normalize_turkic", "normalize_ottoman",
    "kazakh_cyrillic_to_latin", "kyrgyz_cyrillic_to_latin",
    "uzbek_cyrillic_to_latin", "tatar_cyrillic_to_latin",
    "uyghur_arabic_to_latin", "TRANSLITERATORS", "can_transliterate",
    "transliterate_to_latin", "ParallelExample", "make_translation_example",
]
