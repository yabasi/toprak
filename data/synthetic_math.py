# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Sentetik Türkçe Matematik Problemleri Üreticisi

"Düşünen Toprak" (GRPO ile doğrulanabilir ödüllü pekiştirmeli öğrenme) için
kesin, doğrulanabilir cevaplı Türkçe sözel problemler üretir.

Lisans notu: Buradaki TÜM problemler bu dosyadaki şablonlardan rastgele
sayılarla programatik olarak üretilir. ÖSYM, MEB, yayınevi ya da başka bir
telifli kaynaktan soru, metin veya veri KOPYALANMAMIŞTIR; çıktı, projenin
Apache-2.0 lisansı altında serbestçe kullanılabilir.

Özellikler:
  - Tohumlu (seed) ve deterministik: aynı tohum → aynı veri kümesi
  - 11 şablon ailesi × 3 zorluk seviyesi (alışveriş, yaş, hız-zaman-yol,
    işçi-havuz, yüzde/indirim, kesir, bölünebilme, sayı dizisi, denklem,
    olasılık, birim çevirme)
  - Kesin cevaplar `fractions.Fraction` ile hesaplanır (kayan nokta hatası yok)
  - Türkçe dilbilgisi: sayıdan sonra ad tekil kalır ("5 elma", "5 elmalar"
    değil); rakamla yazılan sayılara gelen ekler okunuşa göre uyumlanır
    (`number_suffix`: 5'e, 6'ya, 10'a, 40'a, 3'ü, 2'yi, 9'u, 60'ı, 4'te …)
  - Her örnek için adım adım Türkçe çözüm (SFT ısınması için)

Kayıt biçimi:
    {"id", "question", "answer", "answer_value", "level", "template", "solution"}

Kullanım:
    python data/synthetic_math.py --output-dir data/reasoning \\
        --train 20000 --test 500 --seed 42
    python data/synthetic_math.py --output-dir data/reasoning_sft --format sft
"""

import argparse
import json
import math
import os
import random
import sys
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rewards import REASONING_SYSTEM_PROMPT, format_reasoning  # noqa: E402

# ══════════════════════════════════════════════════════════
#  Türkçe ses uyumu ve sayı ekleri
# ══════════════════════════════════════════════════════════

_ONES = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_BACK_VOWELS = "aıou"
_FRONT_VOWELS = "eiöü"
_VOWELS = _BACK_VOWELS + _FRONT_VOWELS
_VOICELESS = "çfhkpsşt"  # "fıstıkçı şahap"

# Ek kalıpları: (y)/(n)/(s) ünlüyle biten sözcükte kaynaştırma harfidir.
# H = dört yollu ünlü (ı/i/u/ü), A = iki yollu (a/e), D = d/t.
CASE_SPECS = {
    "acc": "(y)H",      # belirtme: 3'ü, 2'yi
    "dat": "(y)A",      # yönelme: 5'e, 6'ya
    "loc": "DA",        # bulunma: 4'te, 6'da
    "abl": "DAn",       # ayrılma: 1'den, 3'ten
    "gen": "(n)Hn",     # ilgi: 2'nin, 3'ün
    "ins": "(y)lA",     # vasıta: 5'le, 6'yla
    "poss3": "(s)H",    # 3. tekil iyelik: 3/5'ü, 12'si, 6'sı
    "poss3_acc": "(s)HnH",  # iyelik + belirtme: 1/4'ini, 2/3'ünü
    "cop": "DHr",       # ek-fiil: 24'tür, 30'dur
}
_CASE_ALIASES = {
    "accusative": "acc", "belirtme": "acc", "-i": "acc",
    "dative": "dat", "yönelme": "dat", "-e": "dat",
    "locative": "loc", "bulunma": "loc", "-de": "loc",
    "ablative": "abl", "ayrılma": "abl", "-den": "abl",
    "genitive": "gen", "ilgi": "gen", "-in": "gen",
    "instrumental": "ins", "vasıta": "ins", "-le": "ins",
    "possessive": "poss3", "iyelik": "poss3",
    "copula": "cop", "ek-fiil": "cop", "-dir": "cop",
}


def _tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _harmonize(word: str, spec: str) -> str:
    """`word` (okunuş, küçük harf) için ek kalıbını somut eke çevir."""
    word = _tr_lower(word)
    vowels = [c for c in word if c in _VOWELS]
    last_vowel = vowels[-1] if vowels else "e"
    ends_vowel = word[-1] in _VOWELS
    voiceless = word[-1] in _VOICELESS

    out = []
    i = 0
    prev = word[-1]
    while i < len(spec):
        ch = spec[i]
        if ch == "(":
            j = spec.index(")", i)
            if ends_vowel and not out:  # kaynaştırma yalnız sözcük sınırında
                out.append(spec[i + 1:j])
                prev = spec[j - 1]
            i = j + 1
            continue
        if ch == "H":
            if last_vowel in "aı":
                ch = "ı"
            elif last_vowel in "ei":
                ch = "i"
            elif last_vowel in "ou":
                ch = "u"
            else:
                ch = "ü"
        elif ch == "A":
            ch = "a" if last_vowel in _BACK_VOWELS else "e"
        elif ch == "D":
            ch = "t" if (not out and voiceless) or (out and prev in _VOICELESS) else "d"
        out.append(ch)
        prev = ch
        i += 1
    return "".join(out)


def suffix_for_word(word: str, case: str) -> str:
    """Okunuşu verilen bir sözcüğe gelecek eki döndür (kesme işaretsiz)."""
    case = _CASE_ALIASES.get(case, case)
    if case not in CASE_SPECS:
        raise ValueError(f"Bilinmeyen hâl/ek: {case!r} (geçerli: {sorted(CASE_SPECS)})")
    return _harmonize(word, CASE_SPECS[case])


def _int_last_word(n: int) -> str:
    """Bir tam sayının okunuşundaki son sözcük (ek uyumunu bu belirler)."""
    n = abs(int(n))
    if n == 0:
        return "sıfır"
    if n % 10:
        return _ONES[n % 10]
    if n % 100:
        return _TENS[(n % 100) // 10]
    if n % 1000:
        return "yüz"
    if n % 10 ** 6:
        return "bin"
    if n % 10 ** 9:
        return "milyon"
    if n % 10 ** 12:
        return "milyar"
    return "trilyon"


def spoken_last_word(display: str) -> str:
    """
    Rakamla yazılmış Türkçe sayının okunuşundaki son sözcük.
        "40" → kırk, "1.000" → bin, "12,5" → beş (on iki virgül beş),
        "3/4" → üç (dörtte üç), "%25" → beş (yüzde yirmi beş)
    """
    s = display.strip().lstrip("-−%")
    if "/" in s:
        return spoken_last_word(s.split("/", 1)[0])
    if "," in s:
        frac = s.split(",", 1)[1]
        return _int_last_word(int(frac)) if frac.strip("0") else "sıfır"
    return _int_last_word(int(s.replace(".", "")))


def format_number(value: Union[int, Fraction, float], prefer_fraction: bool = False) -> str:
    """
    Türkçe sayı yazımı: binlik ayırıcı nokta, ondalık ayırıcı virgül.
        1250 → "1.250", 12.5 → "12,5", 1/3 → "1/3"
    Sonlu ondalık açılımı olan kesirler (en çok 4 basamak) virgüllü yazılır;
    `prefer_fraction` True ise tam olmayan her değer "a/b" biçiminde kalır.
    """
    x = Fraction(value).limit_denominator(10 ** 9) if isinstance(value, float) else Fraction(value)
    sign = "-" if x < 0 else ""
    x = abs(x)

    def _int(n: int) -> str:
        return f"{n:,}".replace(",", ".")

    if x.denominator == 1:
        return sign + _int(x.numerator)
    den = x.denominator
    while den % 2 == 0:
        den //= 2
    while den % 5 == 0:
        den //= 5
    if den == 1 and not prefer_fraction:
        integer = x.numerator // x.denominator
        rest = x - integer
        digits = 0
        while rest.denominator != 1:
            rest *= 10
            digits += 1
        if digits <= 4:
            return f"{sign}{_int(integer)},{str(rest.numerator).zfill(digits)}"
    return f"{sign}{x.numerator}/{x.denominator}"


def number_suffix(n: Union[int, Fraction, float, str], case: str) -> str:
    """
    Rakamla yazılmış sayıya kesme işaretiyle doğru Türkçe eki ekle.

        number_suffix(5, "dat")  → "5'e"     number_suffix(6, "dat")  → "6'ya"
        number_suffix(10, "dat") → "10'a"    number_suffix(100, "dat") → "100'e"
        number_suffix(3, "acc")  → "3'ü"     number_suffix(60, "acc") → "60'ı"
        number_suffix(4, "loc")  → "4'te"    number_suffix("3/5", "poss3") → "3/5'ü"

    `n` sayı ya da hazır biçimlenmiş Türkçe sayı dizesi olabilir. Hâl adları:
    acc, dat, loc, abl, gen, ins, poss3, poss3_acc, cop (ya da Türkçe adları).
    """
    display = n if isinstance(n, str) else format_number(n)
    return f"{display}'{suffix_for_word(spoken_last_word(display), case)}"


def name_suffix(name: str, case: str) -> str:
    """Özel ada kesme işaretiyle ek getir: Ali'nin, Can'a, Mehmet'e, Ayşe'ye."""
    return f"{name}'{suffix_for_word(name, case)}"


# ══════════════════════════════════════════════════════════
#  Şablonlar
# ══════════════════════════════════════════════════════════

NAMES = [
    "Ali", "Ayşe", "Mehmet", "Zeynep", "Elif", "Can", "Deniz", "Murat", "Emre",
    "Selin", "Burak", "Ece", "Kerem", "Defne", "Okan", "Yusuf", "Merve", "Oğuz",
    "Duru", "Arda",
]

# (çoğul ayrılma hâli, birim fiyat aralığı TL)
SHOP_ITEMS = [
    ("kalemlerden", (3, 15)), ("defterlerden", (10, 40)), ("silgilerden", (2, 9)),
    ("elmalardan", (2, 8)), ("ekmeklerden", (5, 15)), ("kitaplardan", (40, 120)),
    ("çikolatalardan", (8, 30)), ("su şişelerinden", (3, 12)),
]

N = format_number  # kısaltma
S = number_suffix

# Şablon fonksiyonu: (rng, level) → (soru, değer, çözüm adımları, tür)
# tür: "auto" (tam/ondalık/kesir), "fraction" (kesir tercih), "percent"
TemplateFn = Callable[[random.Random, int], Tuple[str, Fraction, List[str], str]]
TEMPLATES: Dict[str, TemplateFn] = {}


def template(name: str):
    """Yeni şablon kaydı için dekoratör (bkz. REASONING.md "Şablon ekleme")."""
    def wrap(fn: TemplateFn) -> TemplateFn:
        TEMPLATES[name] = fn
        return fn
    return wrap


def _two_names(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.sample(NAMES, 2)
    return a, b


@template("alisveris")
def _alisveris(rng, level):
    name = rng.choice(NAMES)
    (item1, (lo1, hi1)), (item2, (lo2, hi2)) = rng.sample(SHOP_ITEMS, 2)
    p1, q1 = rng.randint(lo1, hi1), rng.randint(2, 9)
    if level == 1:
        total = p1 * q1
        q = f"{name} tanesi {p1} TL olan {item1} {q1} tane aldı. {name} toplam kaç TL ödedi?"
        steps = [f"Toplam tutar = adet × birim fiyat = {q1} × {p1} = {N(total)} TL."]
        return q, Fraction(total), steps, "auto"
    p2, q2 = rng.randint(lo2, hi2), rng.randint(1, 6)
    total = p1 * q1 + p2 * q2
    if level == 2:
        paid = (total // 50 + 1) * 50 + rng.choice([0, 50, 100])
        change = paid - total
        q = (f"{name} tanesi {p1} TL olan {item1} {q1} tane, tanesi {p2} TL olan "
             f"{item2} {q2} tane aldı. Kasaya {paid} TL verdi. "
             f"{name} kaç TL para üstü alır?")
        steps = [
            f"Birinci ürünler: {q1} × {p1} = {p1 * q1} TL.",
            f"İkinci ürünler: {q2} × {p2} = {p2 * q2} TL.",
            f"Toplam: {p1 * q1} + {p2 * q2} = {total} TL.",
            f"Para üstü: {paid} − {total} = {N(change)} TL.",
        ]
        return q, Fraction(change), steps, "auto"
    disc = rng.choice([10, 20, 25, 50])
    discounted = Fraction(total * (100 - disc), 100)
    q = (f"{name} tanesi {p1} TL olan {item1} {q1} tane, tanesi {p2} TL olan "
         f"{item2} {q2} tane aldı. Kasada toplam tutara %{disc} indirim uygulandı. "
         f"{name} kaç TL ödedi?")
    steps = [
        f"Toplam: {q1} × {p1} + {q2} × {p2} = {p1 * q1} + {p2 * q2} = {total} TL.",
        f"%{disc} indirimle tutarın %{100 - disc}'{suffix_for_word(_int_last_word(100 - disc), 'poss3')} ödenir.",
        f"Ödenen: {total} × {100 - disc} / 100 = {N(discounted)} TL.",
    ]
    return q, discounted, steps, "auto"


@template("yas")
def _yas(rng, level):
    a, b = _two_names(rng)
    if level == 1:
        age_b, diff, k = rng.randint(4, 30), rng.randint(1, 15), rng.randint(2, 12)
        ans = age_b + diff + k
        q = (f"{a}, {name_suffix(b, 'abl')} {diff} yaş büyüktür. {b} {age_b} yaşında "
             f"olduğuna göre {a} {k} yıl sonra kaç yaşında olur?")
        steps = [
            f"{a} şimdi {age_b} + {diff} = {age_b + diff} yaşındadır.",
            f"{k} yıl sonra: {age_b + diff} + {k} = {ans}.",
        ]
        return q, Fraction(ans), steps, "auto"
    if level == 2:
        while True:
            child, x, k = rng.randint(3, 15), rng.randint(1, 20), rng.choice([2, 3])
            mother = k * (child + x) - x
            if 20 <= mother - child and mother <= 65:
                break
        q = (f"{a} {child} yaşında, annesi ise {mother} yaşındadır. Kaç yıl sonra "
             f"annesinin yaşı {name_suffix(a, 'gen')} yaşının {k} katı olur?")
        steps = [
            f"x yıl sonra annenin yaşı {mother} + x, {name_suffix(a, 'gen')} yaşı {child} + x olur.",
            f"Denklem: {mother} + x = {k} × ({child} + x).",
            f"{mother} + x = {k * child} + {k}x ⇒ {mother - k * child} = {k - 1}x.",
            f"x = {mother - k * child} / {k - 1} = {x}.",
        ]
        return q, Fraction(x), steps, "auto"
    while True:
        b0, n, k = rng.randint(1, 12), rng.randint(2, 15), rng.choice([2, 3, 4])
        age_a, age_b = k * b0 + n, b0 + n
        if age_a <= 70:
            break
    s = age_a + age_b
    q = (f"{a} ile {name_suffix(b, 'gen')} yaşları toplamı {S(s, 'cop')}. {n} yıl önce "
         f"{name_suffix(a, 'gen')} yaşı {name_suffix(b, 'gen')} yaşının {k} katıydı. "
         f"{a} bugün kaç yaşındadır?")
    steps = [
        f"{name_suffix(b, 'gen')} bugünkü yaşı b olsun; {a} {s} − b yaşındadır.",
        f"{n} yıl önce: ({s} − b − {n}) = {k} × (b − {n}).",
        f"{s - n} − b = {k}b − {k * n} ⇒ {s - n + k * n} = {k + 1}b ⇒ b = {age_b}.",
        f"{a}: {s} − {age_b} = {age_a}.",
    ]
    return q, Fraction(age_a), steps, "auto"


@template("hiz")
def _hiz(rng, level):
    if level == 1:
        v, t = rng.choice(range(40, 125, 5)), rng.randint(2, 8)
        q = f"Saatte {v} km hızla giden bir araç {t} saatte kaç km yol alır?"
        steps = [f"Yol = hız × zaman = {v} × {t} = {N(v * t)} km."]
        return q, Fraction(v * t), steps, "auto"
    if level == 2:
        v = rng.choice(range(40, 125, 10))
        t = Fraction(rng.choice([3, 4, 5, 6, 7, 8, 9, 10]), 2)
        d = v * t
        q = (f"{N(d)} kilometrelik bir yolu saatte {v} km hızla giden bir araç "
             f"kaç saatte tamamlar?")
        steps = [f"Zaman = yol / hız = {N(d)} / {v} = {N(t)} saat."]
        return q, t, steps, "auto"
    v1, v2 = rng.choice(range(40, 100, 10)), rng.choice(range(50, 130, 10))
    t = rng.randint(2, 6)
    d = (v1 + v2) * t
    q = (f"Aralarında {N(d)} km bulunan iki şehirden iki araç aynı anda birbirine doğru "
         f"yola çıkıyor. Araçların hızları saatte {v1} km ve saatte {v2} kilometredir. "
         f"Araçlar kaç saat sonra karşılaşır?")
    steps = [
        f"Araçlar birbirine yaklaştığı için hızlar toplanır: {v1} + {v2} = {v1 + v2} km/sa.",
        f"Karşılaşma süresi = {N(d)} / {v1 + v2} = {t} saat.",
    ]
    return q, Fraction(t), steps, "auto"


@template("isci_havuz")
def _isci_havuz(rng, level):
    if level == 1:
        w1, d1 = rng.randint(2, 12), rng.randint(2, 15)
        work = w1 * d1
        divisors = [w for w in range(2, 31) if work % w == 0 and w != w1]
        w2 = rng.choice(divisors) if divisors else w1 * 2
        ans = Fraction(work, w2)
        q = (f"{w1} işçi bir işi {d1} günde bitiriyor. Aynı hızda çalışan {w2} işçi "
             f"aynı işi kaç günde bitirir?")
        steps = [
            f"İşin tamamı {w1} × {d1} = {work} işçi-gün eder.",
            f"{w2} işçi ile: {work} / {w2} = {N(ans)} gün.",
        ]
        return q, ans, steps, "auto"
    if level == 2:
        a, b = _two_names(rng)
        x, y = rng.sample(range(2, 21), 2)
        ans = Fraction(x * y, x + y)
        q = (f"{a} bir işi tek başına {x} günde, {b} ise {y} günde bitirebiliyor. "
             f"İkisi birlikte çalışırsa iş kaç günde biter?")
        steps = [
            f"{a} bir günde işin 1/{x}'{suffix_for_word('bir', 'poss3')}, {b} 1/{y}'{suffix_for_word('bir', 'poss3')} yapar.",
            f"Birlikte bir günde: 1/{x} + 1/{y} = {N(Fraction(1, x) + Fraction(1, y), True)}.",
            f"Süre: 1 / ({N(Fraction(1, x) + Fraction(1, y), True)}) = {N(ans)} gün.",
        ]
        return q, ans, steps, "auto"
    fill = rng.randint(2, 12)
    drain = rng.randint(fill + 1, fill + 12)
    ans = Fraction(fill * drain, drain - fill)
    q = (f"Boş bir havuzu A musluğu tek başına {fill} saatte dolduruyor; B musluğu ise "
         f"dolu havuzu {drain} saatte boşaltıyor. İki musluk birlikte açılırsa boş "
         f"havuz kaç saatte dolar?")
    rate = Fraction(1, fill) - Fraction(1, drain)
    steps = [
        f"Bir saatte dolan kısım: 1/{fill} − 1/{drain} = {N(rate, True)}.",
        f"Süre: 1 / ({N(rate, True)}) = {N(ans)} saat.",
    ]
    return q, ans, steps, "auto"


@template("yuzde")
def _yuzde(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            n = rng.choice([20, 25, 40, 50, 80, 100, 200])
            pct = rng.choice([p for p in (5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 75, 80)
                              if n * p % 100 == 0])
            k = n * pct // 100
            q = (f"Bir okulda {n} öğrenci vardır. Bu öğrencilerin {S(k, 'poss3')} "
                 f"gözlüklüdür. Gözlüklü öğrencilerin oranı yüzde kaçtır?")
            steps = [f"Oran = {k} / {n} = {N(Fraction(k, n))}.",
                     f"Yüzde olarak: {N(Fraction(k, n))} × 100 = %{pct}."]
            return q, Fraction(pct), steps, "percent"
        price = rng.choice(range(40, 1001, 20))
        disc = rng.choice([10, 15, 20, 25, 30, 40, 50])
        ans = Fraction(price * (100 - disc), 100)
        q = (f"Fiyatı {N(price)} TL olan bir ürün %{disc} indirimle satılıyor. "
             f"Ürünün indirimli fiyatı kaç TL olur?")
        steps = [f"İndirim tutarı: {N(price)} × {disc} / 100 = {N(Fraction(price * disc, 100))} TL.",
                 f"İndirimli fiyat: {N(price)} − {N(Fraction(price * disc, 100))} = {N(ans)} TL."]
        return q, ans, steps, "auto"
    if level == 2:
        price = rng.choice(range(100, 2001, 50))
        up, down = rng.choice([10, 20, 25, 50]), rng.choice([10, 20, 25, 50])
        mid = Fraction(price * (100 + up), 100)
        ans = mid * Fraction(100 - down, 100)
        q = (f"Fiyatı {N(price)} TL olan bir ürüne önce %{up} zam yapılıyor, ardından yeni "
             f"fiyat üzerinden %{down} indirim uygulanıyor. Ürünün son fiyatı kaç TL olur?")
        steps = [f"Zamlı fiyat: {N(price)} × {100 + up} / 100 = {N(mid)} TL.",
                 f"İndirimli fiyat: {N(mid)} × {100 - down} / 100 = {N(ans)} TL."]
        return q, ans, steps, "auto"
    disc = rng.choice([10, 20, 25, 40, 50, 60, 75])
    step = 100 // math.gcd(100, 100 - disc)
    original = step * rng.randint(2, max(3, 2000 // step))
    final = Fraction(original * (100 - disc), 100)
    q = (f"%{disc} indirimle {N(final)} TL'ye satılan bir ürünün indirimden önceki "
         f"fiyatı kaç TL'dir?")
    steps = [f"İndirimli fiyat, eski fiyatın %{100 - disc}'{suffix_for_word(_int_last_word(100 - disc), 'poss3')} eder.",
             f"Eski fiyat: {N(final)} × 100 / {100 - disc} = {N(original)} TL."]
    return q, Fraction(original), steps, "auto"


def _fraction(rng: random.Random, max_den: int = 9) -> Fraction:
    den = rng.randint(2, max_den)
    return Fraction(rng.randint(1, den - 1), den)


@template("kesir")
def _kesir(rng, level):
    if level == 1:
        f = _fraction(rng)
        x = f.denominator * rng.randint(2, 15)
        part = x * f
        fs = f"{f.numerator}/{f.denominator}"
        q = f"Bir sayının {S(fs, 'poss3')} {S(N(part), 'cop')}. Bu sayı kaçtır?"
        steps = [f"Sayı x olsun: x × {fs} = {N(part)}.",
                 f"x = {N(part)} × {f.denominator} / {f.numerator} = {N(x)}."]
        return q, Fraction(x), steps, "auto"
    if level == 2:
        a, b = _two_names(rng)
        while True:
            f1, f2 = _fraction(rng, 8), _fraction(rng, 8)
            rest = 1 - f1 - f2
            if rest > 0:
                break
        s1, s2 = f"{f1.numerator}/{f1.denominator}", f"{f2.numerator}/{f2.denominator}"
        q = (f"{a} bir pastanın {S(s1, 'poss3_acc')}, {b} ise {S(s2, 'poss3_acc')} yedi. "
             f"Pastanın kaçta kaçı kaldı?")
        steps = [f"Yenen kısım: {s1} + {s2} = {N(f1 + f2, True)}.",
                 f"Kalan: 1 − {N(f1 + f2, True)} = {N(rest, True)}."]
        return q, rest, steps, "fraction"
    while True:
        f1, f2 = _fraction(rng, 6), _fraction(rng, 6)
        if f2 > f1:
            break
    lcm = f1.denominator * f2.denominator // math.gcd(f1.denominator, f2.denominator)
    cap = lcm * rng.randint(2, 20)
    added = (f2 - f1) * cap
    s1, s2 = f"{f1.numerator}/{f1.denominator}", f"{f2.numerator}/{f2.denominator}"
    q = (f"Bir su deposunun {S(s1, 'poss3')} doludur. Depoya {N(added)} litre su "
         f"eklenince deponun {S(s2, 'poss3')} dolu oluyor. Deponun tamamı kaç litre su alır?")
    steps = [f"Eklenen su deponun {s2} − {s1} = {N(f2 - f1, True)} kadarıdır.",
             f"Kapasite: {N(added)} / ({N(f2 - f1, True)}) = {N(cap)} litre."]
    return q, Fraction(cap), steps, "auto"


@template("bolunebilme")
def _bolunebilme(rng, level):
    if level == 1:
        n, k = rng.choice(range(30, 201, 10)), rng.randint(3, 13)
        ans = n // k
        q = (f"1'den {S(n, 'dat')} kadar (ikisi de dahil) olan doğal sayılardan kaç "
             f"tanesi {k} ile tam bölünür?")
        steps = [f"{k} ile bölünenler {k}, {2 * k}, … biçimindedir.",
                 f"Sayıları: ⌊{n} / {k}⌋ = {ans}."]
        return q, Fraction(ans), steps, "auto"
    if level == 2:
        a = rng.randint(10, 200)
        b = a + rng.randint(30, 300)
        k = rng.randint(3, 17)
        ans = b // k - (a - 1) // k
        q = (f"{a} ile {b} arasındaki (ikisi de dahil) doğal sayılardan kaç tanesi "
             f"{k} ile tam bölünür?")
        steps = [f"1'den {S(b, 'dat')} kadar: ⌊{b} / {k}⌋ = {b // k}.",
                 f"1'den {S(a - 1, 'dat')} kadar: ⌊{a - 1} / {k}⌋ = {(a - 1) // k}.",
                 f"Fark: {b // k} − {(a - 1) // k} = {ans}."]
        return q, Fraction(ans), steps, "auto"
    n = rng.choice(range(50, 501, 10))
    p, r = rng.sample([2, 3, 4, 5, 6, 7], 2)
    lcm = p * r // math.gcd(p, r)
    ans = n // p + n // r - n // lcm
    q = (f"1'den {S(n, 'dat')} kadar (ikisi de dahil) olan doğal sayılardan kaç tanesi "
         f"{S(p, 'dat')} ya da {S(r, 'dat')} tam bölünür?")
    steps = [f"{S(p, 'dat')} bölünenler: ⌊{n} / {p}⌋ = {n // p}.",
             f"{S(r, 'dat')} bölünenler: ⌊{n} / {r}⌋ = {n // r}.",
             f"İkisine birden bölünenler (EKOK = {lcm}): ⌊{n} / {lcm}⌋ = {n // lcm}.",
             f"İçerme-dışlama: {n // p} + {n // r} − {n // lcm} = {ans}."]
    return q, Fraction(ans), steps, "auto"


@template("sayi_dizisi")
def _sayi_dizisi(rng, level):
    a1 = rng.randint(-20, 30)
    d = rng.choice([x for x in range(-9, 13) if x != 0])
    terms = [a1 + i * d for i in range(4)]
    shown = ", ".join(str(t) for t in terms)
    if level == 1:
        ans = a1 + 4 * d
        q = f"{shown}, … dizisinde bir sonraki terim kaçtır?"
        steps = [f"Ardışık terimler arasındaki fark {d}.",
                 f"Sonraki terim: {terms[-1]} + ({d}) = {ans}."]
        return q, Fraction(ans), steps, "auto"
    if level == 2:
        n = rng.randint(8, 40)
        ans = a1 + (n - 1) * d
        q = f"{shown}, … aritmetik dizisinin {n}. terimi kaçtır?"
        steps = [f"İlk terim {a1}, ortak fark {d}.",
                 f"n. terim = a₁ + (n − 1) × d = {a1} + {n - 1} × ({d}) = {ans}."]
        return q, Fraction(ans), steps, "auto"
    if rng.random() < 0.5:
        b, r = rng.randint(1, 5), rng.choice([2, 3, -2])
        terms = [b * r ** i for i in range(4)]
        ans = b * r ** 4
        q = f"{', '.join(str(t) for t in terms)}, … dizisinde bir sonraki terim kaçtır?"
        steps = [f"Her terim bir öncekinin {r} katıdır (geometrik dizi).",
                 f"Sonraki terim: {terms[-1]} × ({r}) = {ans}."]
        return q, Fraction(ans), steps, "auto"
    n = rng.randint(5, 30)
    last = a1 + (n - 1) * d
    ans = Fraction(n * (a1 + last), 2)
    q = f"{shown}, … aritmetik dizisinin ilk {n} teriminin toplamı kaçtır?"
    steps = [f"{n}. terim: {a1} + {n - 1} × ({d}) = {last}.",
             f"Toplam = n × (ilk + son) / 2 = {n} × ({a1} + {last}) / 2 = {N(ans)}."]
    return q, ans, steps, "auto"


@template("denklem")
def _denklem(rng, level):
    if level == 1:
        x, k, c = rng.randint(2, 40), rng.randint(2, 9), rng.randint(1, 30)
        if rng.random() < 0.5:
            v = k * x + c
            q = f"Bir sayının {k} katının {c} fazlası {S(v, 'cop')}. Bu sayı kaçtır?"
            steps = [f"{k}x + {c} = {v}", f"{k}x = {v - c}", f"x = {x}"]
        else:
            v = k * x - c
            q = f"Bir sayının {k} katının {c} eksiği {S(v, 'cop')}. Bu sayı kaçtır?"
            steps = [f"{k}x − {c} = {v}", f"{k}x = {v + c}", f"x = {x}"]
        return q, Fraction(x), steps, "auto"
    if level == 2:
        x = rng.randint(-15, 20)
        a, c = rng.sample(range(2, 10), 2)
        b = rng.randint(-20, 20)
        d = a * x + b - c * x
        def lin(coef, const):
            sign = "+" if const >= 0 else "−"
            return f"{coef}x {sign} {abs(const)}"
        q = f"{lin(a, b)} = {lin(c, d)} ise x kaçtır?"
        steps = [f"x'li terimleri bir tarafa topla: {a}x − {c}x = {d} − ({b}).",
                 f"{a - c}x = {d - b}.", f"x = {d - b} / {a - c} = {x}."]
        return q, Fraction(x), steps, "auto"
    small = rng.randint(1, 60)
    diff = rng.randint(1, 40)
    big = small + diff
    s = big + small
    ask_big = rng.random() < 0.5
    q = (f"İki sayının toplamı {s}, farkı {S(diff, 'cop')}. "
         f"{'Büyük' if ask_big else 'Küçük'} sayı kaçtır?")
    steps = [f"Büyük sayı = (toplam + fark) / 2 = ({s} + {diff}) / 2 = {big}.",
             f"Küçük sayı = (toplam − fark) / 2 = ({s} − {diff}) / 2 = {small}."]
    return q, Fraction(big if ask_big else small), steps, "auto"


_COLORS = [("kırmızı", "kırmızı"), ("mavi", "mavi"), ("yeşil", "yeşil"), ("sarı", "sarı")]


@template("olasilik")
def _olasilik(rng, level):
    if level == 1:
        counts = [rng.randint(1, 9) for _ in range(3)]
        colors = [c for c, _ in rng.sample(_COLORS, 3)]
        idx = rng.randrange(3)
        total = sum(counts)
        ans = Fraction(counts[idx], total)
        q = (f"Bir torbada {counts[0]} {colors[0]}, {counts[1]} {colors[1]} ve "
             f"{counts[2]} {colors[2]} top vardır. Torbadan rastgele çekilen bir topun "
             f"{colors[idx]} olma olasılığı kaçtır?")
        steps = [f"Toplam top: {counts[0]} + {counts[1]} + {counts[2]} = {total}.",
                 f"İstenen durum: {counts[idx]}. Olasılık: {counts[idx]}/{total} = {N(ans, True)}."]
        return q, ans, steps, "fraction"
    if level == 2:
        if rng.random() < 0.5:
            k = rng.randint(1, 5)
            if rng.random() < 0.5:
                good = [v for v in range(1, 7) if v > k]
                cond = f"{S(k, 'abl')} büyük"
            else:
                k += 1
                good = [v for v in range(1, 7) if v < k]
                cond = f"{S(k, 'abl')} küçük"
            ans = Fraction(len(good), 6)
            q = (f"Hilesiz bir zar atılıyor. Üst yüze gelen sayının {cond} olma "
                 f"olasılığı kaçtır?")
            steps = [f"Uygun sonuçlar: {', '.join(map(str, good))} ({len(good)} tane).",
                     f"Olasılık: {len(good)}/6 = {N(ans, True)}."]
            return q, ans, steps, "fraction"
        s = rng.randint(2, 12)
        ways = sum(1 for i in range(1, 7) for j in range(1, 7) if i + j == s)
        ans = Fraction(ways, 36)
        q = (f"İki hilesiz zar birlikte atılıyor. Üst yüze gelen sayıların toplamının "
             f"{s} olma olasılığı kaçtır?")
        steps = [f"Toplam 36 eşit olasılıklı sonuç vardır.",
                 f"Toplamı {s} olan sonuç sayısı: {ways}.",
                 f"Olasılık: {ways}/36 = {N(ans, True)}."]
        return q, ans, steps, "fraction"
    r, m = rng.randint(2, 9), rng.randint(1, 9)
    n = r + m
    ans = Fraction(r, n) * Fraction(r - 1, n - 1)
    q = (f"Bir torbada {r} kırmızı ve {m} mavi top vardır. Torbadan art arda ve geri "
         f"atılmadan iki top çekiliyor. İki topun da kırmızı olma olasılığı kaçtır?")
    steps = [f"İlk topun kırmızı olma olasılığı: {r}/{n}.",
             f"Sonra {n - 1} top kalır, {r - 1} tanesi kırmızı: {r - 1}/{n - 1}.",
             f"Çarpım: {r}/{n} × {r - 1}/{n - 1} = {N(ans, True)}."]
    return q, ans, steps, "fraction"


@template("birim_cevirme")
def _birim_cevirme(rng, level):
    if level == 1:
        kind = rng.randrange(3)
        x = Fraction(rng.randint(2, 99), rng.choice([1, 2, 10]))
        if kind == 0:
            ans = x * 1000
            q = f"{N(x)} kilogram kaç gramdır?"
            steps = [f"1 kilogram = 1.000 gram.", f"{N(x)} × 1.000 = {N(ans)} gram."]
        elif kind == 1:
            ans = x * 100
            q = f"{N(x)} metre kaç santimetredir?"
            steps = [f"1 metre = 100 santimetre.", f"{N(x)} × 100 = {N(ans)} santimetre."]
        else:
            ans = x * 1000
            q = f"{N(x)} litre kaç mililitredir?"
            steps = [f"1 litre = 1.000 mililitre.", f"{N(x)} × 1.000 = {N(ans)} mililitre."]
        return q, ans, steps, "auto"
    if level == 2:
        kind = rng.randrange(3)
        if kind == 0:
            h, m = rng.randint(1, 9), rng.randint(1, 59)
            ans = h * 60 + m
            q = f"{h} saat {m} dakika toplam kaç dakikadır?"
            steps = [f"{h} saat = {h} × 60 = {h * 60} dakika.", f"{h * 60} + {m} = {ans} dakika."]
            return q, Fraction(ans), steps, "auto"
        if kind == 1:
            minutes = 15 * rng.randint(2, 40)
            ans = Fraction(minutes, 60)
            q = f"{minutes} dakika kaç saattir?"
            steps = [f"1 saat = 60 dakika.", f"{minutes} / 60 = {N(ans)} saat."]
            return q, ans, steps, "auto"
        grams = rng.randint(1, 400) * 25
        ans = Fraction(grams, 1000)
        q = f"{N(grams)} gram kaç kilogramdır?"
        steps = [f"1 kilogram = 1.000 gram.", f"{N(grams)} / 1.000 = {N(ans)} kilogram."]
        return q, ans, steps, "auto"
    if rng.random() < 0.5:
        v = 18 * rng.randint(1, 8)
        ans = Fraction(v * 1000, 3600)
        q = f"Saatte {v} km hızla giden bir aracın hızı saniyede kaç metredir?"
        steps = [f"1 km = 1.000 m, 1 saat = 3.600 saniye.",
                 f"{v} × 1.000 / 3.600 = {N(ans)} m/sn."]
        return q, ans, steps, "auto"
    cup = rng.choice([200, 250, 400, 500])
    liters = Fraction(cup * rng.randint(4, 40), 1000)
    ans = liters * 1000 / cup
    q = (f"{N(liters)} litre su, {cup} mililitrelik bardaklara dolduruluyor. "
         f"Kaç bardak dolar?")
    steps = [f"{N(liters)} litre = {N(liters * 1000)} mililitre.",
             f"{N(liters * 1000)} / {cup} = {N(ans)} bardak."]
    return q, ans, steps, "auto"


# ══════════════════════════════════════════════════════════
#  Üretim
# ══════════════════════════════════════════════════════════

LEVELS = (1, 2, 3)


def canonical_answer(value: Fraction, kind: str) -> Tuple[str, Union[int, float, str]]:
    """Kesin değerden (kanonik cevap dizesi, answer_value) üret."""
    value = Fraction(value)
    if kind == "percent":
        return "%" + format_number(value), (int(value) if value.denominator == 1 else float(value))
    text = format_number(value, prefer_fraction=(kind == "fraction"))
    if value.denominator == 1:
        return text, int(value)
    if "/" in text:
        return text, f"{value.numerator}/{value.denominator}"
    return text, float(value)


def generate_problem(
    rng: random.Random,
    template_name: Optional[str] = None,
    level: Optional[int] = None,
) -> Dict:
    """Tek problem üret (id hariç)."""
    template_name = template_name or rng.choice(sorted(TEMPLATES))
    level = level or rng.choice(LEVELS)
    question, value, steps, kind = TEMPLATES[template_name](rng, level)
    answer, answer_value = canonical_answer(value, kind)
    return {
        "question": question,
        "answer": answer,
        "answer_value": answer_value,
        "level": level,
        "template": template_name,
        "solution": "\n".join(steps + [f"Sonuç: {answer}"]),
    }


def generate_dataset(
    n: int,
    seed: int,
    templates: Optional[Sequence[str]] = None,
    levels: Sequence[int] = LEVELS,
    id_prefix: str = "sm",
) -> List[Dict]:
    """
    `n` problem üret. Her örnek kendi tohumundan (seed, i) türetilen bağımsız
    bir RNG ile üretilir; böylece çıktı deterministiktir ve alt kümeler
    tekrarlanabilir.
    """
    names = sorted(templates) if templates else sorted(TEMPLATES)
    unknown = set(names) - set(TEMPLATES)
    if unknown:
        raise ValueError(f"Bilinmeyen şablon(lar): {sorted(unknown)}")
    items = []
    for i in range(n):
        rng = random.Random(f"{seed}:{i}")
        item = generate_problem(rng, rng.choice(names), rng.choice(list(levels)))
        items.append({"id": f"{id_prefix}-{seed}-{i:06d}", **item})
    return items


def sft_target(item: Dict) -> str:
    """SFT ısınması için hedef asistan çıktısı (`<düşünce>` biçimi)."""
    return format_reasoning(item["solution"], item["answer"])


def to_sft_record(item: Dict, system_prompt: Optional[str] = REASONING_SYSTEM_PROMPT) -> Dict:
    """Sohbet şablonuna uygun {"messages": [...]} kaydı."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": item["question"]})
    messages.append({"role": "assistant", "content": sft_target(item)})
    return {"id": item["id"], "messages": messages}


def split_train_test(
    n_train: int,
    n_test: int,
    seed: int,
    test_seed: Optional[int] = None,
    **kwargs,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Ayrık tohumlarla eğitim/test kümeleri üret. Test kümesinde eğitimdeki bir
    soruyla birebir aynı olan örnekler atılır (sızıntı önleme).
    """
    test_seed = seed + 1_000_003 if test_seed is None else test_seed
    if test_seed == seed:
        raise ValueError("Eğitim ve test tohumları farklı olmalı.")
    train = generate_dataset(n_train, seed, id_prefix="train", **kwargs)
    seen = {item["question"] for item in train}
    test = []
    i = 0
    while len(test) < n_test:
        batch = generate_dataset(n_test, test_seed + i, id_prefix="test", **kwargs)
        for item in batch:
            if item["question"] not in seen and len(test) < n_test:
                seen.add(item["question"])
                test.append(item)
        i += 1
        if i > 50:
            break
    return train, test


def _write_jsonl(path: str, records: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Toprak — sentetik Türkçe matematik problemleri (lisanssız, doğrulanabilir cevaplı)"
    )
    parser.add_argument("--output-dir", default="data/reasoning", help="Çıktı klasörü")
    parser.add_argument("--train", type=int, default=20000, help="Eğitim örneği sayısı")
    parser.add_argument("--test", type=int, default=500, help="Test örneği sayısı")
    parser.add_argument("--seed", type=int, default=42, help="Eğitim tohumu")
    parser.add_argument("--test-seed", type=int, default=None,
                        help="Test tohumu (varsayılan: seed + 1000003; eğitimden farklı olmalı)")
    parser.add_argument("--templates", default=None,
                        help=f"Virgülle ayrılmış şablonlar (varsayılan: hepsi): {','.join(sorted(TEMPLATES))}")
    parser.add_argument("--levels", default="1,2,3", help="Zorluk seviyeleri, ör. 1,2")
    parser.add_argument("--format", choices=["raw", "sft"], default="raw",
                        help="raw: soru/cevap kayıtları (GRPO); sft: sohbet mesajları (SFT ısınması)")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="SFT kayıtlarına sistem mesajı ekleme")
    args = parser.parse_args(argv)

    templates = args.templates.split(",") if args.templates else None
    levels = tuple(int(x) for x in args.levels.split(","))
    train, test = split_train_test(args.train, args.test, args.seed, args.test_seed,
                                   templates=templates, levels=levels)
    if args.format == "sft":
        system = None if args.no_system_prompt else REASONING_SYSTEM_PROMPT
        train = [to_sft_record(item, system) for item in train]
        test = [to_sft_record(item, system) for item in test]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    _write_jsonl(train_path, train)
    _write_jsonl(test_path, test)
    print(f"✅ {len(train)} eğitim örneği → {train_path}")
    print(f"✅ {len(test)} test örneği → {test_path}")


if __name__ == "__main__":
    main()
