# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Token Düzeyinde Dilbilimsel Etiketleyiciler

Sondalar (probe) ve SAE özellik ilişkilendirmesi için her token'a, token
metninden ve içinde bulunduğu kelimenin o ana kadarki kısmından türetilen
etiketler verir. Etiket tanımsızsa ``-1`` (yok say) kullanılır.

UYARI — sezgisel (heuristic) etiketler:
    Toprak BPE (SentencePiece) parçalarıyla çalışır; parçalar morfem sınırına
    denk gelmek zorunda değildir ("kitapların" → "▁kitap" + "ların" veya
    "▁kitapla" + "rın" olabilir). Buradaki etiketler morfolojik çözümleyici
    çıktısı DEĞİLDİR; parçaların yüzey biçimine bakan basit kurallardır.
    Özellikle "ek_turu" yalnız yaygın ek biçimlerini (çoğul, bulunma,
    ayrılma, -DI geçmiş zaman, -mIş) yakalar ve "da/de" bağlacı gibi
    belirsizlikleri ayıramaz. Sonuçlar bu gürültüyle birlikte okunmalıdır.

Özellikler (FEATURES):
    kelime_basi        token kelime başı mı (▁) yoksa devam parçası mı
    morf_sinifi        kök / ek / özel (ToprakLM.token_morph_classes ile aynı kural)
    son_unlu           kelimenin bu token dahil son ünlüsü: kalın / ince / yok
    sonraki_ek         bir sonraki token bu kelimeye eklenen bir ek mi
                       (nedensel/causal sonda: model sıradaki eki "bekliyor" mu)
    beklenen_uyum      sıradaki ek ünlü içeriyorsa, büyük ünlü uyumunun
                       gerektirdiği sınıf (kelimenin şimdiye dek son ünlüsü)
    sonraki_ek_unlusu  sıradaki ek parçasının gerçek ilk ünlüsü (kalın / ince)
    sert_unsuz         kelime şimdiye dek sert ünsüzle (f s t k ç ş h p) bitiyor mu
                       — "fıstıkçı şahap"; sonraki ek d→t, c→ç, g→k sertleşir
    ek_turu            ek parçasının türü: diğer / çoğul / bulunma / ayrılma /
                       geçmiş zaman (-DI) / -mIş
    sonraki_ek_turu    sıradaki token'ın ek türü (yalnız sıradaki token ek ise)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from model.consonant_harmony import VOICELESS_CONSONANTS
from model.vowel_harmony import _classify_first_vowel, _classify_last_vowel

IGNORE = -1
WORD_START = "▁"


@dataclass(frozen=True)
class FeatureSpec:
    """Bir etiketleyicinin tanımı."""

    name: str
    classes: tuple
    description: str

    @property
    def num_classes(self) -> int:
        return len(self.classes)


FEATURES: Dict[str, FeatureSpec] = {
    spec.name: spec
    for spec in [
        FeatureSpec("kelime_basi", ("devam", "kelime başı"),
                    "Token kelime başı (▁) mı, kelime devamı mı?"),
        FeatureSpec("morf_sinifi", ("kök", "ek", "özel"),
                    "Kök / ek / özel (noktalama, sayı, özel token)."),
        FeatureSpec("son_unlu", ("kalın", "ince", "yok"),
                    "Kelimenin bu token dahil son ünlüsünün sınıfı."),
        FeatureSpec("sonraki_ek", ("hayır", "evet"),
                    "Sıradaki token aynı kelimeye eklenen bir ek mi? (ileriye dönük)"),
        FeatureSpec("beklenen_uyum", ("kalın", "ince"),
                    "Sıradaki ek için büyük ünlü uyumunun gerektirdiği sınıf."),
        FeatureSpec("sonraki_ek_unlusu", ("kalın", "ince"),
                    "Sıradaki ek parçasının gerçek ilk ünlüsü."),
        FeatureSpec("sert_unsuz", ("hayır", "evet"),
                    "Kelime şimdiye dek sert ünsüzle mi bitiyor? (fıstıkçı şahap)"),
        FeatureSpec("ek_turu", ("diğer", "çoğul", "bulunma", "ayrılma", "geçmiş -DI", "-mIş"),
                    "Ek parçasının türü (sezgisel, yaygın ekler)."),
        FeatureSpec("sonraki_ek_turu", ("diğer", "çoğul", "bulunma", "ayrılma", "geçmiş -DI", "-mIş"),
                    "Sıradaki ek parçasının türü (ileriye dönük, sezgisel)."),
    ]
}

SUFFIX_TYPES = FEATURES["ek_turu"].classes

_DI_ENDINGS = ("", "m", "n", "k", "nız", "niz", "nuz", "nüz", "lar", "ler")
_DI_FORMS = {c + v + e for c in "dt" for v in "ıiuü" for e in _DI_ENDINGS}
_MIS_PREFIXES = tuple("m" + v + "ş" for v in "ıiuü")
_LOC_FORMS = {c + v + e for c in "dt" for v in "ae" for e in ("", "ki")}
_ABL_FORMS = {c + v + "n" + e for c in "dt" for v in "ae" for e in ("", "ki")}


def turkish_lower(text: str) -> str:
    """Türkçe'ye duyarlı küçük harf (I→ı, İ→i)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def is_special_token(token: str, token_id: Optional[int] = None) -> bool:
    """Özel token (<s>, <pad>...), noktalama veya harf içermeyen parça mı?"""
    if token_id is not None and token_id < 4:
        return True
    if token.startswith("<") and token.endswith(">"):
        return True
    return not any(c.isalpha() for c in token)


def is_word_start(token: str) -> bool:
    return token.startswith(WORD_START)


def morph_class(token: str, token_id: Optional[int] = None) -> int:
    """0=kök, 1=ek, 2=özel — ToprakLM.token_morph_classes ile aynı kural."""
    if is_special_token(token, token_id) or token in ("<sep>", "<cls>", "<mask>"):
        return 2
    return 0 if is_word_start(token) else 1


def _clean(token: str) -> str:
    return turkish_lower(token.lstrip(WORD_START).replace("'", "").replace("’", ""))


def suffix_type(piece: str) -> int:
    """
    Ek parçasının türü (SUFFIX_TYPES indeksi). Sezgisel:
    çoğul "lar/ler..." ile başlar; bulunma da/de/ta/te(+ki); ayrılma
    dan/den/tan/ten(+ki); geçmiş zaman dı/di/du/dü/tı/ti/tu/tü (+m/n/k/nız/lar);
    -mIş mış/miş/muş/müş ile başlar. Diğer her ek parçası 0'dır.
    """
    p = _clean(piece)
    if p.startswith(("lar", "ler")):
        return 1
    if p in _LOC_FORMS:
        return 2
    if p in _ABL_FORMS:
        return 3
    if p in _DI_FORMS:
        return 4
    if p.startswith(_MIS_PREFIXES):
        return 5
    return 0


def _ends_voiceless(word: str) -> bool:
    for ch in reversed(word):
        if ch.isalpha():
            return ch in VOICELESS_CONSONANTS
    return False


def label_sentence(tokens: Sequence[str], token_ids: Optional[Sequence[int]] = None) -> Dict[str, List[int]]:
    """
    Bir cümlenin token'larını tüm özellikler için etiketle.

    Args:
        tokens: SentencePiece parçaları (ör. ["<s>", "▁kitap", "lar", "da"])
        token_ids: opsiyonel ID'ler (ID < 4 özel sayılır)

    Returns:
        {özellik_adı: [etiket, ...]} — tanımsız konumlarda -1
    """
    n = len(tokens)
    ids = list(token_ids) if token_ids is not None else [None] * n
    special = [is_special_token(t, i) for t, i in zip(tokens, ids)]
    suffix = [(not special[j]) and not is_word_start(tokens[j]) for j in range(n)]
    out = {name: [IGNORE] * n for name in FEATURES}

    word = ""
    for j, tok in enumerate(tokens):
        out["morf_sinifi"][j] = morph_class(tok, ids[j])
        if special[j]:
            word = ""
            continue
        piece = _clean(tok)
        word = piece if is_word_start(tok) else word + piece
        out["kelime_basi"][j] = int(is_word_start(tok))
        out["son_unlu"][j] = _classify_last_vowel(word)
        out["sert_unsuz"][j] = int(_ends_voiceless(word))
        if suffix[j]:
            out["ek_turu"][j] = suffix_type(tok)
        if j + 1 < n:
            nxt_is_suffix = suffix[j + 1]
            out["sonraki_ek"][j] = int(nxt_is_suffix)
            if nxt_is_suffix:
                out["sonraki_ek_turu"][j] = suffix_type(tokens[j + 1])
                nxt_vowel = _classify_first_vowel(_clean(tokens[j + 1]))
                word_vowel = _classify_last_vowel(word)
                if nxt_vowel != 2:
                    out["sonraki_ek_unlusu"][j] = nxt_vowel
                    if word_vowel != 2:
                        out["beklenen_uyum"][j] = word_vowel
    return out


def compute_features(
    token_strings: Sequence[Sequence[str]],
    token_ids: Optional[Sequence[Sequence[int]]] = None,
    features: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Cümle listesi için etiketleri hesapla ve düzleştir.

    Args:
        token_strings: cümle başına token metinleri (CollectedActivations.token_strings)
        token_ids: cümle başına ID'ler (opsiyonel)
        features: hesaplanacak özellik adları (None = hepsi)

    Returns:
        {özellik: (N,) LongTensor}; satırlar ``collect`` çıktısıyla hizalıdır.
    """
    names = list(FEATURES) if features is None else list(features)
    unknown = [f for f in names if f not in FEATURES]
    if unknown:
        raise ValueError(f"Bilinmeyen özellik: {unknown}")
    flat: Dict[str, List[int]] = {name: [] for name in names}
    for s, toks in enumerate(token_strings):
        labels = label_sentence(toks, token_ids[s] if token_ids is not None else None)
        for name in names:
            flat[name].extend(labels[name])
    return {name: torch.tensor(v, dtype=torch.long) for name, v in flat.items()}
