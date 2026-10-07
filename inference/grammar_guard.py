# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Uyum Korumalı Kod Çözme (Grammar Guard)

Eğitim maliyeti SIFIR olan bir logit işlemcisi: üretim sırasında, mevcut
kelimenin son ünlüsüne ve son ünsüzüne bakarak Türkçe ek uyumunu bozan
kelime-devamı tokenlerini maskeler (-inf) veya cezalandırır.

Kurallar:
  1. Büyük ünlü uyumu (2 yönlü): kelimenin son ünlüsü kalınsa (a,ı,o,u) ilk
     ünlüsü ince olan ek tokenleri ("ler", "de"), inceyse kalın olanlar
     ("lar", "da") ihlaldir.
  2. Ünsüz benzeşmesi: kelime sert ünsüzle bitiyorsa (f,s,t,k,ç,ş,p) "d"/"c"
     + ünlü ile başlayan ek tokenleri ihlaldir ("kitap"+"da" → "kitapta").
     "h" varsayılan olarak hariçtir (tahdit, istihdam, mahdut gibi kökler).

Muafiyetler:
  - değişmez ekler (yor, ken, ki, leyin, gil): asla maskelenmez,
  - kökü alıntı istisna listesinde olan kelimeler (saat, kalp, hal ...):
    bu kelimelerde HİÇBİR uyum maskesi uygulanmaz (saat+lar da saat+ler de
    serbest). "hal"+"ı" = "halı" gibi gerçek kelimeler bozulmasın diye
    inceye çevirmek yerine muafiyet seçildi,
  - ünlüsüz tokenler, harf dışı tokenler, kelime başı (▁) tokenleri,
  - `restrict_to_suffixes=True` (varsayılan) iken yalnız bilinen ek
    allomorflarına tam bölünebilen tokenler denetlenir; "▁ki"+"tap" gibi kök
    içi bölünmeler yanlışlıkla maskelenmez,
  - ASLA her şey maskelenmez: en olası `fallback_top_k` adayın hepsi
    maskelenecekse adım değiştirilmeden geçilir.

Verimlilik: (son ünlü sınıfı × sert ünsüz) için 4 boolean maske önceden
hesaplanır; her adımda yalnız kelime geri yürüyüşü + tek bir masked_fill.

Kullanım:
    guard = build_grammar_guard(tokenizer, mode="mask")
    text = generate_text(model, tokenizer, prompt, logits_processors=[guard])
    print(guard.stats)
"""

import os
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.archiphoneme import (
    BACK_VOWELS,
    FRONT_VOWELS,
    DEFAULT_HARMONY_EXCEPTIONS,
    SURFACE_TO_ABSTRACT,
    tr_lower,
)


DEFAULT_INVARIANT_PREFIXES = ("yor", "ken", "leyin", "gil")
DEFAULT_INVARIANT_PIECES = {"ki", "kiler", "kini", "kinin", "kinde", "kine", "kinden"}
GUARD_VOICELESS = set("fstkçşp")   # "h" hariç (tahdit, istihdam)
_APOSTROPHE_PIECES = {"'", "’"}


@dataclass
class GuardStats:
    """Grammar guard sayaçları."""
    steps: int = 0            # çağrı sayısı
    applied: int = 0          # maske/ceza uygulanan adımlar
    top1_changed: int = 0     # en olası token ihlal ediyordu (müdahale)
    fallbacks: int = 0        # her şey maskelenecekti → no-op
    exempt: int = 0           # istisna kelime nedeniyle atlanan adımlar

    def as_dict(self) -> dict:
        return asdict(self)

    def reset(self) -> None:
        self.steps = self.applied = self.top1_changed = self.fallbacks = self.exempt = 0


def _first_vowel_class(text: str) -> int:
    for ch in text:
        if ch in BACK_VOWELS:
            return 0
        if ch in FRONT_VOWELS:
            return 1
    return 2


def _last_vowel_class(text: str) -> int:
    for ch in reversed(text):
        if ch in BACK_VOWELS:
            return 0
        if ch in FRONT_VOWELS:
            return 1
    return 2


def _is_special(piece: str, token_id: int) -> bool:
    return token_id < 4 or (piece.startswith("<") and piece.endswith(">"))


class HarmonyGuard:
    """
    Ünlü uyumu / ünsüz benzeşmesi kısıtlı kod çözme işlemcisi.

    Çağrı imzası: guard(generated_ids: List[int], logits: Tensor(1, V)) -> Tensor
    """

    def __init__(
        self,
        tokenizer,
        mode: str = "mask",
        penalty: float = 5.0,
        exceptions: Optional[Iterable[str]] = None,
        invariant_prefixes: Iterable[str] = DEFAULT_INVARIANT_PREFIXES,
        invariant_pieces: Iterable[str] = DEFAULT_INVARIANT_PIECES,
        voiceless: Optional[Iterable[str]] = None,
        restrict_to_suffixes: bool = True,
        consonant_rule: bool = True,
        fallback_top_k: int = 50,
        max_word_pieces: int = 32,
    ):
        """
        Args:
            tokenizer: get_vocab_size() ve id_to_token() sağlayan tokenizer.
            mode: "mask" (-inf) veya "penalty" (logit - penalty).
            penalty: "penalty" modunda çıkarılacak değer.
            exceptions: Muaf alıntı kökler (varsayılan DEFAULT_HARMONY_EXCEPTIONS).
            invariant_prefixes / invariant_pieces: değişmez ek parçaları.
            voiceless: Ünsüz kuralı için sert ünsüzler (varsayılan "fstkçşp").
            restrict_to_suffixes: True ise yalnız ek allomorflarına bölünebilen
                tokenler denetlenir (kök içi bölünmeleri korur).
            consonant_rule: d/c → t/ç kuralı açık mı.
            fallback_top_k: Bu kadar en olası adayın hepsi maskelenirse no-op.
            max_word_pieces: Kelime başını ararken geriye en fazla kaç token.
        """
        if mode not in ("mask", "penalty"):
            raise ValueError("mode 'mask' veya 'penalty' olmalı")
        self.mode = mode
        self.penalty = float(penalty)
        self.exceptions: Set[str] = {
            tr_lower(w) for w in (DEFAULT_HARMONY_EXCEPTIONS if exceptions is None else exceptions)
        }
        self.voiceless = set(GUARD_VOICELESS if voiceless is None else voiceless)
        self.consonant_rule = consonant_rule
        self.fallback_top_k = fallback_top_k
        self.max_word_pieces = max_word_pieces
        self.stats = GuardStats()

        inv_prefixes = tuple(invariant_prefixes)
        inv_pieces = set(invariant_pieces)
        surfaces = set(SURFACE_TO_ABSTRACT)
        max_surf = max(len(s) for s in surfaces)

        def splits_into_suffixes(text: str) -> bool:
            ok = [False] * (len(text) + 1)
            ok[0] = True
            for i in range(len(text)):
                if not ok[i]:
                    continue
                for ln in range(1, min(max_surf, len(text) - i) + 1):
                    if text[i:i + ln] in surfaces:
                        ok[i + ln] = True
            return ok[len(text)]

        V = tokenizer.get_vocab_size()
        self.vocab_size = V
        self.pieces: List[str] = []        # küçük harf, ▁ temizlenmiş
        self.word_start: List[bool] = []
        self.boundary: List[bool] = []     # harf dışı / özel token: kelimeyi keser
        first_cls = torch.full((V,), 2, dtype=torch.long)
        dc_start = torch.zeros(V, dtype=torch.bool)
        candidate = torch.zeros(V, dtype=torch.bool)

        for tid in range(V):
            raw = tokenizer.id_to_token(tid)
            special = _is_special(raw, tid)
            ws = special or raw.startswith("▁")
            clean = tr_lower(raw.lstrip("▁"))
            self.pieces.append("" if special else clean)
            self.word_start.append(ws)
            is_letters = bool(clean) and clean.isalpha()
            self.boundary.append(special or (not is_letters and clean not in _APOSTROPHE_PIECES))
            if ws or not is_letters:
                continue
            fc = _first_vowel_class(clean)
            if fc == 2:
                continue
            if clean in inv_pieces or clean.startswith(inv_prefixes):
                continue
            if restrict_to_suffixes and not splits_into_suffixes(clean):
                continue
            candidate[tid] = True
            first_cls[tid] = fc
            if len(clean) >= 2 and clean[0] in "dc" and clean[1] in (BACK_VOWELS | FRONT_VOWELS):
                dc_start[tid] = True

        # Ünlü ihlali: son ünlü kalın (0) → ince ilk ünlülü adaylar; tersi
        viol_vowel = [candidate & (first_cls == 1), candidate & (first_cls == 0)]
        viol_cons = candidate & dc_start
        # masks[(sınıf, sert_mi)] ; sınıf 2 = ünlü yok
        self._masks = {}
        for cls in (0, 1, 2):
            for hard in (False, True):
                m = viol_vowel[cls] if cls < 2 else torch.zeros(V, dtype=torch.bool)
                if hard and consonant_rule:
                    m = m | viol_cons
                self._masks[(cls, hard)] = m if bool(m.any()) else None
        self._device_masks = {}

    # ── Kelime bağlamı ──────────────────────────────────────

    def current_word(self, generated_ids: List[int]) -> str:
        """Üretilen dizinin sonundaki (henüz bitmemiş) kelimenin küçük harfli metni."""
        parts = []
        n = len(generated_ids)
        for k in range(n - 1, max(-1, n - 1 - self.max_word_pieces), -1):
            tid = generated_ids[k]
            if tid < 0 or tid >= self.vocab_size or self.boundary[tid]:
                break
            parts.append(self.pieces[tid])
            if self.word_start[tid]:
                break
        return "".join(reversed(parts))

    def _mask_for(self, word: str, V: int, device):
        stem = word.replace("’", "'").split("'")[0]
        if stem in self.exceptions or word in self.exceptions:
            return None, True
        letters = [c for c in word if c.isalpha()]
        if not letters:
            return None, False
        cls = _last_vowel_class(word)
        hard = letters[-1] in self.voiceless
        key = (cls, hard, V, str(device))
        if key not in self._device_masks:
            m = self._masks[(cls, hard)]
            if m is not None:
                if V > m.numel():
                    m = torch.cat([m, torch.zeros(V - m.numel(), dtype=torch.bool)])
                m = m[:V].to(device)
            self._device_masks[key] = m
        return self._device_masks[key], False

    # ── Çağrı ───────────────────────────────────────────────

    def __call__(self, generated_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        self.stats.steps += 1
        if not generated_ids:
            return logits
        word = self.current_word(generated_ids)
        if not word:
            return logits
        mask, exempt = self._mask_for(word, logits.size(-1), logits.device)
        if exempt:
            self.stats.exempt += 1
            return logits
        if mask is None:
            return logits

        row = logits[0]
        k = min(self.fallback_top_k, row.numel())
        top_idx = torch.topk(row, k).indices
        top_masked = mask[top_idx]
        finite_top = torch.isfinite(row[top_idx])
        if not bool((~top_masked & finite_top).any()):
            self.stats.fallbacks += 1
            return logits

        self.stats.applied += 1
        if bool(top_masked[0]):
            self.stats.top1_changed += 1
        if self.mode == "mask":
            return logits.masked_fill(mask.unsqueeze(0), float("-inf"))
        return logits - mask.unsqueeze(0).to(logits.dtype) * self.penalty


def build_grammar_guard(tokenizer, **kw) -> HarmonyGuard:
    """HarmonyGuard fabrikası (ör. build_grammar_guard(tok, mode="penalty", penalty=3.0))."""
    return HarmonyGuard(tokenizer, **kw)
