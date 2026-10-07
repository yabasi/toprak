# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Sohbet Şablonu (Chat Template)

Talimat eğitimi (SFT), tercih hizalaması (DPO), pekiştirmeli öğrenme (GRPO)
ve sohbet arayüzü aynı şablonu kullanır; böylece eğitim ve çıkarım arasında
format kayması olmaz.

Biçim (özel tokenlar tokenizer'da varsa tek token olarak kodlanır):

    <s><|sistem|>…<|son|><|kullanıcı|>…<|son|><|toprak|>…<|son|>

Mevcut 32K tokenizer bu tokenları içermiyorsa geriye uyumlu metin
işaretleyicilerine düşülür ve tur sonu olarak tokenizer'da zaten tek token
olan `<sep>` kullanılır:

    <s>Sistem: …<sep>Kullanıcı: …<sep>Toprak: …<sep>

Kayıp maskesi yalnız asistan içeriği ve onun tur sonu token'ı için 1'dir;
model kullanıcının yazdıklarını taklit etmeyi değil, cevap vermeyi öğrenir.
"""

from typing import Dict, List, Optional, Sequence, Tuple

ROLES = ("system", "user", "assistant")

# Yeni tokenizer eğitirken user_defined_symbols'a eklenecek tokenlar
CHAT_SPECIAL_TOKENS = ["<|sistem|>", "<|kullanıcı|>", "<|toprak|>", "<|son|>"]

_SPECIAL_ROLE_TOKENS = {
    "system": "<|sistem|>",
    "user": "<|kullanıcı|>",
    "assistant": "<|toprak|>",
}
_TEXT_ROLE_MARKERS = {
    "system": "Sistem:",
    "user": "Kullanıcı:",
    "assistant": "Toprak:",
}

IGNORE_INDEX = -100


class ChatTemplate:
    """Mesaj listesini token ID'lerine ve kayıp maskesine dönüştürür."""

    def __init__(self, tokenizer, system_prompt: Optional[str] = None):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        unk = getattr(tokenizer, "unk_token_id", 1)

        special_ids = {tok: tokenizer.token_to_id(tok) for tok in CHAT_SPECIAL_TOKENS}
        self.uses_special_tokens = all(i != unk for i in special_ids.values())

        if self.uses_special_tokens:
            self.role_prefix_ids = {
                role: [special_ids[tok]] for role, tok in _SPECIAL_ROLE_TOKENS.items()
            }
            self.end_id = special_ids["<|son|>"]
        else:
            self.role_prefix_ids = {
                role: self._encode(marker) for role, marker in _TEXT_ROLE_MARKERS.items()
            }
            sep = tokenizer.token_to_id("<sep>")
            self.end_id = sep if sep != unk else tokenizer.eos_token_id

        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    # ── yardımcılar ───────────────────────────────────────

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_bos=False, add_eos=False)

    @property
    def stop_ids(self) -> Tuple[int, ...]:
        """Üretimi durduran token ID'leri (tur sonu ve EOS)."""
        return (self.end_id, self.eos_id)

    def _with_system(self, messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = list(messages)
        if self.system_prompt and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    # ── kodlama ──────────────────────────────────────────

    def encode_with_mask(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """
        Mesajları kodla.

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
            add_generation_prompt: True ise sona asistan rol öneki eklenir
                (model cevabı buradan devam ettirir).

        Returns:
            (input_ids, loss_mask) — aynı uzunlukta; mask 1 = eğitimde hedef
        """
        ids = [self.bos_id]
        mask = [0]
        for message in self._with_system(messages):
            role = message.get("role")
            if role not in ROLES:
                raise ValueError(f"Geçersiz rol: {role!r} (beklenen: {ROLES})")
            prefix = self.role_prefix_ids[role]
            content = self._encode(message.get("content", ""))
            trainable = 1 if role == "assistant" else 0
            ids += prefix
            mask += [0] * len(prefix)
            ids += content + [self.end_id]
            mask += [trainable] * (len(content) + 1)
        if add_generation_prompt:
            prefix = self.role_prefix_ids["assistant"]
            ids += prefix
            mask += [0] * len(prefix)
        return ids, mask

    def encode_prompt(self, messages: Sequence[Dict[str, str]]) -> List[int]:
        """Asistan cevabını üretmek için hazır prompt ID'leri."""
        ids, _ = self.encode_with_mask(messages, add_generation_prompt=True)
        return ids

    def build_labels(
        self,
        messages: Sequence[Dict[str, str]],
        max_len: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Sonraki-token eğitimi için (input_ids, labels) üret.
        labels[t] = input_ids[t+1] (maskeli konumlar IGNORE_INDEX).
        """
        ids, mask = self.encode_with_mask(messages)
        if max_len is not None:
            ids, mask = ids[: max_len + 1], mask[: max_len + 1]
        inputs = ids[:-1]
        labels = [tok if m else IGNORE_INDEX for tok, m in zip(ids[1:], mask[1:])]
        return inputs, labels

    def truncate_history(
        self,
        messages: Sequence[Dict[str, str]],
        max_prompt_tokens: int,
    ) -> List[Dict[str, str]]:
        """
        Prompt bütçesine sığana kadar en eski tur(lar)ı at.
        Sistem mesajı ve son kullanıcı mesajı korunur.
        """
        messages = self._with_system(messages)
        system = [m for m in messages[:1] if m["role"] == "system"]
        rest = messages[len(system):]
        while len(rest) > 1 and len(self.encode_prompt(system + rest)) > max_prompt_tokens:
            rest = rest[1:]
        return system + rest
