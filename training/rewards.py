# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Doğrulanabilir Ödül Fonksiyonları (RLVR / GRPO için)

"Düşünen Toprak" pekiştirmeli öğrenmesinde ödül, bir insan ya da ödül modeli
tarafından değil, kesin kurallarla hesaplanır:

- `extract_answer`   : Türkçe model çıktısından son cevabı çıkarır
                       ("Cevap: 42", "cevap 3/4", "Sonuç: 12,5", "1.250",
                       "-7", "%25", "yüzde 25" …).
- `answers_match`    : Sayı / kesir / yüzde farkındalıklı karşılaştırma.
- `correctness_reward`: Cevap doğruysa 1, değilse 0.
- `format_reward`    : `<düşünce>…</düşünce>` ardından `Cevap: …` biçimi.
- `length_penalty`   : Aşırı uzun çıktılar için 0 … -1 arası ceza.
- `language_reward`  : Harflerin Türkçe/Latin alfabesinden olma oranı
                       (Kiril, CJK, bozuk bayt çöpü vb. cezalandırılır).
- `combine_rewards`  : Ağırlıklı toplam.

Sayı biçimi kuralları (Türkçe öncelikli):
    "12,5"      → 12.5   (virgül ondalık ayırıcıdır)
    "1.250"     → 1250   (noktadan sonra tam 3'lü gruplar → binlik ayırıcı)
    "1.250,75"  → 1250.75
    "3.5"       → 3.5    (3'lü grup değilse nokta ondalık sayılır)
    "1,250,000" → 1250000 (birden çok virgül → İngilizce binlik)
    "3/4", "-3/4", "%25", "25%", "yüzde 25"
"""

import re
from fractions import Fraction
from typing import Callable, Dict, Optional, Tuple, Union

# ── Akıl yürütme biçimi ───────────────────────────────────

THINK_OPEN = "<düşünce>"
THINK_CLOSE = "</düşünce>"
ANSWER_PREFIX = "Cevap:"

REASONING_SYSTEM_PROMPT = (
    "Sen Toprak'sın; soruları adım adım düşünerek çözen bir Türkçe asistansın. "
    "Önce çözüm adımlarını <düşünce> ile </düşünce> etiketleri arasına yaz. "
    "Etiketi kapattıktan sonra son satıra yalnızca 'Cevap: <sonuç>' yaz."
)


def format_reasoning(thought: str, answer: str) -> str:
    """Düşünce metni ve cevabı beklenen çıktı biçimine dönüştür."""
    return f"{THINK_OPEN}\n{thought.strip()}\n{THINK_CLOSE}\n{ANSWER_PREFIX} {answer}"


# ── Sayı ayrıştırma ───────────────────────────────────────

_MINUS = "-−–"
_NUM_CORE = r"\d+(?:[.,]\d+)*"
_NUMBER_RE = re.compile(
    r"(?P<pct1>%\s*|yüzde\s+)?"
    r"(?P<num>(?:(?<![\w)\]])[-−–]\s?)?" + _NUM_CORE
    + r"(?:\s*/\s*" + _NUM_CORE + r")?)"
    r"(?P<pct2>\s*%)?",
    re.IGNORECASE,
)
_MARKER_RE = re.compile(r"(?:cevap|sonuç|yanıt)(?:ım|ımız)?\s*[:：=]?", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _parse_plain(token: str) -> Optional[Fraction]:
    """İşaretsiz/işaretli tek sayı ("1.250,5", "12,5", "3.5") → Fraction."""
    token = token.strip()
    sign = 1
    if token and token[0] in _MINUS:
        sign = -1
        token = token[1:].strip()
    if not token or not token[0].isdigit():
        return None
    if not re.fullmatch(_NUM_CORE, token):
        return None

    has_dot, has_comma = "." in token, "," in token
    if has_dot and has_comma:
        if token.rfind(",") > token.rfind("."):
            integer, decimal = token.replace(".", "").split(",", 1)  # TR: 1.250,75
        else:
            integer, decimal = token.replace(",", "").rsplit(".", 1)  # EN: 1,250.75
        if "," in decimal or "." in decimal:
            return None
    elif has_comma:
        parts = token.split(",")
        if len(parts) == 2:
            integer, decimal = parts  # TR ondalık
        elif all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3:
            integer, decimal = "".join(parts), ""  # EN binlik
        else:
            return None
    elif has_dot:
        parts = token.split(".")
        if all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3 and parts[0] != "0":
            integer, decimal = "".join(parts), ""  # TR binlik
        elif len(parts) == 2:
            integer, decimal = parts  # "3.5"
        else:
            return None
    else:
        integer, decimal = token, ""

    value = Fraction(int(integer))
    if decimal:
        value += Fraction(int(decimal), 10 ** len(decimal))
    return sign * value


def parse_number(text: Union[str, int, float, Fraction]) -> Optional[Tuple[Fraction, bool]]:
    """
    Tek bir sayı ifadesini çözümle.

    Returns:
        (değer, yüzde_mi) ya da çözümlenemezse None.
        "%25" → (25, True); "3/4" → (3/4, False); "1.250" → (1250, False)
    """
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, Fraction)):
        return Fraction(text), False
    if isinstance(text, float):
        return Fraction(text).limit_denominator(10 ** 9), False
    if not isinstance(text, str):
        return None

    s = text.strip().rstrip(".")
    percent = False
    low = s.lower()
    if low.startswith("yüzde"):
        percent, s = True, s[5:].strip()
    if s.startswith("%"):
        percent, s = True, s[1:].strip()
    if s.endswith("%"):
        percent, s = True, s[:-1].strip()
    if not s:
        return None

    if "/" in s:
        num, _, den = s.partition("/")
        n, d = _parse_plain(num), _parse_plain(den)
        if n is None or d is None or d == 0:
            return None
        return n / d, percent
    value = _parse_plain(s)
    if value is None:
        return None
    return value, percent


def _match_to_string(match: "re.Match") -> str:
    num = re.sub(r"\s+", "", match.group("num")).replace("−", "-").replace("–", "-")
    if match.group("pct1") or match.group("pct2"):
        return "%" + num
    return num


def _numbers_in(text: str):
    return [m for m in _NUMBER_RE.finditer(text)]


def extract_answer(text: str) -> Optional[str]:
    """
    Model çıktısından son cevabı bir dize olarak çıkar.

    Öncelik sırası:
      1. Son "Cevap / Sonuç / Yanıt" işaretinden sonraki ilk sayı
      2. Son \\boxed{…} içeriği
      3. Düşünce bloğundan sonraki (yoksa tüm metindeki) son sayı

    Dönen dize normalleştirilmiştir: boşluksuz, eksi işareti "-", yüzdeler
    "%" önekli (ör. "%25"). Sayı bulunamazsa None.
    """
    if not text:
        return None

    markers = list(_MARKER_RE.finditer(text))
    for marker in reversed(markers):
        tail = text[marker.end():marker.end() + 80]
        match = _NUMBER_RE.search(tail)
        if match:
            return _match_to_string(match)

    boxed = _BOXED_RE.findall(text)
    if boxed:
        matches = _numbers_in(boxed[-1])
        if matches:
            return _match_to_string(matches[0])

    tail_text = text.split(THINK_CLOSE)[-1] if THINK_CLOSE in text else text
    matches = _numbers_in(tail_text) or _numbers_in(text)
    if matches:
        return _match_to_string(matches[-1])
    return None


def answers_match(
    pred: Union[str, int, float, Fraction, None],
    gold: Union[str, int, float, Fraction],
    tol: float = 1e-6,
) -> bool:
    """
    Tahmin ile altın cevabı sayısal olarak karşılaştır.

    - Kesir/ondalık eşdeğerliği: "0,75" == "3/4" == "6/8"
    - Yüzde: altın "%25" ise "25", "%25" ve "0,25" kabul edilir
    - `tol`: göreli tolerans (|p-g| <= tol * max(1, |g|))
    `pred` doğrudan sayı olarak çözülemezse önce `extract_answer` uygulanır.
    """
    if pred is None:
        return False
    parsed_gold = parse_number(gold)
    if parsed_gold is None:
        return isinstance(pred, str) and pred.strip() == str(gold).strip()
    parsed_pred = parse_number(pred)
    if parsed_pred is None and isinstance(pred, str):
        extracted = extract_answer(pred)
        parsed_pred = parse_number(extracted) if extracted else None
    if parsed_pred is None:
        return False

    p, p_pct = parsed_pred
    g, g_pct = parsed_gold
    candidates = [p]
    if g_pct and not p_pct:
        candidates.append(p * 100)
    if p_pct and not g_pct:
        candidates.append(p / 100)
    bound = Fraction(tol).limit_denominator(10 ** 12) * max(1, abs(g))
    return any(abs(c - g) <= bound for c in candidates)


# ── Ödül fonksiyonları ────────────────────────────────────

def correctness_reward(completion: str, gold, tol: float = 1e-6, **_) -> float:
    """Çıkarılan cevap altın cevapla eşleşiyorsa 1.0, değilse 0.0."""
    pred = extract_answer(completion)
    return 1.0 if pred is not None and answers_match(pred, gold, tol=tol) else 0.0


_STRICT_FORMAT_RE = re.compile(
    r"^\s*" + re.escape(THINK_OPEN) + r"(?P<thought>.+?)" + re.escape(THINK_CLOSE)
    + r"\s*Cevap\s*:\s*(?P<answer>[^\n]+?)\s*$",
    re.DOTALL,
)


def format_reward(completion: str, **_) -> float:
    """
    Biçim ödülü:
      1.0  — tam biçim: `<düşünce>…</düşünce>` (tek blok, boş değil) + son satır `Cevap: …`
      0.5  — iki etiket ve "Cevap:" var ama sıra/fazlalık hatalı
      0.25 — yalnız "Cevap:" satırı var
      0.0  — hiçbiri
    """
    text = completion or ""
    match = _STRICT_FORMAT_RE.match(text)
    if (
        match
        and match.group("thought").strip()
        and text.count(THINK_OPEN) == 1
        and text.count(THINK_CLOSE) == 1
    ):
        return 1.0
    has_answer = re.search(r"Cevap\s*:", text) is not None
    if THINK_OPEN in text and THINK_CLOSE in text and has_answer:
        return 0.5
    if has_answer:
        return 0.25
    return 0.0


def length_penalty(
    completion: Union[str, int],
    soft_limit: int = 800,
    hard_limit: int = 1600,
    **_,
) -> float:
    """
    Uzunluk cezası (0 … -1). `completion` metin (karakter sayılır) ya da
    doğrudan uzunluk (ör. token sayısı) olabilir. soft_limit'e kadar 0,
    hard_limit'te -1'e lineer iner.
    """
    length = completion if isinstance(completion, int) else len(completion or "")
    if length <= soft_limit:
        return 0.0
    if length >= hard_limit:
        return -1.0
    return -(length - soft_limit) / max(hard_limit - soft_limit, 1)


TURKISH_LETTERS = frozenset(
    "abcçdefgğhıijklmnoöprsştuüvyz" "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
)
_TOLERATED_LETTERS = frozenset("qwxQWX")  # denklemlerde x, alıntı kelimeler


def language_reward(completion: str, allow_qwx: bool = True, min_letters: int = 1, **_) -> float:
    """
    Harflerin (str.isalpha) Türkçe alfabeden olma oranı (0 … 1).

    Kiril/Yunan/CJK harfler, aksanlı Latin harfler (é, ß …) ve bozuk çıktılar
    oranı düşürür. `allow_qwx` True ise q/w/x (ör. denklemlerdeki x) geçerli
    sayılır. Harf sayısı `min_letters`'tan azsa 0 döner.
    """
    letters = [c for c in (completion or "") if c.isalpha()]
    if len(letters) < max(min_letters, 1):
        return 0.0
    valid = TURKISH_LETTERS | _TOLERATED_LETTERS if allow_qwx else TURKISH_LETTERS
    return sum(1 for c in letters if c in valid) / len(letters)


REWARD_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "correctness": correctness_reward,
    "format": format_reward,
    "length": length_penalty,
    "language": language_reward,
}

DEFAULT_WEIGHTS = {"correctness": 1.0, "format": 0.2, "language": 0.1, "length": 0.1}


class CombinedReward:
    """Ağırlıklı ödül toplamı. `reward(completion, gold)` → float."""

    def __init__(self, weights: Dict[str, float], functions: Optional[Dict[str, Callable]] = None):
        self.functions = dict(REWARD_FUNCTIONS)
        if functions:
            self.functions.update(functions)
        unknown = set(weights) - set(self.functions)
        if unknown:
            raise ValueError(f"Bilinmeyen ödül bileşeni: {sorted(unknown)}")
        self.weights = dict(weights)

    def detailed(self, completion: str, gold=None, **kwargs) -> Dict[str, float]:
        """Bileşen bazında ham ödüller ve ağırlıklı "total"."""
        parts = {}
        total = 0.0
        for name, weight in self.weights.items():
            value = float(self.functions[name](completion, gold=gold, **kwargs))
            parts[name] = value
            total += weight * value
        parts["total"] = total
        return parts

    def __call__(self, completion: str, gold=None, **kwargs) -> float:
        return self.detailed(completion, gold, **kwargs)["total"]


def combine_rewards(
    weights: Optional[Dict[str, float]] = None,
    functions: Optional[Dict[str, Callable]] = None,
) -> CombinedReward:
    """
    Ödülleri ağırlıklarla birleştir.

    Örnek:
        reward = combine_rewards({"correctness": 1.0, "format": 0.2})
        reward("<düşünce>2+2=4</düşünce>\\nCevap: 4", gold="4")  # → 1.2
    Ek bileşenler `functions` ile (ad → fn(completion, gold=…)) eklenebilir.
    """
    return CombinedReward(weights if weights is not None else DEFAULT_WEIGHTS, functions)
