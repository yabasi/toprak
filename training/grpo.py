# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — GRPO (Group Relative Policy Optimization) ile "Düşünen Toprak"

Doğrulanabilir ödüllü pekiştirmeli öğrenme (RLVR). Her prompt için G tamamlama
örneklenir, kurallı ödüller (training/rewards.py) hesaplanır ve grup içinde
normalize edilen avantajlarla politika güncellenir. Ayrı bir değer (critic)
ağı gerekmez; taban çizgisi grubun ortalama ödülüdür.

Kayıp (token düzeyinde, yalnız tamamlama tokenları üzerinden ortalama):

    A_i      = (r_i − mean(r)) / (std(r) + eps)          (std = 0 → A = 0)
    ρ_t      = exp(log π_θ(o_t) − log π_old(o_t))
    L_pg     = −min(ρ_t A_i, clip(ρ_t, 1−ε, 1+ε_high) A_i)
    KL_t     = exp(ref − pol) − (ref − pol) − 1          (k3 tahmincisi, ≥ 0)
    L        = Σ_t mask_t (L_pg + β KL_t) / Σ_t mask_t

π_old örnekleme anındaki (güncelleme öncesi) politikadır; `num_iterations`
> 1 ise aynı rollout'lar üzerinde birden çok güncelleme yapılır ve oran
kırpması devreye girer. Referans model (β > 0 ise) başlangıç politikasının
dondurulmuş kopyasıdır.

Kullanım:
    python training/grpo.py --checkpoint checkpoints/toprak_sft.pt \\
        --data data/reasoning/train.jsonl --output checkpoints/grpo
"""

import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from training.scheduler import CosineWarmupScheduler

# ══════════════════════════════════════════════════════════
#  Yapılandırma
# ══════════════════════════════════════════════════════════


@dataclass
class GRPOConfig:
    """GRPO hiperparametreleri."""

    group_size: int = 8              # prompt başına tamamlama (G)
    prompts_per_step: int = 4        # adım başına prompt sayısı
    steps: int = 500                 # toplam GRPO adımı (rollout turu)
    num_iterations: int = 1          # aynı rollout'lar üzerinde güncelleme sayısı (μ)
    lr: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 10
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    beta_kl: float = 0.04            # KL cezası katsayısı (0 → referans model yok)
    clip_eps: float = 0.2            # alt kırpma (1 − ε)
    clip_eps_high: Optional[float] = None  # üst kırpma (None → clip_eps; DAPO: 0.28)
    adv_eps: float = 1e-4            # avantaj normalizasyonu paydası
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0                   # 0 → kapalı
    mask_truncated: bool = False     # durmadan kesilen tamamlamaları kayıptan çıkar
    seed: int = 42
    save_every: int = 100
    log_every: int = 1


# ══════════════════════════════════════════════════════════
#  Saf fonksiyonlar (birim testli)
# ══════════════════════════════════════════════════════════


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Grup içi normalize avantaj: (r − mean) / (std + eps).

    Args:
        rewards: (G,) ya da (N, G) — son boyut grup
    Grup içindeki tüm ödüller eşitse (std ≈ 0) avantaj tam 0 olur; böyle
    gruplar politika gradyanı üretmez. std popülasyon std'sidir (G=1'de 0).
    """
    rewards = rewards.float()
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    adv = (rewards - mean) / (std + eps)
    return torch.where(std < 1e-8, torch.zeros_like(adv), adv)


def k3_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """
    Token düzeyinde KL(π_θ ‖ π_ref) için k3 tahmincisi (Schulman):
        exp(ref − pol) − (ref − pol) − 1   (her zaman ≥ 0; eşitse 0)
    """
    diff = ref_logp - logp
    return torch.exp(diff) - diff - 1


def clipped_policy_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    clip_eps_high: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PPO tarzı kırpılmış vekil kayıp (token başına).

    Args:
        logp, old_logp: (G, L)
        advantages: (G,) ya da (G, 1) — tamamlama başına avantaj
    Returns:
        (kayıp (G, L), kırpılan token göstergesi (G, L) bool)
    """
    if advantages.dim() == logp.dim() - 1:
        advantages = advantages.unsqueeze(-1)
    high = clip_eps if clip_eps_high is None else clip_eps_high
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + high)
    surr1 = ratio * advantages
    surr2 = clipped * advantages
    loss = -torch.min(surr1, surr2)
    was_clipped = surr2 < surr1
    return loss, was_clipped


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: Optional[torch.Tensor],
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.0,
    clip_eps_high: Optional[float] = None,
    normalizer: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GRPO kaybı: kırpılmış politika kaybı + β·k3-KL, yalnız mask=1
    (tamamlama) tokenları üzerinden token düzeyinde ortalama.

    `normalizer` verilirse toplam bu sayıya bölünür (birden çok grubun
    gradyanı biriktirilirken toplam tamamlama token sayısı).
    """
    pg, was_clipped = clipped_policy_loss(logp, old_logp, advantages, clip_eps, clip_eps_high)
    per_token = pg
    kl = None
    if ref_logp is not None and beta > 0:
        kl = k3_kl(logp, ref_logp)
        per_token = per_token + beta * kl
    maskf = mask.to(per_token.dtype)
    denom = float(normalizer) if normalizer is not None else maskf.sum().clamp(min=1.0)
    loss = (per_token * maskf).sum() / denom
    with torch.no_grad():
        stats = {
            "pg_loss": masked_mean(pg, maskf).item(),
            "clip_frac": masked_mean(was_clipped.float(), maskf).item(),
            "kl": masked_mean(kl, maskf).item() if kl is not None else 0.0,
        }
    return loss, stats


def completion_logprobs(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Tam diziler (prompt + tamamlama) için tamamlama tokenlarının log-olasılıkları.

    Args:
        input_ids: (G, P + L) — prompt ortak ve uzunluğu `prompt_len`
    Returns:
        (G, L) — log π(o_t | prompt, o_<t); örneklemeyle tutarlı olsun diye
        logitler `temperature` ile ölçeklenir.
    """
    logits, _, _ = model(input_ids[:, :-1])
    logits = logits[:, prompt_len - 1:, :].float() / max(temperature, 1e-6)
    targets = input_ids[:, prompt_len:]
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def sample_completions(
    model,
    prompt_ids: Sequence[int],
    num_samples: int,
    max_new_tokens: int,
    stop_ids: Sequence[int],
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: int = 0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], List[bool]]:
    """
    KV cache'li paralel sıcaklık örneklemesi (aynı prompt'tan G tamamlama).

    Returns:
        completions: (G, L) — durdurma tokenı dahil, sonrası pad
        mask: (G, L) — 1 = tamamlama tokenı (durdurma tokenı dahil)
        texts_ids: durdurma tokenı hariç tamamlama ID listeleri
        truncated: durdurma tokenı görülmeden max_new_tokens'a ulaşanlar
    """
    device = next(model.parameters()).device
    max_pos = model.freqs_cis.size(0)
    max_new_tokens = min(max_new_tokens, max_pos - len(prompt_ids))
    if max_new_tokens <= 0:
        raise ValueError(f"Prompt ({len(prompt_ids)} token) bağlam sınırını ({max_pos}) dolduruyor.")
    stop = torch.tensor(sorted(set(stop_ids)), device=device)

    x = torch.tensor([list(prompt_ids)] * num_samples, dtype=torch.long, device=device)
    logits, _, past = model(x)
    next_logits = logits[:, -1, :]
    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)
    tokens, masks = [], []
    for _ in range(max_new_tokens):
        scaled = next_logits.float() / max(temperature, 1e-6)
        if top_k and top_k > 0:
            kth = torch.topk(scaled, min(top_k, scaled.size(-1))).values[:, -1:]
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))
        probs = torch.softmax(scaled, dim=-1)
        if generator is not None and generator.device == probs.device:
            tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        else:
            tok = torch.multinomial(probs, 1).squeeze(-1)
        active = ~finished
        tok = torch.where(active, tok, torch.full_like(tok, pad_id))
        tokens.append(tok)
        masks.append(active)
        finished = finished | (active & torch.isin(tok, stop))
        if bool(finished.all()):
            break
        logits, _, past = model(tok.unsqueeze(-1), past_kvs=past)
        next_logits = logits[:, -1, :]

    completions = torch.stack(tokens, dim=1)
    mask = torch.stack(masks, dim=1)
    stop_set = set(int(s) for s in stop_ids)
    ids_list, truncated = [], []
    for row, row_mask in zip(completions.tolist(), mask.tolist()):
        ids = [t for t, m in zip(row, row_mask) if m]
        if ids and ids[-1] in stop_set:
            ids_list.append(ids[:-1])
            truncated.append(False)
        else:
            ids_list.append(ids)
            truncated.append(True)
    return completions, mask, ids_list, truncated


# ══════════════════════════════════════════════════════════
#  Eğitici
# ══════════════════════════════════════════════════════════

# reward_fn(tamamlama_metni, örnek, tamamlama_id'leri) → float
RewardFn = Callable[[str, Dict, List[int]], float]


class GRPOTrainer:
    """
    GRPO eğiticisi.

    Args:
        model: Eğitilecek politika (ToprakLM)
        config: GRPOConfig
        reward_fn: (metin, örnek, id'ler) → ödül
        ref_model: Referans model (None ve beta_kl > 0 ise politikanın kopyası)
        tokenizer: `decode` için (yoksa reward_fn'e boş metin gider)
        stop_ids: Üretimi durduran tokenlar (ör. ChatTemplate.stop_ids)
        correct_fn: (metin, örnek, id'ler) → bool; "accuracy" metriği için
    Veri örnekleri en az "prompt_ids" (List[int]) içermelidir.
    """

    def __init__(
        self,
        model,
        config: GRPOConfig,
        reward_fn: RewardFn,
        ref_model=None,
        tokenizer=None,
        stop_ids: Optional[Sequence[int]] = None,
        pad_id: Optional[int] = None,
        correct_fn: Optional[Callable[[str, Dict, List[int]], bool]] = None,
    ):
        self.model = model
        self.config = config
        self.reward_fn = reward_fn
        self.correct_fn = correct_fn
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        mcfg = getattr(model, "config", None)
        self.pad_id = pad_id if pad_id is not None else getattr(mcfg, "pad_token_id", 0)
        self.stop_ids = tuple(stop_ids) if stop_ids else (getattr(mcfg, "eos_token_id", 3),)

        if ref_model is None and config.beta_kl > 0:
            ref_model = copy.deepcopy(model)
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr, betas=(0.9, 0.99), weight_decay=config.weight_decay,
        )
        total_updates = max(config.steps * config.num_iterations, 1)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=min(config.warmup_steps, total_updates),
            max_steps=total_updates,
            max_lr=config.lr,
            min_lr=config.lr * config.min_lr_ratio,
        )
        self.global_step = 0
        self.generator = None
        try:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(config.seed)
        except (RuntimeError, TypeError):
            torch.manual_seed(config.seed)

    # ── yardımcılar ───────────────────────────────────────

    def _decode(self, ids: List[int]) -> str:
        if self.tokenizer is None or not hasattr(self.tokenizer, "decode"):
            return ""
        return self.tokenizer.decode(ids)

    def _score(self, text: str, item: Dict, ids: List[int]) -> float:
        out = self.reward_fn(text, item, ids)
        return float(out)

    # ── rollout ──────────────────────────────────────────

    @torch.no_grad()
    def rollout(self, item: Dict) -> Dict:
        """Tek prompt için G tamamlama, ödül, avantaj ve eski/ref log-olasılıklar."""
        cfg = self.config
        prompt = list(item["prompt_ids"])
        self.model.eval()
        completions, mask, ids_list, truncated = sample_completions(
            self.model, prompt, cfg.group_size, cfg.max_new_tokens, self.stop_ids,
            pad_id=self.pad_id, temperature=cfg.temperature, top_k=cfg.top_k,
            generator=self.generator,
        )
        texts = [self._decode(ids) for ids in ids_list]
        rewards = torch.tensor(
            [self._score(t, item, ids) for t, ids in zip(texts, ids_list)],
            dtype=torch.float32,
        )
        correct = None
        if self.correct_fn is not None:
            correct = [bool(self.correct_fn(t, item, ids)) for t, ids in zip(texts, ids_list)]
        if cfg.mask_truncated:
            keep = torch.tensor([not t for t in truncated], device=mask.device)
            mask = mask & keep.unsqueeze(-1)

        prompt_t = torch.tensor([prompt] * cfg.group_size, dtype=torch.long, device=self.device)
        full = torch.cat([prompt_t, completions], dim=1)
        old_logp = completion_logprobs(self.model, full, len(prompt), cfg.temperature)
        ref_logp = None
        if self.ref_model is not None and cfg.beta_kl > 0:
            ref_logp = completion_logprobs(self.ref_model, full, len(prompt), cfg.temperature)
        return {
            "input_ids": full,
            "prompt_len": len(prompt),
            "mask": mask,
            "old_logp": old_logp,
            "ref_logp": ref_logp,
            "rewards": rewards,
            "advantages": group_advantages(rewards, cfg.adv_eps).to(self.device),
            "lengths": [len(ids) for ids in ids_list],
            "truncated": truncated,
            "correct": correct,
            "texts": texts,
        }

    # ── güncelleme ──────────────────────────────────────

    def step(self, batch: Sequence[Dict]) -> Dict[str, float]:
        """
        Bir GRPO adımı: rollout + `num_iterations` politika güncellemesi.

        Returns:
            metrikler: reward_mean, reward_std, accuracy, kl, completion_length,
            clip_frac, pg_loss, loss, zero_std_frac, truncated_frac, lr
        """
        cfg = self.config
        rollouts = [self.rollout(item) for item in batch]
        total_tokens = float(sum(r["mask"].sum().item() for r in rollouts)) or 1.0

        stats_acc: Dict[str, float] = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "clip_frac": 0.0}
        n_stat = 0
        self.model.eval()  # dropout kapalı: oran π_θ/π_old örneklemeyle tutarlı
        for _ in range(cfg.num_iterations):
            self.optimizer.zero_grad(set_to_none=True)
            for r in rollouts:
                if r["mask"].sum() == 0:
                    continue
                if (r["advantages"] == 0).all() and r["ref_logp"] is None:
                    continue  # gradyan yok
                logp = completion_logprobs(self.model, r["input_ids"], r["prompt_len"], cfg.temperature)
                loss, st = grpo_loss(
                    logp, r["old_logp"], r["ref_logp"], r["advantages"], r["mask"],
                    clip_eps=cfg.clip_eps, beta=cfg.beta_kl,
                    clip_eps_high=cfg.clip_eps_high, normalizer=total_tokens,
                )
                loss.backward()
                weight = r["mask"].sum().item() / total_tokens
                stats_acc["loss"] += loss.item()
                for key in ("pg_loss", "kl", "clip_frac"):
                    stats_acc[key] += st[key] * weight
                n_stat += 1
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

        iters = max(cfg.num_iterations, 1)
        rewards = torch.cat([r["rewards"] for r in rollouts])
        lengths = [l for r in rollouts for l in r["lengths"]]
        truncated = [t for r in rollouts for t in r["truncated"]]
        zero_std = [float((r["advantages"] == 0).all().item()) for r in rollouts]
        metrics = {
            "step": self.global_step + 1,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std(unbiased=False).item(),
            "completion_length": sum(lengths) / max(len(lengths), 1),
            "truncated_frac": sum(truncated) / max(len(truncated), 1),
            "zero_std_frac": sum(zero_std) / max(len(zero_std), 1),
            "loss": stats_acc["loss"] / iters,
            "pg_loss": stats_acc["pg_loss"] / iters,
            "kl": stats_acc["kl"] / iters,
            "clip_frac": stats_acc["clip_frac"] / iters,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        if all(r["correct"] is not None for r in rollouts):
            flags = [c for r in rollouts for c in r["correct"]]
            metrics["accuracy"] = sum(flags) / max(len(flags), 1)
        self.global_step += 1
        return metrics

    # ── kayıt ───────────────────────────────────────────

    def save_checkpoint(self, path: str, extra: Optional[Dict] = None) -> str:
        """
        training/trainer.py ile aynı biçimde kaydet ("model_state_dict",
        "config": architecture_dict()); inference/generate.load_model ile açılır.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        model = getattr(self.model, "_orig_mod", self.model)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "training_recipe": {"stage": "grpo", "grpo": asdict(self.config)},
            "config": model.config.architecture_dict(),
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        return path

    def train(
        self,
        items: Sequence[Dict],
        output_dir: Optional[str] = None,
        log_fn: Callable[[Dict[str, float]], None] = None,
    ) -> List[Dict[str, float]]:
        """`config.steps` adım eğit; prompt'lar her turda karıştırılır."""
        cfg = self.config
        rng = random.Random(cfg.seed)
        order: List[int] = []
        history = []
        for _ in range(cfg.steps):
            batch = []
            while len(batch) < cfg.prompts_per_step:
                if not order:
                    order = list(range(len(items)))
                    rng.shuffle(order)
                batch.append(items[order.pop()])
            metrics = self.step(batch)
            history.append(metrics)
            if log_fn and (self.global_step % max(cfg.log_every, 1) == 0):
                log_fn(metrics)
            if output_dir and cfg.save_every and self.global_step % cfg.save_every == 0:
                self.save_checkpoint(os.path.join(output_dir, f"toprak_grpo_step_{self.global_step}.pt"))
        if output_dir:
            self.save_checkpoint(os.path.join(output_dir, "toprak_grpo_last.pt"))
        return history


# ══════════════════════════════════════════════════════════
#  Veri ve CLI
# ══════════════════════════════════════════════════════════


def load_reasoning_items(
    path: str,
    template,
    max_prompt_tokens: int,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    JSONL'den {"question", "answer"} (synthetic_math biçimi) ya da
    {"messages": [...], "answer"} kayıtlarını oku ve prompt'ları kodla.
    Bütçeyi aşan prompt'lar atlanır.
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "messages" in record:
                messages = [m for m in record["messages"] if m["role"] != "assistant"]
            else:
                question = record.get("question") or record.get("prompt")
                messages = [{"role": "user", "content": question}]
            if "answer" not in record:
                continue
            prompt_ids = template.encode_prompt(messages)
            if len(prompt_ids) > max_prompt_tokens:
                continue
            items.append({**record, "prompt_ids": prompt_ids})
            if limit and len(items) >= limit:
                break
    return items


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    from inference.generate import load_model
    from model.chat_template import ChatTemplate
    from model.config import detect_device
    from model.tokenizer import ToprakTokenizer
    from training.rewards import REASONING_SYSTEM_PROMPT, answers_match, combine_rewards, extract_answer

    defaults = GRPOConfig()
    parser = argparse.ArgumentParser(description="🌱 Toprak — GRPO ile akıl yürütme eğitimi (Düşünen Toprak)")
    parser.add_argument("--checkpoint", required=True, help="Başlangıç modeli (tercihen SFT ısınması yapılmış)")
    parser.add_argument("--data", required=True, help="JSONL: question/answer (data/synthetic_math.py çıktısı)")
    parser.add_argument("--output", required=True, help="Checkpoint klasörü")
    parser.add_argument("--tokenizer", default="toprak_tokenizer.model", help="SentencePiece modeli")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (varsayılan: otomatik)")
    parser.add_argument("--system-prompt", default=REASONING_SYSTEM_PROMPT, help="Sistem mesajı")
    parser.add_argument("--steps", type=int, default=defaults.steps, help="GRPO adım sayısı")
    parser.add_argument("--group-size", type=int, default=defaults.group_size, help="Prompt başına tamamlama (G)")
    parser.add_argument("--prompts-per-step", type=int, default=defaults.prompts_per_step, help="Adım başına prompt")
    parser.add_argument("--num-iterations", type=int, default=defaults.num_iterations, help="Rollout başına güncelleme (μ)")
    parser.add_argument("--lr", type=float, default=defaults.lr, help="Öğrenme oranı")
    parser.add_argument("--beta-kl", type=float, default=defaults.beta_kl, help="KL cezası katsayısı")
    parser.add_argument("--clip-eps", type=float, default=defaults.clip_eps, help="Oran kırpma ε")
    parser.add_argument("--clip-eps-high", type=float, default=None, help="Üst kırpma ε (DAPO 'clip-higher')")
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens, help="Tamamlama sınırı")
    parser.add_argument("--max-prompt-tokens", type=int, default=256, help="Daha uzun prompt'lar atlanır")
    parser.add_argument("--temperature", type=float, default=defaults.temperature, help="Örnekleme sıcaklığı")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k örnekleme (0 = kapalı)")
    parser.add_argument("--save-every", type=int, default=defaults.save_every, help="Kaç adımda bir kayıt")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Rastgelelik tohumu")
    parser.add_argument("--limit", type=int, default=None, help="En çok bu kadar örnek kullan")
    parser.add_argument("--w-correct", type=float, default=1.0, help="Doğruluk ödülü ağırlığı")
    parser.add_argument("--w-format", type=float, default=0.2, help="Biçim ödülü ağırlığı")
    parser.add_argument("--w-language", type=float, default=0.1, help="Dil ödülü ağırlığı")
    parser.add_argument("--w-length", type=float, default=0.1, help="Uzunluk cezası ağırlığı")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or detect_device()
    tokenizer = ToprakTokenizer(args.tokenizer)
    model, model_config = load_model(args.checkpoint, device=device)
    template = ChatTemplate(tokenizer, system_prompt=args.system_prompt)

    max_prompt = min(args.max_prompt_tokens, model.freqs_cis.size(0) - 16)
    items = load_reasoning_items(args.data, template, max_prompt, args.limit)
    if not items:
        raise SystemExit("❌ Veri bulunamadı ya da tüm prompt'lar bütçeyi aşıyor.")

    config = GRPOConfig(
        group_size=args.group_size, prompts_per_step=args.prompts_per_step, steps=args.steps,
        num_iterations=args.num_iterations, lr=args.lr, beta_kl=args.beta_kl,
        clip_eps=args.clip_eps, clip_eps_high=args.clip_eps_high,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
        save_every=args.save_every, seed=args.seed,
    )
    reward = combine_rewards({
        "correctness": args.w_correct, "format": args.w_format,
        "language": args.w_language, "length": args.w_length,
    })
    soft = 4 * args.max_new_tokens  # ~4 karakter/token: sınıra yaklaşınca ceza

    def reward_fn(text, item, ids):
        return reward(text, item["answer"], soft_limit=soft // 2, hard_limit=soft)

    def correct_fn(text, item, ids):
        pred = extract_answer(text)
        return pred is not None and answers_match(pred, item["answer"])

    trainer = GRPOTrainer(
        model, config, reward_fn, tokenizer=tokenizer,
        stop_ids=template.stop_ids, pad_id=tokenizer.pad_token_id, correct_fn=correct_fn,
    )
    os.makedirs(args.output, exist_ok=True)
    print(f"🌱 GRPO: {len(items)} prompt, G={config.group_size}, cihaz={device}")
    start = time.time()

    def log(m):
        print(
            f"  adım {m['step']:>5} | ödül {m['reward_mean']:.3f} | doğruluk {m.get('accuracy', 0):.3f} "
            f"| KL {m['kl']:.4f} | uzunluk {m['completion_length']:.1f} | kırpma {m['clip_frac']:.3f} "
            f"| std=0 {m['zero_std_frac']:.2f} | {time.time() - start:.0f} sn",
            flush=True,
        )
        with open(os.path.join(args.output, "grpo_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(m) + "\n")

    trainer.train(items, output_dir=args.output, log_fn=log)
    print(f"✅ Bitti → {os.path.join(args.output, 'toprak_grpo_last.pt')}")


if __name__ == "__main__":
    main()
