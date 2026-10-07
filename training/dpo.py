# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Tercih Hizalaması (DPO / ORPO)

Direct Preference Optimization (Rafailov ve ark., 2023): politika modeli,
dondurulmuş bir referans modele göre "tercih edilen" (chosen) cevabın
olasılığını "reddedilen" (rejected) cevaba kıyasla artırmayı öğrenir:

    L = -log σ( β · [(log π(y⁺) - log π_ref(y⁺)) - (log π(y⁻) - log π_ref(y⁻))] )

ORPO (Hong ve ark., 2024) referanssız bir alternatiftir; SFT kaybına
olasılık-oranı (odds ratio) terimi ekler ve ikinci bir model gerektirmez.

Veri biçimi (JSONL):

    {"prompt": "soru" | [{"role": ..., "content": ...}, ...],
     "chosen": "iyi cevap", "rejected": "kötü cevap"}

Kullanım:
    python training/dpo.py --base-checkpoint checkpoints/toprak_sft.pt \\
        --data alignment/examples/dpo_sample.jsonl --output checkpoints/toprak_dpo.pt \\
        --beta 0.1 --lora-r 16
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.chat_template import IGNORE_INDEX, ChatTemplate
from training.scheduler import CosineWarmupScheduler
from training.sft import (
    DEFAULT_LORA_TARGETS,
    add_common_training_args,
    apply_lora,
    autocast_context,
    has_lora,
    infinite_batches,
    load_base_model,
    lora_disabled,
    lora_parameters,
    merge_lora,
    move_batch,
    pad_sequences,
    read_jsonl,
    save_aligned_checkpoint,
    set_seed,
    split_records,
)


# ═══════════════════════════════════════════════════════════
# Log-olasılıklar ve kayıplar
# ═══════════════════════════════════════════════════════════

def sequence_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    average: bool = False,
) -> torch.Tensor:
    """
    Her dizi için etiketli (labels != -100) konumlardaki token log-olasılıklarının
    toplamı (average=True ise ortalaması). labels[t], input_ids[t]'den sonra
    gelen token'dır (ChatTemplate.build_labels ile aynı kaydırma).

    Returns:
        (B,) tensör
    """
    logits = model(input_ids)[0].float()
    mask = labels != IGNORE_INDEX
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    total = token_logps.sum(dim=-1)
    if average:
        return total / mask.sum(dim=-1).clamp(min=1)
    return total


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DPO kaybı (label_smoothing > 0 ise gürültülü etiketlere dayanıklı "cDPO").

    Args:
        policy_*/ref_*: (B,) dizi log-olasılıkları
        beta: KL düzenlileştirme gücü (tipik 0.05–0.5)
        label_smoothing: Tercih etiketinin yanlış olma olasılığı ε

    Returns:
        (loss, reward_margins, accuracy)
        loss: skaler ortalama kayıp
        reward_margins: (B,) β·Δchosen − β·Δrejected (gradyansız)
        accuracy: margin > 0 olan çiftlerin oranı (skaler)
    """
    chosen_rewards = beta * (policy_chosen - ref_chosen)
    rejected_rewards = beta * (policy_rejected - ref_rejected)
    logits = chosen_rewards - rejected_rewards
    losses = (
        -(1.0 - label_smoothing) * F.logsigmoid(logits)
        - label_smoothing * F.logsigmoid(-logits)
    )
    margins = logits.detach()
    accuracy = (margins > 0).float().mean()
    return losses.mean(), margins, accuracy


def _log1m_exp(x: torch.Tensor) -> torch.Tensor:
    """Sayısal kararlı log(1 - exp(x)), x < 0."""
    x = x.clamp(max=-1e-6)
    return torch.where(
        x > -math.log(2.0),
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def orpo_loss(
    chosen_avg_logps: torch.Tensor,
    rejected_avg_logps: torch.Tensor,
    chosen_nll: Optional[torch.Tensor] = None,
    lam: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ORPO kaybı (referanssız):

        L = NLL(y⁺) − λ · log σ( log odds(y⁺) − log odds(y⁻) ),
        log odds(y) = log p − log(1 − p),  p = token başına ortalama olasılık

    Args:
        chosen_avg_logps / rejected_avg_logps: (B,) token başına ortalama log-olasılık
        chosen_nll: Tercih edilen cevabın SFT kaybı (None → -mean(chosen_avg_logps))
        lam: Oran teriminin ağırlığı (makalede 0.1)

    Returns:
        (loss, log_odds_ratio (B,), accuracy)
    """
    log_odds = (chosen_avg_logps - _log1m_exp(chosen_avg_logps)) - (
        rejected_avg_logps - _log1m_exp(rejected_avg_logps)
    )
    ratio_loss = -F.logsigmoid(log_odds).mean()
    if chosen_nll is None:
        chosen_nll = -chosen_avg_logps.mean()
    loss = chosen_nll + lam * ratio_loss
    margins = log_odds.detach()
    return loss, margins, (margins > 0).float().mean()


# ═══════════════════════════════════════════════════════════
# Veri
# ═══════════════════════════════════════════════════════════

def prompt_to_messages(prompt: Union[str, Sequence[dict]]) -> List[Dict[str, str]]:
    """Düz metin prompt'u tek kullanıcı mesajına çevir; liste ise kopyala."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": m["role"], "content": m.get("content", "")} for m in prompt]


class PreferenceDataset(Dataset):
    """
    {"prompt", "chosen", "rejected"} kayıtlarını şablonlu (input_ids, labels)
    çiftlerine dönüştürür. Her iki cevapta da öğrenilecek token'ı olmayan
    örnekler atlanır.
    """

    def __init__(
        self,
        source: Union[str, Sequence[dict]],
        tokenizer=None,
        max_len: int = 512,
        system_prompt: Optional[str] = None,
        template: Optional[ChatTemplate] = None,
    ):
        self.template = template or ChatTemplate(tokenizer, system_prompt=system_prompt)
        records = read_jsonl(source) if isinstance(source, str) else list(source)
        self.examples: List[Dict[str, List[int]]] = []
        self.skipped = 0
        for record in records:
            try:
                messages = prompt_to_messages(record["prompt"])
                chosen, rejected = record["chosen"], record["rejected"]
            except (KeyError, TypeError):
                self.skipped += 1
                continue
            item = {}
            for key, answer in (("chosen", chosen), ("rejected", rejected)):
                inputs, labels = self.template.build_labels(
                    messages + [{"role": "assistant", "content": answer}], max_len=max_len
                )
                item[f"{key}_input_ids"], item[f"{key}_labels"] = inputs, labels
            if all(l == IGNORE_INDEX for l in item["chosen_labels"]) or all(
                l == IGNORE_INDEX for l in item["rejected_labels"]
            ):
                self.skipped += 1
                continue
            self.examples.append(item)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


def preference_collate(batch: Sequence[Dict[str, List[int]]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Tercih edilen ve reddedilen dizileri tek bir (2B, T) yığınına koy:
    ilk B satır chosen, son B satır rejected (tek ileri geçiş).
    """
    seqs = [b["chosen_input_ids"] for b in batch] + [b["rejected_input_ids"] for b in batch]
    labels = [b["chosen_labels"] for b in batch] + [b["rejected_labels"] for b in batch]
    length = max(len(s) for s in seqs)
    return {
        "input_ids": pad_sequences(seqs, pad_id, length),
        "labels": pad_sequences(labels, IGNORE_INDEX, length),
    }


# ═══════════════════════════════════════════════════════════
# Eğitim
# ═══════════════════════════════════════════════════════════

def make_reference_model(model: nn.Module) -> nn.Module:
    """Politikanın dondurulmuş kopyası (referans π_ref)."""
    ref = copy.deepcopy(getattr(model, "_orig_mod", model))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def preference_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    method: str = "dpo",
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    orpo_lambda: float = 0.1,
    ref_model: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bir yığın için (loss, margins, accuracy) hesapla."""
    input_ids, labels = batch["input_ids"], batch["labels"]
    half = input_ids.size(0) // 2
    if method == "orpo":
        avg = sequence_logprobs(model, input_ids, labels, average=True)
        return orpo_loss(avg[:half], avg[half:], lam=orpo_lambda)
    if method != "dpo":
        raise ValueError(f"Bilinmeyen yöntem: {method!r} (dpo | orpo)")

    policy = sequence_logprobs(model, input_ids, labels)
    with torch.no_grad():
        if ref_model is not None:
            ref = sequence_logprobs(ref_model, input_ids, labels)
        elif has_lora(model):
            # LoRA'da referans = LoRA'sız temel model; ikinci kopya gerekmez
            with lora_disabled(model):
                ref = sequence_logprobs(model, input_ids, labels)
        else:
            raise ValueError("DPO için referans model (ref_model) veya LoRA gerekli")
    return dpo_loss(policy[:half], policy[half:], ref[:half], ref[half:],
                    beta=beta, label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate_preferences(model, loader, device, ref_model=None, **kwargs) -> Dict[str, float]:
    """Değerlendirme: ortalama kayıp, ödül marjı ve doğruluk."""
    was_training = model.training
    model.eval()
    losses, margins, accs, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device):
            loss, margin, acc = preference_step(model, batch, ref_model=ref_model, **kwargs)
        losses += loss.item()
        margins += margin.mean().item()
        accs += acc.item()
        n += 1
    if was_training:
        model.train()
    n = max(n, 1)
    return {"loss": losses / n, "margin": margins / n, "accuracy": accs / n}


def train_preference(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_path: Optional[str] = None,
    method: str = "dpo",
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    orpo_lambda: float = 0.1,
    ref_model: Optional[nn.Module] = None,
    epochs: float = 1.0,
    max_steps: Optional[int] = None,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-6,
    min_lr: Optional[float] = None,
    warmup_steps: int = 10,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    lora_r: int = 0,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
    device: str = "cpu",
    eval_every: int = 50,
    log_every: int = 10,
    seed: int = 42,
    pad_id: int = 0,
    shuffle: bool = True,
    merge_lora_at_end: bool = True,
    log_fn=print,
) -> Dict[str, list]:
    """
    DPO / ORPO eğitim döngüsü (sft.train_sft ile aynı iskelet).

    DPO'da referans model: verilmezse LoRA kapalıyken politikanın dondurulmuş
    kopyası oluşturulur; LoRA açıkken LoRA'sız temel model referanstır.

    Returns:
        {"loss", "margin", "accuracy", "eval"} geçmişi
    """
    if len(train_dataset) == 0:
        raise ValueError("Tercih veri seti boş")
    set_seed(seed)
    model.to(device)
    if method == "dpo" and ref_model is None and lora_r <= 0 and not has_lora(model):
        ref_model = make_reference_model(model).to(device)
    if lora_r > 0:
        apply_lora(model, r=lora_r, alpha=lora_alpha, target=lora_targets, dropout=lora_dropout)
        model.to(device)
        params = lora_parameters(model)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    collate = lambda b: preference_collate(b, pad_id=pad_id)  # noqa: E731
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              collate_fn=collate, generator=generator)
    eval_loader = (DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate)
                   if eval_dataset is not None and len(eval_dataset) > 0 else None)
    if max_steps is None:
        max_steps = max(1, math.ceil(math.ceil(len(train_loader) / grad_accum_steps) * epochs))

    unique_params = list({id(p): p for p in params}.values())
    optimizer = torch.optim.AdamW(unique_params, lr=learning_rate, weight_decay=weight_decay,
                                  betas=(0.9, 0.95))
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=min(warmup_steps, max_steps), max_steps=max_steps,
        max_lr=learning_rate, min_lr=learning_rate * 0.1 if min_lr is None else min_lr,
    )
    step_kwargs = dict(method=method, beta=beta, label_smoothing=label_smoothing,
                       orpo_lambda=orpo_lambda)
    recipe = {"method": method, "beta": beta, "label_smoothing": label_smoothing,
              "orpo_lambda": orpo_lambda, "learning_rate": learning_rate,
              "lora_r": lora_r, "max_steps": max_steps, "seed": seed}
    history: Dict[str, list] = {"loss": [], "margin": [], "accuracy": [], "eval": []}
    batches = infinite_batches(train_loader)
    best_eval = float("inf")
    start = time.time()

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = step_margin = step_acc = 0.0
        for _ in range(grad_accum_steps):
            batch = move_batch(next(batches), device)
            with autocast_context(device):
                loss, margins, acc = preference_step(model, batch, ref_model=ref_model, **step_kwargs)
            (loss / grad_accum_steps).backward()
            step_loss += loss.item() / grad_accum_steps
            step_margin += margins.mean().item() / grad_accum_steps
            step_acc += acc.item() / grad_accum_steps
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(unique_params, grad_clip)
        optimizer.step()
        scheduler.step()
        history["loss"].append((step, step_loss))
        history["margin"].append((step, step_margin))
        history["accuracy"].append((step, step_acc))

        if log_fn and (step % max(log_every, 1) == 0 or step == 1 or step == max_steps):
            log_fn(f"  [{method.upper()}] adım {step}/{max_steps} | kayıp {step_loss:.4f} | "
                   f"marj {step_margin:+.4f} | doğruluk {step_acc:.2f} | {time.time() - start:.0f}s")

        if eval_loader is not None and (step % max(eval_every, 1) == 0 or step == max_steps):
            metrics = evaluate_preferences(model, eval_loader, device, ref_model=ref_model, **step_kwargs)
            history["eval"].append((step, metrics))
            if log_fn:
                log_fn(f"  [{method.upper()}] değerlendirme: kayıp {metrics['loss']:.4f} | "
                       f"marj {metrics['margin']:+.4f} | doğruluk {metrics['accuracy']:.2f}")
            if output_path and metrics["loss"] < best_eval:
                best_eval = metrics["loss"]
                root, ext = os.path.splitext(output_path)
                save_aligned_checkpoint(model, f"{root}_best{ext or '.pt'}", step, method, recipe, best_eval)

    if output_path:
        save_aligned_checkpoint(model, output_path, max_steps, method, recipe,
                                best_eval if best_eval < float("inf") else None)
        if log_fn:
            log_fn(f"  💾 {method.upper()} checkpoint kaydedildi: {output_path}")
    if lora_r > 0 and merge_lora_at_end:
        merge_lora(model)
    return history


# Geriye/isim uyumu: train_dpo, ORPO için de çalışır (method="orpo")
train_dpo = train_preference


def main(argv=None):
    import argparse

    from model.config import detect_device
    from model.tokenizer import ToprakTokenizer
    from utils.validation import setup_error_handler, validate_checkpoint, validate_tokenizer

    setup_error_handler()
    parser = argparse.ArgumentParser(description="🌱 Toprak — Tercih hizalaması (DPO / ORPO)")
    add_common_training_args(parser)
    parser.add_argument("--method", choices=["dpo", "orpo"], default="dpo",
                        help="dpo (referanslı) veya orpo (referanssız)")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO β (KL gücü)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Gürültülü tercih etiketleri için ε (cDPO)")
    parser.add_argument("--orpo-lambda", type=float, default=0.1, help="ORPO oran terimi ağırlığı")
    parser.add_argument("--ref-checkpoint", default=None,
                        help="Referans checkpoint (varsayılan: başlangıç modeli)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Öğrenme oranı (varsayılan: tam ayarda 5e-6, LoRA'da 5e-5)")
    parser.set_defaults(epochs=1.0, batch_size=2)
    args = parser.parse_args(argv)

    validate_checkpoint(args.base_checkpoint)
    validate_tokenizer(args.tokenizer)
    device = args.device or detect_device()
    lr = args.lr if args.lr is not None else (5e-5 if args.lora_r > 0 else 5e-6)

    print(f"🌱 Toprak — {args.method.upper()}")
    model, config = load_base_model(args.base_checkpoint, device)
    ref_model = None
    if args.method == "dpo" and args.ref_checkpoint:
        validate_checkpoint(args.ref_checkpoint)
        ref_model, _ = load_base_model(args.ref_checkpoint, device)
        ref_model = make_reference_model(ref_model)
    tokenizer = ToprakTokenizer(args.tokenizer)
    max_len = args.max_len or config.max_seq_len
    template = ChatTemplate(tokenizer, system_prompt=args.system)

    train_records = read_jsonl(args.data)
    eval_records = read_jsonl(args.eval_data) if args.eval_data else []
    if not eval_records:
        train_records, eval_records = split_records(train_records, args.eval_fraction, args.seed)
    train_ds = PreferenceDataset(train_records, max_len=max_len, template=template)
    eval_ds = PreferenceDataset(eval_records, max_len=max_len, template=template) if eval_records else None
    print(f"  Model: {model.count_parameters() / 1e6:.1f}M | cihaz: {device} | "
          f"çift: {len(train_ds)} (atlanan {train_ds.skipped})")

    train_preference(
        model, train_ds, eval_ds, output_path=args.output, method=args.method,
        beta=args.beta, label_smoothing=args.label_smoothing, orpo_lambda=args.orpo_lambda,
        ref_model=ref_model, epochs=args.epochs, max_steps=args.max_steps,
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum, learning_rate=lr,
        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        lora_targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
        device=device, eval_every=args.eval_every, seed=args.seed, pad_id=config.pad_token_id,
    )


if __name__ == "__main__":
    main()
