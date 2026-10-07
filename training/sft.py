# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Talimat İnce Ayarı (Supervised Fine-Tuning, SFT)

Ön eğitilmiş bir Toprak checkpoint'ini sohbet verisiyle ince ayarlar.
Veri ve çıkarım aynı sohbet şablonunu (model/chat_template.py) kullanır;
kayıp yalnız asistan cevaplarından hesaplanır (diğer konumlar -100).

Desteklenen JSONL satırları:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}
    {"instruction": "...", "input": "...", "output": "..."}   # Alpaca biçimi

İsteğe bağlı LoRA (düşük ranklı uyarlayıcılar) ile 80M–1B modeller bir Mac
üzerinde de ince ayarlanabilir; kayıtta LoRA ağırlıkları ana ağırlıklara
katlanır, böylece `inference/generate.py::load_model` checkpoint'i doğrudan
açar.

Kullanım:
    python training/sft.py --base-checkpoint checkpoints/toprak_best.pt \\
        --data alignment/examples/sft_sample.jsonl --output checkpoints/toprak_sft.pt \\
        --lora-r 16 --epochs 3
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib
import json
import math
import random
import time
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.chat_template import IGNORE_INDEX, ChatTemplate
from training.scheduler import CosineWarmupScheduler

DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


# ═══════════════════════════════════════════════════════════
# Veri
# ═══════════════════════════════════════════════════════════

def read_jsonl(path: str) -> List[dict]:
    """JSONL dosyasını oku; boş satırları atla."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} geçersiz JSON: {exc}") from exc
    return records


def write_jsonl(path: str, records: Iterable[dict]) -> int:
    """Kayıtları UTF-8 JSONL olarak yaz; yazılan satır sayısını döndür."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def record_to_messages(record: dict) -> List[Dict[str, str]]:
    """
    Bir veri kaydını mesaj listesine çevir.

    `messages` alanı varsa olduğu gibi kullanılır; yoksa Alpaca biçimi
    (instruction / input / output) kullanıcı + asistan turuna dönüştürülür.
    """
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list):
            raise ValueError("'messages' alanı liste olmalı")
        return [{"role": m["role"], "content": m.get("content", "")} for m in messages]
    if "instruction" in record:
        instruction = (record.get("instruction") or "").strip()
        extra = (record.get("input") or "").strip()
        user = f"{instruction}\n\n{extra}" if extra else instruction
        messages = []
        if record.get("system"):
            messages.append({"role": "system", "content": record["system"]})
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": record.get("output", "")})
        return messages
    raise ValueError("Kayıtta 'messages' veya 'instruction' alanı bulunamadı")


class SFTDataset(Dataset):
    """
    Sohbet örneklerini (input_ids, labels) çiftlerine dönüştüren veri seti.

    Args:
        source: JSONL dosya yolu veya kayıt listesi
        tokenizer: ToprakTokenizer (veya aynı arayüze sahip nesne)
        max_len: Örnek başına en fazla token (girdi uzunluğu)
        system_prompt: Kayıtta sistem mesajı yoksa eklenecek varsayılan
        template: Hazır ChatTemplate (verilirse tokenizer/system_prompt yok sayılır)
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
        self.max_len = max_len
        self.examples: List[Dict[str, List[int]]] = []
        self.skipped = 0
        for record in records:
            try:
                messages = record_to_messages(record)
            except (ValueError, KeyError):
                self.skipped += 1
                continue
            inputs, labels = self.template.build_labels(messages, max_len=max_len)
            if not inputs or all(label == IGNORE_INDEX for label in labels):
                # Kırpma sonrası öğrenilecek asistan token'ı kalmadı
                self.skipped += 1
                continue
            self.examples.append({"input_ids": inputs, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


def pad_sequences(sequences: Sequence[Sequence[int]], value: int, length: Optional[int] = None) -> torch.Tensor:
    """Dizileri sağdan doldurarak (B, T) tensörüne çevir."""
    length = length or max(len(s) for s in sequences)
    out = torch.full((len(sequences), length), value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = torch.tensor(list(seq), dtype=torch.long)
    return out


def sft_collate(batch: Sequence[Dict[str, List[int]]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Girdileri pad_id (0), etiketleri IGNORE_INDEX (-100) ile doldur."""
    length = max(len(item["input_ids"]) for item in batch)
    return {
        "input_ids": pad_sequences([item["input_ids"] for item in batch], pad_id, length),
        "labels": pad_sequences([item["labels"] for item in batch], IGNORE_INDEX, length),
    }


def sft_loss(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Maskeli çapraz entropi. Model hedefsiz çağrılır; kayıp burada
    ignore_index=-100 ile hesaplanır (modelin pad=0 maskesi kullanılmaz,
    çünkü 0 bir etiket değildir — maskelemeyi etiketler belirler).
    """
    logits = model(batch["input_ids"])[0]
    labels = batch["labels"]
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


# ═══════════════════════════════════════════════════════════
# LoRA
# ═══════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Dondurulmuş bir nn.Linear etrafında düşük ranklı uyarlayıcı:

        y = base(x) + (alpha / r) · B(A(dropout(x)))

    B sıfırla başlatılır; böylece eğitim başında model birebir aynıdır.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rankı (r) pozitif olmalı")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.enabled = True
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        weight = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=weight.device, dtype=weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def delta_weight(self) -> torch.Tensor:
        """Katlanacak ağırlık farkı: scaling · B @ A."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        lora = F.linear(F.linear(self.lora_dropout(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return out + lora * self.scaling

    def merged_linear(self) -> nn.Linear:
        """LoRA katlanmış bağımsız bir nn.Linear döndür."""
        merged = nn.Linear(
            self.base.in_features, self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device, dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_weight().to(self.base.weight.dtype))
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    target: Sequence[str] = DEFAULT_LORA_TARGETS,
    dropout: float = 0.0,
) -> List[str]:
    """
    Adı `target` içindeki bir parçayla biten nn.Linear katmanlarını LoRALinear
    ile sar ve modelin geri kalanını dondur.

    Returns:
        Sarılan modüllerin tam adları
    """
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    if not replaced:
        raise ValueError(f"LoRA hedefi bulunamadı: {tuple(target)}")
    return replaced


def merge_lora(model: nn.Module) -> int:
    """Tüm LoRALinear katmanlarını katlanmış nn.Linear ile değiştir (yerinde)."""
    count = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged_linear())
                count += 1
    return count


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Yalnız LoRA A/B parametreleri."""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params += [m.lora_A, m.lora_B]
    return params


@contextlib.contextmanager
def lora_disabled(model: nn.Module):
    """LoRA katkısını geçici olarak kapat (DPO'da referans = temel model)."""
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    previous = [m.enabled for m in modules]
    for m in modules:
        m.enabled = False
    try:
        yield
    finally:
        for m, state in zip(modules, previous):
            m.enabled = state


def merged_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    LoRA katlanmış, standart ToprakLM anahtarlarıyla state_dict üret
    (modeli değiştirmeden; ara checkpoint'ler için).
    """
    model = getattr(model, "_orig_mod", model)
    state = model.state_dict()
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"{name}." if name else ""
        merged = module.base.weight.detach() + module.delta_weight().detach().to(module.base.weight.dtype)
        state.pop(f"{prefix}lora_A", None)
        state.pop(f"{prefix}lora_B", None)
        state.pop(f"{prefix}base.weight", None)
        state[f"{prefix}weight"] = merged
        if module.base.bias is not None:
            state[f"{prefix}bias"] = state.pop(f"{prefix}base.bias")
    return {k: v.detach().cpu().clone() for k, v in state.items()}


# ═══════════════════════════════════════════════════════════
# Eğitim yardımcıları (DPO ile ortak)
# ═══════════════════════════════════════════════════════════

def autocast_context(device: str):
    """CUDA'da bf16 autocast; CPU/MPS'te float32 (bağlamsız)."""
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def infinite_batches(loader: DataLoader) -> Iterator:
    """DataLoader'ı epoch'lar boyunca sonsuz döngüye al."""
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("Veri seti boş")


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_aligned_checkpoint(
    model: nn.Module,
    path: str,
    step: int = 0,
    stage: str = "sft",
    recipe: Optional[dict] = None,
    best_eval_loss: Optional[float] = None,
) -> str:
    """
    `training/trainer.py` ile uyumlu checkpoint yaz (LoRA katlanmış).
    `inference.generate.load_model` bu dosyayı doğrudan yükler.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    base = getattr(model, "_orig_mod", model)
    checkpoint = {
        "model_state_dict": merged_state_dict(base),
        "config": base.config.architecture_dict(),
        "global_step": step,
        "best_eval_loss": best_eval_loss if best_eval_loss is not None else float("inf"),
        "training_recipe": {"stage": stage, **(recipe or {})},
    }
    torch.save(checkpoint, path)
    return path


def load_base_model(checkpoint_path: str, device: str):
    """Temel checkpoint'i yükle ve eğitim moduna al."""
    from inference.generate import load_model

    model, config = load_model(checkpoint_path, device)
    model.to(torch.float32)
    model.train()
    return model, config


@torch.no_grad()
def evaluate_sft(model: nn.Module, loader: DataLoader, device: str, max_batches: Optional[int] = None) -> float:
    """Token ağırlıklı ortalama değerlendirme kaybı."""
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_batch(batch, device)
        with autocast_context(device):
            n = int((batch["labels"] != IGNORE_INDEX).sum())
            if n == 0:
                continue
            total += sft_loss(model, batch).item() * n
            tokens += n
    if was_training:
        model.train()
    return total / max(tokens, 1)


def train_sft(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_path: Optional[str] = None,
    epochs: float = 3.0,
    max_steps: Optional[int] = None,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 2e-5,
    min_lr: Optional[float] = None,
    warmup_steps: int = 20,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    lora_r: int = 0,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
    device: str = "cpu",
    eval_every: int = 50,
    log_every: int = 10,
    save_every: int = 0,
    seed: int = 42,
    pad_id: int = 0,
    merge_lora_at_end: bool = True,
    log_fn=print,
) -> Dict[str, list]:
    """
    SFT eğitim döngüsü: AdamW + doğrusal ısınmalı kosinüs LR, gradyan biriktirme,
    gradyan kırpma, değerlendirme kaybı ve checkpoint kaydı.

    lora_r > 0 ise yalnız LoRA parametreleri eğitilir; sonda katlanır.

    Returns:
        {"train_loss": [(adım, kayıp)], "eval_loss": [(adım, kayıp)], "lr": [...]}
    """
    if len(train_dataset) == 0:
        raise ValueError("Eğitim veri seti boş (öğrenilecek asistan token'ı yok)")
    set_seed(seed)
    model.to(device)
    if lora_r > 0:
        apply_lora(model, r=lora_r, alpha=lora_alpha, target=lora_targets, dropout=lora_dropout)
        model.to(device)
        params = lora_parameters(model)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    generator = torch.Generator().manual_seed(seed)
    collate = lambda b: sft_collate(b, pad_id=pad_id)  # noqa: E731
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, generator=generator)
    eval_loader = (DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
                   if eval_dataset is not None and len(eval_dataset) > 0 else None)

    if max_steps is None:
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        max_steps = max(1, math.ceil(steps_per_epoch * epochs))

    # Tekrarlanan (bağlı) parametreleri bir kez ver
    unique_params = list({id(p): p for p in params}.values())
    optimizer = torch.optim.AdamW(unique_params, lr=learning_rate, weight_decay=weight_decay,
                                  betas=(0.9, 0.95))
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=min(warmup_steps, max_steps), max_steps=max_steps,
        max_lr=learning_rate, min_lr=learning_rate * 0.1 if min_lr is None else min_lr,
    )

    history: Dict[str, list] = {"train_loss": [], "eval_loss": [], "lr": []}
    best_eval = float("inf")
    batches = infinite_batches(train_loader)
    recipe = {
        "learning_rate": learning_rate, "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps, "max_steps": max_steps,
        "lora_r": lora_r, "lora_alpha": lora_alpha if lora_r > 0 else None, "seed": seed,
    }
    start = time.time()

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            batch = move_batch(next(batches), device)
            with autocast_context(device):
                loss = sft_loss(model, batch) / grad_accum_steps
            loss.backward()
            step_loss += loss.item()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(unique_params, grad_clip)
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history["train_loss"].append((step, step_loss))
        history["lr"].append((step, lr))

        if log_fn and (step % max(log_every, 1) == 0 or step == 1 or step == max_steps):
            log_fn(f"  [SFT] adım {step}/{max_steps} | kayıp {step_loss:.4f} | lr {lr:.2e} "
                   f"| {time.time() - start:.0f}s")

        if eval_loader is not None and (step % max(eval_every, 1) == 0 or step == max_steps):
            eval_loss = evaluate_sft(model, eval_loader, device)
            history["eval_loss"].append((step, eval_loss))
            if log_fn:
                log_fn(f"  [SFT] değerlendirme kaybı {eval_loss:.4f}")
            if eval_loss < best_eval:
                best_eval = eval_loss
                if output_path:
                    root, ext = os.path.splitext(output_path)
                    save_aligned_checkpoint(model, f"{root}_best{ext or '.pt'}", step, "sft", recipe, best_eval)

        if output_path and save_every and step % save_every == 0 and step != max_steps:
            root, ext = os.path.splitext(output_path)
            save_aligned_checkpoint(model, f"{root}_step{step}{ext or '.pt'}", step, "sft", recipe)

    if output_path:
        save_aligned_checkpoint(model, output_path, max_steps, "sft", recipe,
                                best_eval if best_eval < float("inf") else None)
        if log_fn:
            log_fn(f"  💾 SFT checkpoint kaydedildi: {output_path}")
    if lora_r > 0 and merge_lora_at_end:
        merge_lora(model)
    return history


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def add_common_training_args(parser) -> None:
    """SFT ve DPO CLI'larında ortak argümanlar."""
    parser.add_argument("--base-checkpoint", required=True, help="Başlangıç checkpoint'i (.pt)")
    parser.add_argument("--data", required=True, help="Eğitim JSONL dosyası")
    parser.add_argument("--eval-data", default=None, help="Değerlendirme JSONL dosyası (opsiyonel)")
    parser.add_argument("--eval-fraction", type=float, default=0.0,
                        help="--eval-data yoksa eğitim verisinden ayrılacak oran (ör. 0.05)")
    parser.add_argument("--output", required=True, help="Çıktı checkpoint yolu (.pt)")
    parser.add_argument("--tokenizer", default="toprak_tokenizer.model", help="Tokenizer model dosyası")
    parser.add_argument("--system", default=None, help="Varsayılan sistem mesajı")
    parser.add_argument("--max-len", type=int, default=None,
                        help="Örnek başına maks token (varsayılan: modelin max_seq_len değeri)")
    parser.add_argument("--epochs", type=float, default=3.0, help="Epoch sayısı")
    parser.add_argument("--max-steps", type=int, default=None, help="Optimizer adımı (epoch yerine)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradyan biriktirme adımı")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=0, help="LoRA rankı (0 = tam ince ayar)")
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default=",".join(DEFAULT_LORA_TARGETS),
                        help="Virgülle ayrılmış hedef katman adları")
    parser.add_argument("--device", default=None, help="Cihaz (varsayılan: otomatik)")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=0, help="Ara checkpoint sıklığı (0 = kapalı)")
    parser.add_argument("--seed", type=int, default=42)


def split_records(records: List[dict], fraction: float, seed: int):
    """Kayıtları deterministik olarak eğitim/değerlendirme diye böl."""
    if fraction <= 0 or len(records) < 2:
        return records, []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


def main(argv=None):
    import argparse

    from model.config import detect_device
    from model.tokenizer import ToprakTokenizer
    from utils.validation import setup_error_handler, validate_checkpoint, validate_tokenizer

    setup_error_handler()
    parser = argparse.ArgumentParser(description="🌱 Toprak — Talimat ince ayarı (SFT, opsiyonel LoRA)")
    add_common_training_args(parser)
    parser.add_argument("--lr", type=float, default=None,
                        help="Öğrenme oranı (varsayılan: tam ayarda 2e-5, LoRA'da 2e-4)")
    args = parser.parse_args(argv)

    validate_checkpoint(args.base_checkpoint)
    validate_tokenizer(args.tokenizer)
    device = args.device or detect_device()
    lr = args.lr if args.lr is not None else (2e-4 if args.lora_r > 0 else 2e-5)

    print("🌱 Toprak — SFT")
    model, config = load_base_model(args.base_checkpoint, device)
    tokenizer = ToprakTokenizer(args.tokenizer)
    max_len = args.max_len or config.max_seq_len

    train_records = read_jsonl(args.data)
    eval_records = read_jsonl(args.eval_data) if args.eval_data else []
    if not eval_records:
        train_records, eval_records = split_records(train_records, args.eval_fraction, args.seed)
    template = ChatTemplate(tokenizer, system_prompt=args.system)
    train_ds = SFTDataset(train_records, max_len=max_len, template=template)
    eval_ds = SFTDataset(eval_records, max_len=max_len, template=template) if eval_records else None
    print(f"  Model: {model.count_parameters() / 1e6:.1f}M | cihaz: {device} | "
          f"örnek: {len(train_ds)} (atlanan {train_ds.skipped})"
          + (f" | değerlendirme: {len(eval_ds)}" if eval_ds else ""))
    print(f"  Şablon: {'özel tokenlar' if template.uses_special_tokens else '<sep> metin işaretleyicileri'}")

    train_sft(
        model, train_ds, eval_ds, output_path=args.output,
        epochs=args.epochs, max_steps=args.max_steps, batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum, learning_rate=lr, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        lora_targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
        device=device, eval_every=args.eval_every, save_every=args.save_every,
        seed=args.seed, pad_id=config.pad_token_id,
    )


if __name__ == "__main__":
    main()
