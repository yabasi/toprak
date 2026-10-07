# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Bilgi Damıtma (Knowledge Distillation)
Büyük öğretmen modelin (ör. Large) bilgisini küçük öğrenciye (ör. Small)
aktarır; telefonda çalışacak model için ilk adım.

Kayıp:
    L = alpha * T² * KL(p_öğretmen^T || p_öğrenci^T) + (1 - alpha) * CE(öğrenci, etiket)

- T (sıcaklık) yumuşak dağılımları düzleştirir; T² gradyan ölçeğini korur
  (Hinton ve ark., 2015).
- Pad tokenları (etiket = pad_token_id) hem KL hem CE'den çıkarılır.
- top_k: öğretmenin yalnız en yüksek k logit'i tutulur ve bu k üzerinde
  yeniden normalize edilir (öğretmen logit'lerini önbelleğe alırken/aktarırken
  bellek: V yerine 2k değer). Öğrenci log-olasılıkları tam sözlük üzerinden
  hesaplanıp bu k indekste toplanır.
- Öğretmen ve öğrenci sözlüğü (vocab) aynı olmalı.

Kullanım:
    python training/distill.py --teacher checkpoints/toprak_large.pt \\
        --student-size small --data-dir data_cache/bin --bin-mode \\
        --max-steps 20000 --temperature 2.0 --alpha 0.5 --top-k 64
"""

import argparse
import copy
import math
import os
import sys
import time
from typing import Iterable, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

TeacherLogits = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


# ── Kayıp fonksiyonları ────────────────────────────────────

def sparsify_teacher_logits(teacher_logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Öğretmen logit'lerinin en yüksek k değerini ve indekslerini döndür.

    Returns:
        (values (..., k), indices (..., k))
    """
    k = min(int(k), teacher_logits.size(-1))
    values, indices = torch.topk(teacher_logits, k, dim=-1)
    return values, indices


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: TeacherLogits,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = 0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Damıtma kaybı.

    Args:
        student_logits: (B, T, V)
        teacher_logits: (B, T, V) tam logit'ler veya (values, indices) top-k çifti
        labels: (B, T) sert etiketler (None → yalnız KL, alpha zorla 1)
        temperature: Yumuşatma sıcaklığı T
        alpha: KL ağırlığı (0 → saf CE, 1 → saf KL)
        ignore_index: Kayba katılmayacak etiket (varsayılan pad = 0)
        top_k: Tam öğretmen logit'i verildiyse, yalnız top-k tutulur

    Returns:
        (loss, {"kl", "ce", "loss"} — float değerler)
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha [0, 1] aralığında olmalı, {alpha} verildi")
    if temperature <= 0:
        raise ValueError("temperature pozitif olmalı")
    V = student_logits.size(-1)
    T = float(temperature)

    sparse = isinstance(teacher_logits, (tuple, list))
    if not sparse:
        assert teacher_logits.size(-1) == V, (
            f"Öğretmen ve öğrenci sözlüğü uyuşmuyor: {teacher_logits.size(-1)} != {V}"
        )
        if top_k is not None and top_k < V:
            teacher_logits = sparsify_teacher_logits(teacher_logits, top_k)
            sparse = True
    if sparse:
        t_values, t_indices = teacher_logits
        assert int(t_indices.max()) < V, "Öğretmen top-k indeksleri öğrenci sözlüğünü aşıyor"

    # Maske: geçerli (pad olmayan) pozisyonlar
    if labels is not None:
        mask = (labels != ignore_index).float()
    else:
        mask = torch.ones(student_logits.shape[:-1], device=student_logits.device)
        alpha = 1.0
    n_valid = mask.sum().clamp(min=1.0)

    s_logp = F.log_softmax(student_logits.float() / T, dim=-1)
    if sparse:
        t_logp = F.log_softmax(t_values.float() / T, dim=-1)  # k üzerinde yeniden normalize
        s_logp_k = torch.gather(s_logp, -1, t_indices)
        kl_tok = (t_logp.exp() * (t_logp - s_logp_k)).sum(-1)
    else:
        t_logp = F.log_softmax(teacher_logits.float() / T, dim=-1)
        kl_tok = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    kl = (kl_tok * mask).sum() / n_valid

    if labels is not None and alpha < 1.0:
        ce = F.cross_entropy(
            student_logits.float().reshape(-1, V), labels.reshape(-1), ignore_index=ignore_index
        )
    else:
        ce = student_logits.new_zeros((), dtype=torch.float32)

    loss = alpha * (T ** 2) * kl + (1.0 - alpha) * ce
    return loss, {"kl": kl.item(), "ce": ce.item(), "loss": loss.item()}


# ── Eğitici ────────────────────────────────────────────────

class DistillTrainer:
    """
    Öğretmen → öğrenci damıtma döngüsü.

    Öğretmen dondurulur (eval + no_grad). Öğrenci AdamW + cosine warmup ile
    eğitilir. Veri yükleyici `{"input_ids", "labels"}` sözlükleri üretmelidir
    (data/dataset.py: ToprakDataset / ToprakShardDataset).

    Checkpoint biçimi standart Toprak biçimidir: "model_state_dict",
    "config" (architecture_dict), "global_step" (+ optimizer ve damıtma meta).
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_loader: Iterable,
        lr: float = 3e-4,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        min_lr: float = 1e-5,
        temperature: float = 2.0,
        alpha: float = 0.5,
        top_k: Optional[int] = None,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
        weight_decay: float = 0.1,
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        save_every: int = 0,
        log_every: int = 50,
        eval_loader: Optional[Iterable] = None,
        teacher_path: Optional[str] = None,
    ):
        s_cfg, t_cfg = student.config, teacher.config
        if s_cfg.vocab_size != t_cfg.vocab_size:
            raise ValueError(
                f"Sözlük uyuşmazlığı: öğretmen {t_cfg.vocab_size}, öğrenci {s_cfg.vocab_size}. "
                "Damıtma için aynı tokenizer şart."
            )
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
            if getattr(s_cfg, attr) != getattr(t_cfg, attr):
                raise ValueError(f"{attr} öğretmen ve öğrencide farklı")

        self.device = device
        self.student = student.to(device)
        self.teacher = teacher.to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_steps = max_steps
        self.temperature = temperature
        self.alpha = alpha
        self.top_k = top_k
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.log_every = log_every
        self.teacher_path = teacher_path
        self.pad_token_id = s_cfg.pad_token_id
        self.global_step = 0
        self.history = []

        decay = [p for n, p in self.student.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in self.student.named_parameters() if p.requires_grad and p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.95),
        )
        from training.scheduler import CosineWarmupScheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps=warmup_steps, max_steps=max_steps,
            max_lr=lr, min_lr=min(min_lr, lr),
        )

    def _batches(self):
        while True:
            produced = False
            for batch in self.train_loader:
                produced = True
                yield batch
            if not produced:
                raise RuntimeError("Eğitim veri yükleyicisi boş")

    def _student_aux_loss(self) -> torch.Tensor:
        """Öğrenci MoE ise yük dengeleme kaybı."""
        aux = [b.last_aux_loss for b in getattr(self.student, "blocks", [])
               if getattr(b, "use_moe", False) and b.last_aux_loss is not None]
        if not aux:
            return None
        return self.student.config.moe_aux_loss_coef * torch.stack(aux).mean()

    def compute_loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        with torch.no_grad():
            t_logits = self.teacher(input_ids)[0]
            if self.top_k:
                t_logits = sparsify_teacher_logits(t_logits, self.top_k)
        s_logits = self.student(input_ids)[0]
        loss, parts = distillation_loss(
            s_logits, t_logits, labels, temperature=self.temperature,
            alpha=self.alpha, ignore_index=self.pad_token_id,
        )
        aux = self._student_aux_loss()
        if aux is not None:
            loss = loss + aux
        return loss, parts

    @torch.no_grad()
    def evaluate(self, loader: Optional[Iterable] = None, max_batches: int = 50) -> dict:
        """Ortalama KL (T=1 değil, eğitim sıcaklığında) ve CE."""
        loader = loader or self.eval_loader
        if loader is None:
            return {}
        self.student.eval()
        totals, n = {"kl": 0.0, "ce": 0.0}, 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            t_logits = self.teacher(ids)[0]
            s_logits = self.student(ids)[0]
            _, parts_kl = distillation_loss(s_logits, t_logits, labels, self.temperature, 1.0,
                                            self.pad_token_id)
            _, parts_ce = distillation_loss(s_logits, t_logits, labels, self.temperature, 0.0,
                                            self.pad_token_id)
            totals["kl"] += parts_kl["kl"]
            totals["ce"] += parts_ce["ce"]
            n += 1
        self.student.train()
        return {k: v / max(n, 1) for k, v in totals.items()}

    def train(self, max_steps: Optional[int] = None) -> list:
        """Damıtma döngüsü. Adım başına {"step", "loss", "kl", "ce", "lr"} geçmişi döndürür."""
        steps = max_steps or self.max_steps
        self.student.train()
        batches = self._batches()
        start = time.time()
        while self.global_step < steps:
            self.optimizer.zero_grad(set_to_none=True)
            acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0}
            for _ in range(self.grad_accum_steps):
                batch = next(batches)
                ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                loss, parts = self.compute_loss(ids, labels)
                (loss / self.grad_accum_steps).backward()
                for key in acc:
                    acc[key] += parts[key] / self.grad_accum_steps
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
            self.optimizer.step()
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            self.global_step += 1
            record = {"step": self.global_step, "lr": lr, **acc}
            self.history.append(record)

            if self.log_every and self.global_step % self.log_every == 0:
                elapsed = time.time() - start
                print(f"  adım {self.global_step:>6} | kayıp {acc['loss']:.4f} | "
                      f"KL {acc['kl']:.4f} | CE {acc['ce']:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            if self.checkpoint_dir and self.save_every and self.global_step % self.save_every == 0:
                self.save_checkpoint()
        if self.checkpoint_dir:
            self.save_checkpoint(tag="last")
        return self.history

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        """Standart Toprak checkpoint biçiminde öğrenciyi kaydet."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        name = f"toprak_distill_{tag}.pt" if tag else f"toprak_distill_step_{self.global_step}.pt"
        path = os.path.join(self.checkpoint_dir, name)
        student = getattr(self.student, "_orig_mod", self.student)
        torch.save({
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": student.config.architecture_dict(),
            "distillation": {
                "teacher": self.teacher_path,
                "teacher_config": self.teacher.config.architecture_dict(),
                "temperature": self.temperature,
                "alpha": self.alpha,
                "top_k": self.top_k,
            },
        }, path)
        print(f"  💾 Öğrenci checkpoint'i kaydedildi: {path}")
        return path


# ── CLI ────────────────────────────────────────────────────

def parse_args(argv=None):
    from model.config import CONFIGS
    parser = argparse.ArgumentParser(description="🌱 Toprak — bilgi damıtma (öğretmen → öğrenci)")
    parser.add_argument("--teacher", required=True, help="Öğretmen checkpoint'i (.pt)")
    parser.add_argument("--student-size", default="small", choices=list(CONFIGS),
                        help="Öğrenci preset boyutu (varsayılan: small)")
    parser.add_argument("--student-init", default=None,
                        help="Öğrenciyi var olan checkpoint'ten başlat (opsiyonel)")
    parser.add_argument("--data-dir", default="data_cache/clean/train",
                        help="Veri dizini (JSONL/TXT veya --bin-mode ile manifest.json'lu shard dizini)")
    parser.add_argument("--bin-mode", action="store_true", help="Pre-tokenize .bin shard'larını kullan")
    parser.add_argument("--tokenizer", default="toprak_tokenizer.model")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Dizi uzunluğu (varsayılan: öğrenci preset'i, öğretmenle sınırlı)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=2.0, help="Damıtma sıcaklığı T")
    parser.add_argument("--alpha", type=float, default=0.5, help="KL ağırlığı (0=saf CE, 1=saf KL)")
    parser.add_argument("--top-k", type=int, default=None, help="Öğretmen top-k seyrekleştirme")
    parser.add_argument("--checkpoint-dir", default="checkpoints/distill")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default=None, help="Cihaz (varsayılan: otomatik)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from model.config import CONFIGS, detect_device
    from model.transformer import ToprakLM
    from model.tokenizer import ToprakTokenizer
    from data.dataset import ToprakDataset, ToprakShardDataset, create_dataloader
    from export.hf_llama import config_from_dict

    torch.manual_seed(args.seed)
    device = args.device or detect_device()

    print(f"🎓 Öğretmen yükleniyor: {args.teacher}")
    ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
    t_config = config_from_dict(ckpt.get("config", {}))
    tokenizer = ToprakTokenizer(args.tokenizer)
    teacher = ToprakLM(t_config, tokenizer=tokenizer)
    teacher.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    print(f"  Öğretmen: {teacher.count_parameters() / 1e6:.1f}M parametre")

    s_config = copy.deepcopy(CONFIGS[args.student_size])
    s_config.vocab_size = t_config.vocab_size
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        setattr(s_config, attr, getattr(t_config, attr))
    seq_len = args.seq_len or s_config.max_seq_len
    seq_len = min(seq_len, t_config.max_seq_len)
    s_config.max_seq_len = max(s_config.max_seq_len, seq_len)
    s_config.device = device
    student = ToprakLM(s_config, tokenizer=tokenizer)
    if args.student_init:
        s_ckpt = torch.load(args.student_init, map_location="cpu", weights_only=False)
        student.load_state_dict(s_ckpt["model_state_dict"])
    print(f"  Öğrenci ({args.student_size}): {student.count_parameters() / 1e6:.1f}M parametre")
    if tokenizer.get_vocab_size() != t_config.vocab_size:
        raise ValueError("Tokenizer sözlüğü öğretmen sözlüğüyle uyuşmuyor")

    if args.bin_mode:
        dataset = ToprakShardDataset(
            bin_dir=args.data_dir, split="train", max_seq_len=seq_len,
            expected_vocab_size=t_config.vocab_size, seed=args.seed,
        )
        try:
            eval_dataset = ToprakShardDataset(
                bin_dir=args.data_dir, split="eval", max_seq_len=seq_len,
                expected_vocab_size=t_config.vocab_size,
            )
        except (RuntimeError, ValueError):
            eval_dataset = None
    else:
        dataset = ToprakDataset(args.data_dir, tokenizer, max_seq_len=seq_len, seed=args.seed)
        eval_dataset = None
    loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    eval_loader = (create_dataloader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                     drop_last=False) if eval_dataset is not None else None)

    trainer = DistillTrainer(
        student, teacher, loader,
        lr=args.lr, max_steps=args.max_steps, warmup_steps=args.warmup_steps,
        temperature=args.temperature, alpha=args.alpha, top_k=args.top_k,
        grad_accum_steps=args.grad_accum, device=device,
        checkpoint_dir=args.checkpoint_dir, save_every=args.save_every,
        log_every=args.log_every, eval_loader=eval_loader, teacher_path=args.teacher,
    )
    print(f"🔥 Damıtma: T={args.temperature}, alpha={args.alpha}, top_k={args.top_k}, "
          f"seq_len={seq_len}, {args.max_steps} adım")
    trainer.train()
    if eval_loader is not None:
        metrics = trainer.evaluate()
        print(f"  Eval: KL {metrics['kl']:.4f} | CE {metrics['ce']:.4f} "
              f"| PPL {math.exp(min(metrics['ce'], 50)):.2f}")


if __name__ == "__main__":
    main()
