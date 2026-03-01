"""
Toprak — Trainer Sınıfı
Eğitim döngüsü, checkpoint yönetimi ve logging.
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model.config import ModelConfig
from model.transformer import ToprakGPT
from training.scheduler import CosineWarmupScheduler


class ToprakTrainer:
    """
    Toprak model eğitim sınıfı.

    Özellikler:
    - Apple M4 Pro MPS optimizasyonu
    - bfloat16 mixed precision
    - Gradient accumulation
    - Checkpoint save/resume
    - Detaylı logging
    """

    def __init__(
        self,
        model: ToprakGPT,
        config: ModelConfig,
        train_dataloader,
        eval_dataloader=None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer: AdamW with weight decay
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # LR Scheduler: Cosine warmup
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            max_lr=config.learning_rate,
        )

        # Eğitim durumu
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.train_losses = []

        # Device
        self.device = config.device

    def train(self, resume_from: Optional[str] = None):
        """
        Ana eğitim döngüsü.

        Args:
            resume_from: Checkpoint dosyası (opsiyonel, eğitime devam etmek için)
        """
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"✓ Checkpoint'ten devam ediliyor: step {self.global_step}")

        self.model.train()
        self.model.to(self.device)

        print(f"\n{'='*60}")
        print(f"🌱 Toprak Eğitimi Başlıyor")
        print(f"{'='*60}")
        print(f"  Device:              {self.device}")
        print(f"  Parametreler:        {self.model.count_parameters()/1e6:.1f}M")
        print(f"  Batch Size:          {self.config.batch_size}")
        print(f"  Grad Accumulation:   {self.config.grad_accum_steps}")
        print(f"  Efektif Batch:       {self.config.batch_size * self.config.grad_accum_steps}")
        print(f"  Learning Rate:       {self.config.learning_rate}")
        print(f"  Max Steps:           {self.config.max_steps}")
        print(f"  Warmup Steps:        {self.config.warmup_steps}")
        print(f"{'='*60}\n")

        accumulation_loss = 0.0
        start_time = time.time()
        tokens_processed = 0

        data_iter = iter(self.train_dataloader)

        while self.global_step < self.config.max_steps:
            self.optimizer.zero_grad()

            # Gradient accumulation
            for micro_step in range(self.config.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Mixed precision forward pass
                if self.device == "mps":
                    # MPS için autocast
                    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                        logits, loss = self.model(input_ids, targets=labels)
                else:
                    logits, loss = self.model(input_ids, targets=labels)

                # Gradient accumulation: loss'u böl
                loss = loss / self.config.grad_accum_steps
                loss.backward()
                accumulation_loss += loss.item()
                tokens_processed += input_ids.numel()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )

            # Optimizer step
            self.optimizer.step()
            lr = self.scheduler.step()
            self.global_step += 1

            # Logging
            if self.global_step % 100 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed
                print(
                    f"  Step {self.global_step:>6d}/{self.config.max_steps} | "
                    f"Loss: {accumulation_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Tok/s: {tokens_per_sec:.0f} | "
                    f"Elapsed: {elapsed/60:.1f}min"
                )
                self.train_losses.append((self.global_step, accumulation_loss))

            accumulation_loss = 0.0

            # Checkpoint kaydet
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

                # Eval
                if self.eval_dataloader:
                    eval_loss = self.evaluate()
                    print(f"  📊 Eval Loss: {eval_loss:.4f}")
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint(tag="best")
                        print(f"  🏆 Yeni en iyi model kaydedildi!")
                    self.model.train()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Eğitim tamamlandı!")
        print(f"  Toplam süre: {total_time/3600:.1f} saat")
        print(f"  Son loss: {self.train_losses[-1][1] if self.train_losses else 'N/A'}")
        print(f"  En iyi eval loss: {self.best_eval_loss:.4f}")
        print(f"{'='*60}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Eval seti üzerinde loss hesapla."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            _, loss = self.model(input_ids, targets=labels)
            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 50:  # Eval'i 50 batch ile sınırla
                break

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, tag: Optional[str] = None):
        """Checkpoint kaydet."""
        if tag:
            filename = f"toprak_{tag}.pt"
        else:
            filename = f"toprak_step_{self.global_step}.pt"

        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "d_ff": self.config.d_ff,
                "max_seq_len": self.config.max_seq_len,
            },
        }

        torch.save(checkpoint, filepath)
        print(f"  💾 Checkpoint kaydedildi: {filepath}")

        # Eski checkpoint'leri temizle
        self._cleanup_checkpoints()

    def load_checkpoint(self, filepath: str):
        """Checkpoint'ten devam et."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Optimizer state'lerini doğru device'a taşı (CPU → MPS fix)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

    def _cleanup_checkpoints(self):
        """Eski checkpoint'leri sil (son N tanesini tut)."""
        checkpoints = sorted(
            Path(self.checkpoint_dir).glob("toprak_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        while len(checkpoints) > self.config.keep_last_n:
            old = checkpoints.pop(0)
            old.unlink()
            print(f"  🗑️  Eski checkpoint silindi: {old.name}")
