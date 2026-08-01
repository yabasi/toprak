# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Trainer Sınıfı
Eğitim döngüsü, checkpoint yönetimi, TensorBoard logging.

Optimizasyonlar:
- torch.compile() ile model derleme
- Gradient checkpointing (bellek tasarrufu)
- TensorBoard ile eğitim metrikleri
- bfloat16 mixed precision (MPS)
"""

import hashlib
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch.optim import AdamW

from model.config import ModelConfig
from model.transformer import ToprakLM
from training.scheduler import CosineWarmupScheduler
from utils.reproducibility import write_manifest

# TensorBoard — opsiyonel
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class ToprakTrainer:
    """
    Toprak model eğitim sınıfı.

    Özellikler:
    - Apple M4 Pro MPS optimizasyonu
    - bfloat16 mixed precision
    - torch.compile() ile model derleme
    - Gradient checkpointing (bellek tasarrufu)
    - Gradient accumulation
    - Checkpoint save/resume
    - TensorBoard logging
    """

    def __init__(
        self,
        model: ToprakLM,
        config: ModelConfig,
        train_dataloader,
        eval_dataloader=None,
        checkpoint_dir: str = "checkpoints",
        use_compile: bool = True,
        use_gradient_checkpointing: bool = True,
        log_dir: str = "logs",
        vowel_harmony_loss=None,
        morph_weight_loss=None,
        consonant_harmony_loss=None,
        syllable_rhyme_loss=None,
        use_bf16: bool = False,
        training_recipe: Optional[dict] = None,
        experiment_manifest: Optional[dict] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.checkpoint_dir = checkpoint_dir
        self.vowel_harmony_loss = vowel_harmony_loss
        self.morph_weight_loss = morph_weight_loss
        self.consonant_harmony_loss = consonant_harmony_loss
        self.syllable_rhyme_loss = syllable_rhyme_loss
        self.training_recipe = training_recipe or {}
        self.experiment_manifest = experiment_manifest or {}
        self._data_epoch_generator_state = None
        self._data_epoch_sampler_state = None
        self._data_batch_in_epoch = 0
        self._resume_data_state = None

        # bf16 yalnız CUDA'da anlamlı; bf16 destekli kart kontrolü
        self.use_bf16 = False
        if use_bf16:
            if config.device == "cuda" and torch.cuda.is_bf16_supported():
                self.use_bf16 = True
                print("  ✓ bfloat16 mixed precision aktif (CUDA)")
            else:
                print("  ⚠ --bf16 istendi ama desteklenmiyor (cihaz/cuda kontrolü), fp16'ya düşülüyor")

        # FP16 küçük gradientleri sıfıra yuvarlayabildiğinden loss scaling
        # gerekir. BF16'ın dinamik aralığı geniştir ve scaler kullanılmaz.
        scaler_enabled = config.device == "cuda" and not self.use_bf16
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.grad_scaler = torch.amp.GradScaler(
                "cuda", enabled=scaler_enabled
            )
        else:  # PyTorch 2.0–2.2 uyumluluğu
            self.grad_scaler = torch.cuda.amp.GradScaler(
                enabled=scaler_enabled
            )

        os.makedirs(checkpoint_dir, exist_ok=True)

        # ─── Gradient Checkpointing ───
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing = True
            print("  ✓ Gradient checkpointing aktif (bellek tasarrufu)")

        # ─── torch.compile() ───
        self.compiled = False
        if use_compile:
            try:
                self.model = torch.compile(self.model)
                self.compiled = True
                print("  ✓ torch.compile() aktif (eğitim hızlandırma)")
            except Exception as e:
                print(f"  ⚠ torch.compile() başarısız: {e}")

        # ─── TensorBoard ───
        self.writer = None
        if HAS_TENSORBOARD:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"  ✓ TensorBoard aktif → {log_dir}/")
            print(f"    Görüntüle: tensorboard --logdir {log_dir}")
        else:
            print("  ⚠ TensorBoard bulunamadı (pip install tensorboard)")

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
        self.consecutive_nan_count = 0
        self.max_consecutive_nan = 10  # Bu kadar arka arkaya nan olursa dur

        # Device
        self.device = config.device

    def _capture_rng_state(self) -> dict:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: Optional[dict]) -> None:
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _capture_data_state(self) -> dict:
        generator_state = self._data_epoch_generator_state
        if isinstance(generator_state, torch.Tensor):
            generator_state = generator_state.clone()
        return {
            "epoch_generator_state": generator_state,
            "sampler_state": self._data_epoch_sampler_state,
            "batch_in_epoch": self._data_batch_in_epoch,
        }

    def _training_sampler(self):
        batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
        return getattr(batch_sampler, "sampler", None)

    def _create_data_iterator(self):
        """Aynı epoch permütasyonunu ve batch cursor'unu yeniden kur."""
        generator = getattr(self.train_dataloader, "generator", None)
        sampler = self._training_sampler()
        resume_state = self._resume_data_state
        if resume_state:
            epoch_state = resume_state.get("epoch_generator_state")
            batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
            if epoch_state is not None:
                if generator is None:
                    raise RuntimeError(
                        "Checkpoint veri cursor'u için seed'li DataLoader generator gerekli"
                    )
                generator.set_state(epoch_state)
            sampler_state = resume_state.get("sampler_state")
            if sampler_state is not None:
                if not hasattr(sampler, "load_state_dict"):
                    raise RuntimeError("Checkpoint sampler durumu yüklenemiyor")
                sampler.load_state_dict(sampler_state)
            self._data_epoch_generator_state = (
                epoch_state.clone() if isinstance(epoch_state, torch.Tensor) else epoch_state
            )
            self._data_epoch_sampler_state = sampler_state
            data_iter = iter(self.train_dataloader)
            for _ in range(batch_in_epoch):
                try:
                    next(data_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        "Checkpoint batch cursor'u DataLoader uzunluğunu aşıyor"
                    ) from exc
            self._data_batch_in_epoch = batch_in_epoch
            self._resume_data_state = None
            return data_iter

        if hasattr(sampler, "set_training_step"):
            sampler.set_training_step(self.global_step)
        self._data_epoch_generator_state = (
            generator.get_state().clone() if generator is not None else None
        )
        self._data_epoch_sampler_state = (
            sampler.state_dict() if hasattr(sampler, "state_dict") else None
        )
        self._data_batch_in_epoch = 0
        return iter(self.train_dataloader)

    def _next_training_batch(self, data_iter):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = self._create_data_iterator()
            batch = next(data_iter)
        self._data_batch_in_epoch += 1
        return batch, data_iter

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

        # Ünlü Uyumu Loss — cihaza taşı ve warmup ayarla
        if self.vowel_harmony_loss is not None:
            self.vowel_harmony_loss.to(self.device)
            if resume_from:
                self.vowel_harmony_loss.start_step = self.global_step
                print(f"  ✓ Ünlü uyumu loss warmup: step {self.global_step} → {self.global_step + self.vowel_harmony_loss.warmup_steps}")

        # Morfolojik Ağırlıklı Kayıp — cihaza taşı ve warmup ayarla
        if self.morph_weight_loss is not None:
            self.morph_weight_loss.to(self.device)
            if resume_from:
                self.morph_weight_loss.start_step = self.global_step
                print(f"  ✓ Morfolojik ağırlık warmup: step {self.global_step} → {self.global_step + self.morph_weight_loss.warmup_steps}")

        # Ünsüz Benzeşmesi Loss — cihaza taşı ve warmup ayarla
        if self.consonant_harmony_loss is not None:
            self.consonant_harmony_loss.to(self.device)
            if resume_from:
                self.consonant_harmony_loss.start_step = self.global_step
                print(f"  ✓ Ünsüz benzeşmesi loss warmup: step {self.global_step} → {self.global_step + self.consonant_harmony_loss.warmup_steps}")

        # Hece ve Kafiye Loss — cihaza taşı ve warmup ayarla
        if self.syllable_rhyme_loss is not None:
            self.syllable_rhyme_loss.to(self.device)
            if resume_from:
                self.syllable_rhyme_loss.start_step = self.global_step
                print(f"  ✓ Hece ve kafiye loss warmup: step {self.global_step} → {self.global_step + self.syllable_rhyme_loss.warmup_steps}")

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
        print(f"  torch.compile:       {'✅' if self.compiled else '❌'}")
        print(f"  Grad Checkpoint:     {'✅' if hasattr(self.model, 'gradient_checkpointing') and self.model.gradient_checkpointing else '❌'}")
        print(f"  TensorBoard:         {'✅' if self.writer else '❌'}")
        print(f"  Ünlü Uyumu Loss:     {'✅ (λ=' + str(self.vowel_harmony_loss.lambda_weight) + ')' if self.vowel_harmony_loss else '❌'}")
        print(f"  Morph Ağırlık:       {'✅ (w=' + str(self.morph_weight_loss.suffix_weight) + ')' if self.morph_weight_loss else '❌'}")
        print(f"  Ünsüz Benzeşmesi:    {'✅ (λ=' + str(self.consonant_harmony_loss.lambda_weight) + ')' if self.consonant_harmony_loss else '❌'}")
        print(f"  Hece & Kafiye Loss:  {'✅ (λ_hece=' + str(self.syllable_rhyme_loss.lambda_syllable) + ', λ_kaf=' + str(self.syllable_rhyme_loss.lambda_rhyme) + ')' if self.syllable_rhyme_loss else '❌'}")
        
        # Morfolojik Başlık durumunu orijinal modelden çek
        model_orig = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        use_mh = getattr(model_orig, 'use_morph_head', False)
        mh_lambda = getattr(model_orig, 'morph_lambda', 0.2)
        print(f"  Morfolojik Başlık:   {'✅ (λ=' + str(mh_lambda) + ')' if use_mh else '❌'}")
        
        if self.writer:
            print(f"")
            print(f"  📊 Eğitim loglarını takip etmek için yeni bir terminalde:")
            print(f"     tensorboard --logdir {self.writer.log_dir}")
            print(f"     Tarayıcıda: http://localhost:6006")
        print(f"{'='*60}\n")

        accumulation_loss = 0.0
        accumulation_vh_loss = 0.0
        accumulation_ch_loss = 0.0
        accumulation_mh_loss = 0.0
        accumulation_sr_syllable_loss = 0.0
        accumulation_sr_rhyme_loss = 0.0
        start_time = time.time()
        step_start_time = time.time()
        tokens_processed = 0
        nan_in_batch = False

        data_iter = self._create_data_iterator()
        step_rng_state = self._capture_rng_state()
        step_data_state = self._capture_data_state()

        try:
            while self.global_step < self.config.max_steps:
                # Kesinti optimizer adımının ortasında olursa son tamamlanmış
                # adımın RNG/veri cursor'una geri dönülebilsin.
                step_rng_state = self._capture_rng_state()
                step_data_state = self._capture_data_state()
                self.optimizer.zero_grad()

                # Gradient accumulation
                nan_in_batch = False
                for micro_step in range(self.config.grad_accum_steps):
                    batch, data_iter = self._next_training_batch(data_iter)

                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    # Forward pass
                    # NOT: MPS'de bfloat16 autocast, RoPE complex tensor
                    # işlemleriyle uyumsuz ve nan üretiyor. MPS zaten kendi
                    # optimizasyonlarını yapıyor, bu yüzden float32 kullanıyoruz.
                    amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
                    if self.morph_weight_loss is not None:
                        # Morfolojik ağırlıklı kayıp — CE loss'u ayrı hesapla
                        if self.device == "cuda":
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                logits, morph_head_loss, _ = self.model(
                                    input_ids,
                                    targets=labels,
                                    compute_lm_loss=False,
                                )
                                loss = self.morph_weight_loss(logits, labels, self.global_step)
                                loss = loss + morph_head_loss
                        else:
                            logits, morph_head_loss, _ = self.model(
                                input_ids,
                                targets=labels,
                                compute_lm_loss=False,
                            )
                            loss = self.morph_weight_loss(logits, labels, self.global_step)
                            loss = loss + morph_head_loss
                    else:
                        if self.device == "cuda":
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                logits, loss, _ = self.model(input_ids, targets=labels)
                        else:
                            # MPS ve CPU — float32 (MPS autocast nan üretiyor)
                            logits, loss, _ = self.model(input_ids, targets=labels)

                    # Ünlü Uyumu Auxiliary Loss
                    if self.vowel_harmony_loss is not None:
                        vh_loss = self.vowel_harmony_loss(logits, labels, self.global_step)
                        loss = loss + vh_loss
                        accumulation_vh_loss += vh_loss.item() / self.config.grad_accum_steps

                    # Ünsüz Benzeşmesi Auxiliary Loss
                    if self.consonant_harmony_loss is not None:
                        ch_loss = self.consonant_harmony_loss(logits, labels, self.global_step)
                        loss = loss + ch_loss
                        accumulation_ch_loss += ch_loss.item() / self.config.grad_accum_steps

                    # Hece ve Kafiye Auxiliary Loss
                    if self.syllable_rhyme_loss is not None:
                        sr_syllable_loss, sr_rhyme_loss = self.syllable_rhyme_loss(logits, labels, self.global_step)
                        loss = loss + sr_syllable_loss + sr_rhyme_loss
                        accumulation_sr_syllable_loss += sr_syllable_loss.item() / self.config.grad_accum_steps
                        accumulation_sr_rhyme_loss += sr_rhyme_loss.item() / self.config.grad_accum_steps

                    # Morfolojik Başlık metriği. Kayıp model forward'ında
                    # eklenir; morph-weight yolunda da targets geçirildiği için
                    # başlık aynı şekilde eğitilir.
                    if getattr(model_orig, 'use_morph_head', False):
                        accumulation_mh_loss += (
                            model_orig._last_morph_loss
                            / self.config.grad_accum_steps
                        )

                    # NaN/Inf loss kontrolü
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_in_batch = True
                        break

                    # Gradient accumulation: loss'u böl
                    loss = loss / self.config.grad_accum_steps
                    self.grad_scaler.scale(loss).backward()
                    accumulation_loss += loss.item()
                    tokens_processed += input_ids.numel()

                # NaN batch kontrolü — gradient step'i atla
                if nan_in_batch:
                    self.optimizer.zero_grad()  # Corrupt gradient'ları temizle
                    self.consecutive_nan_count += 1
                    lr = self.scheduler.step()  # Scheduler'ı yine ilerlet
                    self.global_step += 1
                    print(
                        f"  ⚠ Step {self.global_step}: NaN loss tespit edildi, "
                        f"gradient atlanıyor ({self.consecutive_nan_count}/{self.max_consecutive_nan})"
                    )
                    if self.consecutive_nan_count >= self.max_consecutive_nan:
                        print(f"\n❌ Arka arkaya {self.max_consecutive_nan} NaN loss!")
                        print(f"   Olası sebepler:")
                        print(f"   - Learning rate çok yüksek (şu anki: {lr:.2e})")
                        print(f"   - Veri bozuk (pad/unk token oranı çok yüksek)")
                        print(f"   - Model ağırlıkları patladı")
                        print(f"   Eğitim durduruluyor.")
                        break
                    accumulation_loss = 0.0
                    accumulation_vh_loss = 0.0
                    accumulation_ch_loss = 0.0
                    accumulation_mh_loss = 0.0
                    accumulation_sr_syllable_loss = 0.0
                    accumulation_sr_rhyme_loss = 0.0
                    continue

                # Başarılı step — nan sayacını sıfırla
                self.consecutive_nan_count = 0

                # Gradient clipping
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

                # NaN gradient kontrolü
                if isinstance(grad_norm, torch.Tensor) and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.optimizer.zero_grad()
                    self.grad_scaler.update()
                    lr = self.scheduler.step()
                    self.global_step += 1
                    print(f"  ⚠ Step {self.global_step}: NaN gradient norm, step atlanıyor")
                    accumulation_loss = 0.0
                    accumulation_vh_loss = 0.0
                    accumulation_ch_loss = 0.0
                    accumulation_mh_loss = 0.0
                    accumulation_sr_syllable_loss = 0.0
                    accumulation_sr_rhyme_loss = 0.0
                    continue

                # Optimizer step
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                lr = self.scheduler.step()
                self.global_step += 1

                # Logging (her 10 adımda)
                if self.global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    step_time = time.time() - step_start_time
                    tokens_per_sec = tokens_processed / elapsed

                    avg_loss = accumulation_loss / 10.0
                    avg_vh_loss = accumulation_vh_loss / 10.0
                    avg_ch_loss = accumulation_ch_loss / 10.0
                    avg_mh_loss = accumulation_mh_loss / 10.0
                    avg_sr_syllable_loss = accumulation_sr_syllable_loss / 10.0
                    avg_sr_rhyme_loss = accumulation_sr_rhyme_loss / 10.0

                    print(
                        f"  Step {self.global_step:>6d}/{self.config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Tok/s: {tokens_per_sec:.0f} | "
                        f"Elapsed: {elapsed/60:.1f}min"
                    )
                    self.train_losses.append((self.global_step, avg_loss))

                    # TensorBoard logging
                    if self.writer:
                        self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                        self.writer.add_scalar("train/learning_rate", lr, self.global_step)
                        self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.global_step)
                        self.writer.add_scalar("train/grad_norm", grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, self.global_step)
                        self.writer.add_scalar("train/epoch_time_min", elapsed / 60, self.global_step)
                        sampler = self._training_sampler()
                        if hasattr(sampler, "weights_at"):
                            for group, weight in sampler.weights_at(self.global_step).items():
                                self.writer.add_scalar(
                                    f"data_mixture/{group}_weight",
                                    weight,
                                    self.global_step,
                                )
                        if self.vowel_harmony_loss is not None:
                            self.writer.add_scalar("train/vh_loss", avg_vh_loss, self.global_step)
                        if self.consonant_harmony_loss is not None:
                            self.writer.add_scalar("train/ch_loss", avg_ch_loss, self.global_step)
                        if self.syllable_rhyme_loss is not None:
                            self.writer.add_scalar("train/sr_syllable_loss", avg_sr_syllable_loss, self.global_step)
                            self.writer.add_scalar("train/sr_rhyme_loss", avg_sr_rhyme_loss, self.global_step)
                        if getattr(model_orig, 'use_morph_head', False):
                            self.writer.add_scalar("train/mh_loss", avg_mh_loss, self.global_step)
                        if self.morph_weight_loss is not None:
                            self.writer.add_scalar("train/root_loss", self.morph_weight_loss._last_root_loss, self.global_step)
                            self.writer.add_scalar("train/suffix_loss", self.morph_weight_loss._last_suffix_loss, self.global_step)
                            self.writer.add_scalar("train/suffix_weight_effective", self.morph_weight_loss._effective_weight, self.global_step)

                    step_start_time = time.time()
                    accumulation_loss = 0.0  # Sadece loglama sonrası sıfırla
                    accumulation_vh_loss = 0.0
                    accumulation_ch_loss = 0.0
                    accumulation_mh_loss = 0.0
                    accumulation_sr_syllable_loss = 0.0
                    accumulation_sr_rhyme_loss = 0.0
                # Loglama step'i değilse — sonraki 10-step döngüsü için biriktirmeye devam et

                # Checkpoint kaydet
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

                    # Eval
                    if self.eval_dataloader:
                        eval_loss = self.evaluate()
                        print(f"  📊 Eval Loss: {eval_loss:.4f}")

                        if self.writer:
                            self.writer.add_scalar("eval/loss", eval_loss, self.global_step)
                            import math
                            self.writer.add_scalar("eval/perplexity", math.exp(eval_loss), self.global_step)

                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint(tag="best")
                            print(f"  🏆 Yeni en iyi model kaydedildi!")
                        self.model.train()

        except KeyboardInterrupt:
            print(f"\n{'='*60}")
            print(f"🛑 Eğitim kullanıcı tarafından durduruldu (Ctrl+C)")
            print(f"  Son step: {self.global_step}")
            print(f"  💾 Son checkpoint kaydediliyor...")
            self.optimizer.zero_grad()
            self.save_checkpoint(
                rng_state=step_rng_state,
                data_state=step_data_state,
            )
            print(f"  ✓ Güvenli çıkış yapıldı.")
            print(f"  📌 Devam etmek için:")
            print(f"     --resume checkpoints/toprak_step_{self.global_step}.pt")
            print(f"{'='*60}")
            if self.writer:
                self.writer.close()
            return

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Eğitim tamamlandı!")
        print(f"  Toplam süre: {total_time/3600:.1f} saat")
        print(f"  Son loss: {self.train_losses[-1][1] if self.train_losses else 'N/A'}")
        print(f"  En iyi eval loss: {self.best_eval_loss:.4f}")
        print(f"{'='*60}")

        # Her durumda son modeli ayrı bir checkpoint olarak kaydet
        # (ör. kısa eğitimlerde eval hiç çalışmasa bile inference için hazır olsun)
        self.save_checkpoint(tag="last")

        if self.writer:
            self.writer.close()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Eval seti üzerinde loss hesapla."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss, _ = self.model(input_ids, targets=labels)
            else:
                _, loss, _ = self.model(input_ids, targets=labels)
            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 50:  # Eval'i 50 batch ile sınırla
                break

        return total_loss / max(num_batches, 1)

    def save_checkpoint(
        self,
        tag: Optional[str] = None,
        rng_state: Optional[dict] = None,
        data_state: Optional[dict] = None,
    ):
        """Checkpoint kaydet."""
        if tag:
            filename = f"toprak_{tag}.pt"
        else:
            filename = f"toprak_step_{self.global_step}.pt"

        filepath = os.path.join(self.checkpoint_dir, filename)

        # torch.compile model'den orijinal state_dict çıkar
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_save = self.model._orig_mod

        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "grad_scaler_state_dict": self.grad_scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "training_recipe": self.training_recipe,
            "experiment_manifest": self.experiment_manifest,
            "rng_state": rng_state or self._capture_rng_state(),
            "data_state": data_state or self._capture_data_state(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "num_heads": self.config.num_heads,
                "num_kv_heads": self.config.num_kv_heads,
                "num_layers": self.config.num_layers,
                "d_ff": self.config.d_ff,
                "max_seq_len": self.config.max_seq_len,
                "rope_theta": self.config.rope_theta,
                "norm_eps": self.config.norm_eps,
            },
        }

        torch.save(checkpoint, filepath)
        print(f"  💾 Checkpoint kaydedildi: {filepath}")

        # Eski checkpoint'leri temizle
        self._cleanup_checkpoints()

    def load_checkpoint(self, filepath: str):
        """Checkpoint'ten devam et."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        digest = hashlib.sha256()
        with open(filepath, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.training_recipe.update({
            "resume_checkpoint_sha256": digest.hexdigest(),
            "resume_global_step": checkpoint.get("global_step"),
            "resume_scheduler_state": checkpoint.get("scheduler_state_dict"),
        })
        if self.experiment_manifest:
            self.experiment_manifest["training_recipe"] = self.training_recipe
            manifest_dirs = [self.checkpoint_dir]
            if self.writer is not None:
                manifest_dirs.append(self.writer.log_dir)
            write_manifest(self.experiment_manifest, *manifest_dirs)

        # torch.compile model'e yükleme
        model_to_load = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_load = self.model._orig_mod

        model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("grad_scaler_state_dict")
        if scaler_state:
            self.grad_scaler.load_state_dict(scaler_state)

        # Optimizer state'lerini doğru device'a taşı (CPU → MPS fix)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))
        self._resume_data_state = checkpoint.get("data_state")
        self._restore_rng_state(checkpoint.get("rng_state"))

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
