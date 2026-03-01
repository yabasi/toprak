"""
Toprak — LR Scheduler
Cosine Annealing with Linear Warmup.
"""

import math


class CosineWarmupScheduler:
    """
    Learning rate scheduler: linear warmup → cosine annealing.

    Warmup süresi boyunca LR 0'dan max_lr'a lineer artar,
    ardından cosine eğrisi ile min_lr'a iner.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float = 3e-4,
        min_lr: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        """Mevcut adım için learning rate hesapla."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * (self.current_step / max(self.warmup_steps, 1))
        elif self.current_step >= self.max_steps:
            return self.min_lr
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

    def step(self):
        """Bir adım ilerle ve LR'yi güncelle."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr

    def state_dict(self) -> dict:
        """Scheduler durumunu kaydet."""
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict: dict):
        """Scheduler durumunu yükle."""
        self.current_step = state_dict["current_step"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.max_steps = state_dict["max_steps"]
        self.max_lr = state_dict["max_lr"]
        self.min_lr = state_dict["min_lr"]
