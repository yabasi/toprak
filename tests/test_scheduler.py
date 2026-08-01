# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import unittest

import torch

from training.scheduler import CosineWarmupScheduler


class TestCosineWarmupScheduler(unittest.TestCase):
    def test_first_optimizer_step_uses_warmup_lr(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=10,
            max_steps=100,
            max_lr=1e-3,
        )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        next_lr = scheduler.step()
        self.assertAlmostEqual(next_lr, 2e-4)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-4)

    def test_loaded_state_applies_resumed_lr(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=10,
            max_steps=100,
            max_lr=1e-3,
        )
        state = scheduler.state_dict()
        state["current_step"] = 4
        scheduler.load_state_dict(state)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-4)


if __name__ == "__main__":
    unittest.main()
