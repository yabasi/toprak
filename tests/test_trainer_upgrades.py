# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Uçtan uca eğitim testi: MTP + MoE + morfolojik başlık açıkken ToprakTrainer
çalışır, checkpoint tam mimari config'i taşır, inference yükleyicisi modeli
birebir kurar ve kaldığı yerden devam bit-düzeyinde aynı sonucu verir.
"""

import os
import tempfile
import unittest

import torch
from torch.utils.data import Dataset

from data.dataset import create_dataloader
from inference.generate import load_model
from model.config import ModelConfig
from model.transformer import ToprakLM
from training.trainer import ToprakTrainer
from utils.reproducibility import seed_everything


class LanguageModelDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        values = [4 + ((index + offset) % 12) for offset in range(9)]
        return {
            "input_ids": torch.tensor(values[:-1], dtype=torch.long),
            "labels": torch.tensor(values[1:], dtype=torch.long),
        }


class TinyTokenizer:
    def id_to_token(self, token_id):
        return f"▁t{token_id}" if token_id % 2 else f"ek{token_id}"


def upgraded_config():
    return ModelConfig(
        vocab_size=16, d_model=16, num_heads=2, num_kv_heads=1, num_layers=2,
        d_ff=32, max_seq_len=8, device="cpu", learning_rate=1e-3,
        warmup_steps=1, max_steps=4, batch_size=2, grad_accum_steps=1,
        save_every=2, keep_last_n=3,
        num_mtp_heads=2, num_experts=4, experts_top_k=2,
        rope_scaling={"type": "yarn", "factor": 2.0, "original_max_seq_len": 4},
    )


def build(temp_dir, name):
    config = upgraded_config()
    model = ToprakLM(config, tokenizer=TinyTokenizer())
    model.use_morph_head = True
    loader = create_dataloader(LanguageModelDataset(), batch_size=2, shuffle=True, seed=7)
    trainer = ToprakTrainer(
        model, config, loader,
        checkpoint_dir=os.path.join(temp_dir, name),
        log_dir=os.path.join(temp_dir, name + "_logs"),
        use_compile=False, use_gradient_checkpointing=False,
    )
    return model, trainer


class TestTrainerWithUpgrades(unittest.TestCase):

    def test_train_save_load_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            seed_everything(2026, deterministic=True)
            full_model, full_trainer = build(temp_dir, "full")
            full_trainer.train()
            self.assertGreater(full_model._last_mtp_loss, 0)
            self.assertGreater(full_model._last_moe_aux_loss, 0)
            full_state = {k: v.detach().clone() for k, v in full_model.state_dict().items()}

            step_two = os.path.join(temp_dir, "full", "toprak_step_2.pt")
            checkpoint = torch.load(step_two, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"]["num_mtp_heads"], 2)
            self.assertEqual(checkpoint["config"]["num_experts"], 4)
            self.assertEqual(checkpoint["config"]["rope_scaling"]["type"], "yarn")

            loaded, config = load_model(step_two, device="cpu")
            self.assertEqual(len(loaded.mtp_heads), 2)
            self.assertTrue(all(block.use_moe for block in loaded.blocks))
            loaded.load_state_dict(checkpoint["model_state_dict"], strict=True)

            seed_everything(999, deterministic=True)
            resumed_model, resumed_trainer = build(temp_dir, "resumed")
            resumed_trainer.train(resume_from=step_two)
            for name, tensor in resumed_model.state_dict().items():
                self.assertTrue(torch.equal(tensor, full_state[name]), name)


if __name__ == "__main__":
    unittest.main()
