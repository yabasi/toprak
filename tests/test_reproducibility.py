# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import json
import os
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import create_dataloader
from model.config import ModelConfig
from model.transformer import ToprakLM
from training.trainer import ToprakTrainer
from utils.reproducibility import (
    build_experiment_manifest,
    fingerprint_data,
    seed_everything,
    write_manifest,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NumberDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {"value": torch.tensor(index)}


class LanguageModelDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        values = [4 + ((index + offset) % 12) for offset in range(5)]
        return {
            "input_ids": torch.tensor(values[:-1], dtype=torch.long),
            "labels": torch.tensor(values[1:], dtype=torch.long),
        }


class TinyTokenizer:
    def id_to_token(self, token_id):
        return f"▁t{token_id}"


def tiny_config():
    return ModelConfig(
        vocab_size=16,
        d_model=8,
        num_heads=2,
        num_kv_heads=1,
        num_layers=1,
        d_ff=16,
        max_seq_len=4,
        device="cpu",
        learning_rate=1e-3,
        warmup_steps=1,
        max_steps=4,
        batch_size=2,
        grad_accum_steps=1,
        save_every=2,
        keep_last_n=3,
    )


def cursor_trainer(loader):
    trainer = ToprakTrainer.__new__(ToprakTrainer)
    trainer.train_dataloader = loader
    trainer._data_epoch_generator_state = None
    trainer._data_epoch_sampler_state = None
    trainer._data_batch_in_epoch = 0
    trainer._resume_data_state = None
    return trainer


class TestReproducibility(unittest.TestCase):
    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            seed_everything(2026, deterministic=True)
            full_config = tiny_config()
            full_model = ToprakLM(full_config, tokenizer=TinyTokenizer())
            full_loader = create_dataloader(
                LanguageModelDataset(), batch_size=2, shuffle=True, seed=19
            )
            full_trainer = ToprakTrainer(
                full_model,
                full_config,
                full_loader,
                checkpoint_dir=os.path.join(temp_dir, "full"),
                log_dir=os.path.join(temp_dir, "full_logs"),
                use_compile=False,
                use_gradient_checkpointing=False,
            )
            full_trainer.train()
            step_two = os.path.join(temp_dir, "full", "toprak_step_2.pt")
            full_state = {
                name: tensor.detach().clone()
                for name, tensor in full_model.state_dict().items()
            }

            seed_everything(999, deterministic=True)
            resumed_config = tiny_config()
            resumed_model = ToprakLM(resumed_config, tokenizer=TinyTokenizer())
            resumed_loader = create_dataloader(
                LanguageModelDataset(), batch_size=2, shuffle=True, seed=19
            )
            resumed_trainer = ToprakTrainer(
                resumed_model,
                resumed_config,
                resumed_loader,
                checkpoint_dir=os.path.join(temp_dir, "resumed"),
                log_dir=os.path.join(temp_dir, "resumed_logs"),
                use_compile=False,
                use_gradient_checkpointing=False,
            )
            resumed_trainer.train(resume_from=step_two)

            for name, tensor in resumed_model.state_dict().items():
                self.assertTrue(
                    torch.equal(tensor, full_state[name]),
                    f"Resume sonrası parametre farklı: {name}",
                )

    def test_seed_everything_repeats_all_rngs(self):
        seed_everything(123)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        first = (random.random(), float(np.random.rand()), torch.rand(3))
        seed_everything(123)
        second = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_rng_state_roundtrip(self):
        trainer = ToprakTrainer.__new__(ToprakTrainer)
        seed_everything(77)
        state = trainer._capture_rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(2))
        trainer._restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), torch.rand(2))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_resume_replays_same_dataloader_cursor(self):
        first_loader = create_dataloader(
            NumberDataset(), batch_size=2, shuffle=True, drop_last=False, seed=9
        )
        first = cursor_trainer(first_loader)
        iterator = first._create_data_iterator()
        for _ in range(3):
            _, iterator = first._next_training_batch(iterator)
        state = first._capture_data_state()
        expected = []
        for _ in range(3):
            batch, iterator = first._next_training_batch(iterator)
            expected.extend(batch["value"].tolist())

        second_loader = create_dataloader(
            NumberDataset(), batch_size=2, shuffle=True, drop_last=False, seed=9
        )
        second = cursor_trainer(second_loader)
        second._resume_data_state = state
        iterator = second._create_data_iterator()
        actual = []
        for _ in range(3):
            batch, iterator = second._next_training_batch(iterator)
            actual.extend(batch["value"].tolist())
        self.assertEqual(actual, expected)

    def test_full_fingerprint_changes_with_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "data.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("bir")
            first = fingerprint_data(temp_dir, "full")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("iki")
            second = fingerprint_data(temp_dir, "full")
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_experiment_manifest_records_runtime_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_path = os.path.join(temp_dir, "tokenizer.model")
            data_path = os.path.join(temp_dir, "data.txt")
            with open(tokenizer_path, "wb") as handle:
                handle.write(b"tokenizer")
            with open(data_path, "w", encoding="utf-8") as handle:
                handle.write("deney verisi")
            manifest = build_experiment_manifest(
                PROJECT_ROOT,
                {"seed": 42},
                tokenizer_path,
                data_path,
                "full",
                ["training/train.py"],
            )
            paths = write_manifest(manifest, os.path.join(temp_dir, "run"))
            with open(paths[0], "r", encoding="utf-8") as handle:
                written = json.load(handle)
        self.assertEqual(written["training_recipe"]["seed"], 42)
        self.assertIsNotNone(written["tokenizer"]["sha256"])
        self.assertIsNotNone(written["data"]["sha256"])
        self.assertIn("torch", written["runtime"]["packages"])


if __name__ == "__main__":
    unittest.main()
