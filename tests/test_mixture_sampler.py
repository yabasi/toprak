# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.dataset import ToprakShardDataset, create_dataloader
from data.mixture import (
    CurriculumMixtureSampler,
    legacy_curriculum_config,
    resolve_group,
    validate_mixture_config,
)
from training.trainer import ToprakTrainer
from scripts.pretokenize import tokenize_corpus


class GroupedDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        high_end = size // 5
        self.group_ranges = {
            "high_quality": [(0, high_end)],
            "general": [(high_end, size)],
        }

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"value": torch.tensor(index)}


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        return [3 + (ord(char) % 29) for char in text]

    def get_vocab_size(self):
        return 64


def cursor_trainer(loader):
    trainer = ToprakTrainer.__new__(ToprakTrainer)
    trainer.train_dataloader = loader
    trainer.global_step = 0
    trainer._data_epoch_generator_state = None
    trainer._data_epoch_sampler_state = None
    trainer._data_batch_in_epoch = 0
    trainer._resume_data_state = None
    return trainer


class TestMixtureSampler(unittest.TestCase):
    def setUp(self):
        self.config = legacy_curriculum_config(
            ["wiki"], curriculum_steps=100, initial_high_weight=0.8,
            final_high_weight=0.4,
        )

    def test_schedule_and_source_resolution(self):
        sampler = CurriculumMixtureSampler(GroupedDataset(), self.config)
        self.assertAlmostEqual(sampler.weights_at(0)["high_quality"], 0.8)
        self.assertAlmostEqual(sampler.weights_at(50)["high_quality"], 0.6)
        self.assertAlmostEqual(sampler.weights_at(100)["high_quality"], 0.4)
        self.assertEqual(resolve_group("wiki", self.config), "high_quality")
        self.assertEqual(resolve_group("fineweb2", self.config), "general")

    def test_sampling_is_deterministic_and_weighted(self):
        dataset = GroupedDataset()
        first = CurriculumMixtureSampler(
            dataset, self.config, seed=7, samples_per_step=100_000,
            num_samples=2000, chunk_size=100,
        )
        second = CurriculumMixtureSampler(
            dataset, self.config, seed=7, samples_per_step=100_000,
            num_samples=2000, chunk_size=100,
        )
        first_indices = list(first)
        self.assertEqual(first_indices, list(second))
        high_ratio = sum(index < 20 for index in first_indices) / len(first_indices)
        # 2000 örneklik deterministik çekimde başlangıç ağırlığı çevresinde olmalı.
        self.assertGreater(high_ratio, 0.72)
        self.assertLess(high_ratio, 0.88)

    def test_sampler_cursor_resumes_exactly(self):
        dataset = GroupedDataset(size=20)
        first_sampler = CurriculumMixtureSampler(
            dataset, self.config, seed=11, samples_per_step=2, chunk_size=2
        )
        first_loader = create_dataloader(
            dataset, batch_size=2, shuffle=False, seed=11, sampler=first_sampler
        )
        first = cursor_trainer(first_loader)
        iterator = first._create_data_iterator()
        for _ in range(4):
            _, iterator = first._next_training_batch(iterator)
        state = first._capture_data_state()
        expected = []
        for _ in range(3):
            batch, iterator = first._next_training_batch(iterator)
            expected.extend(batch["value"].tolist())

        second_sampler = CurriculumMixtureSampler(
            dataset, self.config, seed=11, samples_per_step=2, chunk_size=2
        )
        second_loader = create_dataloader(
            dataset, batch_size=2, shuffle=False, seed=11, sampler=second_sampler
        )
        second = cursor_trainer(second_loader)
        second._resume_data_state = state
        iterator = second._create_data_iterator()
        actual = []
        for _ in range(3):
            batch, iterator = second._next_training_batch(iterator)
            actual.extend(batch["value"].tolist())
        self.assertEqual(actual, expected)

    def test_invalid_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "default"):
            validate_mixture_config({
                "groups": {"only": {"initial_weight": 1, "final_weight": 1}}
            })
        with self.assertRaisesRegex(ValueError, "birden fazla"):
            validate_mixture_config({
                "groups": {
                    "a": {"sources": ["wiki"], "default": True},
                    "b": {"sources": ["wiki"]},
                }
            })

    def test_pretokenize_writes_grouped_manifest_and_dataset_ranges(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            with (input_dir / "docs.jsonl").open("w", encoding="utf-8") as handle:
                for source, text in (
                    ("wiki", "Anadolu tarihi"),
                    ("wiki", "Türkçe ansiklopedi"),
                    ("fineweb2", "Güncel web metni"),
                    ("culturax", "Genel derlem metni"),
                ):
                    handle.write(json.dumps({"source": source, "text": text}) + "\n")

            tokenize_corpus(
                str(input_dir), FakeTokenizer(), str(output_dir),
                shard_size=32, eval_ratio=0, curriculum=True,
                high_quality_sources=("wiki",), curriculum_steps=100,
            )

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["curriculum"])
            self.assertEqual(
                {shard["group"] for shard in manifest["train"]["shards"]},
                {"high_quality", "general"},
            )
            self.assertEqual(
                manifest["train"]["group_docs"],
                {"high_quality": 2, "general": 2},
            )
            dataset = ToprakShardDataset(
                str(output_dir), "train", max_seq_len=4, expected_vocab_size=64
            )
            self.assertTrue(dataset.group_ranges["high_quality"])
            self.assertTrue(dataset.group_ranges["general"])


if __name__ == "__main__":
    unittest.main()
