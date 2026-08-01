"""Mini sanity test: ToprakShardDataset shard sınırlarını doğru çözüyor mu?"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ToprakShardDataset, create_dataloader


def test_shard_dataset_basic():
    with tempfile.TemporaryDirectory() as d:
        s1 = np.arange(1000, dtype=np.uint16)
        s1.tofile(os.path.join(d, "train_00000.bin"))
        s2 = np.arange(1500, 2500, dtype=np.uint16)
        s2.tofile(os.path.join(d, "train_00001.bin"))
        e1 = np.arange(500, dtype=np.uint16)
        e1.tofile(os.path.join(d, "eval_00000.bin"))

        manifest = {
            "tokenizer_vocab_size": 32000,
            "dtype": "uint16",
            "shard_size_tokens": 1000,
            "eval_ratio": 0.01,
            "curriculum": False,
            "high_quality_sources": [],
            "train": {
                "shards": [
                    {"path": "train_00000.bin", "tokens": 1000},
                    {"path": "train_00001.bin", "tokens": 1000},
                ],
                "total_tokens": 2000,
            },
            "eval": {
                "shards": [{"path": "eval_00000.bin", "tokens": 500}],
                "total_tokens": 500,
            },
            "total_docs": 0,
            "eval_docs": 0,
        }
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        # max_seq_len=100 -> her shard: (1000-1)//100 = 9 blok
        ds_tr = ToprakShardDataset(bin_dir=d, split="train", max_seq_len=100)
        assert len(ds_tr) == 18, f"train len yanlış: {len(ds_tr)}"

        b0 = ds_tr[0]
        assert b0["input_ids"][0].item() == 0
        assert b0["input_ids"][1].item() == 1

        # 9. blok = shard 1'in başı (1500'den başlamalı)
        b9 = ds_tr[9]
        assert b9["input_ids"][0].item() == 1500, b9["input_ids"][0].item()

        ds_ev = ToprakShardDataset(bin_dir=d, split="eval", max_seq_len=100)
        assert len(ds_ev) == 4

        print("✅ ToprakShardDataset OK (18 train + 4 eval blok, shard sınırı doğru)")


class TestShardManifestValidation(unittest.TestCase):
    def _write_fixture(self, directory, **manifest_overrides):
        tokens = np.arange(201, dtype=np.uint16)
        tokens.tofile(os.path.join(directory, "train_00000.bin"))
        manifest = {
            "tokenizer_vocab_size": 32000,
            "dtype": "uint16",
            "curriculum": True,
            "train": {
                "shards": [{"path": "train_00000.bin", "tokens": 201}],
                "total_tokens": 201,
            },
            "eval": {"shards": [], "total_tokens": 0},
        }
        manifest.update(manifest_overrides)
        with open(os.path.join(directory, "manifest.json"), "w") as f:
            json.dump(manifest, f)

    def test_manifest_metadata_is_exposed_and_validated(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_fixture(d)
            dataset = ToprakShardDataset(
                d,
                split="train",
                max_seq_len=100,
                expected_vocab_size=32000,
            )
            self.assertTrue(dataset.curriculum)
            self.assertEqual(dataset.dtype, np.dtype(np.uint16))

    def test_vocab_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_fixture(d)
            with self.assertRaisesRegex(ValueError, "vocab uyuşmazlığı"):
                ToprakShardDataset(
                    d,
                    split="train",
                    max_seq_len=100,
                    expected_vocab_size=123,
                )

    def test_declared_token_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_fixture(d)
            manifest_path = os.path.join(d, "manifest.json")
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest["train"]["shards"][0]["tokens"] = 999
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            with self.assertRaisesRegex(ValueError, "token sayısı uyuşmuyor"):
                ToprakShardDataset(d, split="train", max_seq_len=100)

    def test_eval_dataloader_keeps_incomplete_batch(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_fixture(d)
            dataset = ToprakShardDataset(d, split="train", max_seq_len=100)
            loader = create_dataloader(
                dataset,
                batch_size=8,
                shuffle=False,
                drop_last=False,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0]["input_ids"].shape[0], 2)


if __name__ == "__main__":
    test_shard_dataset_basic()
    unittest.main()
