# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import copy
import os
import tempfile
import unittest

import torch
from torch.utils.data import TensorDataset

from data.dataset import create_dataloader
from evaluation.ablation import (
    bootstrap_mean_interval,
    build_ablation_report,
    paired_deltas,
    render_markdown,
    validate_ablation_pair,
)
from model.config import ModelConfig
from model.transformer import ToprakLM
from training.trainer import ToprakTrainer


def make_report(name="baseline", scores=(0.0, 1.0), seed=42):
    results = [
        {
            "id": f"sample-{index}",
            "benchmark": "seed",
            "category": "morphology" if index == 0 else "grammar",
            "type": "multiple_choice",
            "score": score,
            "correct": bool(score),
        }
        for index, score in enumerate(scores)
    ]
    categories = {
        result["category"]: {
            "count": 1,
            "mean_score": result["score"],
            "correct": int(result["correct"]),
        }
        for result in results
    }
    return {
        "suite_version": "toprak-eval-v1",
        "summary": {
            "samples": len(results),
            "micro_score": sum(scores) / len(scores),
            "macro_score": sum(scores) / len(scores),
            "correct": sum(bool(score) for score in scores),
        },
        "categories": categories,
        "results": results,
        "metadata": {
            "experiment_name": name,
            "tokenizer_sha256": "tokenizer",
            "benchmark_sha256": {"seed.jsonl": "benchmark"},
            "model_config": {"d_model": 16},
            "global_step": 100,
            "data_fingerprint_sha256": "data",
            "training_recipe": {
                "experiment_name": name,
                "data_dir": "/data",
                "seed": seed,
                "max_steps": 100,
                "auxiliary_losses": {
                    "vowel_harmony": {"enabled": name != "baseline"}
                },
            },
        },
    }


class TestAblation(unittest.TestCase):
    def test_training_recipe_is_persisted_in_checkpoint(self):
        config = ModelConfig(
            vocab_size=16,
            d_model=8,
            num_heads=2,
            num_kv_heads=1,
            num_layers=1,
            d_ff=16,
            max_seq_len=8,
            device="cpu",
        )
        recipe = {"experiment_name": "ablation-baseline", "seed": 42}
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ToprakTrainer(
                model=ToprakLM(config),
                config=config,
                train_dataloader=[],
                checkpoint_dir=temp_dir,
                log_dir=os.path.join(temp_dir, "logs"),
                use_compile=False,
                use_gradient_checkpointing=False,
                training_recipe=recipe,
            )
            trainer.save_checkpoint(tag="recipe")
            if trainer.writer:
                trainer.writer.close()
            checkpoint = torch.load(
                os.path.join(temp_dir, "toprak_recipe.pt"),
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(checkpoint["training_recipe"], recipe)
        self.assertIn("rng_state", checkpoint)
        self.assertIn("data_state", checkpoint)

    def test_seeded_dataloader_order_is_repeatable(self):
        dataset = TensorDataset(torch.arange(20))
        first = create_dataloader(
            dataset, batch_size=4, shuffle=True, drop_last=False, seed=123
        )
        second = create_dataloader(
            dataset, batch_size=4, shuffle=True, drop_last=False, seed=123
        )
        first_order = torch.cat([batch[0] for batch in first]).tolist()
        second_order = torch.cat([batch[0] for batch in second]).tolist()
        self.assertEqual(first_order, second_order)

    def test_paired_report_and_markdown(self):
        baseline = make_report()
        candidate = make_report("vowel", (1.0, 1.0))
        report = build_ablation_report(
            "baseline.json", baseline, {"vowel": candidate},
            bootstrap_samples=100, seed=7,
        )
        paired = report["candidates"]["vowel"]["paired"]
        self.assertEqual(paired["wins"], 1)
        self.assertEqual(paired["ties"], 1)
        self.assertEqual(paired["losses"], 0)
        self.assertAlmostEqual(paired["mean_delta"], 0.5)
        self.assertIn("vowel", render_markdown(report))

    def test_bootstrap_is_deterministic(self):
        first = bootstrap_mean_interval([1.0, 0.0, -1.0], samples=100, seed=9)
        second = bootstrap_mean_interval([1.0, 0.0, -1.0], samples=100, seed=9)
        self.assertEqual(first, second)

    def test_recipe_mismatch_is_rejected(self):
        baseline = make_report()
        candidate = make_report("vowel", seed=99)
        with self.assertRaisesRegex(ValueError, "eğitim tarifleri"):
            validate_ablation_pair(candidate, baseline)

    def test_sample_mismatch_is_rejected(self):
        baseline = make_report()
        candidate = make_report("vowel")
        candidate["results"] = candidate["results"][:-1]
        with self.assertRaisesRegex(ValueError, "örnek ID"):
            paired_deltas(candidate, baseline)

    def test_global_step_mismatch_is_rejected(self):
        baseline = make_report()
        candidate = copy.deepcopy(make_report("vowel"))
        candidate["metadata"]["global_step"] = 101
        with self.assertRaisesRegex(ValueError, "adımları"):
            validate_ablation_pair(candidate, baseline)

    def test_data_fingerprint_mismatch_is_rejected(self):
        baseline = make_report()
        candidate = copy.deepcopy(make_report("vowel"))
        candidate["metadata"]["data_fingerprint_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "veri parmak"):
            validate_ablation_pair(candidate, baseline)


if __name__ == "__main__":
    unittest.main()
