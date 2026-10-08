# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from evaluation.compare_lm_eval import (
    build_table,
    check_compatibility,
    load_report,
    normalized_gain,
)
from evaluation.run_lm_eval import TASK_PRESETS
from scripts.run_baselines import build_command, report_path, select_models

ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "configs", "lm_eval_baselines.json")


def make_report(name, scores, model_type="hf", params=100_000_000, **settings):
    base = {"xcopa_tr": 0.5, "belebele_tur_Latn": 0.25}
    return {
        "lm_eval_version": "0.4.13",
        "model": {"type": model_type, "name": name, "num_parameters": params},
        "settings": {"num_fewshot": None, "seed": 1234, "max_length": 1024,
                     "limit": None, **settings},
        "versions": {task: 1.0 for task in scores},
        "summary": [
            {"task": task, "metric": "acc", "value": value, "stderr": 0.01,
             "random_baseline": base[task]}
            for task, value in scores.items()
        ],
    }


class BaselineConfigTest(unittest.TestCase):
    def setUp(self):
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            self.config = json.load(handle)

    def test_config_is_well_formed(self):
        self.assertIn(self.config["preset"], TASK_PRESETS)
        ids = [m["id"] for m in self.config["models"]]
        self.assertEqual(len(ids), len(set(ids)))
        for model in self.config["models"]:
            self.assertIn(model["tier"], self.config["tiers"])

    def test_select_models_by_tier_and_id(self):
        small = select_models(self.config, tiers=["small"])
        self.assertTrue(small)
        self.assertTrue(all(m["tier"] == "small" for m in small))
        picked = select_models(self.config, models=["Qwen/Qwen2.5-0.5B"])
        self.assertEqual([m["id"] for m in picked], ["Qwen/Qwen2.5-0.5B"])
        with self.assertRaises(ValueError):
            select_models(self.config, models=["yok/model"])

    def test_build_command_passes_backend_and_shared_settings(self):
        entry = {"id": "boun-tabi-LMG/TURNA", "backend": "seq2seq"}
        args = SimpleNamespace(preset=None, batch_size=4, limit=None,
                               max_length=512, device="cpu")
        cmd = build_command(entry, self.config, args, "out.json")
        self.assertEqual(cmd[cmd.index("--hf-backend") + 1], "seq2seq")
        self.assertEqual(cmd[cmd.index("--preset") + 1], self.config["preset"])
        self.assertEqual(cmd[cmd.index("--max-length") + 1], "512")
        self.assertNotIn("--limit", cmd)

    def test_report_path_is_filesystem_safe(self):
        self.assertEqual(
            os.path.basename(report_path("/tmp", "Qwen/Qwen2.5-0.5B")),
            "Qwen__Qwen2.5-0.5B.json",
        )


class CompareReportsTest(unittest.TestCase):
    def test_normalized_gain(self):
        self.assertAlmostEqual(normalized_gain(0.5, 0.5), 0.0)
        self.assertAlmostEqual(normalized_gain(1.0, 0.25), 1.0)
        self.assertAlmostEqual(normalized_gain(0.625, 0.25), 0.5)
        self.assertIsNone(normalized_gain(0.5, None))

    def test_table_sorts_by_gain_and_bolds_toprak(self):
        reports = [
            make_report("weak", {"xcopa_tr": 0.52, "belebele_tur_Latn": 0.26}),
            make_report("toprak_last", {"xcopa_tr": 0.60, "belebele_tur_Latn": 0.40},
                        model_type="toprak"),
        ]
        table = build_table(reports)
        lines = table.splitlines()
        self.assertTrue(lines[2].startswith("| **toprak_last**"))
        self.assertIn("weak", lines[3])
        self.assertIn("Şans seviyesi", lines[-1])

    def test_incomplete_model_has_no_average(self):
        reports = [
            make_report("full", {"xcopa_tr": 0.6, "belebele_tur_Latn": 0.3}),
            make_report("partial", {"xcopa_tr": 0.9}),
        ]
        partial_line = [l for l in build_table(reports).splitlines() if "partial" in l][0]
        self.assertTrue(partial_line.rstrip(" |").endswith("-"))

    def test_compatibility_warnings(self):
        reports = [
            make_report("a", {"xcopa_tr": 0.6}),
            make_report("b", {"xcopa_tr": 0.6}, max_length=512, num_fewshot=5),
        ]
        warnings = "\n".join(check_compatibility(reports))
        self.assertIn("num_fewshot", warnings)
        self.assertIn("Bağlam", warnings)

    def test_limited_reports_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(make_report("x", {"xcopa_tr": 0.5}, limit=10), handle)
            with self.assertRaises(ValueError):
                load_report(path)


if __name__ == "__main__":
    unittest.main()
