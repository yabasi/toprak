# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Talimat ince ayarı (SFT) ve LoRA testleri."""

import json
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from model.chat_template import IGNORE_INDEX, ChatTemplate
from model.config import ModelConfig
from model.transformer import ToprakLM
from training.sft import (
    LoRALinear,
    SFTDataset,
    apply_lora,
    lora_parameters,
    merge_lora,
    merged_state_dict,
    record_to_messages,
    save_aligned_checkpoint,
    sft_collate,
    sft_loss,
    train_sft,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = 48


class MockTokenizer:
    """Karakter tabanlı sahte tokenizer (sohbet şablonu + model kurulumu için)."""

    pad_token_id, unk_token_id, bos_token_id, eos_token_id = 0, 1, 2, 3
    specials = ["<pad>", "<unk>", "<s>", "</s>", "<sep>"]

    def get_vocab_size(self):
        return V

    def id_to_token(self, token_id):
        return self.specials[token_id] if token_id < 5 else f"▁t{token_id}"

    def token_to_id(self, token):
        return self.specials.index(token) if token in self.specials else self.unk_token_id

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [5 + ord(c) % (V - 5) for c in text]
        return ([2] if add_bos else []) + ids + ([3] if add_eos else [])

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids if i > 4)


def tiny_model(seed=0, max_seq_len=64):
    torch.manual_seed(seed)
    config = ModelConfig(vocab_size=V, d_model=32, num_heads=4, num_kv_heads=2, num_layers=2,
                         d_ff=64, max_seq_len=max_seq_len, device="cpu")
    return ToprakLM(config, tokenizer=MockTokenizer())


RECORDS = [
    {"messages": [{"role": "user", "content": "selam"}, {"role": "assistant", "content": "merhaba"}]},
    {"instruction": "topla", "input": "2 3", "output": "5"},
    {"messages": [{"role": "user", "content": "yalnız kullanıcı"}]},  # öğrenilecek token yok
    {"foo": "bar"},  # geçersiz biçim
]


class TestSFTData(unittest.TestCase):

    def test_alpaca_conversion(self):
        msgs = record_to_messages({"instruction": "Çevir", "input": "hello", "output": "merhaba"})
        self.assertEqual(msgs, [
            {"role": "user", "content": "Çevir\n\nhello"},
            {"role": "assistant", "content": "merhaba"},
        ])
        msgs = record_to_messages({"instruction": "Say", "output": "bir"})
        self.assertEqual(msgs[0]["content"], "Say")

    def test_dataset_reads_jsonl_and_skips_untrainable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sft.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in RECORDS:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.write("\n")
            ds = SFTDataset(path, MockTokenizer(), max_len=64)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.skipped, 2)
        template = ChatTemplate(MockTokenizer())
        expected = template.build_labels(RECORDS[0]["messages"])
        self.assertEqual((ds[0]["input_ids"], ds[0]["labels"]), expected)
        # Etiketler yalnız asistan cevabı + tur sonu için
        trainable = [l for l in ds[0]["labels"] if l != IGNORE_INDEX]
        self.assertEqual(trainable, MockTokenizer().encode("merhaba") + [4])

    def test_max_len_truncation(self):
        long = {"messages": [{"role": "user", "content": "a" * 10},
                             {"role": "assistant", "content": "b" * 40}]}
        ds = SFTDataset([long], MockTokenizer(), max_len=40)
        self.assertEqual(len(ds), 1)
        self.assertEqual(len(ds[0]["input_ids"]), 40)
        self.assertEqual(len(ds[0]["labels"]), 40)
        # Asistan kısmı sınırın dışında kalırsa örnek atlanır
        ds = SFTDataset([long], MockTokenizer(), max_len=8)
        self.assertEqual(len(ds), 0)
        self.assertEqual(ds.skipped, 1)

    def test_collate_padding(self):
        batch = sft_collate([
            {"input_ids": [2, 5, 6], "labels": [-100, 6, 7]},
            {"input_ids": [2, 5], "labels": [-100, 9]},
        ])
        self.assertEqual(batch["input_ids"].tolist(), [[2, 5, 6], [2, 5, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 6, 7], [-100, 9, -100]])

    def test_sft_loss_matches_manual_ce(self):
        model = tiny_model().eval()
        ds = SFTDataset(RECORDS[:2], MockTokenizer(), max_len=64)
        batch = sft_collate([ds[0], ds[1]])
        loss = sft_loss(model, batch)
        logits = model(batch["input_ids"])[0]
        mask = batch["labels"] != IGNORE_INDEX
        manual = F.cross_entropy(logits[mask], batch["labels"][mask])
        self.assertAlmostEqual(loss.item(), manual.item(), places=5)

    def test_example_file_loads_with_real_tokenizer(self):
        from model.tokenizer import ToprakTokenizer

        tok = ToprakTokenizer(os.path.join(ROOT, "toprak_tokenizer.model"))
        ds = SFTDataset(os.path.join(ROOT, "alignment", "examples", "sft_sample.jsonl"), tok, max_len=512)
        self.assertEqual(len(ds), 8)
        self.assertEqual(ds.skipped, 0)


class TestLoRA(unittest.TestCase):

    def test_apply_lora_is_identity_at_init_and_freezes_base(self):
        model = tiny_model().eval()
        x = torch.randint(5, V, (2, 10))
        before = model(x)[0]
        names = apply_lora(model, r=4, alpha=8)
        self.assertEqual(len(names), 2 * 4)  # 2 katman × q/k/v/out
        after = model(x)[0]
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual({id(p) for p in trainable}, {id(p) for p in lora_parameters(model)})

    def test_merge_preserves_function(self):
        model = tiny_model().eval()
        apply_lora(model, r=4, alpha=8)
        torch.manual_seed(1)
        for p in lora_parameters(model):
            p.data.normal_(0, 0.1)
        x = torch.randint(5, V, (2, 10))
        with torch.no_grad():
            lora_out = model(x)[0]
            state = merged_state_dict(model)
            fresh = tiny_model(seed=99).eval()
            fresh.load_state_dict(state)
            self.assertTrue(torch.allclose(lora_out, fresh(x)[0], atol=1e-5))
            self.assertEqual(merge_lora(model), 8)
            self.assertFalse(any(isinstance(m, LoRALinear) for m in model.modules()))
            self.assertTrue(torch.allclose(lora_out, model(x)[0], atol=1e-5))


class TestTrainSFT(unittest.TestCase):

    def _dataset(self):
        records = [
            {"messages": [{"role": "user", "content": f"soru {i}"},
                          {"role": "assistant", "content": "cevap " + "ab" * (i % 3 + 1)}]}
            for i in range(6)
        ]
        return SFTDataset(records, MockTokenizer(), max_len=48)

    def test_full_finetune_reduces_loss(self):
        model = tiny_model()
        ds = self._dataset()
        hist = train_sft(model, ds, eval_dataset=ds, max_steps=30, batch_size=3, grad_accum_steps=1,
                         learning_rate=3e-3, warmup_steps=2, eval_every=30, log_fn=None)
        self.assertLess(hist["train_loss"][-1][1], hist["train_loss"][0][1])
        self.assertEqual(len(hist["eval_loss"]), 1)

    def test_lora_checkpoint_loads_with_generate_load_model(self):
        from inference.generate import load_model

        model = tiny_model()
        ds = self._dataset()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sft.pt")
            hist = train_sft(model, ds, output_path=path, max_steps=4, batch_size=2,
                             grad_accum_steps=2, learning_rate=1e-2, warmup_steps=1,
                             lora_r=4, lora_alpha=8, log_fn=None)
            self.assertEqual(len(hist["train_loss"]), 4)
            self.assertFalse(any(isinstance(m, LoRALinear) for m in model.modules()))
            ckpt = torch.load(path, weights_only=False)
            self.assertEqual(ckpt["training_recipe"]["stage"], "sft")
            self.assertEqual(ckpt["config"], model.config.architecture_dict())
            loaded, _ = load_model(path, "cpu")
        model.eval()
        x = torch.randint(5, V, (1, 12))
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[0], loaded(x)[0], atol=1e-5))

    def test_save_checkpoint_without_lora(self):
        model = tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_aligned_checkpoint(model, os.path.join(tmp, "a", "m.pt"), step=3)
            ckpt = torch.load(path, weights_only=False)
        self.assertEqual(set(ckpt["model_state_dict"]), set(model.state_dict()))
        self.assertEqual(ckpt["global_step"], 3)


if __name__ == "__main__":
    unittest.main()
