# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Morfoloji Mikroskobu (interpret/) testleri: aktivasyon kaydedici (yoğun ve
MoE), dilbilimsel etiketleyiciler, doğrusal sondalar ve kontrol görevi,
TopK SAE, özellik-etiket ilişkisi, müdahale/yamalama ve HTML raporları.
"""

import json
import os
import re
import tempfile
import unittest

import torch

from interpret.activations import ActivationRecorder, collect
from interpret.features import (
    FEATURES,
    compute_features,
    label_sentence,
    suffix_type,
)
from interpret.patching import ablate_feature, next_token_logprob, patch_activations
from interpret.probes import control_labels, probe_all_layers, train_linear_probe
from interpret.report import (
    build_compare_report,
    build_probe_report,
    build_sae_report,
    render_compare_html,
    render_probe_html,
    render_sae_html,
    write_report,
)
from interpret.sae import (
    TopKSAE,
    encode_all,
    feature_label_association,
    feature_top_tokens,
    train_sae,
)
from model.config import ModelConfig
from model.transformer import ToprakLM

VOCAB = [
    "<pad>", "<unk>", "<s>", "</s>",
    "▁kitap", "lar", "da", "▁ev", "ler", "de", "▁saat", "ten",
    "▁okul", "dan", "▁gel", "di", "▁git", "miş", ".", "▁göz", "lük", "▁ağaç", "tan", "▁bahçe", "den",
]
TOK2ID = {t: i for i, t in enumerate(VOCAB)}

SENTENCES = [
    "▁kitap lar da ▁ev ler de .",
    "▁saat ten ▁okul dan ▁gel di .",
    "▁git miş ▁göz lük ler de .",
    "▁ağaç tan ▁bahçe de ▁kitap lar .",
    "▁ev de ▁saat lar da ▁gel di .",
    "▁okul lar dan ▁git miş .",
    "▁göz lük ten ▁ağaç lar da .",
    "▁bahçe ler de ▁kitap tan ▁gel di .",
    "▁kitap da ▁ev ler den .",
    "▁saat ler ▁okul da ▁git miş .",
    "▁ağaç lar ▁göz lük da ▁gel di .",
    "▁bahçe den ▁ev ler ▁kitap lar da .",
]


class MockTokenizer:
    """Boşlukla ayrılmış parçaları kelime dağarcığından ID'ye çeviren sahte tokenizer."""

    def __init__(self, vocab=VOCAB):
        self.vocab = list(vocab)

    def get_vocab_size(self):
        return len(self.vocab)

    def id_to_token(self, token_id):
        return self.vocab[token_id]

    def encode(self, text, add_bos=True, add_eos=False):
        ids = [TOK2ID[p] for p in text.split()]
        return ([2] if add_bos else []) + ids + ([3] if add_eos else [])


def tiny_model(seed=0, **overrides):
    torch.manual_seed(seed)
    base = dict(vocab_size=len(VOCAB), d_model=32, num_heads=4, num_kv_heads=2,
                num_layers=2, d_ff=64, max_seq_len=32, device="cpu")
    base.update(overrides)
    model = ToprakLM(ModelConfig(**base), tokenizer=MockTokenizer())
    model.eval()
    return model


def _hook_count(model):
    n = 0
    for m in model.modules():
        n += len(m._forward_hooks) + len(m._forward_pre_hooks)
    return n


class TestActivationRecorder(unittest.TestCase):

    def _check(self, model, moe=False):
        ids = torch.tensor([[2, 4, 5, 6, 7, 8, 9]])
        T, d = ids.size(1), model.config.d_model
        before = _hook_count(model)
        sites = ("resid_pre", "resid_post", "attn_out", "ffn_out")
        with ActivationRecorder(model, sites=sites) as rec:
            self.assertGreater(_hook_count(model), before)
            with torch.no_grad():
                _, _, _, hidden = model(ids, return_hidden=True)
            for site in sites:
                for i in range(model.config.num_layers):
                    self.assertEqual(tuple(rec.get(site, i).shape), (1, T, d))
            # Artık akış ayrışımı: post = pre + attn + ffn
            for i in range(model.config.num_layers):
                recon = rec.get("resid_pre", i) + rec.get("attn_out", i) + rec.get("ffn_out", i)
                torch.testing.assert_close(recon, rec.get("resid_post", i), atol=1e-5, rtol=1e-5)
            last = rec.get("resid_post", model.config.num_layers - 1)
            torch.testing.assert_close(model.ln_f(last), hidden, atol=1e-5, rtol=1e-5)
            if moe:
                for i in range(model.config.num_layers):
                    r = rec.get_routing(i)
                    self.assertEqual(tuple(r.shape), (1, T, model.config.experts_top_k))
        self.assertEqual(_hook_count(model), before)
        n_records = len(rec.records["resid_post"][0])
        with torch.no_grad():
            model(ids)
        self.assertEqual(len(rec.records["resid_post"][0]), n_records)

    def test_dense(self):
        self._check(tiny_model())

    def test_moe(self):
        self._check(tiny_model(num_experts=4, experts_top_k=2), moe=True)

    def test_layer_subset_and_bad_site(self):
        model = tiny_model()
        with ActivationRecorder(model, sites=("resid_post",), layers=[1]) as rec:
            with torch.no_grad():
                model(torch.tensor([[2, 4, 5]]))
        self.assertEqual(list(rec.records["resid_post"]), [1])
        with self.assertRaises(ValueError):
            ActivationRecorder(model, sites=("bilinmeyen",))

    def test_collect_alignment(self):
        model = tiny_model(num_experts=4, experts_top_k=2, moe_layer_freq=2)
        c = collect(model, MockTokenizer(), SENTENCES[:3], max_len=16)
        n = sum(len(s.split()) + 1 for s in SENTENCES[:3])
        self.assertEqual(c.num_tokens, n)
        self.assertEqual(tuple(c.acts["resid_post"][1].shape), (n, 32))
        self.assertEqual(tuple(c.acts["embed"][0].shape), (n, 32))
        self.assertEqual(list(c.layer_dict()), ["emb", "L1", "L2"])
        self.assertEqual(c.token_strings[0][:3], ["<s>", "▁kitap", "lar"])
        self.assertEqual(c.morph_classes[:3].tolist(), [2, 0, 1])
        self.assertEqual(list(c.routing), [1])  # yalnız 2. blok MoE
        self.assertEqual(tuple(c.routing[1].shape), (n, 2))
        self.assertEqual(c.sentence_index.tolist().count(0), len(c.token_ids[0]))


class TestFeatures(unittest.TestCase):

    def test_label_sentence(self):
        toks = ["<s>", "▁kitap", "lar", "da", "▁ev", "ler", "de", "▁saat", "ten", "."]
        L = label_sentence(toks)
        self.assertEqual(L["kelime_basi"], [-1, 1, 0, 0, 1, 0, 0, 1, 0, -1])
        self.assertEqual(L["morf_sinifi"], [2, 0, 1, 1, 0, 1, 1, 0, 1, 2])
        self.assertEqual(L["son_unlu"], [-1, 0, 0, 0, 1, 1, 1, 0, 1, -1])
        self.assertEqual(L["sonraki_ek"], [-1, 1, 1, 0, 1, 1, 0, 1, 0, -1])
        # "▁saat" → "ten": uyum kalın bekler ama "ten" incedir (yabancı köken istisnası)
        self.assertEqual(L["beklenen_uyum"][7], 0)
        self.assertEqual(L["sonraki_ek_unlusu"][7], 1)
        self.assertEqual(L["beklenen_uyum"][1], 0)
        self.assertEqual(L["beklenen_uyum"][4], 1)
        self.assertEqual(L["sert_unsuz"][1], 1)   # kitap → p
        self.assertEqual(L["sert_unsuz"][4], 0)   # ev
        self.assertEqual(L["sert_unsuz"][7], 1)   # saat → t
        self.assertEqual(L["ek_turu"], [-1, -1, 1, 2, -1, 1, 2, -1, 3, -1])
        self.assertEqual(L["sonraki_ek_turu"][1], 1)
        self.assertEqual(L["sonraki_ek_turu"][7], 3)

    def test_suffix_types(self):
        self.assertEqual(suffix_type("lar"), 1)
        self.assertEqual(suffix_type("ları"), 1)
        self.assertEqual(suffix_type("te"), 2)
        self.assertEqual(suffix_type("daki"), 2)
        self.assertEqual(suffix_type("den"), 3)
        self.assertEqual(suffix_type("tan"), 3)
        self.assertEqual(suffix_type("dü"), 4)
        self.assertEqual(suffix_type("tık"), 4)
        self.assertEqual(suffix_type("dir"), 0)   # ek-fiil, geçmiş zaman değil
        self.assertEqual(suffix_type("mişler"), 5)
        self.assertEqual(suffix_type("ğ"), 0)

    def test_compute_features_flat(self):
        toks = [["<s>", "▁kitap", "lar"], ["<s>", "▁ev", "de", "."]]
        feats = compute_features(toks)
        self.assertEqual(set(feats), set(FEATURES))
        self.assertEqual(feats["morf_sinifi"].tolist(), [2, 0, 1, 2, 0, 1, 2])
        with self.assertRaises(ValueError):
            compute_features(toks, features=["yok_boyle"])


class TestProbes(unittest.TestCase):

    def test_planted_signal_and_random_labels(self):
        g = torch.Generator().manual_seed(0)
        X = torch.randn(600, 16, generator=g)
        w = torch.randn(16, generator=g)
        y = (X @ w > 0).long()
        res = train_linear_probe(X, y, num_classes=2, epochs=150, seed=0)
        self.assertGreater(res.accuracy, 0.9)
        self.assertGreater(res.macro_f1, 0.9)
        y_rand = torch.randint(0, 2, (600,), generator=g)
        res_r = train_linear_probe(X, y_rand, num_classes=2, epochs=150, seed=0)
        self.assertLess(res_r.accuracy, 0.65)
        self.assertLess(abs(res_r.accuracy - 0.5), 0.15)
        probs = res.predict_proba(X[:5])
        torch.testing.assert_close(probs.sum(-1), torch.ones(5))

    def test_ignore_and_groups(self):
        g = torch.Generator().manual_seed(1)
        X = torch.randn(300, 8, generator=g)
        y = (X[:, 0] > 0).long()
        y[::5] = -1
        groups = torch.arange(300) // 10
        res = train_linear_probe(X, y, epochs=100, groups=groups)
        self.assertEqual(res.num_train + res.num_val, int((y >= 0).sum()))
        self.assertGreater(res.accuracy, 0.85)

    def test_control_labels_per_type(self):
        ids = torch.tensor([5, 6, 5, 7, 6, 5, 9])
        y = torch.tensor([0, 1, 1, 0, -1, 1, 0])
        c = control_labels(ids, y, seed=3)
        self.assertEqual(c[4].item(), -1)
        self.assertEqual(c[0].item(), c[2].item())
        self.assertEqual(c[0].item(), c[5].item())
        torch.testing.assert_close(c, control_labels(ids, y, seed=3))

    def test_probe_all_layers_selectivity(self):
        g = torch.Generator().manual_seed(2)
        n = 500
        token_ids = torch.randint(0, 200, (n,), generator=g)
        y = torch.randint(0, 2, (n,), generator=g)
        signal = torch.randn(n, 12, generator=g)
        signal[:, 0] += 3.0 * (2 * y.float() - 1)
        noise = torch.randn(n, 12, generator=g)
        out = probe_all_layers({"emb": noise, "L1": signal}, y, token_ids=token_ids,
                               groups=torch.arange(n) // 10, epochs=100)
        rows = {r["layer"]: r for r in out["rows"]}
        self.assertGreater(rows["L1"]["accuracy"], 0.9)
        self.assertLess(rows["emb"]["accuracy"], 0.7)
        self.assertGreater(rows["L1"]["selectivity"], 0.25)
        self.assertIn("control_accuracy", rows["emb"])
        self.assertEqual(set(out["probes"]), {"emb", "L1"})


def _sparse_data(n=1500, d=16, atoms=32, active=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(atoms, d, generator=g)
    D = D / D.norm(dim=1, keepdim=True)
    idx = torch.stack([torch.randperm(atoms, generator=g)[:active] for _ in range(n)])
    coef = torch.zeros(n, atoms)
    coef.scatter_(1, idx, torch.rand(n, active, generator=g) * 2 + 0.5)
    return coef @ D, coef, D


class TestSAE(unittest.TestCase):

    def test_training_reduces_error_and_topk(self):
        X, _, _ = _sparse_data()
        res = train_sae(X, d_hidden=64, k=4, epochs=15, lr=3e-3, seed=0)
        self.assertLess(res.nmse, res.initial_nmse)
        self.assertLess(res.history[-1], res.history[0])
        self.assertLess(res.nmse, 0.5)
        self.assertAlmostEqual(res.fve, 1 - res.nmse, places=6)
        z = encode_all(res.sae, X)
        self.assertLessEqual(int((z > 0).sum(-1).max()), 4)
        norms = res.sae.W_dec.norm(dim=0)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)
        self.assertGreaterEqual(res.dead_fraction, 0.0)
        self.assertLessEqual(res.dead_fraction, 1.0)

    def test_deterministic(self):
        X, _, _ = _sparse_data(n=300)
        a = train_sae(X, d_hidden=32, k=3, epochs=3, seed=1)
        b = train_sae(X, d_hidden=32, k=3, epochs=3, seed=1)
        self.assertEqual(a.history, b.history)

    def _identity_sae(self, d):
        sae = TopKSAE(d, d, d)
        with torch.no_grad():
            sae.W_enc.copy_(torch.eye(d))
            sae.W_dec.copy_(torch.eye(d))
            sae.b_pre.zero_()
            sae.b_enc.zero_()
        return sae

    def test_association_finds_planted_feature(self):
        g = torch.Generator().manual_seed(0)
        d, n = 12, 400
        labels = torch.randint(0, 2, (n,), generator=g)
        X = torch.rand(n, d, generator=g) * 0.5
        X[:, 5] += 2.0 * labels.float()
        labels[:10] = -1
        ranked = feature_label_association(self._identity_sae(d), X, labels, positive_class=1, top=3)
        self.assertEqual(ranked[0]["feature"], 5)
        self.assertGreater(ranked[0]["r"], 0.8)
        self.assertGreater(ranked[0]["mean_diff"], 1.5)

    def test_association_trained_sae(self):
        X, coef, _ = _sparse_data(seed=3)
        labels = (coef[:, 0] > 0).long()
        res = train_sae(X, d_hidden=64, k=4, epochs=15, lr=3e-3, seed=0)
        ranked = feature_label_association(res.sae, X, labels, top=1)
        self.assertGreater(ranked[0]["r"], 0.5)

    def test_feature_top_tokens(self):
        d = 4
        sae = self._identity_sae(d)
        X = torch.zeros(6, d)
        X[3, 2] = 5.0
        X[1, 2] = 1.0
        toks = ["<s>", "▁kitap", "lar", "da", "▁ev", "ler"]
        sid = torch.tensor([0, 0, 0, 0, 1, 1])
        tops = feature_top_tokens(sae, X, toks, n=2, features=[2, 0], sentence_ids=sid)
        self.assertEqual(tops[2][0]["token"], "da")
        self.assertEqual(tops[2][0]["left"], "<s> kitaplar")
        self.assertEqual(tops[2][0]["right"], "")
        self.assertEqual(tops[0], [])


class TestPatching(unittest.TestCase):

    def test_ablate_direction_removes_component(self):
        model = tiny_model()
        ids = torch.tensor([[2, 4, 5, 6]])
        u = torch.randn(32)
        before = _hook_count(model)
        with ablate_feature(model, 0, direction=u):
            with ActivationRecorder(model, sites=("resid_post",), layers=[0]) as rec, torch.no_grad():
                model(ids)
        proj = rec.get("resid_post", 0) @ (u / u.norm())
        self.assertLess(float(proj.abs().max()), 1e-4)
        self.assertEqual(_hook_count(model), before)

    def test_patch_all_positions_reproduces_source(self):
        model = tiny_model()
        src, dst = torch.tensor([[2, 4, 5, 6]]), torch.tensor([[2, 7, 8, 9]])
        last = model.config.num_layers - 1
        with ActivationRecorder(model, sites=("resid_post",), layers=[last]) as rec, torch.no_grad():
            src_logits = model(src)[0]
        with patch_activations(model, last, rec.get("resid_post", last)), torch.no_grad():
            patched = model(dst)[0]
        torch.testing.assert_close(patched, src_logits, atol=1e-5, rtol=1e-5)
        with patch_activations(model, last, rec.get("resid_post", last), positions=[3]), torch.no_grad():
            partial = model(dst)[0]
        torch.testing.assert_close(partial[:, 3], src_logits[:, 3], atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(partial[:, 1], src_logits[:, 1]))

    def test_sae_feature_ablation_and_logprob(self):
        model = tiny_model()
        ids = torch.tensor([[2, 4, 5, 6]])
        sae = TestSAE()._identity_sae(32)
        base = next_token_logprob(model, ids, [5, 8])
        self.assertEqual(tuple(base.shape), (1, 4))
        self.assertTrue(bool((base <= 0).all()))
        with ablate_feature(model, 0, sae=sae, feature=3):
            with ActivationRecorder(model, sites=("resid_post",), layers=[0]) as rec, torch.no_grad():
                model(ids)
        self.assertLessEqual(float(rec.get("resid_post", 0)[..., 3].max()), 1e-6)
        with self.assertRaises(ValueError):
            with ablate_feature(model, 0):
                pass


class TestReports(unittest.TestCase):

    def _assert_self_contained(self, text):
        self.assertTrue(text.startswith("<!doctype html>"))
        self.assertIsNone(re.search(r"https?://", text))
        self.assertNotIn("<link", text)
        self.assertNotIn(" src=", text)
        self.assertIn("prefers-color-scheme", text)
        self.assertIn('name="viewport"', text)
        data = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', text, re.S)
        self.assertIsNotNone(data)
        return json.loads(data.group(1))

    def test_probe_report_html_moe(self):
        model = tiny_model(num_experts=4, experts_top_k=2)
        c = collect(model, MockTokenizer(), SENTENCES, max_len=16)
        feats = ["morf_sinifi", "son_unlu", "beklenen_uyum"]
        report = build_probe_report(c, features=feats, epochs=30, view_sentences=5,
                                    metadata={"checkpoint": "test.pt"}, num_experts=4)
        self.assertEqual(report["layers"], ["emb", "L1", "L2"])
        self.assertEqual(list(report["features"]), feats)
        self.assertEqual(len(report["features"]["morf_sinifi"]["rows"]), 3)
        self.assertEqual(len(report["token_view"]["sentences"]), 5)
        probs = report["token_view"]["probs"]["son_unlu"]["L1"]
        self.assertEqual(len(probs[0]), len(c.token_strings[0]))
        self.assertEqual(len(probs[0][0]), 3)
        self.assertIsNotNone(report["moe"])
        tbl = report["moe"]["layers"]["L1"]
        self.assertEqual(len(tbl["counts"]), 4)
        self.assertEqual(sum(map(sum, tbl["counts"])), c.num_tokens * 2)
        html_text = render_probe_html(report)
        data = self._assert_self_contained(html_text)
        self.assertEqual(data["kind"], "probe")
        for needle in ("Katman başına sonda doğruluğu", "Token görünümü", "MoE yönlendirme", "<svg", 'id="tv-tokens"'):
            self.assertIn(needle, html_text)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_report(tmp, "probe", report, html_text)
            self.assertTrue(os.path.exists(paths["html"]))
            with open(paths["json"], encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["kind"], "probe")

        cmp_report = build_compare_report(report, report, "a", "b")
        self.assertEqual(cmp_report["warnings"], [])
        for f in cmp_report["features"].values():
            for r in f["rows"]:
                self.assertEqual(r["delta_accuracy"], 0.0)
                self.assertFalse(r["notable"])
        cmp_html = render_compare_html(cmp_report)
        self._assert_self_contained(cmp_html)
        self.assertIn("Checkpoint Karşılaştırması", cmp_html)

    def test_dense_report_has_no_moe(self):
        model = tiny_model()
        c = collect(model, MockTokenizer(), SENTENCES, max_len=16)
        report = build_probe_report(c, features=["kelime_basi"], epochs=20, view_sentences=2)
        self.assertIsNone(report["moe"])
        self.assertNotIn('id="moe"', render_probe_html(report))

    def test_sae_report_html(self):
        model = tiny_model()
        c = collect(model, MockTokenizer(), SENTENCES, max_len=16, sites=("resid_post",))
        acts = c.acts["resid_post"][1]
        res = train_sae(acts, d_hidden=64, k=4, epochs=3, seed=0)
        report = build_sae_report(res, c, acts, "L2", top_features=5, contexts_per_feature=3,
                                  associate=["son_unlu", "morf_sinifi"])
        self.assertLessEqual(len(report["top_features"]), 5)
        html_text = render_sae_html(report)
        data = self._assert_self_contained(html_text)
        self.assertEqual(data["kind"], "sae")
        self.assertIn("Etiketlerle en ilişkili latentler", html_text)


class TestCLI(unittest.TestCase):

    def test_probe_and_compare_cli(self):
        from interpret.cli import main, read_texts
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tok_path = os.path.join(root, "toprak_tokenizer.model")
        texts_path = os.path.join(root, "interpret", "examples", "sentences.txt")
        self.assertGreaterEqual(len(read_texts(texts_path)), 30)
        if not os.path.exists(tok_path):
            self.skipTest("tokenizer yok")
        from model.tokenizer import ToprakTokenizer
        tok = ToprakTokenizer(tok_path)
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=tok.get_vocab_size(), d_model=16, num_heads=2, num_kv_heads=1,
                          num_layers=2, d_ff=32, max_seq_len=64, device="cpu")
        model = ToprakLM(cfg, tokenizer=tok)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "m.pt")
            torch.save({"model_state_dict": model.state_dict(), "config": cfg.architecture_dict(),
                        "global_step": 1}, ckpt)
            out = os.path.join(tmp, "rep")
            paths = main(["probe", "--checkpoint", ckpt, "--tokenizer", tok_path, "--texts", texts_path,
                          "--out", out, "--features", "kelime_basi", "son_unlu", "--epochs", "20",
                          "--view-sentences", "3"])
            with open(paths["json"], encoding="utf-8") as fh:
                rep = json.load(fh)
            self.assertEqual(rep["metadata"]["global_step"], 1)
            self.assertEqual(rep["layers"], ["emb", "L1", "L2"])
            cmp_paths = main(["compare", "--report-a", paths["json"], "--report-b", paths["json"],
                              "--out", out])
            self.assertTrue(os.path.exists(cmp_paths["html"]))
            sae_paths = main(["sae", "--checkpoint", ckpt, "--tokenizer", tok_path, "--texts", texts_path,
                              "--out", out, "--layer", "L2", "--d-hidden", "32", "--k", "4", "--epochs", "2"])
            self.assertTrue(os.path.exists(sae_paths["html"]))


if __name__ == "__main__":
    unittest.main()
