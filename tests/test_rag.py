# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
RAG paketi testleri: Türkçe metin işleme, madde yapılı parçalama, BM25
indeks (madde/sayı/ifade ek puanları, JSON kayıt), token bütçeli kaynaklı
prompt, atıf doğrulama ve komut satırı (index / search / ask).
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

import torch

from rag.chunker import Chunk, Document, chunk_document, load_documents
from rag.citations import parse_citation_ids, render_report, verify_citations
from rag.index import BM25Index, SearchResult
from rag.prompt import DEFAULT_SYSTEM_PROMPT, NO_ANSWER, build_rag_messages, render_source
from rag.text import (
    analyze,
    article_key,
    normalize,
    parse_article_header,
    split_sentences,
    stem,
    tokenize,
    turkish_lower,
    turkish_upper,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "rag", "examples")
TOKENIZER_PATH = os.path.join(ROOT, "toprak_tokenizer.model")


class WordTokenizer:
    """Kelime başına bir token sayan sahte tokenizer (bütçe testleri için)."""

    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.vocab.setdefault(w, 10 + len(self.vocab)) for w in text.split()]
        return ([self.bos_token_id] if add_bos else []) + ids + ([self.eos_token_id] if add_eos else [])

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "") for i in ids)

    def token_to_id(self, token):
        return self.unk_token_id


def make_chunk(cid, text, article=None, title="Örnek Yönetmelik"):
    return Chunk(id=cid, doc_id="d", title=title, article=article, text=text, start=0, end=len(text))


class TestTurkishText(unittest.TestCase):

    def test_turkish_case_mapping(self):
        self.assertEqual(turkish_lower("İSTANBUL IŞIK"), "istanbul ışık")
        self.assertEqual(turkish_upper("istanbul ılık"), "İSTANBUL ILIK")
        self.assertEqual(normalize("  KANUN’UN   Maddesi "), "kanun'un maddesi")
        self.assertEqual(normalize("Çiğ Şüphe", strip_accents=True), "cig suphe")
        self.assertIn("ç", normalize("Çiğ"))  # varsayılan: aksanlar korunur

    def test_stemmer_is_consistent_across_inflections(self):
        self.assertEqual({stem(w) for w in ["kanun", "kanunun", "kanuna", "kanunlar"]}, {"kanun"})
        self.assertEqual(stem("kitaplarından"), stem("kitap"))
        self.assertEqual(stem("çocuğu"), "çocuk")
        self.assertEqual(stem("maddesi"), stem("maddede"))
        self.assertEqual(stem("ev"), "ev")             # kısa kelimeye dokunulmaz
        self.assertEqual(stem("5237"), "5237")

    def test_tokenize_numbers_and_article_refs(self):
        toks = [t.text for t in tokenize(
            "5237 sayılı Kanun'un 12 nci maddesi, m. 7/3 ve Geçici Madde 1; 01.01.2020")]
        self.assertIn("5237", toks)
        self.assertIn("kanun", toks)                  # kesmeden sonraki ek atılır
        self.assertIn("madde:12", toks)
        self.assertIn("madde:7", toks)
        self.assertIn("madde:7/3", toks)
        self.assertIn("geçici_madde:1", toks)
        self.assertIn("01.01.2020", toks)

    def test_analyze_removes_stopwords(self):
        terms = analyze("Bu ve şu için kitap ödünç alınır")
        self.assertNotIn("ve", terms)
        self.assertNotIn("için", terms)
        self.assertIn("kitap", terms)

    def test_article_header_parsing(self):
        self.assertEqual(parse_article_header("MADDE 5 – (1) Metin"), "Madde 5")
        self.assertEqual(parse_article_header("Madde 12- Metin"), "Madde 12")
        self.assertEqual(parse_article_header("Geçici Madde 1 – Metin"), "Geçici Madde 1")
        self.assertEqual(parse_article_header("GEÇİCİ MADDE 3 – Metin"), "Geçici Madde 3")
        self.assertEqual(parse_article_header("EK MADDE 2 – Metin"), "Ek Madde 2")
        self.assertIsNone(parse_article_header("Bu madde 5 kez okundu."))
        self.assertEqual(article_key("Geçici Madde 1"), "geçici_madde:1")
        self.assertEqual(article_key("Madde 5"), "madde:5")

    def test_sentence_splitter_abbreviations_and_offsets(self):
        text = ("Amaç\nMADDE 1 – (1) T.C. Örnek Bel. ile Dr. Ali bu işi yapar. "
                "Bkz. Md. 5 vb. hükümler 15. maddede yazılıdır. No. 3 dosya açıldı!\n"
                "(2) İkinci fıkra.")
        sents = split_sentences(text)
        texts = [s.text for s in sents]
        self.assertEqual(texts[0], "Amaç")
        self.assertTrue(texts[1].startswith("MADDE 1"))
        self.assertTrue(texts[1].endswith("yapar."))
        self.assertEqual(texts[2], "Bkz. Md. 5 vb. hükümler 15. maddede yazılıdır.")
        self.assertEqual(texts[3], "No. 3 dosya açıldı!")
        self.assertEqual(texts[4], "(2) İkinci fıkra.")
        for s in sents:
            self.assertEqual(text[s.start:s.end], s.text)


class TestChunker(unittest.TestCase):

    def test_example_documents_are_fictional_and_have_provenance(self):
        docs = load_documents(EXAMPLES)
        self.assertEqual(len(docs), 3)
        for doc in docs:
            self.assertIn("KURGUSAL", doc.text)
            self.assertTrue(doc.url and doc.license and doc.date)
            self.assertEqual(doc.metadata.get("fictional"), "true")

    def test_article_structure_and_offsets(self):
        docs = {d.id: d for d in load_documents(EXAMPLES)}
        doc = docs["ornek-kutuphane"]
        chunks = chunk_document(doc)
        articles = [c.article for c in chunks if c.article]
        self.assertEqual(articles, [f"Madde {i}" for i in range(1, 9)])
        for c in chunks:
            self.assertEqual(doc.text[c.start:c.end], c.text)
            self.assertEqual(c.url, doc.url)
            self.assertEqual(c.license, doc.license)
            self.assertEqual(c.date, doc.date)
        madde6 = next(c for c in chunks if c.article == "Madde 6")
        self.assertTrue(madde6.text.startswith("Ödünç verme\nMADDE 6"))  # başlık satırı dahil
        bike = chunk_document(docs["ornek-bisiklet"])
        self.assertIn("Geçici Madde 1", [c.article for c in bike])

    def test_budget_and_overlap_within_article(self):
        sentences = " ".join(f"Bu fıkranın {i}. cümlesi oldukça uzun bir açıklama içerir." for i in range(30))
        doc = Document("uzun", "Uzun Yönetmelik", f"MADDE 1 – {sentences}\nMADDE 2 – Kısa madde.")
        chunks = chunk_document(doc, max_chars=200, overlap_sentences=1)
        m1 = [c for c in chunks if c.article == "Madde 1"]
        self.assertGreater(len(m1), 3)
        for c in m1:
            self.assertLessEqual(len(c.text), 200)
        for a, b in zip(m1, m1[1:]):
            self.assertLess(b.start, a.end)            # örtüşme
            self.assertGreater(b.end, a.end)           # ilerleme
        m2 = [c for c in chunks if c.article == "Madde 2"]
        self.assertEqual(len(m2), 1)
        self.assertGreaterEqual(m2[0].start, m1[-1].end)  # örtüşme madde sınırını aşmaz

    def test_token_budget_and_long_sentence_split(self):
        long_sentence = " ".join(["kelime"] * 100) + "."
        doc = Document("x", "X", f"MADDE 1 – {long_sentence}")
        chunks = chunk_document(doc, token_counter=lambda s: len(s.split()), max_tokens=20)
        self.assertGreater(len(chunks), 4)
        for c in chunks:
            self.assertLessEqual(len(c.text.split()), 20)

    def test_jsonl_with_governance_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "docs.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "a", "title": "A", "text": "MADDE 1 – Metin.",
                                    "url": "https://example.org/a", "license": "CC0-1.0",
                                    "date": "2026-01-01"}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"text": "MADDE 1 – Başka metin.", "source_url": "https://example.org/b",
                                    "licenses": ["ODC-By-1.0"], "downloaded_at": "2026-02-02",
                                    "source": "web"}, ensure_ascii=False) + "\n")
            docs = load_documents(path)
        self.assertEqual(docs[0].id, "a")
        self.assertEqual(docs[0].license, "CC0-1.0")
        self.assertEqual(docs[1].url, "https://example.org/b")
        self.assertEqual(docs[1].license, "ODC-By-1.0")
        self.assertEqual(docs[1].date, "2026-02-02")
        self.assertEqual(docs[1].metadata["source"], "web")


class TestBM25Index(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.index = BM25Index.from_path(EXAMPLES)

    def test_relevant_article_ranks_first(self):
        top = self.index.search("Gecikme bedeli ne kadar?", k=3)[0]
        self.assertEqual((top.chunk.doc_id, top.chunk.article), ("ornek-kutuphane", "Madde 7"))
        top = self.index.search("parsel sulama saatleri", k=3)[0]
        self.assertEqual((top.chunk.doc_id, top.chunk.article), ("ornek-bahce", "Madde 3"))

    def test_article_and_number_boosts(self):
        res = self.index.search("Geçici Madde 1", k=3)
        self.assertEqual(res[0].chunk.article, "Geçici Madde 1")
        self.assertIn("article", res[0].boosts)
        res = self.index.search("9902 sayılı kanun", k=3)
        self.assertEqual(res[0].chunk.doc_id, "ornek-bahce")
        self.assertEqual(res[0].boosts["number"], 2 * self.index.number_boost)  # 4+ hane → 2×
        plain = self.index.search("9902 sayılı kanun", k=3, boost=False)
        self.assertEqual(plain[0].boosts, {})
        self.assertGreater(res[0].score, plain[0].score)

    def test_phrase_boost(self):
        res = self.index.search('"ilk otuz dakikası ücretsizdir"', k=1)
        self.assertEqual(res[0].chunk.doc_id, "ornek-bisiklet")
        self.assertIn("phrase", res[0].boosts)

    def test_bm25_scores_are_sorted_and_deterministic(self):
        a = self.index.search("üye ödünç materyal", k=5)
        b = self.index.search("üye ödünç materyal", k=5)
        self.assertEqual([r.chunk.id for r in a], [r.chunk.id for r in b])
        scores = [r.score for r in a]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(self.index.search("zzzz qqqq", k=5), [])

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            self.index.save(path)
            loaded = BM25Index.load(path)
        q = "bisiklet istasyon dışında bırakılırsa ne olur"
        self.assertEqual(
            [(r.chunk.id, round(r.score, 6)) for r in self.index.search(q, k=4)],
            [(r.chunk.id, round(r.score, 6)) for r in loaded.search(q, k=4)],
        )
        self.assertEqual(len(loaded), len(self.index))

    def test_k1_b_parameters_change_scores(self):
        docs = load_documents(EXAMPLES)
        a = BM25Index.from_documents(docs, k1=0.5, b=0.0).search("ödünç süresi", k=1)[0]
        b = BM25Index.from_documents(docs, k1=2.0, b=1.0).search("ödünç süresi", k=1)[0]
        self.assertNotAlmostEqual(a.bm25, b.bm25)


class TestPrompt(unittest.TestCase):

    def setUp(self):
        self.results = [
            SearchResult(make_chunk("c1", "Ödünç süresi on dört gündür.", "Madde 6"), 9.0),
            SearchResult(make_chunk("c2", " ".join(["dolgu"] * 50), "Madde 2"), 3.0),
            SearchResult(make_chunk("c3", "Gecikme bedeli günlük 2 TL'dir.", "Madde 7"), 6.0),
        ]

    def test_sources_numbered_by_score(self):
        rag = build_rag_messages("Ödünç süresi nedir?", self.results)
        self.assertEqual([s.chunk.id for s in rag.sources], ["c1", "c3", "c2"])
        self.assertEqual([m["role"] for m in rag.messages], ["system", "user"])
        self.assertEqual(rag.messages[0]["content"], DEFAULT_SYSTEM_PROMPT)
        self.assertIn(NO_ANSWER, DEFAULT_SYSTEM_PROMPT)
        user = rag.messages[1]["content"]
        self.assertIn("[1] Örnek Yönetmelik — Madde 6: Ödünç süresi on dört gündür.", user)
        self.assertIn("[2] Örnek Yönetmelik — Madde 7:", user)
        self.assertTrue(user.index("[1]") < user.index("[2]") < user.index("Soru:"))
        self.assertEqual(render_source(4, make_chunk("x", "a  b")), "[4] Örnek Yönetmelik: a b")

    def test_budget_drops_lowest_scored(self):
        tok = WordTokenizer()
        full = build_rag_messages("Ödünç süresi nedir?", self.results, tokenizer=tok)
        budget = full.prompt_tokens - 30          # 50 kelimelik kaynak sığmaz
        rag = build_rag_messages("Ödünç süresi nedir?", self.results, tokenizer=tok,
                                 max_prompt_tokens=budget)
        self.assertEqual([s.chunk.id for s in rag.sources], ["c1", "c3"])
        self.assertEqual([r.chunk.id for r in rag.dropped], ["c2"])
        self.assertLessEqual(rag.prompt_tokens, budget)

    def test_single_oversized_source_is_truncated(self):
        tok = WordTokenizer()
        big = [SearchResult(make_chunk("big", " ".join(f"k{i}" for i in range(500)), "Madde 1"), 1.0)]
        empty = build_rag_messages("Soru?", [], tokenizer=tok)
        rag = build_rag_messages("Soru?", big, tokenizer=tok, max_prompt_tokens=empty.prompt_tokens + 80)
        self.assertEqual(len(rag.sources), 1)
        self.assertTrue(rag.sources[0].truncated)
        self.assertLessEqual(rag.prompt_tokens, empty.prompt_tokens + 80)

    @unittest.skipUnless(os.path.exists(TOKENIZER_PATH), "tokenizer yok")
    def test_exact_budget_with_chat_template_and_real_tokenizer(self):
        from model.chat_template import ChatTemplate
        from model.tokenizer import ToprakTokenizer
        tok = ToprakTokenizer(TOKENIZER_PATH)
        template = ChatTemplate(tok)
        results = BM25Index.from_path(EXAMPLES).search("üye ödünç kitap gecikme bedeli", k=10)
        for budget in (300, 600, 4096):
            rag = build_rag_messages("Gecikme bedeli nedir?", results, tokenizer=tok,
                                     max_prompt_tokens=budget, template=template)
            ids = template.encode_prompt(rag.messages)
            self.assertLessEqual(len(ids), budget)
            self.assertEqual(rag.prompt_tokens, len(ids))
            self.assertEqual(len(rag.sources) + len(rag.dropped), len(results))
        small = build_rag_messages("x", results, tokenizer=tok, max_prompt_tokens=300, template=template)
        large = build_rag_messages("x", results, tokenizer=tok, max_prompt_tokens=4096, template=template)
        self.assertLess(len(small.sources), len(large.sources))


class TestCitations(unittest.TestCase):

    SOURCES = {
        1: "Örnek Kütüphane Yönetmeliği Madde 6: Bir üye aynı anda en fazla beş materyal ödünç alabilir. "
           "Ödünç verme süresi on dört gündür.",
        2: "Örnek Kütüphane Yönetmeliği Madde 7: Gecikilen her gün için günlük 2 TL gecikme bedeli uygulanır.",
    }

    def test_parse_citation_ids(self):
        self.assertEqual(parse_citation_ids("a [1] b [2, 3] c [4-6] d [1][7]"), [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(parse_citation_ids("atıf yok"), [])

    def test_supported_answer(self):
        answer = "Ödünç verme süresi on dört gündür [1]. Gecikme bedeli günlük 2 TL'dir [2]."
        rep = verify_citations(answer, self.SOURCES)
        self.assertEqual(len(rep.sentences), 2)
        self.assertTrue(all(s.supported for s in rep.sentences))
        self.assertEqual(rep.citation_precision, 1.0)
        self.assertEqual(rep.supported_ratio, 1.0)
        self.assertEqual(rep.fabricated_ids, [])

    def test_fabricated_and_wrong_numbers(self):
        answer = "Ödünç süresi on dört gündür [3]. Gecikme bedeli günlük 5 TL'dir [2]."
        rep = verify_citations(answer, self.SOURCES)
        self.assertEqual(rep.fabricated_ids, [3])
        self.assertIn("fabricated_ids", rep.flags)
        self.assertFalse(rep.sentences[0].supported)
        self.assertEqual(rep.sentences[1].missing_numbers, ["5"])
        self.assertFalse(rep.sentences[1].supported)
        self.assertIn("unsupported_numbers", rep.flags)
        self.assertLess(rep.supported_ratio, 1.0)

    def test_uncited_factual_and_suggestion(self):
        answer = "Gecikme bedeli günlük 2 TL olarak uygulanır."
        rep = verify_citations(answer, self.SOURCES)
        s = rep.sentences[0]
        self.assertTrue(s.uncited_factual)
        self.assertEqual(s.suggested_id, 2)
        self.assertIsNone(rep.citation_precision)
        self.assertEqual(rep.supported_ratio, 0.0)

    def test_detached_citation_and_abstention(self):
        rep = verify_citations("Ödünç verme süresi on dört gündür. [1]", self.SOURCES)
        self.assertEqual(len(rep.sentences), 1)
        self.assertEqual(rep.sentences[0].cited_ids, [1])
        self.assertTrue(rep.sentences[0].supported)
        rep = verify_citations(NO_ANSWER, self.SOURCES)
        self.assertTrue(rep.abstained)
        self.assertIsNone(rep.supported_ratio)

    def test_accepts_numbered_source_objects_and_renders_report(self):
        results = [SearchResult(make_chunk("c1", self.SOURCES[1]), 2.0)]
        rag = build_rag_messages("soru", results)
        rep = verify_citations("Bir üye en fazla beş materyal ödünç alabilir [1]. Uydurma [9].", rag.sources)
        self.assertTrue(rep.sentences[0].supported)
        self.assertEqual(rep.fabricated_ids, [9])
        text = render_report(rep)
        self.assertIn("Atıf Doğrulama Raporu", text)
        self.assertIn("UYDURMA ATIF", text)
        self.assertIn("DESTEKLENİYOR", text)
        json.dumps(rep.to_dict(), ensure_ascii=False)


class TestCli(unittest.TestCase):

    def _run(self, argv):
        from rag.cli import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(argv)
        self.assertEqual(code, 0)
        return buf.getvalue()

    def test_index_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "idx.json")
            self.assertIn("İndeks kaydedildi", self._run(["index", "--docs", EXAMPLES, "--out", out]))
            text = self._run(["search", "--index", out, "--query", "gecikme bedeli"])
            self.assertIn("Madde 7", text)
            data = json.loads(self._run(["search", "--index", out, "--query", "gecikme bedeli", "--json", "--k", "2"]))
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["chunk"]["article"], "Madde 7")

    @unittest.skipUnless(os.path.exists(TOKENIZER_PATH), "tokenizer yok")
    def test_ask_end_to_end_with_tiny_model(self):
        from model.config import ModelConfig
        from model.tokenizer import ToprakTokenizer
        from model.transformer import ToprakLM
        tok = ToprakTokenizer(TOKENIZER_PATH)
        torch.manual_seed(0)
        config = ModelConfig(vocab_size=tok.get_vocab_size(), d_model=32, num_heads=4, num_kv_heads=2,
                             num_layers=1, d_ff=64, max_seq_len=512, device="cpu")
        model = ToprakLM(config, tokenizer=tok)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "tiny.pt")
            torch.save({"config": config.architecture_dict(), "model_state_dict": model.state_dict()}, ckpt)
            idx = os.path.join(tmp, "idx.json")
            self._run(["index", "--docs", EXAMPLES, "--out", idx])
            for extra in (["--speculative"], []):
                out = self._run(["ask", "--index", idx, "--checkpoint", ckpt, "--tokenizer", TOKENIZER_PATH,
                                 "--query", "Gecikme bedeli ne kadar?", "--max-new-tokens", "6",
                                 "--device", "cpu", "--json"] + extra)
                data = json.loads(out)
                self.assertLessEqual(data["prompt_tokens"], 512 - 6)
                self.assertGreater(len(data["sources"]), 0)
                self.assertIn("citation_precision", data["citations"])
            self.assertIsNotNone(data["sources"][0]["chunk"]["url"])


if __name__ == "__main__":
    unittest.main()
