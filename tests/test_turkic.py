# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Türk dünyası (çok dilli Türk dilleri) desteği testleri."""

import contextlib
import io
import json
import os
import random
import tempfile
import unittest

from data.mixture import load_mixture_config
from data.turkic import (
    TRANSLATION_TOKEN,
    TURKIC_LANGUAGES,
    ParallelExample,
    can_transliterate,
    detect_script,
    guess_turkic_language,
    kazakh_cyrillic_to_latin,
    kyrgyz_cyrillic_to_latin,
    language_tag_tokens,
    make_translation_example,
    normalize_ottoman,
    normalize_turkic,
    tag_text,
    tatar_cyrillic_to_latin,
    transliterate_to_latin,
    turkic_tokenizer_symbols,
    uyghur_arabic_to_latin,
    uzbek_cyrillic_to_latin,
)
from evaluation.turkic_tokenizer_report import (
    analyze_turkic_tokenizer,
    build_report,
    load_turkic_samples,
    measure_texts,
    render_table,
)
from model.chat_template import CHAT_SPECIAL_TOKENS
from model.tokenizer import ToprakTokenizer
from scripts.train_turkic_tokenizer import (
    build_balanced_corpus,
    main as train_main,
    plan_quotas,
    select_indices,
    temperature_probabilities,
    tokenizer_extra_symbols,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_PATH = os.path.join(PROJECT_ROOT, "evaluation", "turkic_seed.json")
MIXTURE_PATH = os.path.join(PROJECT_ROOT, "configs", "turkic_mixture.json")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "toprak_tokenizer.model")
EXPECTED_CODES = {"tr", "az", "tk", "uz", "kk", "ky", "tt", "ba", "ug", "crh", "gag", "ota"}


def _load_seed():
    with open(SEED_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)["samples"]


class TestRegistry(unittest.TestCase):
    def test_all_languages_present(self):
        self.assertEqual(set(TURKIC_LANGUAGES), EXPECTED_CODES)

    def test_language_fields(self):
        for code, language in TURKIC_LANGUAGES.items():
            self.assertEqual(language.code, code)
            self.assertEqual(language.tag, f"<dil:{code}>")
            self.assertTrue(language.name)
            self.assertEqual(len(language.iso639_3), 3)
            self.assertTrue(language.scripts)
            self.assertIn(language.primary_script, language.scripts)
            self.assertTrue(set(language.scripts) <= {"latin", "cyrillic", "arabic"})
            self.assertTrue(language.extra_letters)

    def test_extra_letter_notes(self):
        self.assertIn("ə", TURKIC_LANGUAGES["az"].extra_letters)
        for letter in ("ä", "ň", "ý", "ž"):
            self.assertIn(letter, TURKIC_LANGUAGES["tk"].extra_letters)
        for letter in "әғқңөұүһі":
            self.assertIn(letter, TURKIC_LANGUAGES["kk"].extra_letters)
        self.assertEqual(TURKIC_LANGUAGES["ota"].scripts, ("arabic",))
        self.assertEqual(TURKIC_LANGUAGES["ug"].primary_script, "arabic")

    def test_tokens_and_tags(self):
        tags = language_tag_tokens()
        self.assertEqual(len(tags), len(EXPECTED_CODES))
        self.assertEqual(len(set(tags)), len(tags))
        symbols = turkic_tokenizer_symbols()
        self.assertEqual(symbols[-1], TRANSLATION_TOKEN)
        self.assertEqual(TRANSLATION_TOKEN, "<çeviri>")
        self.assertEqual(tag_text("Сәлем", "kk"), "<dil:kk> Сәлем")
        with self.assertRaises(ValueError):
            tag_text("x", "xx")
        extra = tokenizer_extra_symbols()
        self.assertEqual(extra[: len(CHAT_SPECIAL_TOKENS)], list(CHAT_SPECIAL_TOKENS))
        self.assertEqual(len(extra), len(set(extra)))


class TestScriptAndLanguage(unittest.TestCase):
    def test_detect_script(self):
        self.assertEqual(detect_script("Bugün hava güzel")["dominant"], "latin")
        self.assertEqual(detect_script("Қазақстан")["dominant"], "cyrillic")
        self.assertEqual(detect_script("ئۇيغۇر تىلى")["dominant"], "arabic")
        self.assertIsNone(detect_script("123 !?")["dominant"])
        mixed = detect_script("abc где")
        self.assertAlmostEqual(mixed["latin"], 0.5)
        self.assertAlmostEqual(mixed["cyrillic"], 0.5)
        self.assertEqual(mixed["letters"], 6)
        self.assertEqual(detect_script("Oʻzbekiston")["latin"], 1.0)

    def test_seed_top1_accuracy(self):
        samples = _load_seed()
        correct = sum(
            guess_turkic_language(s["text"])[0][0] == s["lang"] for s in samples
        )
        self.assertGreaterEqual(correct / len(samples), 0.9)

    def test_clear_cases(self):
        cases = {
            "Mən Bakıda yaşayıram.": "az",
            "Men Aşgabatda ýaşaýaryn.": "tk",
            "Oʻzbekiston goʻzal va qadimiy yurt.": "uz",
            "O'zbekiston go'zal yurt.": "uz",
            "Біз кешке шай ішеміз.": "kk",
            "Бүгүн аба ырайы абдан жакшы.": "ky",
            "Татарстан — матур җир.": "tt",
            "Беҙ киске сәй эсәбеҙ.": "ba",
            "بۈگۈن ھاۋا ناھايىتى ياخشى.": "ug",
            "هوا بوگون پك گوزل.": "ota",
            "Ben İstanbul'da yaşıyorum.": "tr",
            "Bän Komratta yaşêêrim.": "gag",
            "Qırım — güzel bir yurt.": "crh",
        }
        for text, expected in cases.items():
            ranked = guess_turkic_language(text)
            self.assertEqual(ranked[0][0], expected, f"{text}: {ranked[:3]}")
            self.assertAlmostEqual(sum(score for _, score in ranked), 1.0)

    def test_guess_edge_cases(self):
        self.assertEqual(guess_turkic_language("1234 ..."), [])
        self.assertEqual(len(guess_turkic_language("Bugün hava güzel", top_k=2)), 2)
        ranked = guess_turkic_language("Қазақстан")
        self.assertTrue(all(TURKIC_LANGUAGES[c].scripts for c, _ in ranked))
        self.assertNotIn("tr", dict(ranked))


class TestTransliteration(unittest.TestCase):
    def test_kazakh(self):
        pairs = {
            "Қазақстан": "Qazaqstan",
            "Алматы": "Almaty",
            "Әліпби": "Älıpbi",
            "Шымкент": "Şymkent",
            "Ұлттық": "Ūlttyq",
            "Қайрат": "Qairat",
            "Тәуелсіздік": "Täuelsızdık",
            "ҚАЗАҚСТАН": "QAZAQSTAN",
            "Ірі": "Irı",
            "Ия": "İia",
        }
        for cyrillic, latin in pairs.items():
            self.assertEqual(kazakh_cyrillic_to_latin(cyrillic), latin)

    def test_uzbek(self):
        pairs = {
            "Ўзбекистон": "Oʻzbekiston",
            "шаҳар": "shahar",
            "Тошкент": "Toshkent",
            "ёшлар": "yoshlar",
            "ер": "yer",
            "поезд": "poyezd",
            "театр": "teatr",
            "ғалаба": "gʻalaba",
            "цирк": "sirk",
            "милиция": "militsiya",
            "маъно": "maʼno",
            "ШАҲАР": "SHAHAR",
            "Шаҳар": "Shahar",
            "Яхши": "Yaxshi",
        }
        for cyrillic, latin in pairs.items():
            self.assertEqual(uzbek_cyrillic_to_latin(cyrillic), latin)

    def test_kyrgyz(self):
        pairs = {
            "Кыргызстан": "Kırgızstan",
            "Бишкек": "Bişkek",
            "Ысык-Көл": "Isık-Köl",
            "жакшы": "jakşı",
            "мең": "meñ",
        }
        for cyrillic, latin in pairs.items():
            self.assertEqual(kyrgyz_cyrillic_to_latin(cyrillic), latin)

    def test_tatar(self):
        pairs = {
            "Казан": "Qazan",
            "китап": "kitap",
            "Татарстан": "Tatarstan",
            "авыл": "awıl",
            "яшим": "yäşim",
            "бүген": "bügen",
            "җир": "cir",
            "тугай": "tuğay",
        }
        for cyrillic, latin in pairs.items():
            self.assertEqual(tatar_cyrillic_to_latin(cyrillic), latin)

    def test_uyghur(self):
        pairs = {
            "ئۇيغۇر": "uyghur",
            "تىل": "til",
            "مەسئۇل": "mes'ul",
            "جەمئىيەت": "jem'iyet",
            "خوش": "xosh",
            "بۈگۈن ھاۋا ياخشى": "bügün hawa yaxshi",
            "رەھمەت، خەير خوش.": "rehmet, xeyr xosh.",
        }
        for arabic, latin in pairs.items():
            self.assertEqual(uyghur_arabic_to_latin(arabic), latin)
        self.assertEqual(uyghur_arabic_to_latin("ئۇيغۇر", capitalize_sentences=True), "Uyghur")
        # ن + گ dizisi ڭ (ng) ile karışmasın
        self.assertEqual(uyghur_arabic_to_latin("نگ"), "n'g")
        self.assertEqual(uyghur_arabic_to_latin("ڭ"), "ng")

    def test_seed_latin_fields_match_transliterators(self):
        for sample in _load_seed():
            if sample.get("latin") and sample["lang"] in ("kk", "ky", "tt"):
                self.assertEqual(
                    transliterate_to_latin(sample["text"], sample["lang"]), sample["latin"]
                )
            if sample.get("latin") and sample["lang"] == "ug":
                self.assertEqual(
                    uyghur_arabic_to_latin(sample["text"], capitalize_sentences=True),
                    sample["latin"],
                )

    def test_dispatch_and_ottoman_refusal(self):
        self.assertEqual(transliterate_to_latin("Bugün", "tr"), "Bugün")
        self.assertEqual(transliterate_to_latin("Алматы", "kk"), "Almaty")
        self.assertTrue(can_transliterate("az"))
        self.assertFalse(can_transliterate("ota"))
        self.assertFalse(can_transliterate("ba"))
        with self.assertRaises(ValueError):
            transliterate_to_latin("هوا", "ota")
        with self.assertRaises(ValueError):
            transliterate_to_latin("Башҡорт", "ba")


class TestNormalizationAndParallel(unittest.TestCase):
    def test_uzbek_apostrophes(self):
        for variant in ("'", "`", "‘", "’", "ʼ", "ʻ"):
            self.assertEqual(
                normalize_turkic(f"O{variant}zbekiston g{variant}alaba", "uz"),
                "Oʻzbekiston gʻalaba",
            )
        self.assertEqual(normalize_turkic("ma'no", "uz"), "maʼno")

    def test_other_normalizations(self):
        self.assertEqual(normalize_turkic("Ankara’da", "tr"), "Ankara'da")
        self.assertEqual(normalize_turkic("Азәrbaycan".replace("Аз", "Az"), "az"),
                         "Azərbaycan")
        self.assertEqual(normalize_turkic("бiз", "kk"), "біз")
        self.assertEqual(normalize_turkic("hello бiз", "kk"), "hello біз")
        decomposed = "güzel"
        self.assertEqual(normalize_turkic(decomposed, "tr"), "güzel")

    def test_ottoman_normalizer(self):
        text = "كتابـي"
        self.assertEqual(normalize_ottoman(text), "کتابی")
        harakat = "كِتَاب"
        self.assertEqual(normalize_ottoman(harakat, remove_harakat=True), "کتاب")
        self.assertIn("ِ", normalize_ottoman(harakat))

    def test_translation_example(self):
        text = make_translation_example("هوا بوگون پك گوزل.", "ota", "Hava bugün pek güzel.", "tr")
        self.assertEqual(text, "<dil:ota> هوا بوگون پک گوزل. <çeviri> <dil:tr> Hava bugün pek güzel.")
        example = ParallelExample("Сәлем", "kk", "Selam", "tr", source="parallel_turkic_tr",
                                  license="CC-BY-4.0")
        record = example.to_record()
        self.assertEqual(record["source"], "parallel_turkic_tr")
        self.assertTrue(record["text"].startswith("<dil:kk> Сәлем <çeviri> <dil:tr>"))
        with self.assertRaises(ValueError):
            ParallelExample("", "ota", "x", "tr")
        with self.assertRaises(ValueError):
            ParallelExample("x", "xx", "x", "tr")


class TestMixtureConfig(unittest.TestCase):
    def test_turkic_mixture_validates(self):
        config = load_mixture_config(MIXTURE_PATH)
        groups = config["groups"]
        self.assertEqual(
            set(groups), {"turkish", "oghuz", "kipchak", "karluk", "historical", "parallel"}
        )
        self.assertTrue(groups["turkish"]["default"])
        initial = sum(g["initial_weight"] for g in groups.values())
        final = sum(g["final_weight"] for g in groups.values())
        self.assertAlmostEqual(initial, 1.0)
        self.assertAlmostEqual(final, 1.0)
        self.assertAlmostEqual(groups["turkish"]["initial_weight"], 0.75)
        self.assertAlmostEqual(groups["turkish"]["final_weight"], 0.60)
        for name, group in groups.items():
            if name != "turkish":
                self.assertGreater(group["final_weight"], group["initial_weight"])

    def test_every_non_turkish_language_has_a_source(self):
        config = load_mixture_config(MIXTURE_PATH)
        all_sources = {s for g in config["groups"].values() for s in g["sources"]}
        for code in EXPECTED_CODES - {"tr", "ota"}:
            self.assertTrue(any(s.endswith(f"_{code}") for s in all_sources), code)
        self.assertIn("parallel_ota_tr", config["groups"]["parallel"]["sources"])


class TestBalancedSampling(unittest.TestCase):
    def test_temperature_probabilities(self):
        sizes = {"tr": 900, "kk": 90, "gag": 10}
        raw = temperature_probabilities(sizes, alpha=1.0)
        self.assertAlmostEqual(raw["tr"], 0.9)
        uniform = temperature_probabilities(sizes, alpha=0.0)
        for value in uniform.values():
            self.assertAlmostEqual(value, 1 / 3)
        probs = temperature_probabilities(sizes, alpha=0.3)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        expected = {k: (v / 1000) ** 0.3 for k, v in sizes.items()}
        norm = sum(expected.values())
        for lang in sizes:
            self.assertAlmostEqual(probs[lang], expected[lang] / norm)
        self.assertGreater(probs["gag"], 0.01)
        self.assertGreater(probs["tr"], probs["kk"])
        self.assertEqual(temperature_probabilities({"tr": 5, "ba": 0})["ba"], 0.0)
        with self.assertRaises(ValueError):
            temperature_probabilities({"tr": 0})
        with self.assertRaises(ValueError):
            temperature_probabilities({"tr": 1}, alpha=2.0)

    def test_quotas_respect_capacity_and_total(self):
        sizes = {"tr": 10_000, "kk": 500, "gag": 20}
        quotas = plan_quotas(sizes, 3000, alpha=0.3)
        self.assertEqual(sum(quotas.values()), 3000)
        self.assertEqual(quotas["gag"], 20)
        self.assertLessEqual(quotas["kk"], 500)
        self.assertEqual(plan_quotas(sizes, 3000, alpha=0.3), quotas)
        repeated = plan_quotas(sizes, 3000, alpha=0.3, max_repeat=3.0)
        self.assertEqual(repeated["gag"], 60)
        small = plan_quotas({"tr": 5, "kk": 5}, 100)
        self.assertEqual(small, {"tr": 5, "kk": 5})

    def test_select_indices_deterministic(self):
        first = select_indices(10, 4, random.Random("42:kk"))
        second = select_indices(10, 4, random.Random("42:kk"))
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 4)
        self.assertEqual(select_indices(3, 7, random.Random(0)).count(0), 2)
        self.assertEqual(select_indices(0, 5, random.Random(0)), [])

    def test_build_corpus_with_transliteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            kk = os.path.join(tmp, "kk.txt")
            tr = os.path.join(tmp, "tr.jsonl")
            with open(kk, "w", encoding="utf-8") as handle:
                handle.write("Алматы\nҚазақстан\n")
            with open(tr, "w", encoding="utf-8") as handle:
                for i in range(20):
                    handle.write(json.dumps({"text": f"Satır {i}\nİkinci {i}"}) + "\n")
            out = os.path.join(tmp, "corpus.txt")
            stats = build_balanced_corpus(
                {"kk": [kk], "tr": [tr]}, out, total_lines=12, alpha=0.3,
                transliteration="both",
            )
            self.assertEqual(stats["languages"]["tr"]["available_lines"], 40)
            self.assertEqual(stats["languages"]["kk"]["selected_lines"], 2)
            with open(out, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            self.assertIn("Almaty", lines)
            self.assertIn("Қазақстан", lines)
            self.assertEqual(len(lines), 12 + 2)
            with open(out, encoding="utf-8") as handle:
                first_run = handle.read()
            build_balanced_corpus({"kk": [kk], "tr": [tr]}, out, 12, 0.3,
                                  transliteration="both")
            with open(out, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), first_run)


class TestTokenizerReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = load_turkic_samples([SEED_PATH])

    def test_seed_loading(self):
        for code in EXPECTED_CODES:
            self.assertGreaterEqual(len(self.samples[code]), 5, code)
        self.assertIn("ota-latn", self.samples)
        self.assertIn("kk-latn", self.samples)
        self.assertEqual(list(self.samples)[0], "tr")
        no_latin = load_turkic_samples([SEED_PATH], include_latin=False)
        self.assertNotIn("kk-latn", no_latin)

    def test_report_on_real_tokenizer(self):
        if not os.path.exists(TOKENIZER_PATH):
            self.skipTest("toprak_tokenizer.model yok")
        tokenizer = ToprakTokenizer(TOKENIZER_PATH)
        analysis = analyze_turkic_tokenizer(tokenizer, self.samples, "current", TOKENIZER_PATH)
        languages = analysis["languages"]
        self.assertAlmostEqual(languages["tr"]["fertility_vs_tr"], 1.0)
        # Türkçe tokenizer akraba dillerde belirgin biçimde daha fazla token üretir.
        self.assertGreater(languages["kk"]["tokens_per_word"], languages["tr"]["tokens_per_word"])
        self.assertGreater(languages["kk"]["tokens_per_word"],
                           languages["kk-latn"]["tokens_per_word"])
        for metrics in languages.values():
            self.assertEqual(metrics["unknown_rate"], 0.0)
        table = render_table(build_report([analysis]))
        self.assertIn("| current | kk |", table)

    def test_tiny_trained_tokenizer_via_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            for lang, texts in self.samples.items():
                if lang.endswith("-latn"):
                    continue
                with open(os.path.join(tmp, f"{lang}.txt"), "w", encoding="utf-8") as handle:
                    handle.write("\n".join(texts * 3) + "\n")
            prefix = os.path.join(tmp, "tiny")
            with contextlib.redirect_stdout(io.StringIO()):
                code = train_main([
                    "--input-dir", tmp, "--total-lines", "150",
                    "--corpus-out", os.path.join(tmp, "corpus", "c.txt"),
                    "--model-prefix", prefix, "--vocab-size", "500",
                ])
            self.assertEqual(code, 0)
            tokenizer = ToprakTokenizer(prefix + ".model")
            for symbol in turkic_tokenizer_symbols() + list(CHAT_SPECIAL_TOKENS):
                self.assertNotEqual(tokenizer.token_to_id(symbol), tokenizer.unk_token_id)
            metrics = measure_texts(tokenizer, self.samples["ug"])
            self.assertEqual(metrics["samples"], 5)
            self.assertEqual(metrics["unknown_rate"], 0.0)
            self.assertGreater(metrics["characters_per_token"], 0)
            self.assertIsNotNone(metrics["byte_token_rate"])


if __name__ == "__main__":
    unittest.main()
