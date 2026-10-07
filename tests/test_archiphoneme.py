# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Arkifonemik Biçimbilgisi Birim Testleri
`model/archiphoneme.py`: yüzey gerçekleştirme, ek soyutlama, codec gidiş-dönüşü,
korpus dönüşümü ve tokenizer sembolleri.
"""

import os
import subprocess
import sys
import tempfile
import unittest

from model.archiphoneme import (
    ABSTRACT_SUFFIXES,
    ArchiphonemeCodec,
    RuleBasedSegmenter,
    SURFACE_TO_ABSTRACT,
    abstract_suffix,
    archiphoneme_user_symbols,
    is_abstract_suffix,
    realize,
    tr_lower,
    tr_upper,
    transform_corpus_line,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRealize(unittest.TestCase):

    def check(self, stem, suffixes, expected):
        self.assertEqual(realize(stem, suffixes), expected, f"{stem}+{'+'.join(suffixes)}")

    def test_vowel_harmony_two_way(self):
        self.check("kitap", ["lAr", "DA"], "kitaplarda")
        self.check("ev", ["lAr", "DA"], "evlerde")
        self.check("okul", ["DAn"], "okuldan")

    def test_vowel_harmony_four_way(self):
        self.check("göz", ["lHk"], "gözlük")
        self.check("gel", ["DH"], "geldi")
        self.check("rol", ["(y)H"], "rolü")

    def test_consonant_assimilation(self):
        self.check("kitap", ["DA"], "kitapta")
        self.check("süt", ["CH"], "sütçü")
        self.check("kitap", ["CH"], "kitapçı")
        self.check("sev", ["GH"], "sevgi")

    def test_buffer_consonants(self):
        self.check("araba", ["(y)A"], "arabaya")
        self.check("ev", ["(y)A"], "eve")
        self.check("araba", ["(n)Hn"], "arabanın")
        self.check("araba", ["(y)DH"], "arabaydı")
        self.check("ev", ["(y)DH"], "evdi")
        self.check("oku", ["(H)r"], "okur")
        self.check("yap", ["(A)r"], "yapar")

    def test_softening(self):
        self.check("kitap", ["(s)H"], "kitabı")
        self.check("köpek", ["(s)H"], "köpeği")
        self.check("renk", ["(s)H"], "rengi")
        self.check("ağaç", ["(y)A"], "ağaca")
        self.check("at", ["(s)H"], "atı")            # tek heceli, yumuşamaz
        self.check("göz", ["lHk", "(s)H"], "gözlüğü")  # ek sonu k yumuşar
        self.check("gel", ["(y)AcAk", "(y)Hm"], "geleceğim")

    def test_non_softening_is_configurable(self):
        self.assertEqual(realize("kitap", ["(s)H"], non_softening={"kitap"}), "kitapı")
        self.assertEqual(realize("kitap", ["(s)H"], soften=False), "kitapı")

    def test_loanword_harmony_exceptions(self):
        self.check("saat", ["lAr"], "saatler")
        self.check("kalp", ["(s)H"], "kalbi")        # hem istisna uyum hem yumuşama
        self.check("saat", ["lAr", "DA"], "saatlerde")
        self.assertEqual(realize("saat", ["lAr"], harmony_exceptions=set()), "saatlar")

    def test_future_and_progressive(self):
        self.check("gel", ["(y)AcAk"], "gelecek")
        self.check("oku", ["(y)AcAk"], "okuyacak")
        self.check("gel", ["Hyor"], "geliyor")
        self.check("oku", ["Hyor"], "okuyor")
        self.check("ara", ["Hyor"], "arıyor")
        self.check("bekle", ["Hyor"], "bekliyor")
        self.check("söyle", ["Hyor"], "söylüyor")
        self.check("gel", ["mA", "Hyor"], "gelmiyor")
        self.check("oku", ["mA", "Hyor"], "okumuyor")
        self.check("gel", ["Hyor", "DH"], "geliyordu")

    def test_invariant_suffixes(self):
        self.check("araba", ["(y)ken"], "arabayken")
        self.check("ev", ["DA", "ki", "lAr"], "evdekiler")
        self.check("araba", ["DA", "ki"], "arabadaki")

    def test_turkish_casing(self):
        self.check("İstanbul'", ["DA"], "İstanbul'da")
        self.check("Iğdır'", ["DA"], "Iğdır'da")         # I → ı (kalın)
        self.check("İzmir'", ["DAn"], "İzmir'den")
        self.check("Mehmet'", ["(y)A"], "Mehmet'e")     # özel isimde yazım yumuşaması yok
        self.check("İstanbul", ["DAn"], "İstanbuldan")
        self.check("KİTAP", ["lAr"], "KİTAPLAR")
        self.check("KİTAP", ["(s)H"], "KİTABI")
        self.check("Kitap", ["lAr"], "Kitaplar")
        self.assertEqual(tr_lower("IĞDIR İZMİR"), "ığdır izmir")
        self.assertEqual(tr_upper("iğdır"), "İĞDIR")

    def test_plus_prefix_accepted(self):
        self.check("kitap", ["+lAr", "+DA"], "kitaplarda")


class TestAbstractSuffix(unittest.TestCase):

    def test_table(self):
        expected = {
            "lar": "lAr", "ler": "lAr", "da": "DA", "de": "DA", "ta": "DA", "te": "DA",
            "dan": "DAn", "ten": "DAn", "ı": "(y)H", "ü": "(y)H", "yi": "(y)H",
            "sı": "(s)H", "ya": "(y)A", "nın": "(n)Hn", "dı": "DH", "tü": "DH",
            "miş": "mHş", "acak": "(y)AcAk", "yecek": "(y)AcAk", "eceğ": "(y)AcAk",
            "ıyor": "Hyor", "üyor": "Hyor", "yor": "Hyor", "sa": "sA", "meli": "mAlH",
            "ip": "(y)Hp", "yup": "(y)Hp", "ken": "(y)ken", "yken": "(y)ken", "ki": "ki",
            "lık": "lHk", "lüğ": "lHk", "çı": "CH", "ci": "CH", "ları": "lArH",
            "ımız": "(H)mHz", "dır": "DHr",
        }
        for surf, abstract in expected.items():
            self.assertEqual(abstract_suffix(surf), abstract, surf)

    def test_abstract_passthrough_and_unknown(self):
        self.assertEqual(abstract_suffix("lAr"), "lAr")
        self.assertEqual(abstract_suffix("+DA"), "DA")
        self.assertEqual(abstract_suffix("LAR"), "lAr")
        with self.assertRaises(KeyError):
            abstract_suffix("xyz")

    def test_round_trip_through_realize(self):
        cases = [
            ("kitap", ["lar", "da"]), ("ev", ["ler", "den"]), ("araba", ["ya"]),
            ("kitap", ["ı"]), ("gel", ["ecek"]), ("oku", ["yor"]), ("göz", ["lük"]),
            ("süt", ["çü"]), ("okul", ["dan"]), ("araba", ["nın"]), ("gel", ["di", "m"]),
            ("ev", ["imiz", "de"]), ("saat", ["ler"]), ("kitap", ["ta"]),
        ]
        for stem, surfs in cases:
            expected = realize(stem, []) + "".join(surfs)
            if stem == "kitap" and surfs == ["ı"]:
                expected = "kitabı"
            got = realize(stem, [abstract_suffix(s) for s in surfs])
            self.assertEqual(got, expected, f"{stem}+{surfs}")

    def test_every_table_allomorph_maps_to_known_abstract(self):
        for surf, abstract in SURFACE_TO_ABSTRACT.items():
            self.assertIn(abstract, ABSTRACT_SUFFIXES)
            self.assertEqual(abstract_suffix(surf), abstract)


class TestCodec(unittest.TestCase):

    def setUp(self):
        self.codec = ArchiphonemeCodec()

    def test_encode_segmented_format(self):
        self.assertEqual(self.codec.encode_segmented("kitap", ["lar", "da"]), "kitap+lAr+DA")
        self.assertEqual(self.codec.encode_segmented("araba", ["ya"]), "araba+(y)A")

    def test_decode(self):
        self.assertEqual(
            self.codec.decode("kitap+lAr+DA ev+lAr+DA, araba+(y)A gel+(y)AcAk."),
            "kitaplarda evlerde, arabaya gelecek.",
        )

    def test_decode_leaves_plain_plus_alone(self):
        self.assertEqual(self.codec.decode("a+b C++ 2+2 x+"), "a+b C++ 2+2 x+")

    def test_sentence_round_trip(self):
        line = ("Kitaplarda ve evlerde, İstanbul'dan gelecek öğrenciler okuyor; "
                "Türkiye'nin saatleri  kitabı  gözlükçüler!")
        stats = {}
        abstract = transform_corpus_line(line, stats=stats)
        self.assertIn("Kitap+lAr+DA", abstract)
        self.assertIn("ev+lAr+DA,", abstract)
        self.assertIn("İstanbul'+DAn", abstract)
        self.assertIn("kitap+(y)H", abstract)
        self.assertEqual(self.codec.decode(abstract), line)
        self.assertGreater(stats["segmented"], 5)

    def test_pluggable_segmenter(self):
        lexicon = {"kitaplarda": ("kitap", ["lar", "da"]), "geliyor": ("gel", ["iyor"])}
        line = "kitaplarda geliyor bugün"
        out = transform_corpus_line(line, segmenter=lexicon.get)
        self.assertEqual(out, "kitap+lAr+DA gel+Hyor bugün")
        self.assertEqual(ArchiphonemeCodec().decode(out), line)

    def test_bad_segmentation_is_rejected(self):
        stats = {}
        out = transform_corpus_line("kitapta", segmenter=lambda w: ("kitap", ["da"]), stats=stats)
        self.assertEqual(out, "kitap+DA")   # "da" → DA, yüzey yine "kitapta"
        out = transform_corpus_line("evde", segmenter=lambda w: ("ev", ["xyz"]), stats=stats)
        self.assertEqual(out, "evde")
        out = transform_corpus_line("evde", segmenter=lambda w: ("ok", ["da"]), stats=stats)
        self.assertEqual(out, "evde")
        self.assertEqual(stats["rejected"], 2)   # bilinmeyen ek + yanlış kök


class TestSegmenter(unittest.TestCase):

    def test_heuristic_segmentation(self):
        seg = RuleBasedSegmenter()
        self.assertEqual(seg("kitaplarda"), ("kitap", ["lar", "da"]))
        self.assertEqual(seg("evlerde"), ("ev", ["ler", "de"]))
        self.assertEqual(seg("kitabı"), ("kitap", ["ı"]))
        self.assertEqual(seg("rengi"), ("renk", ["i"]))
        self.assertEqual(seg("bekliyor"), ("bekle", ["yor"]))
        self.assertEqual(seg("Türkiye'nin"), ("Türkiye'", ["nin"]))
        self.assertIsNone(seg("masa"))
        self.assertIsNone(seg("123"))

    def test_user_lexicon(self):
        seg = RuleBasedSegmenter(lexicon=["ağaç"])
        self.assertEqual(seg("ağacı"), ("ağaç", ["ı"]))


class TestUserSymbols(unittest.TestCase):

    def test_symbols(self):
        syms = archiphoneme_user_symbols()
        self.assertIn("+lAr", syms)
        self.assertIn("+DA", syms)
        self.assertIn("+(y)AcAk", syms)
        self.assertEqual(len(syms), len(set(syms)))
        for s in syms:
            self.assertTrue(s.startswith("+"))
            self.assertTrue(is_abstract_suffix(s[1:]))

    def test_sentencepiece_keeps_symbols_whole(self):
        import sentencepiece as spm
        with tempfile.TemporaryDirectory() as tmp:
            corpus = os.path.join(tmp, "c.txt")
            lines = ["kitap+lAr+DA ev+lAr+DAn araba+(y)A oku+Hyor"] * 30
            lines += ["bir iki üç dört beş altı yedi sekiz dokuz on"] * 30
            with open(corpus, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            prefix = os.path.join(tmp, "t")
            spm.SentencePieceTrainer.train(
                input=corpus, model_prefix=prefix, vocab_size=120, model_type="bpe",
                character_coverage=1.0, user_defined_symbols=archiphoneme_user_symbols(),
                minloglevel=2,
            )
            sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
            pieces = sp.encode("kitap+lAr+DAn", out_type=str)
            self.assertIn("+lAr", pieces)
            self.assertIn("+DAn", pieces)
            text = sp.decode(sp.encode("kitap+lAr+DAn araba+(y)A"))
            self.assertEqual(ArchiphonemeCodec().decode(text), "kitaplardan arabaya")


class TestCorpusCLI(unittest.TestCase):

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.txt")
            dst = os.path.join(tmp, "out.txt")
            text = "Kitaplarda ve evlerde okuyor.\nTürkiye'nin başkenti Ankara'dır.\n"
            with open(src, "w", encoding="utf-8") as f:
                f.write(text)
            res = subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts", "archiphoneme_corpus.py"),
                 "--input", src, "--output", dst, "--verify"],
                capture_output=True, text=True, cwd=ROOT, timeout=60,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("Doğrulama hatası: 0", res.stdout)
            with open(dst, encoding="utf-8") as f:
                out = f.read()
            self.assertIn("Kitap+lAr+DA", out)
            self.assertEqual(ArchiphonemeCodec().decode(out), text)


if __name__ == "__main__":
    unittest.main()
