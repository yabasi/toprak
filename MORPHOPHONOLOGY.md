# 🌱 Toprak — Morfofonoloji: Arkifonemik Tokenizasyon ve Uyum Korumalı Kod Çözme

Bu belge Toprak'ın Türkçe ses-biçim kurallarını doğrudan kullanan iki özelliğini anlatır:

1. **Arkifonemik biçim** (`model/archiphoneme.py`): ekler soyut biçimde (`+lAr`, `+DA`, `+(y)AcAk`) öğrenilir, yüzey biçimi kurallarla türetilir.
2. **Grammar guard** (`inference/grammar_guard.py`): eğitim gerektirmeyen, üretim sırasında uyumu bozan ek tokenlerini maskeleyen logit işlemcisi.

Ölçüm aracı: `evaluation/harmony_check.py`.

---

## 1. Motivasyon

Standart BPE tokenizer `lar` ile `ler`'i, `da`/`de`/`ta`/`te`'yi birbirinden bağımsız tokenler olarak görür. Model hangi allomorfun geleceğini her bağlamda yeniden öğrenmek zorundadır ve küçük modellerde `kitapler`, `evlarda`, `kitapda` gibi hatalar görülür. Yardımcı kayıplar (`VowelHarmonyLoss`, `ConsonantHarmonyLoss`) bu hataları azaltır ama ortadan kaldırmaz.

Arkifonemik gösterimde model yalnız **hangi ekin** geleceğine karar verir (`+lAr`, `+DA`); ekin **nasıl yazılacağı** deterministik kurallarla belirlenir. Böylece ünlü uyumu ve ünsüz benzeşmesi hataları **yapısal olarak imkânsız** hale gelir; ayrıca aynı ekin 2–8 allomorfu tek tokende birleştiği için vocab daha verimli kullanılır.

Grammar guard ise mevcut (yüzey biçimli) modellere sıfır eğitim maliyetiyle aynı güvenceyi kısmen sağlar.

---

## 2. Arkifonemler ve Kurallar

| Arkifonem | Yüzey | Kural |
|---|---|---|
| `A` | a / e | Büyük ünlü uyumu: son ünlü kalınsa `a`, inceyse `e` |
| `H` | ı / i / u / ü | Küçük ünlü uyumu: kalın-düz `ı`, ince-düz `i`, kalın-yuvarlak `u`, ince-yuvarlak `ü` |
| `D` | d / t | Sert ünsüzden (f s t k ç ş h p — *fıstıkçı şahap*) sonra `t` |
| `C` | c / ç | Sert ünsüzden sonra `ç` |
| `G` | g / k | Sert ünsüzden sonra `k` (sev+GH → sevgi) |
| `(y)`, `(n)`, `(s)` | — | Kaynaştırma: yalnız ünlüyle biten biçimden sonra (araba+(y)A → arabaya, ev+(y)A → eve) |
| `(H)`, `(A)` | — | Koşullu ünlü: yalnız ünsüzle biten biçimden sonra (kitap+(H)m → kitabım, araba+(H)m → arabam) |

`realize(stem, suffixes)` ayrıca şunları uygular:

- **Süreksiz ünsüz yumuşaması**: ünlüyle başlayan ekten önce p→b, ç→c, t→d, k→ğ, nk→ng (kitap+(s)H → kitabı, renk+(s)H → rengi). Karar sırası:
  1. `non_softening` listesinde → yumuşamaz (at, saç, top, et, ait, hukuk, ahlak, devlet, saat, bırak …);
  2. `softening_words` listesinde → yumuşar (kalp, dip, uç, yurt, dört, kanat, umut …);
  3. `nk` ile bitiyorsa → yumuşar;
  4. tek heceliyse → yumuşamaz;
  5. çok heceli ve `t` ile bitiyorsa → yumuşamaz (çoğu Arapça alıntı veya fiil);
  6. aksi halde (çok heceli p/ç/k) → yumuşar.
  Ek sonundaki `k` (`-lHk`, `-(y)AcAk`, `-DHk`) her zaman yumuşar: göz+lHk+(s)H → gözlüğü, gel+(y)AcAk+(y)Hm → geleceğim.
- **-Hyor öncesi daralma**: ünlüyle biten fiilde `H` düşer; son ünlü a/e ise önceki ünlüye göre daralır: oku+Hyor → okuyor, ara+Hyor → arıyor, bekle+Hyor → bekliyor, söyle+Hyor → söylüyor, gel+mA+Hyor → gelmiyor.
- **Alıntı kelime uyum istisnaları**: son ünlüsü kalın olduğu halde ince ek alan kökler (`DEFAULT_HARMONY_EXCEPTIONS`): saat, kalp, harf, hal, rol, gol, alkol, petrol, istikbal, ihtimal, hayal, kabul, usul, kontrol, sembol, dikkat, itaat, cemaat … (saat+lAr → saatler, kalp+(s)H → kalbi, rol+(y)H → rolü).
- **Değişmez ekler**: `yor`, `(y)ken`, `ki` (evdekiler, arabayken).
- **Türkçe harf düzeni**: `tr_lower` / `tr_upper` (I↔ı, İ↔i). Kökün harf düzeni korunur; tümü büyükse ekler de büyür (KİTAP+(s)H → KİTABI).
- **Özel isimler**: kökün sonundaki kesme işareti (`İstanbul'`) özel isim demektir; yazımda yumuşama yapılmaz (Mehmet'+(y)A → Mehmet'e), Iğdır'+DA → Iğdır'da.

Tüm listeler `realize(..., non_softening=..., harmony_exceptions=..., softening_words=...)` ve `ArchiphonemeCodec(...)` parametreleriyle değiştirilebilir.

### Ek tablosu (`SUFFIX_TABLE`)

| Grup | Soyut ekler |
|---|---|
| Çoğul | `lAr` |
| Hâl | `(y)H`, `(y)A`, `DA`, `DAn`, `(n)Hn`, `(y)lA`, `CA` |
| İyelik | `(H)m`, `(H)n`, `(s)H`, `(H)mHz`, `(H)nHz`, `lArH`; zamir n'li hâl: `nDA`, `nDAn`, `nA`, `nH` |
| Ek-fiil | `(y)DH`, `(y)mHş`, `(y)sA`, `DHr`, `(y)ken` |
| Kişi | `(y)Hm`, `sHn`, `(y)Hz`, `sHnHz`, `m`, `n`, `k`, `nHz` |
| Zaman/kip | `DH`, `mHş`, `(y)AcAk`, `Hyor`, `sA`, `mAlH`, `(H)r`, `(A)r`, `mA`, `mAz` |
| Fiilimsi | `mAk`, `(y)Hp`, `DHk`, `(y)An`, `(y)Abil` |
| Yapım | `lHk`, `CH`, `lH`, `sHz`, `GH`, `ki` |

`abstract_suffix("ler") == "lAr"`, `abstract_suffix("ten") == "DAn"`, `abstract_suffix("ı") == "(y)H"`, `abstract_suffix("yor") == "Hyor"`. Bir yüzey biçimi birden çok soyut eke karşılık geliyorsa (ör. `ı` = belirtme ya da iyelik) tablodaki ilk ek seçilir; bu durumlar aynı bağlamda aynı yüzeyi verdiğinden gidiş-dönüş bozulmaz.

---

## 3. Soyut Metin Biçimi

Tokenizer eğitimi için kesin biçim:

```
kitaplarda            → kitap+lAr+DA
evlerde,              → ev+lAr+DA,
Türkiye'nin           → Türkiye'+(n)Hn
gelecek               → gel+(y)AcAk
masa                  → masa            (çekimsiz / çözümlenemeyen kelime aynen)
```

- Kelimeler boşlukla ayrılır, orijinal boşluklar korunur.
- Ekler köke **boşluksuz** `+` ile bağlanır. SentencePiece `user_defined_symbols` ile her `+Ek` tek token olur (`▁kitap`, `+lAr`, `+DA`) ve `▁` kelime sınırı bilgisi korunur.
- Sondaki noktalama son ekten sonra gelir; baştaki noktalama kökten önce kalır.
- `ArchiphonemeCodec.decode` yalnız geçerli ek dizilerini birleştirir; `a+b`, `C++`, `2+2` olduğu gibi kalır. Ham metninde `+` içeren kelimeler dönüştürülmez.

---

## 4. Korpus → Tokenizer → Çıkarım İş Akışı

```bash
# 1) Korpusu soyut biçime çevir (yerleşik sezgisel segmenter)
python scripts/archiphoneme_corpus.py --input data/corpus.txt \
    --output data/corpus_arch.txt --verify [--lexicon kokler.txt]
```

```python
# 2) Tokenizer'ı arkifonemik sembollerle eğit
from model.tokenizer import train_tokenizer
from model.chat_template import CHAT_SPECIAL_TOKENS
from model.archiphoneme import archiphoneme_user_symbols

train_tokenizer(
    "data/corpus_arch.txt",
    model_prefix="toprak_arch_tokenizer",
    extra_symbols=list(CHAT_SPECIAL_TOKENS) + archiphoneme_user_symbols(),
)

# 3) Model bu tokenizer ile soyut metin üzerinde eğitilir.
# 4) Çıkarımda üretilen soyut metni yüzey Türkçe'ye çevir
from model.archiphoneme import ArchiphonemeCodec
codec = ArchiphonemeCodec()
print(codec.decode("kitap+lAr+DA oku+Hyor+lAr."))   # → "kitaplarda okuyorlar."
```

`transform_corpus_line(line, segmenter)` her kelimede `decode(encode(kelime)) == kelime` doğrulaması yapar; tutmayan analizler reddedilir (`stats["rejected"]`). Bu yüzden dönüşüm **kayıpsızdır**: `codec.decode(transform_corpus_line(line)) == line`.

### Zemberek / TRmorph entegrasyonu

Segmenter, `kelime -> (kök, [yüzey ekleri]) | None` imzalı herhangi bir çağrılabilirdir:

```python
def zemberek_segmenter(word):
    analysis = morphology.analyze_and_disambiguate(word)  # örnek API
    if not analysis:
        return None
    return analysis.lemma, analysis.surface_morphemes      # ör. ("kitap", ["lar", "da"])

line_abs = transform_corpus_line(line, segmenter=zemberek_segmenter)
```

Yerleşik `RuleBasedSegmenter` **sezgiseldir**: sözlük olmadan bilinen ek allomorflarını sağdan soyar, gidiş-dönüş doğrular ve en kısa kökü (≥ 3 harf) ya da sözlükteki kökü seçer. "ağacı → ağa+cı", "gidiyor → gid+iyor" gibi yanlış ama kayıpsız analizler yapabilir. Üretim korpusları için morfolojik çözümleyici önerilir.

---

## 5. Grammar Guard (Uyum Korumalı Kod Çözme)

`HarmonyGuard(tokenizer, mode="mask"|"penalty", penalty=5.0, exceptions=...)` bir `logits_processor`'dır: `(generated_ids: List[int], logits: Tensor(1, V)) -> Tensor`.

**Ön hesaplama** (32K vocab için ~0.1 sn): her token için ilk ünlü sınıfı, kelime başı bayrağı, `d`/`c`+ünlü başlangıcı; buradan (son ünlü sınıfı × sert ünsüz) için 4 boolean maske.

**Her adım** (~0.2 ms): üretilen id'lerde geriye doğru kelime başına (`▁` token) kadar yürünür, parçalar birleştirilir (kesme işareti saydamdır: `▁İstanbul` `'` → `istanbul'`). Kelimenin son ünlüsü ve son harfine göre hazır maske seçilir ve tek bir `masked_fill` (veya ceza çıkarma) uygulanır.

Kurallar:
- son ünlü kalın → ilk ünlüsü ince olan ek tokenleri ihlal (`▁kitap` + `ler`),
- son ünlü ince → ilk ünlüsü kalın olan ek tokenleri ihlal (`▁ev` + `lar`),
- kelime sert ünsüzle bitiyor (f s t k ç ş p) → `d`/`c` + ünlü ile başlayan ek tokenleri ihlal (`▁kitap` + `da`). `h` varsayılan olarak hariçtir (tahdit, istihdam, mahdut).

Muafiyetler:
- değişmez ekler: `yor…`, `ken…`, `leyin`, `gil`, `ki`, `kiler` …;
- istisna kökler (saat, kalp, hal …): **hiç maske uygulanmaz** — `▁saat` + `lar` de `ler` de serbesttir. İnceye çevirmek yerine muafiyet seçildi çünkü `▁hal` + `ı` = *halı* gibi gerçek kelimeler bozulurdu;
- ünlüsüz, harf dışı, kelime başı ve özel tokenler;
- `restrict_to_suffixes=True` (varsayılan): yalnız bilinen ek allomorflarına bölünebilen tokenler denetlenir; böylece `▁ki` + `tap` gibi kök içi bölünmeler maskelenmez;
- **asla her şey maskelenmez**: en olası `fallback_top_k` (50) adayın hepsi ihlalse adım değiştirilmez (`stats.fallbacks`).

`GuardStats`: `steps`, `applied` (maske uygulanan adım), `top1_changed` (en olası token ihlal ediyordu), `fallbacks`, `exempt`.

### Kullanım

```bash
python inference/generate.py --checkpoint checkpoints/toprak_best.pt \
    --prompt "Kitaplar" --grammar-guard mask
python inference/generate.py --grammar-guard penalty --guard-penalty 3.0 ...
```

```python
from inference.generate import generate_text
from inference.grammar_guard import build_grammar_guard

guard = build_grammar_guard(tokenizer, mode="mask")
text = generate_text(model, tokenizer, "Türkiye'nin", logits_processors=[guard])
print(guard.stats.as_dict())
```

---

## 6. Ölçüm: `evaluation/harmony_check.py`

`harmony_violation_rate(text)` metni kelime kelime tarar, sondaki bilinen allomorfları soyar ve her eki önceki yüzey biçiminden `realize` ile beklenenle karşılaştırır:

```python
from evaluation.harmony_check import harmony_violation_rate
harmony_violation_rate("kitapler kitapda saatler İstanbul'de")
# {'checked': 4, 'violations': 3, 'vowel_violations': 2, 'consonant_violations': 1, 'rate': 0.75, ...}
```

Guard'lı ve guard'sız çıktıları karşılaştırmak için:

```bash
python inference/generate.py --num-samples 20 ... > guardsiz.txt
python inference/generate.py --num-samples 20 --grammar-guard mask ... > guardli.txt
python evaluation/harmony_check.py guardsiz.txt guardli.txt
```

Ölçüm sezgiseldir: ek gibi biten kökler (ör. *hikaye* → "hika"+"ye") yanlış alarm verebilir; beklenen d/c yerine t/ç (başkent+i) köke ait olabileceğinden sayılmaz. Mutlak değerden çok **göreli karşılaştırma** için kullanın.

---

## 7. Sınırlamalar

- **Düzensiz fiiller** desteklenmez: git→gidiyor/gider, et→eder, tat→tadar (kurallar `gitiyor`/`etiyor` üretir; bunlar sözlüksel bilgi gerektirir). `de`/`ye` + `Hyor` (diyor, yiyor) desteklenir.
- **Sözlüksel ses olayları**: ünlü düşmesi (ağız→ağzı, burun→burnu), ikizleşme (hak→hakkı, his→hissi), `-mAk`+`(y)A` gibi eski biçimler.
- **Yumuşama** sezgisel listelere dayanır; özellikle çok heceli `-t` ile bitenler (kanat→kanadı) liste gerektirir.
- **Sayılar ve kısaltmalar** (1990'da, TBMM'de) okunuşa göre uyum alır; desteklenmez (bu kelimeler dönüşümde aynen kalır).
- **Uyum istisnaları** tam eşleşmeyle bakılır; birleşik kelimeler (çalarsaat) için listeye eklenmelidir.
- **Grammar guard** yalnız büyük ünlü uyumunu (2 yönlü) ve d/c benzeşmesini denetler; küçük ünlü uyumu (ı/i/u/ü seçimi) ve yumuşama denetlenmez. Kesme sonrası sayılar (1990'da) için bağlam boştur, müdahale edilmez.
- **Yerleşik segmenter** sözlüksüzdür; dilbilimsel doğruluk için Zemberek/TRmorph takılmalıdır (bkz. §4).
