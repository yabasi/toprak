# Türk Dünyası: Çok Dilli Türk Dilleri Desteği

Toprak'ın ana dili Türkiye Türkçesidir. Bu belge, aynı modele akraba Türk
dillerini (Azerbaycan, Türkmen, Özbek, Kazak, Kırgız, Tatar, Başkurt, Uygur,
Kırım Tatar, Gagavuz Türkçeleri ve Osmanlı Türkçesi) **Türkçeyi baskın
tutarak** eklemek için gereken araçları ve tarifleri anlatır.

> Durum: araçlar, yapılandırma ve ölçümler hazırdır. Bu dillerle eğitilmiş bir
> Toprak modeli ve Türk dünyası tokenizer'ı henüz yayımlanmamıştır.

## Vizyon

Türk dilleri ortak bir dil bilgisi iskeletini paylaşır: eklemeli (aglütinatif)
yapı, ünlü uyumu, SOV dizilişi, son çekim edatları, iyelik ve hâl ekleri.
`ev-ler-imiz-de`, Kazakça `үй-лер-іміз-де`, Özbekçe `uy-lar-imiz-da` aynı
morfolojik zinciri izler. Bu nedenle:

- **Diller arası aktarım (cross-lingual transfer):** Türkçede öğrenilen ek
  sıralaması ve ünlü uyumu bilgisi akraba dillere taşınır; düşük kaynaklı
  diller (Gagavuzca, Kırım Tatarcası, Başkurtça) yüksek kaynaklı dillerin
  temsilinden faydalanır. Toprak'ın ünlü uyumu / ünsüz uyumu yardımcı kayıpları
  (bkz. [ABLATION.md](ABLATION.md)) bu ortak yapıyı açıkça hedefler.
- **Ortak söz varlığı:** `su/суу/сув`, `göz/көз/koʻz`, `kitap/кітап/kitob`
  gibi ortak kökler aynı ya da benzer alt-kelime parçalarına düşer — tokenizer
  bu dilleri gördüğü sürece.
- **Erişim:** 200 milyonu aşkın Türk dili konuşuru için tek bir açık model;
  Osmanlıca ↔ Türkçe çeviri ile tarihî metinlerin okunması.

## Dil tablosu

Makine tarafından okunan kayıt: [`data/turkic.py`](data/turkic.py) →
`TURKIC_LANGUAGES`.

| Kod | Dil | ISO 639-3 | Kol | Yazılar (ana yazı kalın) | Ek harf notları | Etiket |
|---|---|---|---|---|---|---|
| tr | Türkiye Türkçesi | tur | Oğuz | **Latin** | ç ğ ı İ ö ş ü | `<dil:tr>` |
| az | Azerbaycan Türkçesi | aze | Oğuz | **Latin**, Kiril (eski), Arap (İran) | ə q x | `<dil:az>` |
| tk | Türkmen Türkçesi | tuk | Oğuz | **Latin**, Kiril (eski) | ä ň ý ž | `<dil:tk>` |
| uz | Özbek Türkçesi | uzb | Karluk | **Latin**, Kiril | oʻ gʻ sh ch ng; Kiril ў ғ қ ҳ | `<dil:uz>` |
| kk | Kazak Türkçesi | kaz | Kıpçak | **Kiril**, Latin (2021), Arap (Çin) | ә ғ қ ң ө ұ ү һ і | `<dil:kk>` |
| ky | Kırgız Türkçesi | kir | Kıpçak | **Kiril**, Arap (Çin) | ң ө ү | `<dil:ky>` |
| tt | Tatar Türkçesi | tat | Kıpçak | **Kiril**, Latin (Zamanälif) | ә ө ү җ ң һ | `<dil:tt>` |
| ba | Başkurt Türkçesi | bak | Kıpçak | **Kiril** | ә ө ү ғ ҡ ң ҙ ҫ һ | `<dil:ba>` |
| ug | Uygur Türkçesi | uig | Karluk | **Arap (UEY)**, Latin (ULY), Kiril | ئا ئە ئو ئۇ ئۆ ئۈ ئې ئى | `<dil:ug>` |
| crh | Kırım Tatar Türkçesi | crh | Kıpçak (Oğuz etkili) | **Latin**, Kiril | â ñ q | `<dil:crh>` |
| gag | Gagavuz Türkçesi | gag | Oğuz | **Latin**, Kiril (eski) | ä ê ţ | `<dil:gag>` |
| ota | Osmanlı Türkçesi | ota | Oğuz (tarihî) | **Arap** | Arap-Fars harfleri, ڭ | `<dil:ota>` |

Ek özel token: `<çeviri>` (paralel/çeviri örneklerinde kaynak ile hedefi ayırır).

## Veri kaynakları

Aşağıdaki kaynaklar **aday** listesidir. Kodlar/konfigürasyon adları zamanla
değişebilir; kullanmadan önce her veri kartını ve lisansı kendiniz doğrulayın.
[DATA_GOVERNANCE.md](DATA_GOVERNANCE.md) kuralları burada da geçerlidir:
`data/governance.py` içindeki `SOURCE_REGISTRY`'de kaydı olmayan her kaynak
`review_required` olarak işaretlenir.

| Kaynak | Diller (kayıttaki ipucu) | Lisans notu |
|---|---|---|
| FineWeb-2 (`HuggingFaceFW/fineweb-2`) | `azj_Latn`, `tuk_Latn`, `uzn_Latn`/`uzn_Cyrl`, `kaz_Cyrl`, `kir_Cyrl`, `tat_Cyrl`, `bak_Cyrl`, `uig_Arab`, `crh_Latn`, `gag_Latn` | ODC-By 1.0 + Common Crawl kullanım şartları (Türkçe için doğrulandı; diğer alt kümeler için veri kartını kontrol edin) |
| CulturaX (`uonlp/CulturaX`) | az, tk, uz, kk, ky, tt, ba, ug (dil kodları) | mC4 ve OSCAR üst kaynak koşulları — ayrı inceleme gerekli |
| MADLAD-400 (`allenai/MADLAD-400`) | Türk dillerinin çoğu; düşük kaynaklı diller için değerli | Veri kartındaki lisans ve Common Crawl şartları — doğrulayın |
| Wikipedia dump'ları (`dumps.wikimedia.org/<kod>wiki/`) | tr, az, tk, uz, kk, ky, tt, ba, ug, crh, gag | GFDL + CC BY-SA (atıf ve aynı lisansla paylaşım yükümlülükleri) |
| Osmanlıca | Vikikaynak, dijitalleştirilmiş kamu malı basma eserler, akademik derlemler | Telif durumu eser bazında değişir; OCR kalitesi denetlenmeli |
| Paralel veri | OPUS (Tatoeba, TED, Bible vb.), FLORES-200 (yalnız değerlendirme!) | Alt derlem başına farklı lisans; FLORES'i eğitime **koymayın** |

Önerilen kaynak adlandırması: `<kaynak>_<dil>` (`fineweb2_kk`, `wiki_uz`,
`culturax_ba`). Böylece JSONL `source` alanı doğrudan
[`configs/turkic_mixture.json`](configs/turkic_mixture.json) gruplarına
eşlenir. Yeni kaynakları `SOURCE_REGISTRY`'e eklemek (lisans kaydıyla) ayrı bir
yönetişim adımıdır.

## Tokenizer

### Neden yeni bir tokenizer?

Mevcut 32K Türkçe BPE tokenizer'ın ([`toprak_tokenizer.model`](toprak_tokenizer.model))
[`evaluation/turkic_seed.json`](evaluation/turkic_seed.json) üzerindeki
ölçümü (dil başına 5 gündelik cümle; `xx-latn` satırları aynı cümlelerin
Latin transliterasyonu, `ota-latn` ise insan yazımı transkripsiyondur):

```bash
python evaluation/turkic_tokenizer_report.py --tokenizer current=toprak_tokenizer.model
```

| Dil | Token/kelime | ×tr | Karakter/token | Byte token | UNK |
|---|---:|---:|---:|---:|---:|
| tr | 1.71 | 1.00 | 3.69 | 0.0% | 0.0% |
| az | 2.56 | 1.50 | 2.13 | 0.0% | 0.0% |
| tk | 3.17 | 1.86 | 1.86 | 3.5% | 0.0% |
| uz | 3.00 | 1.76 | 2.17 | 0.0% | 0.0% |
| kk | 5.10 | 2.99 | 1.03 | 0.0% | 0.0% |
| kk-latn | 3.05 | 1.79 | 1.72 | 0.0% | 0.0% |
| ky | 5.72 | 3.35 | 1.08 | 0.0% | 0.0% |
| ky-latn | 2.78 | 1.63 | 2.22 | 0.0% | 0.0% |
| tt | 5.00 | 2.93 | 1.08 | 4.7% | 0.0% |
| tt-latn | 3.18 | 1.86 | 1.72 | 0.0% | 0.0% |
| ba | 6.00 | 3.52 | 0.93 | 17.6% | 0.0% |
| ug | 7.07 | 4.14 | 0.85 | 11.3% | 0.0% |
| ug-latn | 3.07 | 1.80 | 2.02 | 0.0% | 0.0% |
| crh | 2.35 | 1.38 | 2.25 | 0.0% | 0.0% |
| gag | 2.22 | 1.30 | 2.42 | 0.0% | 0.0% |
| ota | 5.47 | 3.21 | 0.96 | 1.9% | 0.0% |
| ota-latn | 2.00 | 1.17 | 3.19 | 0.0% | 0.0% |

Yorum:

- Latin yazılı Oğuz dilleri (az, crh, gag) Türkçe tokenizer'la makul
  (×1.3–1.5); Türkmence ve Özbekçe ×1.8 civarında.
- Kiril ve Arap yazılı diller **neredeyse harf harf** bölünüyor
  (karakter/token ≈ 1). Kazakça, Kırgızca, Tatarca ve Başkurtça Türkçeden 3–3.5
  kat, Uygurca 4 kattan fazla token harcıyor; bu da aynı bağlam penceresine
  3–4 kat az metin sığması ve eğitim/çıkarım maliyetinin aynı oranda artması
  demektir.
- Başkurtça (ҙ ҫ ҡ) ve Uygurca harflerin bir kısmı vocab'da hiç yok; %11–18
  token byte fallback'e düşüyor.
- Aynı cümleler Latin'e çevrildiğinde fertility ~3'e iniyor: sorun dilin
  kendisi değil, tokenizer'ın bu yazıları hiç görmemiş olması.
- Seed seti çok küçüktür; sayılar yön gösterir, kesin değildir. Karar için
  dil başına binlerce cümlelik held-out örnekle tekrarlayın (`--input`).

### Tarif

[`scripts/train_turkic_tokenizer.py`](scripts/train_turkic_tokenizer.py):

```bash
# data_cache/turkic/ içinde tr.txt, az.jsonl, kk.txt, ... (dosya adı = dil kodu)
python scripts/train_turkic_tokenizer.py \
  --input-dir data_cache/turkic \
  --total-lines 3000000 \
  --alpha 0.3 \
  --seed 42 \
  --transliterate none \
  --corpus-out data_cache/turkic_tokenizer_corpus.txt \
  --stats-out data_cache/turkic_tokenizer_stats.json \
  --model-prefix toprak_turkic_tokenizer \
  --vocab-size 64000
```

- **Dengeli örnekleme:** dil başına satır sayısı `n_i` için
  `p_i ∝ (n_i/N)^α`. `α=1` ham oran, `α=0` eşit dağılım; varsayılan `α=0.3`
  düşük kaynaklı dilleri yukarı çeker ama Türkçeyi en büyük pay olarak tutar.
  Bir dilin kotası mevcut satırını (`--max-repeat` ile çarpılmış) aşamaz;
  artan pay diğer dillere dağıtılır. Seçim `--seed` ile deterministiktir.
  Fonksiyonlar: `temperature_probabilities`, `plan_quotas`, `select_indices`,
  `build_balanced_corpus`.
- **Karakter kapsamı:** `character_coverage=0.99995` (Kiril, Arap ve ek Latin
  harflerin vocab'a girmesi için 0.9999'dan yüksek).
- **Özel tokenlar:** `extra_symbols = CHAT_SPECIAL_TOKENS + turkic_tokenizer_symbols()`
  yani sohbet tokenları + 12 dil etiketi + `<çeviri>`; her biri tek token olur.
- **Vocab:** 12 dil ve 3 yazı için 32K dardır; 48K–64K önerilir. Türkçe
  verimini [TOKENIZER_ANALYSIS.md](TOKENIZER_ANALYSIS.md) araçlarıyla ayrıca
  kontrol edin — yeni tokenizer Türkçede belirgin gerileme yaratmamalıdır.
- **Değerlendirme:** yeni tokenizer'ı eskisiyle aynı tabloda karşılaştırın:

```bash
python evaluation/turkic_tokenizer_report.py \
  --tokenizer current=toprak_tokenizer.model \
  --tokenizer turkic=toprak_turkic_tokenizer.model \
  --output evaluation/reports/turkic-tokenizer.json \
  --markdown evaluation/reports/turkic-tokenizer.md
```

Not: Tokenizer değişirse mevcut checkpoint'ler yeni vocab ile uyumsuzdur;
Türk dünyası modeli yeni bir ön eğitim koşusu (veya embedding genişletme +
devam eğitimi) gerektirir.

## Veri karışımı

[`configs/turkic_mixture.json`](configs/turkic_mixture.json) mevcut
`toprak-mixture-v1` şemasını kullanır ve `validate_mixture_config`'ten geçer
(bkz. [DATA_MIXTURE.md](DATA_MIXTURE.md)):

| Grup | Kaynaklar | Başlangıç | Bitiş |
|---|---|---:|---:|
| `turkish` (default) | `wiki`, `fineweb2`, `culturax`, `*_tr` | 0.75 | 0.60 |
| `oghuz` | `*_az`, `*_tk`, `*_gag`, `*_crh` | 0.08 | 0.12 |
| `kipchak` | `*_kk`, `*_ky`, `*_tt`, `*_ba` | 0.07 | 0.11 |
| `karluk` | `*_uz`, `*_ug` | 0.05 | 0.08 |
| `historical` | `ota_corpus`, `ota_wikisource` | 0.01 | 0.03 |
| `parallel` | `parallel_ota_tr`, `parallel_turkic_tr`, `parallel_translit` | 0.04 | 0.06 |

- Curriculum: model önce sağlam bir Türkçe temeli öğrenir, 20K adım boyunca
  akraba dillerin ve paralel verinin payı lineer artar.
- `turkish` default gruptur: **eşleşmeyen her kaynak Türkçe sayılır.** Yeni
  dil kaynaklarını mutlaka ilgili grubun `sources` listesine ekleyin.
- Kırım Tatarcası dil bilimsel olarak Kıpçak koluna yakındır; Latin yazımı ve
  söz varlığı Türkiye Türkçesine çok yakın olduğu için pratik nedenle `oghuz`
  grubundadır.
- Sampler her grupta veri bekler; `pretokenize.py` boş grupta erken hata verir.
  Elinizde olmayan grupları (ör. `historical`) config'ten çıkarın.
- Grup içinde diller veri büyüklüğüyle orantılı örneklenir; grup içi denge
  gerekiyorsa dilleri ayrı gruplara bölün veya düşük kaynaklı dilin
  shard'ını çoğaltın.

## Transliterasyon politikası

Fonksiyonlar (`data/turkic.py`), deterministik ve büyük/küçük harf korur:

| Fonksiyon | Şema | Örnek |
|---|---|---|
| `kazakh_cyrillic_to_latin` | 2021 resmî Kazak Latin alfabesi (ә→ä, ғ→ğ, қ→q, ң→ñ, ө→ö, ұ→ū, ү→ü, һ→h, і→ı, ы→y, ш→ş, ч→ç, ж→j) | Қазақстан → Qazaqstan, Әліпби → Älıpbi |
| `kyrgyz_cyrillic_to_latin` | Ortak Türk Alfabesi temelli yaygın şema (ы→ı, ң→ñ, ж→j, ч→ç, ш→ş, й→y); resmî Kırgız Latin alfabesi yoktur | Кыргызстан → Kırgızstan |
| `uzbek_cyrillic_to_latin` | Resmî 1995 Özbek Latin alfabesi (ш→sh, ч→ch, ў→oʻ, ғ→gʻ, қ→q, ҳ→h, х→x, ъ→ʼ; е → ye kelime başı/ünlü sonrası, ц → ts ünlü sonrası, aksi hâlde s) | Ўзбекистон → Oʻzbekiston, шаҳар → shahar |
| `tatar_cyrillic_to_latin` | Zamanälif'e yakın (җ→c, ң→ñ, х→x; к/г/я/ю kelimenin ön/art ünlüsüne göre k/g/yä/yü ↔ q/ğ/ya/yu; в ünlüden sonra w) | Казан → Qazan, китап → kitap |
| `uyghur_arabic_to_latin` | UEY → ULY (ünlüler ا ە و ۇ ۆ ۈ ې ى; kelime başı ئ sessiz, kelime içi ئ → `'`; n'g, s'h ayrımı) | ئۇيغۇر → uyghur, تىل → til |

Belgelenmiş yaklaşımlar: Kazakçada и/й → i ve у → u bağlamdan bağımsızdır
(2021 alfabesinde de tek harfe iner); ю/я → iu/ia, ц → ts, щ → şş. Tatarcada
ön/art ünlü kararı kelime düzeyindedir; Rusça alıntılarda yanılabilir.
Başkurtça için transliterasyon tanımlı değildir.

**Native yazıyı korumak mı, Latin'e çevirmek mi?**

| Seçenek | Artı | Eksi |
|---|---|---|
| Native yazı (varsayılan, `--transliterate none`) | Model gerçek dünyadaki metni okur/yazar; kullanıcı Kiril/Arap yazıyla soru sorabilir; bilgi kaybı yok | Ortak kökler farklı yazılarda farklı token olur, aktarım daha zayıf; tokenizer'ın üç yazıyı kapsaması gerekir |
| Hepsini Latin'e çevir (`latin`) | Ortak kökler aynı parçalara düşer, Türkçeden aktarım en güçlü; daha küçük vocab yeter | Çıktı kullanıcının yazısında değildir (geri çeviri gerekir, çoğu zaman belirsiz); transliterasyon hataları korpusa girer |
| İkisi birden (`both`) | Tokenizer iki biçimi de öğrenir; paralel `parallel_translit` verisiyle yazılar arası köprü kurulur | Korpus büyür; aynı içerik iki kez görülür (tekrar etkisi) |

Öneri: ön eğitimde **native yazı** + küçük bir `parallel_translit` grubu
(aynı cümlenin native ve Latin biçimi `<çeviri>` ile) — model hem gerçek
yazıyı öğrenir hem de yazılar arası eşlemeyi görür. Tokenizer korpusunda
`both` modu düşünülebilir.

## Osmanlıca yaklaşımı

Osmanlı Türkçesi Arap harfli bir abjad ile yazılır; ünlüler çoğunlukla
gösterilmez ve aynı yazım birden çok okunabilir (`كوزل` → güzel/gözel/küzel…).
Kurallı bir harf çevirisi **anlamlı bir sonuç vermez**; bu yüzden
`transliterate_to_latin(..., "ota")` bilerek hata verir.

Bunun yerine:

1. **Tek dilli Osmanlıca metin** (`historical` grubu): `normalize_ottoman`
   ile NFKC, keşide temizliği, isteğe bağlı hareke kaldırma ve
   ك/ک, ي/ى/ی birleştirmesi.
2. **Paralel veri** (`parallel` grubu): insan yapımı transkripsiyon ve
   sadeleştirmeler. `ParallelExample` / `make_translation_example`:

```python
from data.turkic import ParallelExample, make_translation_example

make_translation_example("هوا بوگون پك گوزل.", "ota", "Hava bugün pek güzel.", "tr")
# '<dil:ota> هوا بوگون پک گوزل. <çeviri> <dil:tr> Hava bugün pek güzel.'

ParallelExample("Сәлем", "kk", "Selam", "tr",
                source="parallel_turkic_tr", license="CC-BY-4.0").to_record()
# {"text": "<dil:kk> Сәлем <çeviri> <dil:tr> Selam", "source": "parallel_turkic_tr", ...}
```

İki yönü de (ota→tr ve tr→ota) eğitime koymak faydalıdır. Değerlendirme için
eğitimde kullanılmayan, editörlü bir paralel set ayırın.

## Dil tespiti ve `data/` hattında gereken değişiklik

Bugün dil filtresi [`data/crawler.py`](data/crawler.py) içindedir: `langdetect`
ile `detect(text[:1000]) != "tr"` olan sayfalar atılır.
[`data/cleaner.py`](data/cleaner.py) ayrı bir dil filtresi uygulamaz (kalite
skoru `isalpha()` tabanlıdır ve yazıdan bağımsızdır), ancak boilerplate ve PII
kalıpları Türkçeye özgüdür. Çok dilli kullanım için yapılması gerekenler
(bu paket bu dosyaları değiştirmez):

1. **`langdetect` yetersiz:** profilleri az/tk/uz/kk/ky/tt/ba/ug/crh/gag/ota
   içermez; Kazakça/Kırgızca/Tatarca genelde `ru`/`bg`/`mk`, Uygurca ve
   Osmanlıca `ar`/`fa`/`ur` olarak etiketlenir ve atılır. Bunun yerine
   fastText `lid.176` (az, tk, uz, kk, ky, tt, ba, ug, crh dâhil) veya
   GlotLID (gag, crh ve yazı alt etiketleri `kaz_Cyrl` gibi) kullanılmalıdır.
2. Filtre tek bir `"tr"` yerine **izin verilen dil kümesi** almalı
   (`--allowed-languages tr,az,kk,...`) ve tespit edilen dili/yazıyı
   (`lang`, `script`, `lang_confidence`) belge metadata'sına yazmalıdır;
   Toprak dil kodları için `data.turkic.TURKIC_LANGUAGES` anahtarları
   kullanılabilir.
3. Kaynak etiketinden gelen dil (ör. FineWeb-2 `kaz_Cyrl` alt kümesi) ile
   tespit edilen dil uyuşmuyorsa belge reddedilmeli veya işaretlenmelidir.
   `guess_turkic_language` + `detect_script` hızlı bir **ikinci görüş** olarak
   kullanılabilir (ör. Kazakça kaynakta Kiril oranı < %80 → şüpheli).
4. `normalize_turkic(text, lang)` temizlemede NFKC'den önce çağrılmalı
   (Özbekçe oʻ/gʻ kesmeleri, Kazakçada Latin `i` → `і` gibi düzeltmeler).
5. Boilerplate (`reklam`, `devamını oku`…) ve PII (T.C. kimlik, +90 telefon)
   kalıpları dil başına genişletilmeli; aksi hâlde diğer dillerde gürültü
   kalır.
6. Contamination taraması çok dilli değerlendirme setlerini (FLORES-200,
   Belebele, TUMLU vb.) de içermelidir.

`guess_turkic_language` bilinçli olarak **sezgiseldir**: ayırt edici harfler
(ə → az; ň/ý/ž → tk; oʻ/gʻ → uz; қ/ұ/і → kk; ң/ө/ү ama қ/ғ/ә/і yok → ky;
җ → tt; ҙ/ҫ/ҡ → ba; ې/ۆ/ۈ → ug), sık işlev kelimeleri ve birkaç ek kalıbıyla
puan verir. Seed setinde 60 cümlenin 58'ini doğru bilir; yanıldığı ikisi
(ayırt edici harfi olmayan Başkurtça ve Gagavuzca cümleler) zaten
belirsizdir. Türkçe dışı metinleri (Rusça, Farsça, İngilizce) de en yakın Türk
diline atar; üretim filtresi olarak tek başına kullanılmamalıdır.

## Değerlendirme önerileri

- **Tokenizer:** `evaluation/turkic_tokenizer_report.py` ile dil başına
  fertility, karakter/token, byte fallback ve UNK; büyük held-out örneklerle.
- **Dil başına perplexity / bits-per-byte:** tokenizer'dan bağımsız
  karşılaştırma için bits-per-byte tercih edin.
- **FLORES-200** (çeviri, chrF++): tr↔az/tk/uz/kk/ky/tt/ba/ug/crh yönleri.
- **Belebele** (okuduğunu anlama): mevcut Türk dili alt kümeleri (ör. az, kk,
  ky, uz; kapsamı veri kartından kontrol edin).
- **SIB-200** (konu sınıflandırma) ve Türk dillerine özgü çok dilli
  anlama benchmarkları (ör. TUMLU; sürüm ve lisansı doğrulayın).
- **Osmanlıca:** editörlü paralel test seti üzerinde ota→tr chrF ve insan
  değerlendirmesi.
- **Türkçe gerileme:** mevcut `evaluation/` paketi (bkz. [EVALUATION.md](EVALUATION.md))
  ile Türkçe skorların düşmediğini doğrulayın; Türk dünyası modeli Türkçeden
  ödün vermemelidir.

## Sınırlamalar

- Seed seti küçük ve proje ekibi tarafından yazılmıştır; yazımlar özenle
  seçildi ancak anadil konuşurlarınca gözden geçirilmelidir.
- Dil tahmini sezgiseldir; Türk dili olmayan metinleri reddetmez.
- Transliterasyonlar yaklaşık kurallar içerir (Kazakça и/у, Tatarca ön/art
  ünlü kararı, Rusça alıntılar); geri çevrim (Latin → Kiril) sağlanmaz.
  Kırgızca için resmî Latin alfabe yoktur, kullanılan şema bir tercihtir.
- Başkurtça ve Azerbaycan/Kazak Arap yazıları için transliterasyon yoktur;
  Güney Azerbaycan Türkçesi ayrı bir kaynak olarak ele alınmamıştır.
- Osmanlıca için otomatik transliterasyon bilerek sunulmaz; kaliteli paralel
  veri sınırlıdır.
- Mixture config'teki kaynak adları öneridir; verinizdeki `source` alanlarıyla
  eşleştirmeniz gerekir. Eşleşmeyen kaynaklar Türkçe grubuna düşer.
- Veri kaynaklarının lisansları dil alt kümesine göre değişebilir; tablodaki
  notlar hukuki görüş değildir.

## Kod haritası

| Dosya | İçerik |
|---|---|
| [`data/turkic.py`](data/turkic.py) | Kayıt, yazı tespiti, dil tahmini, normalizasyon, transliterasyon, paralel veri |
| [`configs/turkic_mixture.json`](configs/turkic_mixture.json) | Türk dünyası curriculum karışımı |
| [`scripts/train_turkic_tokenizer.py`](scripts/train_turkic_tokenizer.py) | Dengeli korpus + tokenizer eğitimi |
| [`evaluation/turkic_tokenizer_report.py`](evaluation/turkic_tokenizer_report.py) | Dil başına tokenizer raporu |
| [`evaluation/turkic_seed.json`](evaluation/turkic_seed.json) | 12 dil × 5 cümle seed seti |
| [`tests/test_turkic.py`](tests/test_turkic.py) | Birim testleri |
