# Toprak: 32K Bağlam ve Kaynaklı Türkçe RAG

Bu belge iki şeyi anlatıyor. Birincisi, 2K bağlamla ön eğitilmiş bir Toprak
checkpoint'inin **YaRN** ile 16K–32K bağlama nasıl genişletileceği ve bunun
**passkey (iğne) testiyle** nasıl ölçüleceği. İkincisi, mevzuat ve yönetmelik
metinleri için **atıflı (kaynak gösteren) Türkçe RAG** sistemi: indeksleme,
arama, token bütçeli prompt, üretim ve atıf doğrulama.

> **Önemli:** Bu sistem hukuki danışmanlık vermez. Cevaplar yalnızca indekslenen
> metinlere dayanır ve hatalı olabilir. Resmi metni her zaman kaynağından
> doğrulayın.

---

## 1. YaRN: RoPE'u neden ve nasıl genişletiyoruz?

RoPE, her `(q, k)` boyut çiftini konuma bağlı bir açıyla döndürür. Çift `i`
için frekans `θ_i = base^(-2i/d)` olur. Bazı boyutların frekansı **yüksektir**:
birkaç token içinde tam tur atarlar ve yakın tokenlar arasındaki göreli sırayı
kodlarlar. Bazılarının frekansı ise **düşüktür**: eğitim bağlamı boyunca (ör. 2048
token) bir tur bile tamamlamazlar ve uzak mesafeyi kodlarlar.

Bağlamı 2K'nın ötesine taşıyınca sorun düşük frekanslı boyutlarda çıkar. Model
bu boyutlarda eğitimde hiç görmediği açılarla karşılaşır ve attention bozulur.
Üç çözüm var:

| Yöntem | Ne yapar | Sorunu |
|---|---|---|
| Linear (Position Interpolation) | Tüm frekansları `s` faktörüyle böler | Yüksek frekanslar da sıkışır, yerel sıra bilgisi bulanıklaşır |
| NTK-aware | `base` değerini büyütür | Daha iyi, ama bazı boyutlar yine eğitim aralığının dışına çıkar |
| **YaRN** | Boyutları frekansa göre ayrı ele alır | İnce ayarla en iyi sonucu verir |

YaRN (`model/rope.py → scaled_inv_freqs`) şöyle çalışır:

1. **Yüksek frekanslı boyutlar olduğu gibi kalır** (ekstrapolasyon). Bu boyutlar
   eğitim bağlamında zaten defalarca tam tur attı. Yeni konumlarda gördükleri
   açılar eğitimde görülenlerle aynı aralıktadır. Bunları sıkıştırmak, modelin
   kelime içi ve kelimeler arası yerel sırayı ayırt etmesini zorlaştırırdı.
   Bu, Türkçe gibi bir kelimenin 3–6 tokena bölündüğü bir dilde özellikle önemlidir.
2. **Düşük frekanslı boyutlar `s` ile bölünür** (interpolasyon). 2K boyunca bir
   tur bile atmayan boyutlar, 32K bağlamda da eğitimde gördükleri açı aralığında
   kalır.
3. **Aradaki bant bir rampayla karıştırılır.** Sınırlar, eğitim bağlamında kaç
   tam tur atıldığına göre belirlenir: `beta_fast=32` turdan fazlası korunur,
   `beta_slow=1` turdan azı interpole edilir.
4. **Attention sıcaklığı (mscale):** Bağlam uzayınca softmax daha çok tokena
   dağılır ve entropi artar, yani attention "dağınık" hale gelir. YaRN bunu
   `mscale = 0.1·ln(s) + 1` ile dengeler. Toprak bu çarpanı `freqs_cis`
   genliğine uygular. q ve k aynı tablo ile döndürüldüğü için `q·k` skorları
   `mscale²` ile ölçeklenir, yani logitler hafifçe keskinleşir. `s=16` için
   `mscale ≈ 1.277`.

Yapılandırma:

```python
config.max_seq_len = 32768
config.rope_scaling = {"type": "yarn", "factor": 16.0, "original_max_seq_len": 2048}
# RoPE tablosu güvenlik payıyla 2 × max_seq_len konum için hesaplanır
```

YaRN parametre eklemez. Yalnızca RoPE tablosu değişir. Checkpoint'e yazılan
`rope_scaling` alanı, `load_model` ile yükleme sırasında modeli doğru şekilde
yeniden kurar.

---

## 2. Tarif: 2K checkpoint'i 32K'ya genişletmek

### 2.1 Kademeli devam eğitimi (önerilen)

Doğrudan 2K'dan 32K'ya atlamak yerine iki aşama daha kararlı sonuç verir:

```bash
# Aşama 1: 2K → 8K (factor 4)
python training/train.py --model-size large \
    --data-dir data_bin_long --bin-mode \
    --resume checkpoints/toprak_best.pt \
    --max-seq-len 8192 --rope-scaling yarn --rope-factor 4 \
    --rope-original-max-seq-len 2048 \
    --batch-size 1 --grad-accum 32 --lr 2e-5 \
    --max-steps <mevcut_adım + 1000> --bf16 \
    --checkpoint-dir checkpoints/long8k

# Aşama 2: 8K → 32K (factor 16, hâlâ orijinal 2K'ya göre)
python training/train.py --model-size large \
    --data-dir data_bin_long --bin-mode \
    --resume checkpoints/long8k/toprak_last.pt \
    --max-seq-len 32768 --rope-scaling yarn --rope-factor 16 \
    --rope-original-max-seq-len 2048 \
    --batch-size 1 --grad-accum 16 --lr 1e-5 \
    --max-steps <mevcut_adım + 600> --bf16 \
    --checkpoint-dir checkpoints/long32k
```

Görev tanımındaki tek adımlı 16K varyantı da kullanılabilir:
`--max-seq-len 16384 --rope-scaling yarn` (`--rope-factor` verilmezse
`max_seq_len / önceki_max_seq_len` = 8 olarak hesaplanır).

**Notlar:**

- `--rope-factor` her zaman **orijinal ön eğitim bağlamına** göre verilir.
  `--rope-original-max-seq-len 2048` değeri iki aşamada da aynı kalır.
- `--resume` ağırlıkları, optimizer durumunu ve LR scheduler durumunu
  geri yükler. Cosine programı ön eğitimin sonundaysa LR zaten düşük
  olacaktır. Bu, uzun bağlam uyarlaması için genellikle uygundur. Aynı
  sebeple `--max-steps` değerini mevcut adımın **üzerine** ekleyerek verin.
  Taze bir warmup ile yeni LR programı istiyorsanız yalnızca ağırlık
  yükleyen bir başlatma gerekir. Bu seçenek `train.py`'de henüz yoktur.
- **Adım ve token önerisi:** YaRN makalesindeki deneylerde birkaç yüz adım
  (~0,1–0,5 milyar token) yeterli oldu. Toprak boyutundaki modeller için
  başlangıç noktası olarak aşama başına 400–1000 adım öneriyoruz. Adım başına
  efektif batch hedefi ~0,25–0,5M token, LR ise ön eğitim tepe LR'sinin %5–10'u
  kadar olmalı (`1e-5`–`3e-5`). Passkey ve perplexity eğrisi doyduğunda durun.
- **Veri:** Uzun bağlam *gerçek uzun belge* ister. Kısa belgeleri
  birleştirmek model için yalnızca "uzak tokenları yok say" sinyali üretir.
  Uygun kaynaklar: tam Wikipedia makaleleri, kitap bölümleri, mevzuat metninin
  tamamı (kanun ve yönetmelikler), uzun mahkeme kararları, raporlar. Önerilen
  karışım: ~%60 uzun belge (≥8K token), ~%40 normal ön eğitim verisi. Böylece
  kısa bağlam yeteneği unutulmaz. Kaynakların lisansı `DATA_GOVERNANCE.md`
  kurallarına göre kaydedilmelidir.
- **Bellek:**
  - Gradient checkpointing varsayılan olarak açıktır (`--no-grad-checkpoint`
    ile kapatılır) ve 16K–32K bağlamda **zorunludur**.
  - Attention, SDPA (flash/memory-efficient çekirdek) kullanır. Attention
    belleği `O(T)` kalır, ama hesap maliyeti `O(T²)` olarak artar.
  - En büyük tek tensör çoğu zaman **logit**'tir: `T × V × 4 bayt`. Örneğin
    `16384 × 32000 × 4 ≈ 2,1 GB`, 32K'da ≈ 4,2 GB (gradyanıyla iki katı).
    `--batch-size 1` ve `--grad-accum` kullanın, `--bf16` açın.
  - Çıkarım sırasında KV cache: `2 × katman × kv_heads × head_dim × T × 2 bayt`.
    Large modelde (28 katman, 4 KV head, 64 boyut) 32K için ≈ 0,94 GB, Small
    modelde ≈ 0,24 GB. GQA bu maliyeti dörtte bire indirir.
  - Apple Silicon (MPS) üzerinde 32K eğitimi pratik değildir. Tek bir
    A100/H100 80GB ile Large modeli 32K'da eğitmek mümkündür (bkz. `RUNPOD.md`).

### 2.2 Değerlendirme: passkey (iğne) testi

```bash
python scripts/passkey_eval.py \
    --checkpoint checkpoints/long32k/toprak_last.pt \
    --tokenizer toprak_tokenizer.model \
    --lengths 1024 2048 4096 8192 16384 32768 \
    --depths 0 0.25 0.5 0.75 1 --trials 5 --perplexity \
    --json-out passkey_32k.json
```

- Türkçe dolgu metninin içine istenen derinliğe (%0 = baş, %100 = son)
  `"Gizli anahtar sayı: 48213. Bunu hatırla."` cümlesi yerleştirilir. Metin
  `Cevap: Gizli anahtar sayı:` ile biter ve modelin greedy cevabındaki ilk sayı
  beklenen sayıyla **birebir** karşılaştırılır.
- Prompt token düzeyinde kurulur. Uzunluk hedefi asla aşmaz ve iğne bir cümle
  sınırına yerleşir.
- Model bağlamını (`config.max_seq_len`) aşan uzunluklar atlanır ve
  raporlanır. `--allow-extrapolation` verilirse RoPE tablosunun sonuna kadar
  (2×) denenir.
- `--perplexity` her uzunluk için dolgu metninin perplexity'sini hesaplar.
  Hesap 2K'lık pencerelerle ve KV cache üzerinden yapılır, bellek sınırlı
  kalır. Bağlam genişletme başarılıysa perplexity uzunlukla **artmamalıdır**.
- Çıktı uzunluk × derinlik doğruluk tablosu ile JSON kayıtlarıdır.
  Fonksiyonlar (`build_passkey_prompt`, `run_passkey_grid`, `score_passkey`,
  `format_grid_table`) içe aktarılabilir.

Beklenen tablo: genişletme öncesi 2K checkpoint 2K'nın ötesinde ~0 doğruluk
gösterir. Başarılı bir YaRN devam eğitiminden sonra hedef bağlama kadar tüm
derinliklerde yüksek doğruluk beklenir. Ortadaki derinliklerdeki (%25–%75)
düşüş "lost in the middle" etkisidir.

Passkey testi gerekli bir koşuldur ama yeterli değildir. Model uzun metni
gerçekten *kullanıyor* mu, bunu `evaluation/benchmarks/long_context.jsonl`
ve aşağıdaki RAG değerlendirmesi ile birlikte ölçün.

---

## 3. Kaynaklı Türkçe RAG

```
belgeler (.txt/.md/JSONL)
   │  rag/chunker.py   madde yapısını koruyan, örtüşmeli parçalama + provenance
   ▼
BM25 indeksi (JSON)
   │  rag/index.py     Türkçe kök + madde/sayı/ifade ek puanları
   ▼
ilk k parça
   │  rag/prompt.py    numaralı kaynaklar, token bütçesi (2K … 32K)
   ▼
ChatTemplate → generate_text / n-gram spekülatif çözümleme
   │
   ▼
cevap + [n] atıfları
   │  rag/citations.py atıf doğrulama raporu
   ▼
kullanıcı
```

### 3.1 Komutlar

```bash
# 1) İndeks: klasör (.txt/.md, isteğe bağlı front matter) veya JSONL
python -m rag.cli index --docs rag/examples --out rag_index.json
#    (--max-chars 1200 --overlap 1 --k1 1.5 --b 0.75)

# 2) Arama
python -m rag.cli search --index rag_index.json --query "gecikme bedeli ne kadar"
python -m rag.cli search --index rag_index.json --query "Geçici Madde 1" --json
python -m rag.cli search --index rag_index.json --query '"ilk otuz dakikası ücretsizdir"'

# 3) Kaynaklı cevap + atıf doğrulama
python -m rag.cli ask --index rag_index.json \
    --checkpoint checkpoints/toprak_best.pt --tokenizer toprak_tokenizer.model \
    --query "Kütüphaneden aynı anda en fazla kaç materyal ödünç alınabilir?" \
    --speculative
```

`rag/examples/` altındaki üç belge (**Örnek Kütüphane Yönetmeliği**, **Örnek
Belediyesi Bisiklet Paylaşım Yönergesi**, **Örnek Topluluk Bahçesi Kullanım
Esasları**) bu proje için yazılmış **kurgusal** metinlerdir. Gerçek mevzuat
değildirler ve hukuki geçerlilikleri yoktur. Front matter'da `fictional: true`
olarak işaretlidirler.

**Belge biçimleri.** JSONL satırı `{"id","title","text","url","license","date"}`
biçimindedir. Toprak korpus şemasının (`toprak-document-v1`) `source_url`,
`licenses`, `downloaded_at`, `source` alanları ve iç içe bir `provenance` nesnesi
de tanınır. Böylece `data/governance.py` ile işlenmiş bir korpus doğrudan
indekslenebilir. Bu alanlar her parçaya taşınır ve `ask` çıktısında kaynak
listesinde (url | lisans | tarih) gösterilir.

### 3.2 Türkçe metin işleme (`rag/text.py`)

- `turkish_lower`: İ→i ve I→ı. Python'un `lower()` fonksiyonu "İ" harfini
  "i̇" yapar, "I" harfini de "i" yapar.
- Aksanlar (ç, ğ, ı, ö, ş, ü) varsayılan olarak **korunur**.
  `normalize(..., strip_accents=True)` isteğe bağlıdır.
- Tokenizer şu terimleri tanır:
  - madde atıfları: `Madde 12`, `m. 12/3`, `md. 5`, `12 nci maddesi`,
    `Geçici Madde 1` → `madde:12`, `madde:12/3`, `geçici_madde:1`
  - sayılar: `5237`, `01.01.2020`
  - kesme işaretli özel adlar: `Kanun'un` → `kanun`
- Kök bulucu **sezgiseldir**: yaygın çekim ekleri en uzun eşleşme önce
  silinir, en az 3 harflik kök kalır. Ünlü/kaynaştırma ünsüzü kuralları
  uygulanır ve b/c/ğ yumuşaması geri alınır. Hukuki terimler için koruma
  sözlüğü vardır (`kanun`, `madde`, `yönetmelik` …). Morfolojik çözümleyici
  değildir: bazı kökleri bozar (ör. `teslim` → `tesl`). Sorgu ve belgeye aynı
  işlem uygulandığından arama için tutarlıdır.
- Cümle bölücü kısaltmaları (Md., vb., Dr., T.C., s., No., Bkz. …), sıra
  sayılarını ("15. maddede"), madde başlıklarını ve fıkra numaralarını
  ("(2)", "a)") tanır. Karakter ofsetleri korunur.

### 3.3 Parçalama (`rag/chunker.py`)

- `MADDE 5 –`, `Madde 12-`, `Geçici Madde 1 –`, `EK MADDE 2 –` gibi bir satır
  her zaman yeni bir parça başlatır ve parçanın `article` alanı doldurulur.
  Hemen önceki kısa başlık satırları ("Amaç", "BİRİNCİ BÖLÜM") o maddeye
  dahil edilir.
- Uzun maddeler cümle sınırlarında, karakter (`max_chars`) veya token
  (`token_counter` + `max_tokens`) bütçesine göre örtüşmeli parçalara
  bölünür. Örtüşme madde sınırını aşmaz.
- Her parçada şu alanlar bulunur: `id, doc_id, title, article, text, start,
  end, url, license, date, metadata`. `text == belge[start:end]` eşitliği
  geçerlidir.

### 3.4 Arama (`rag/index.py`)

Okapi BM25 (`k1=1.5`, `b=0.75`) köklenmiş terimler üzerinde çalışır. Parça
başlığı ve madde etiketi de indekslenir. Hukuk sorgularında en ayırt edici
bilgi numaralar olduğu için BM25 skoruna açıklanabilir ek puanlar eklenir:

| Ek puan | Koşul | Varsayılan |
|---|---|---|
| `article` | Sorgudaki madde atfı parçanın kendi maddesiyle aynı | +4.0 |
| `number` | Sorgudaki sayı parçada geçiyor (4+ hane = kanun no. → 2×) | +1.5 / +3.0 |
| `phrase` | Sorgudaki "tırnaklı ifade" parçada aynen geçiyor | +3.0 |

Her sonuç `bm25`, `boosts` ve `matched_terms` alanlarını taşır. Bu sayede
neden üst sırada çıktığı görülebilir. İndeks JSON olarak kaydedilir ve dış
bağımlılık gerektirmez.

### 3.5 Prompt ve token bütçesi (`rag/prompt.py`)

Sistem mesajı modele şu talimatları verir: yalnızca numaralı kaynaklardan
cevap ver, her bilgiden sonra `[1]` biçiminde atıf yap, madde ya da kanun
numarası uydurma, bilgi yoksa **"Kaynaklarda bu bilgi yok."** yaz. Kaynaklar
`[n] <başlık> — Madde X: metin` biçiminde, skor sırasıyla numaralandırılır.

Bütçe: `max_prompt_tokens = bağlam − max_new_tokens`. Kaynaklar skor
sırasıyla eklenir. Sığmayan en düşük skorlu kaynaklar düşürülür. Tek kaynak
bile sığmıyorsa metni token düzeyinde kırpılır. `template=ChatTemplate(...)`
verilirse sonuç şablonun gerçek kodlamasıyla kesin olarak doğrulanır. Aynı kod
2K modelde 2–4 madde, 32K YaRN modelde onlarca madde veya bütün bir yönetmelik
sığdırır. `ask` komutu bağlamı checkpoint'teki `config.max_seq_len`
değerinden okur (`--max-context` ile değiştirilebilir).

**Üretim.** `--speculative` greedy n-gram (prompt lookup) spekülatif
çözümleme kullanır. RAG cevapları kaynaklardan uzun ifadeleri kopyaladığı için
taslak kabul oranı yüksektir. Çıktı standart greedy ile token token aynıdır.
Örnekleme modunda kaynaktan alıntıyı cezalandırmamak için
`repetition_penalty=1.05` ve `no_repeat_ngram_size=0` kullanılır.

### 3.6 Atıf doğrulama (`rag/citations.py`)

`verify_citations(answer, sources)` cevabı cümlelere böler ve her cümle için
şunları hesaplar:

- **atıflar:** `[1]`, `[1, 2]`, `[1-3]`, `[1][2]`. Cümleden sonra yalnız
  kalan `[1]` önceki cümleye bağlanır.
- **uydurma numaralar:** kaynak listesinde olmayan `[n]`.
- **sözcüksel destek:** cümlenin köklenmiş içerik terimlerinin, atıf
  yapılan kaynak(lar)da geçme oranı (varsayılan eşik 0,5).
- **kaynakta olmayan sayı veya madde:** cümlede geçen ama atıf yapılan
  kaynakta bulunmayan sayılar ve madde atıfları. Mevzuatta en tehlikeli hata
  budur: "2 TL" yerine "5 TL", "Madde 7" yerine "Madde 9" gibi. Böyle bir
  cümle desteklenmiş sayılmaz.
- **atıfsız olgu cümlesi:** sayı, tarih, ay adı, madde atfı ya da cümle
  ortasında büyük harfli özel ad içeren ama atıf taşımayan cümle. En iyi
  destekleyen kaynak öneri olarak gösterilir.
- **kaçınma:** "Kaynaklarda bu bilgi yok." cevabı ayrıca işaretlenir.

Genel ölçütler:

- `citation_precision`: kaynağı tarafından desteklenen geçerli atıfların
  tüm atıflara oranı.
- `supported_ratio`: desteklenen iddia cümlelerinin tüm iddia cümlelerine
  oranı.

`render_report` Türkçe bir rapor üretir. `ask --json` bütün ayrıntıyı JSON
olarak verir.

---

## 4. Kullanım senaryosu: mevzuat asistanı

Senaryo: bir kurum içi asistan, kanun, yönetmelik ve yönergelerden madde
numarasıyla birlikte cevap verir. Cevaptaki her cümle kaynağa kadar izlenebilir.

- **Kaynaklar:** Mevzuat metinleri (ör. mevzuat bilgi sistemi), Resmî Gazete
  ve yüksek mahkeme (ör. Yargıtay) kararları genellikle kamuya açıktır. Yine
  de **her kaynağın kullanım koşulları ayrıca kontrol edilmelidir**: toplu
  indirme veya tarama kısıtları, yeniden yayım koşulları, kararlardaki kişisel
  verilerin anonimleştirilmesi gibi. `DATA_GOVERNANCE.md`'deki süreci izleyin.
  Her belge için `url`, `license` (ya da kullanım koşulu notu) ve `date`
  (yürürlük veya yayım tarihi) kaydedin. Kişisel veri içeren kararları
  `data/governance.py` içindeki PII temizleyicisinden geçirin.
- **Güncellik:** Mevzuat sık değişir. `date` alanını yürürlük tarihi olarak
  tutun, mülga ve değişik maddeleri ayrı belge sürümleri olarak indeksleyin
  ve indeksi düzenli olarak yeniden kurun.
- **Akış:** `index` (gece toplu) → `ask` (sorgu başına) → atıf raporu.
  `supported_ratio < 1` veya `fabricated_ids` olan cevaplar kullanıcıya uyarı
  ile gösterilmeli ya da reddedilmelidir.
- **Hukuki uyarı:** Asistan **hukuki danışmanlık değildir**. Çıktısı bir
  avukatın veya resmi merciin görüşünün yerini tutmaz. Arayüzde bu açıkça
  belirtilmelidir (`ask` her çıktının sonunda bunu yazar).

---

## 5. Dürüst sınırlamalar

- **YaRN tarifi doğrulanmış sonuç değildir.** Yukarıdaki adım ve LR değerleri
  literatürden ve benzer boyuttaki modellerden çıkarılmış başlangıç
  noktalarıdır. Toprak checkpoint'leri üzerinde 32K passkey sonuçları henüz
  raporlanmamıştır. Küçük modeller (80M–342M) uzun bağlamı büyük modeller
  kadar iyi kullanamaz. Passkey testini geçmek, uzun belgeler üzerinde akıl
  yürütebildiği anlamına gelmez.
- **32K eğitim maliyeti yüksektir:** attention hesabı `O(T²)`, logit belleği
  büyüktür. MPS ile pratik değildir. `--resume` scheduler durumunu da geri
  yükler. Yalnızca ağırlıkla başlatıp taze bir LR programı başlatma seçeneği
  henüz yoktur.
- **Arama yalnızca sözcükseldir (BM25).** Eş anlamlıları ve yeniden ifadeyi
  ("kira" / "icar") yakalamaz. Kök bulucu sezgiseldir: bazı kelimeleri yanlış
  köklendirir ve kısa kelimelere dokunmaz (`ev`/`evler` eşleşmez). Yoğun
  vektörlü (dense) bir retriever veya reranker ile birleştirmek kaliteyi
  artırır.
- **Atıf doğrulama sözcükseldir.** Olumsuzlamayı ("verilir" / "verilmez")
  ayırt edemez. Doğru ama farklı kelimelerle yazılmış cümleyi
  "zayıf destek" sayabilir. Kopyalanmış ama bağlamdan koparılmış bir cümleyi
  de "destekleniyor" sayabilir. Bir NLI modelinin veya insan incelemesinin
  yerini tutmaz.
- **Model talimata uymayabilir.** Atıf biçimi ve "Kaynaklarda bu bilgi yok."
  davranışı, modelin bu formatta SFT/DPO ile eğitilmesine bağlıdır. Yalnızca
  ön eğitimli bir checkpoint atıf üretmeyebilir. Doğrulama raporu tam da bu
  yüzden vardır.
- **Parçalama biçime bağlıdır.** Madde başlığı algılama `MADDE n –` kalıbını
  bekler. Tablolar, dipnotlar, ekler ve taranmış (OCR) PDF metinleri özel
  ön işlem ister.
- Örnek belgeler kurgusaldır. Gerçek bir mevzuat performansı ölçmek için
  lisansı uygun, etiketli bir soru-cevap seti gerekir.
