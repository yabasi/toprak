<p align="center">
  <img src="https://img.shields.io/badge/🌱_TOPRAK-Türkçe_Dil_Modeli-2E7D32?style=for-the-badge&labelColor=1B5E20" alt="Toprak" />
</p>

<h1 align="center">🌱 Toprak</h1>

<p align="center">
  <strong>Sıfırdan Eğitilen, Tamamen Özgün Türkçe Büyük Dil Modeli</strong>
</p>

<p align="center">
  <em>"Toprak" — hem bir bebeğin adı, hem de tohumların yeşerdiği yer.<br>Bu proje, Türk milletinin kendi dilinde kendi yapay zekasını yetiştirmesi için atılmış bir tohumdur.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Apple_Silicon-MPS-000000?style=flat-square&logo=apple&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Dil-Türkçe_🇹🇷-E30A17?style=flat-square" />
  <img src="https://img.shields.io/badge/Durum-Aktif_Geliştirme-yellow?style=flat-square" />
</p>

---

## Neden Toprak?

Dünya genelinde yüzlerce dil modeli geliştirilirken, **Türkçe için sıfırdan yazılmış açık kaynak bir model neredeyse yok**. Mevcut Türkçe modellerin çoğu, İngilizce modellerin üzerine fine-tune edilmiş versiyonlardır — Türkçe'nin zengin morfolojisini, ekleme yapısını ve dilbilgisini tam olarak kavrayamazlar.

**Toprak**, bu eksikliği gidermek için doğdu:

- **Sıfırdan inşa** — Hiçbir mevcut modelden fine-tune yapılmıyor. Mimari, tokenizer, ağırlıklar — her şey bu proje kapsamında yazılıyor.
- **Türkçe'ye özel** — 32.000 kelimelik Türkçe BPE tokenizer, `ç`, `ğ`, `ı`, `ö`, `ş`, `ü` karakterlerine tam destek.
- **Apple Silicon optimizasyonu** — M4 Pro / MPS (Metal GPU) üzerinde optimize float32 ile eğitim.
- **Tamamen açık kaynak** — Kod, mimari, eğitim süreci — her şey şeffaf ve erişilebilir.

> **💡 Bu bir ticari ürün değil, bir araştırma ve milli katkı projesidir.** Türkiye'de yapay zeka alanında bağımsız üretim kapasitesini geliştirmek için atılmış bir adımdır.

> 📖 **Kapsamlı kullanım rehberi için:** [GUIDE.md](GUIDE.md) — Kurulum, eğitim, inference, parametreler ve sık sorulan sorular.

---

## Hızlı Bakış

| | Detay |
|---|---|
| **Mimari** | Decoder-only Transformer (modern 2024 nesil, sıfırdan tasarım) |
| **Small** | ~80M parametre — `d_model=640`, `layers=14`, `heads=10`, `kv_heads=2` |
| **Medium** | ~125M parametre — `d_model=768`, `layers=16`, `heads=12`, `kv_heads=4` |
| **Large** | ~342M parametre — `d_model=1024`, `layers=28`, `heads=16`, `kv_heads=4` |
| **XL** | ~941M parametre — `d_model=1536`, `layers=36`, `heads=16`, `kv_heads=4` |
| **Normalizasyon** | RMSNorm (bias'sız, LayerNorm'dan daha hızlı) |
| **Aktivasyon** | SwiGLU (gated FFN, SiLU tabanlı) |
| **Pozisyon** | RoPE (Rotary Position Embedding) |
| **Attention** | GQA (Grouped Query Attention) + KV Cache + SDPA |
| **Tokenizer** | 32K BPE (SentencePiece, Türkçe morfolojisine uygun) |
| **Eğitim Verisi** | Türkçe Wikipedia + haber siteleri + kamu kaynakları |
| **Cihaz** | Auto-detect: CUDA (NVIDIA) / MPS (Apple Silicon) / CPU |
| **Framework** | PyTorch 2.x + torch.compile() |
| **Optimizer** | AdamW (weight decay=0.1, betas=0.9/0.95) |
| **LR Scheduler** | Cosine annealing with linear warmup |
| **Precision** | MPS: float32 / CUDA: float16 mixed precision |
| **Türkçe Uyumu** | Ünlü Uyumu Auxiliary Loss (dünyada ilk, opsiyonel) |
| **Morfolojik Kayıp** | Ek tokenlerine ağırlıklı CE Loss (dünyada ilk, opsiyonel) |

---

## Mimari

```
┌─────────────────────────────────────────────────────────┐
│                  ToprakLM (2024)                        │
│                                                         │
│  Input IDs ──► Token Embedding                          │
│                      │         (Positional Emb yok,     │
│                      │          RoPE kullanılıyor)      │
│              ┌───────▼──────────────────┐               │
│              │  TransformerBlock × N    │               │
│              │                          │               │
│              │  ┌─────────────┐         │               │
│              │  │ RMSNorm     │         │               │
│              │  │ GQA + RoPE  │         │  Pre-RMSNorm  │
│              │  │ + KV Cache  │         │  Architecture │
│              │  │ + Residual  │         │               │
│              │  ├─────────────┤         │               │
│              │  │ RMSNorm     │         │               │
│              │  │ SwiGLU FFN  │         │               │
│              │  │ + Residual  │         │               │
│              │  └─────────────┘         │               │
│              └───────┬──────────────────┘               │
│                      │                                  │
│              ┌───────▼────────┐                         │
│              │  RMSNorm       │                         │
│              │  LM Head       │◄── Weight Tying         │
│              └───────┬────────┘                         │
│                      │                                  │
│                   Logits                                │
└─────────────────────────────────────────────────────────┘
```

**Temel tasarım kararları (modern 2024 decoder-only standartları):**
- **RMSNorm**: Bias'sız, LayerNorm'dan ~%5-8 daha hızlı normalizasyon
- **SwiGLU**: 3 katmanlı gated FFN (SiLU aktivasyonlu), GELU'dan daha düşük loss
- **RoPE**: Rotary Position Embedding — relative position, extrapolation kabiliyeti
- **GQA**: Grouped Query Attention — daha az KV head ile bellek tasarrufu
- **KV Cache**: Inference'da sadece son token hesaplanır → 5-10x hız artışı
- **Bias-free**: Tüm Linear katmanlardan bias kaldırıldı
- **Weight Tying**: Token embedding ile LM head aynı ağırlıkları paylaşır
- **Causal Masking**: Dinamik üst üçgen mask ile autoregressive üretim
- **Gradient Accumulation**: Küçük batch'lerle büyük efektif batch simülasyonu
- **Ünlü Uyumu Loss**: Türkçe ünlü uyumuna aykırı token tahminlerini cezalandıran auxiliary loss (dünyada ilk)
- **Morfolojik Ağırlıklı Kayıp**: Ek (suffix) tokenlerine daha yüksek CE loss ağırlığı vererek morfoloji öğrenimini güçlendirir (dünyada ilk)

---

## Proje Yapısı

```
toprak/
│
├── model/                        # Model Mimarisi
│   ├── config.py                 #    Model konfigürasyonları (Small/Medium/Large/XL)
│   ├── attention.py              #    GQA + RoPE + KV Cache + SDPA
│   ├── transformer.py            #    ToprakLM (SwiGLU, RMSNorm, Grad Checkpoint)
│   ├── norms.py                  #    RMSNorm — Modern normalizasyon
│   ├── rope.py                   #    RoPE — Rotary Position Embedding
│   ├── tokenizer.py              #    SentencePiece BPE Tokenizer wrapper
│   ├── vowel_harmony.py          #    Ünlü Uyumu Auxiliary Loss (Türkçe'ye özel)
│   └── morph_weighting.py        #    Morfolojik Ağırlıklı CE Loss (dünyada ilk)
│
├── data/                         # Veri Toplama & İşleme
│   ├── sources.py                #    Türkçe kaynak URL'leri ve yapılandırma
│   ├── crawler.py                #    asyncio + aiohttp web crawler
│   ├── cleaner.py                #    7 aşamalı veri temizleme pipeline
│   └── dataset.py                #    PyTorch Dataset + DataLoader
│
├── training/                     # Eğitim
│   ├── train.py                  #    CLI — Ana eğitim entry point
│   ├── trainer.py                #    Eğitim döngüsü, checkpoint, logging
│   └── scheduler.py              #    Cosine warmup LR scheduler
│
├── inference/                    # Çıkarım & Sohbet
│   ├── generate.py               #    Metin üretimi (top-k, top-p, repetition penalty)
│   └── chat.py                   #    Terminal tabanlı interaktif sohbet
│
├── evaluation/                   # Değerlendirme
│   └── eval.py                   #    Perplexity hesaplama
│
├── upload/                       # HuggingFace Entegrasyonu
│   └── push_to_hub.py            #    Model + tokenizer yükleme
│
├── scripts/                      # Yardımcı Araçlar
│   └── prepare_data.py           #    Uçtan uca veri pipeline
│
├── requirements.txt              #    Python bağımlılıkları
└── LICENSE                       #    MIT Lisansı
```

---

## Kurulum

### Gereksinimler

- Python 3.11+
- macOS (Apple Silicon önerilir) veya Linux
- ~10GB disk alanı (veri + model)

### Adımlar

```bash
# 1. Projeyi klonla
git clone https://github.com/yabasi/toprak.git
cd toprak

# 2. Sanal ortam oluştur ve aktif et
python3 -m venv venv
source venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Apple Silicon GPU kontrolü
python3 -c "import torch; print('MPS kullanılabilir:', torch.backends.mps.is_available())"
```

---

## Kullanım

### Veri Hazırlama

Tüm pipeline'ı tek komutla çalıştır — Wikipedia indir → tokenizer eğit → veriyi temizle:

```bash
python3 scripts/prepare_data.py
```

<details>
<summary>Adım adım çalıştırma (isteğe bağlı)</summary>

```bash
# Sadece Wikipedia indir
python3 scripts/prepare_data.py --step download

# Hızlı test (örnek veri ile)
python3 scripts/prepare_data.py --use-sample --sample-count 5000

# Sadece tokenizer eğit
python3 scripts/prepare_data.py --step tokenizer

# Sadece veriyi temizle ve böl
python3 scripts/prepare_data.py --step prepare
```

</details>

### Model Eğitimi

```bash
python3 training/train.py \
  --model-size medium \
  --data-dir data_cache/clean/train \
  --eval-data-dir data_cache/clean/eval \
  --tokenizer toprak_tokenizer.model
```

<details>
<summary>Eğitim parametreleri ve devam etme</summary>

| Parametre | Küçük Model | Orta Model |
|---|---|---|
| `--model-size` | `small` | `medium` |
| `--batch-size` | 8–16 | 8 |
| `--grad-accum` | 4 | 4 |
| `--max-steps` | 100,000 | 100,000 |
| Tahmini süre (M4 Pro) | 1–2 gün | 4–6 gün |

```bash
# Kaldığın yerden devam et
python3 training/train.py \
  --model-size small \
  --data-dir data_cache/clean/train \
  --resume checkpoints/toprak_step_5000.pt
```

</details>

### Sohbet

```bash
python3 inference/chat.py \
  --checkpoint checkpoints/toprak_best.pt \
  --tokenizer toprak_tokenizer.model
```

```
🧑 Sen: Türkiye'nin en güzel şehri hangisidir?
🌱 Toprak: ...
```

### 4️⃣ Metin Üretimi

```bash
python3 inference/generate.py \
  --checkpoint checkpoints/toprak_best.pt \
  --tokenizer toprak_tokenizer.model \
  --prompt "Yapay zekanın geleceği" \
  --temperature 0.8 \
  --num-samples 3
```

### Değerlendirme

```bash
python3 evaluation/eval.py \
  --checkpoint checkpoints/toprak_best.pt \
  --eval-data data_cache/clean/eval \
  --tokenizer toprak_tokenizer.model
```

| Perplexity | Anlam |
|---|---|
| < 50 | ✅ Hedef başarıldı |
| 50–100 | 🟡 İyi yolda |
| > 100 | 🔴 Daha fazla veri/eğitim gerekli |

---

## Geliştirme Döngüsü

```
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  Veri    │────►│  Eğitim  │────►│  Eval    │
   │  Topla   │     │          │     │          │
   └──────────┘     └──────────┘     └─────┬────┘
        ▲                                  │
        │           ┌──────────┐           │
        │           │  Yayınla │◄──────────┘
        │           │  (HF Hub)│       İyileşme varsa
        │           └──────────┘
        │                │
        └────────────────┘
            Yeni veri ile tekrarla
```

```bash
# 1. Yeni veri topla
python3 data/crawler.py --source haber --max-pages 1000

# 2. Temizle
python3 data/cleaner.py --input data_cache --output data_cache/clean

# 3. Son checkpoint'ten eğitime devam et
python3 training/train.py --resume checkpoints/toprak_best.pt \
  --data-dir data_cache/clean/train

# 4. Değerlendir
python3 evaluation/eval.py --checkpoint checkpoints/toprak_best.pt \
  --eval-data data_cache/clean/eval

# 5. HuggingFace'e yükle
python3 upload/push_to_hub.py --checkpoint checkpoints/toprak_best.pt \
  --repo KULLANICI_ADI/toprak-v1
```

---

## Yol Haritası

| Aşama | Hedef | Durum |
|---|---|---|
| **v0.1-alpha** | Altyapı kodu, tokenizer, veri pipeline | ✅ Tamamlandı |
| **v0.2-beta** | 125M model (Medium), 207M token ile eğitim | 🔄 Eğitim devam ediyor |
| **v1.0** | 125M model, 10GB+ veri, stabil versiyon | ⏳ Planlandı |
| **v1.5** | 342M model (Large), RTX 4090 ile eğitim | ⏳ Planlandı |
| **v2.0** | Sürekli güncelleme, topluluk katkıları, fine-tuning | ⏳ Planlandı |

---

## Katkı

Bu proje Türk yapay zeka topluluğuna açıktır. Katkıda bulunmak isterseniz:

> 📖 **Detaylı katkı rehberi için:** [CONTRIBUTING.md](CONTRIBUTING.md)

1. Bu repoyu **fork**'layın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'e push edin (`git push origin feature/yeni-ozellik`)
5. **Pull Request** açın

**Katkı alanları:**
- Yeni Türkçe veri kaynakları ekleme
- Test ve benchmark'lar
- Dokümantasyon iyileştirmeleri
- Bug fix'ler
- Performans optimizasyonları

---

## Teknik Detaylar

<details>
<summary><strong>Model Mimarisi (2024 Nesil)</strong></summary>

- **RMSNorm**: Bias'sız root mean square normalizasyon (LayerNorm yerine)
- **SwiGLU**: 3 katmanlı gated FFN — `SiLU(gate) * up → down` (GELU yerine)
- **RoPE**: Rotary Position Embedding — complex çarpımla pozisyon kodlama
- **GQA**: Grouped Query Attention — 10Q/2KV (small), 12Q/4KV (medium), 16Q/4KV (large/xl)
- **SDPA**: PyTorch native scaled_dot_product_attention (FlashAttention benzeri)
- **KV Cache**: Inference'da geçmiş key/value'ları sakla → her adımda sadece 1 token
- **Bias-free**: Tüm Linear katmanlardan bias kaldırıldı
- **Weight Tying**: Token embedding ↔ LM head aynı ağırlıklar
- **Init**: Scaled init — residual projeksiyonlar `1/√(2N)` ile ölçeklendirilmiş
- **Ünlü Uyumu Loss**: Türkçe büyük ünlü uyumunu auxiliary loss olarak enjekte eder (dünyada ilk)
- **Morfolojik Ağırlıklı Kayıp**: Ek tokenlerine yüksek ağırlık → morfoloji farkındalığı (dünyada ilk)

</details>

<details>
<summary><strong>Veri Pipeline</strong></summary>

- **Crawler**: asyncio + aiohttp, robots.txt uyumlu, 1s rate limit
- **Temizleme**: 7 aşamalı pipeline — HTML artıkları, Unicode (NFKC), boilerplate filtre, kalite skoru, MD5 dedup
- **Kaynaklar**: Wikipedia (~2GB), Haber siteleri (~5GB), Kamu kurumları (~1GB), Edebiyat (~500MB), Akademik (~2GB)
- **Format**: JSONL — `{url, text, source, timestamp, word_count}`

</details>

<details>
<summary><strong>Tokenizer</strong></summary>

- **Algoritma**: BPE (Byte Pair Encoding) — SentencePiece
- **Vocab**: 32,000 token
- **Karakter kapsama**: %99.99 (Türkçe karakterler dahil)
- **Özel tokenler**: `PAD(0)`, `UNK(1)`, `BOS(2)`, `EOS(3)`, `<sep>`, `<cls>`, `<mask>`
- **Normalizasyon**: NFKC
- **Byte fallback**: Etkin (bilinmeyen karakter desteği)

</details>

<details>
<summary><strong>Eğitim Optimizasyonları</strong></summary>

- **Multi-Device**: CUDA (NVIDIA) / MPS (Apple Silicon) / CPU — otomatik algılama
- **SDPA**: PyTorch native scaled_dot_product_attention
- **torch.compile()**: Model derleme ile %10-30 hız artışı
- **Gradient Checkpointing**: FFN katmanlarında bellek tasarrufu
- **Mixed Precision**: CUDA (float16) / MPS & CPU (float32 — RoPE complex tensor uyumluluğu için)
- **NaN Guard**: Loss/gradient nan kontrolü, arka arkaya 10 nan'da erken durdurma
- **Gradient Accumulation**: Küçük batch ile büyük efektif batch simülasyonu
- **Gradient Clipping**: Max norm 1.0
- **Checkpoint Strategy**: Her 5000 adımda kaydet, son 3'ü tut
- **TensorBoard**: Loss, LR, tokens/s, grad norm, eval perplexity takibi
- **Döküman Karıştırma**: Epoch başı döküman seviyesinde shuffle
- **Dropout**: 0.0 (modern modellerde dropout kullanılmıyor)
- **Ünlü Uyumu Auxiliary Loss**: Opsiyonel — Türkçe ünlü uyumuna aykırı token tahminlerini cezalandırır (`--vowel-harmony`)
- **Morfolojik Ağırlıklı Kayıp**: Opsiyonel — Ek tokenlerine yüksek CE ağırlığı, kök/ek loss ayrı takip (`--morph-weight`)

</details>

<details>
<summary><strong>Inference</strong></summary>

- **KV Cache**: Prefill + decode ayrılmış — her adımda sadece son token hesaplanır
- **Top-k Sampling**: En olası k token arasından seçim
- **Top-p (Nucleus) Sampling**: Kümülatif olasılık eşiği
- **Repetition Penalty**: Tekrar eden tokenlere ceza (×1.3)
- **No-repeat N-gram**: Aynı 4-gram'ın tekrarını engelleme
- **Sayısal Stabilite**: NaN ve negatif olasılık kontrolü

</details>

---

## Beklentiler

> **Önemli:** Bu bir araştırma projesidir. İlk modelin mükemmel olmaması başarısızlık değil — sürecin doğal bir parçasıdır.

| Aşama | Beklenti |
|---|---|
| İlk model (1–2 hafta) | Tutarsız, bazen anlamsız cümleler — **tamamen normal** |
| v0.1 (1 ay) | Türkçe cümle yapısını kavramış, hatalar mevcut |
| v0.5 (3 ay) | Konuya uygun cevaplar, tutarlılık artıyor |
| v1.0 (6 ay) | Kullanılabilir Türkçe metin üretici — tutarlı ve anlamlı çıktılar |
| v2.0+ (1 yıl+) | Daha büyük model, daha fazla veri → gerçek kalite |

---

## Geliştirici

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/yabasi">
        <img src="https://github.com/yabasi.png" width="100px;" alt="Abbas Kandemir"/><br />
        <sub><b>Abbas Kandemir</b></sub>
      </a><br />
      <sub>Proje Kurucusu & Ana Geliştirici</sub><br />
      <a href="https://github.com/yabasi">@yabasi</a>
    </td>
  </tr>
</table>

> Katkıda bulunmak ister misiniz? Pull request'lerinizi bekliyoruz! Detaylar için [CONTRIBUTING.md](CONTRIBUTING.md) rehberine bakın.

### 🤝 Katkıda Bulunanlar

Toprak'a katkıda bulunan herkese teşekkür ederiz! 🙏

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ismailocal">
        <img src="https://github.com/ismailocal.png" width="80px;" alt="İsmail Öcal"/><br />
        <sub><b>İsmail Öcal</b></sub>
      </a><br />
      <sub>🐛 Bug Fix</sub>
    </td>
    <!-- Yeni katkıda bulunanlar buraya eklenecek -->
  </tr>
</table>

> 💡 **Sen de bu listeye girebilirsin!** Her kabul edilen PR ile katkıda bulunanlar listesine ekliyoruz. [Nasıl katkıda bulunabileceğini öğren →](CONTRIBUTING.md)

---

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında yayınlanmıştır. Herkes özgürce kullanabilir, değiştirebilir ve dağıtabilir.

---

<p align="center">
  <strong>🌱 Her büyük ağaç, küçük bir tohumla başlar.</strong><br>
  <em>Toprak — Türk milletinin yapay zeka toprağı.</em>
</p>

<p align="center">
  <sub>Made with ❤️ by <a href="https://github.com/yabasi">Abbas Kandemir</a> in Türkiye</sub>
</p>
