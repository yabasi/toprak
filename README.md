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
- **Apple Silicon optimizasyonu** — M4 Pro / MPS (Metal GPU) üzerinde bfloat16 mixed precision ile eğitim.
- **Tamamen açık kaynak** — Kod, mimari, eğitim süreci — her şey şeffaf ve erişilebilir.

> **💡 Bu bir ticari ürün değil, bir araştırma ve milli katkı projesidir.** Türkiye'de yapay zeka alanında bağımsız üretim kapasitesini geliştirmek için atılmış bir adımdır.

---

## Hızlı Bakış

| | Detay |
|---|---|
| **Mimari** | Decoder-only Transformer (GPT-2 tarzı, Pre-LayerNorm) |
| **Küçük Model** | ~85M parametre — `d_model=640`, `layers=14`, `heads=10` |
| **Orta Model** | ~125M parametre — `d_model=768`, `layers=16`, `heads=12` |
| **Tokenizer** | 32K BPE (SentencePiece, Türkçe morfolojisine uygun) |
| **Eğitim Verisi** | Türkçe Wikipedia + haber siteleri + kamu kaynakları |
| **Cihaz** | Apple M4 Pro — 24GB RAM, MPS (Metal GPU) |
| **Framework** | PyTorch 2.x |
| **Aktivasyon** | GELU |
| **Optimizer** | AdamW (weight decay=0.1, betas=0.9/0.95) |
| **LR Scheduler** | Cosine annealing with linear warmup |
| **Precision** | bfloat16 mixed precision |

---

## Mimari

```
┌────────────────────────────────────────────────────────┐
│                  ToprakLM                              │
│                                                        │
│  Input IDs ──► Token Embedding ──┐                     │
│                                  ├──► + ──► Dropout    │
│  Positions ──► Position Embedding┘                     │
│                      │                                 │
│              ┌───────▼─────────────────┐               │
│              │ TransformerBlock × N    │               │
│              │                         │               │
│              │  ┌────────────┐         │               │
│              │  │ LayerNorm  │         │               │
│              │  │ Multi-Head │         │               │
│              │  │ Attention  │         │  Pre-LN       │
│              │  │ + Residual │         │  Architecture │
│              │  ├────────────┤         │               │
│              │  │ LayerNorm  │         │               │
│              │  │ FFN (GELU) │         │               │
│              │  │ + Residual │         │               │
│              │  └────────────┘         │               │
│              └───────┬─────────────────┘               │
│                      │                                 │
│              ┌───────▼────────┐                        │
│              │  Final LN      │                        │
│              │  LM Head       │◄── Weight Tying        │
│              └───────┬────────┘                        │
│                      │                                 │
│                   Logits                               │
└────────────────────────────────────────────────────────┘
```

**Temel tasarım kararları:**
- **Pre-LayerNorm**: Eğitim stabilitesi için (Post-LN'ye göre çok daha kararlı)
- **Weight Tying**: Token embedding ile LM head aynı ağırlıkları paylaşır → parametre tasarrufu
- **Causal Masking**: Üst üçgen mask ile autoregressive üretim
- **Gradient Accumulation**: Küçük batch'lerle büyük efektif batch simülasyonu

---

## Proje Yapısı

```
toprak/
│
├── model/                        # Model Mimarisi
│   ├── config.py                 #    Model konfigürasyonları (Small / Medium)
│   ├── attention.py              #    Multi-Head Self-Attention + Causal Mask
│   ├── transformer.py            #    ToprakLM — Ana model sınıfı
│   └── tokenizer.py              #    SentencePiece BPE Tokenizer wrapper
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
  --model-size small \
  --data-dir data_cache/clean/train \
  --eval-data-dir data_cache/clean/eval \
  --tokenizer toprak_tokenizer.model
```

<details>
<summary>Eğitim parametreleri ve devam etme</summary>

| Parametre | Küçük Model | Orta Model |
|---|---|---|
| `--model-size` | `small` | `medium` |
| `--batch-size` | 8–16 | 4–8 |
| `--grad-accum` | 4 | 8 |
| `--max-steps` | 100,000 | 200,000 |
| Tahmini süre (M4 Pro) | 2–3 gün | 5–7 gün |

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
| **v0.1-alpha** | Altyapı kodu, tokenizer, ilk eğitim | 🔄 Devam ediyor |
| **v0.2-beta** | 85M model, 5GB+ veri ile eğitim | ⏳ Planlandı |
| **v1.0** | 125M model, 10GB veri, stabil versiyon | ⏳ Planlandı |
| **v1.x** | Sürekli güncelleme, topluluk katkıları | ⏳ Planlandı |

---

## Katkı

Bu proje Türk yapay zeka topluluğuna açıktır. Katkıda bulunmak isterseniz:

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
<summary><strong>Eğitim Optimizasyonları (Apple Silicon)</strong></summary>

- **MPS Backend**: `device='mps'` — Apple Metal Performance Shaders
- **Mixed Precision**: `torch.autocast(device_type='mps', dtype=torch.bfloat16)`
- **Gradient Accumulation**: Küçük batch ile büyük efektif batch simülasyonu
- **Gradient Clipping**: Max norm 0.5
- **Checkpoint Strategy**: Her 5000 adımda kaydet, son 3'ü tut

</details>

<details>
<summary><strong>Inference</strong></summary>

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
| v1.0 (6 ay) | Kullanılabilir Türkçe chatbot — GPT-2 seviyesi |
| v2.0+ (1 yıl+) | Daha büyük model, daha fazla veri → gerçek kalite |

---

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında yayınlanmıştır. Herkes özgürce kullanabilir, değiştirebilir ve dağıtabilir.

---

<p align="center">
  <strong>🌱 Her büyük ağaç, küçük bir tohumla başlar.</strong><br>
  <em>Toprak — Türk milletinin yapay zeka toprağı.</em>
</p>

<p align="center">
  <sub>Made with ❤️ in Türkiye</sub>
</p>
