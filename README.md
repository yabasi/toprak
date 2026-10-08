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
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square" />
  <img src="https://img.shields.io/badge/Dil-Türkçe_🇹🇷-E30A17?style=flat-square" />
  <img src="https://img.shields.io/badge/Durum-Aktif_Geliştirme-yellow?style=flat-square" />
</p>

---

## Neden Toprak?

Dünya genelinde yüzlerce dil modeli geliştirilirken, **Türkçe için sıfırdan yazılmış açık kaynak bir model neredeyse yok**. Mevcut Türkçe modellerin çoğu, İngilizce modellerin üzerine fine-tune edilmiş versiyonlardır — Türkçe'nin zengin morfolojisini, ekleme yapısını ve dilbilgisini tam olarak kavrayamazlar.

**Toprak**, bu eksikliği gidermek için doğdu:

- **Sıfırdan inşa** — Hiçbir mevcut modelden fine-tune yapılmıyor. Mimari, tokenizer, ağırlıklar — her şey bu proje kapsamında yazılıyor.
- **Türkçe'ye özel** — 32.000 tokenlık Türkçe BPE tokenizer; `ç`, `ğ`, `ı`, `ö`, `ş`, `ü` ve byte fallback desteği.
- **Apple Silicon optimizasyonu** — M4 Pro / MPS (Metal GPU) üzerinde optimize float32 ile eğitim.
- **Tamamen açık kaynak** — Kod, mimari, eğitim süreci — her şey şeffaf ve erişilebilir.

> **💡 Bu bir ticari ürün değil, bir araştırma ve milli katkı projesidir.** Türkiye'de yapay zeka alanında bağımsız üretim kapasitesini geliştirmek için atılmış bir adımdır.

> 📖 **Kapsamlı kullanım rehberi için:** [GUIDE.md](GUIDE.md). Yardımcı kayıplar için [ABLATION.md](ABLATION.md), tokenizer karşılaştırması için [TOKENIZER_ANALYSIS.md](TOKENIZER_ANALYSIS.md), deney tekrarları için [REPRODUCIBILITY.md](REPRODUCIBILITY.md), veri karışımı için [DATA_MIXTURE.md](DATA_MIXTURE.md) dosyasına bakın.
>
> 🚀 **Yeni nesil yetenekler:** [ARCHITECTURE_UPGRADES.md](ARCHITECTURE_UPGRADES.md) (MTP, MoE, YaRN, spekülatif çözümleme) · [MORPHOPHONOLOGY.md](MORPHOPHONOLOGY.md) (arşifonemik ekler, dilbilgisi koruması) · [ALIGNMENT.md](ALIGNMENT.md) (SFT, DPO, Toprak Anayasası) · [REASONING.md](REASONING.md) (GRPO) · [LONG_CONTEXT_RAG.md](LONG_CONTEXT_RAG.md) (32K bağlam, kaynak gösteren RAG) · [TURKIC.md](TURKIC.md) (Türk dilleri) · [EDGE.md](EDGE.md) (cihazda çalıştırma) · [INTERPRETABILITY.md](INTERPRETABILITY.md) (Morfoloji Mikroskobu)

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
| **Tokenizer** | 32K BPE (SentencePiece); kapsam, fertility ve morfoloji analiz paketi |
| **Eğitim Verisi** | Türkçe Wikipedia + haber siteleri + kamu kaynakları |
| **Cihaz** | Auto-detect: CUDA (NVIDIA) / MPS (Apple Silicon) / CPU |
| **Framework** | PyTorch 2.x + torch.compile() |
| **Optimizer** | AdamW (weight decay=0.1, betas=0.9/0.95) |
| **LR Scheduler** | Cosine annealing with linear warmup |
| **Precision** | MPS: float32 / CUDA: float16 mixed precision |
| **Türkçe Uyumu** | Ünlü Uyumu Auxiliary Loss (dünyada ilk, opsiyonel) |
| **Ünsüz Benzeşmesi** | Ünsüz Benzeşmesi Auxiliary Loss (dünyada ilk, opsiyonel) |
| **Morfolojik Kayıp** | Ek tokenlerine ağırlıklı CE Loss (dünyada ilk, opsiyonel) |
| **Morfolojik Başlık** | Kök, Ek ve Özel Çoklu Görev (Multi-task) Sınıflandırma Başlığı (dünyada ilk, opsiyonel) |
| **Hece & Kafiye** | Türkçe Hece Ölçüsü ve Kafiye Uyumu Auxiliary Loss (dünyada ilk, opsiyonel) |
| **Çoklu Token Tahmini** | MTP başlıkları + kendi kendine spekülatif çözümleme (opsiyonel) |
| **Uzman Karışımı** | Morfoloji yönlendirmeli MoE + yoğun→MoE upcycling (opsiyonel) |
| **Uzun Bağlam** | RoPE ölçekleme: linear / NTK / YaRN (opsiyonel) |
| **Hizalama** | Sohbet şablonu, SFT + LoRA, DPO/ORPO, Toprak Anayasası, GRPO |
| **Dağıtım** | HF Llama dışa aktarımı (GGUF/MLX yolu), int8/int4, damıtma |

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
- **Ünsüz Benzeşmesi Loss**: Sert ünsüzle biten kelimelerden sonra yumuşak ünsüzle başlayan ek tahminlerini cezalandıran auxiliary loss (dünyada ilk)
- **Morfolojik Ağırlıklı Kayıp**: Ek (suffix) tokenlerine daha yüksek CE loss ağırlığı vererek morfoloji öğrenimini güçlendirir (dünyada ilk)
- **Morfolojik Sınır & Sentaks Başlığı (POS Head)**: Tokenları Kök, Ek ve Özel olarak 3 sınıfa ayıran çoklu görev (multi-task) yardımcı başlığı (dünyada ilk)
- **Hece & Kafiye Loss**: Türkçe hece ölçüsü (hece vezni) ve kafiye (uyak) kurallarını eğitimde dinamik kısıt olarak öğreten auxiliary loss (dünyada ilk)

---

## Yeni Nesil Yetenekler

Toprak'ı Türkçeye özgü bir araştırma modelinden uçtan uca bir asistana
taşıyan on yetenek. Hepsi **varsayılan olarak kapalıdır**: mevcut
checkpoint'ler ve komutlar aynen çalışır. Her biri birim ve entegrasyon
testleriyle doğrulanmıştır.

| # | Yetenek | Ne sağlar? | Belge |
|---|---|---|---|
| 1 | **Arşifonemik ekler** | Ekleri soyut biçimde (`+lAr`, `+DA`, `+(y)A`) temsil eden kayıpsız kodek. Ünlü uyumu ve ünsüz benzeşmesi kurallı bir katmanda gerçekleşir, uyum hatası yapısal olarak imkânsızlaşır | [MORPHOPHONOLOGY.md](MORPHOPHONOLOGY.md) |
| 2 | **Dilbilgisi korumalı üretim** | Eğitim gerektirmeyen logit işlemcisi. Uyumsuz ek tokenlarını maskeler (`--grammar-guard mask`) | [MORPHOPHONOLOGY.md](MORPHOPHONOLOGY.md) |
| 3 | **Çoklu Token Tahmini + spekülatif çözümleme** | Model kelimenin geri kalan eklerini önceden tahmin eder. Taslaklar tek ileri geçişte doğrulanır ve çıktı greedy ile birebir aynıdır | [ARCHITECTURE_UPGRADES.md](ARCHITECTURE_UPGRADES.md) |
| 4 | **Morfoloji yönlendirmeli MoE** | Kök, ek ve özel ipuçlu uzman karışımı. Örneğin Medium, aynı aktif hesapla 352M toplam parametreye çıkar | [ARCHITECTURE_UPGRADES.md](ARCHITECTURE_UPGRADES.md) |
| 5 | **Toprak-Sohbet** | Sohbet şablonu, SFT + LoRA, DPO/ORPO ve 15 ilkeli **Toprak Anayasası** ile öz-düzeltme | [ALIGNMENT.md](ALIGNMENT.md) |
| 6 | **Düşünen Toprak** | Lisans sorunu olmayan sentetik Türkçe matematik verisi, doğrulanabilir ödüller, `<düşünce>` formatı ve GRPO | [REASONING.md](REASONING.md) |
| 7 | **32K bağlam + kaynak gösteren RAG** | YaRN ile bağlam genişletme, passkey testi, madde bazlı mevzuat arama ve alıntı doğrulama | [LONG_CONTEXT_RAG.md](LONG_CONTEXT_RAG.md) |
| 8 | **Türk Dünyası** | 12 Türk dili için kayıt, betik tespiti, Kiril ve Arap harflerinden transliterasyon, Osmanlıca paralel veri ve dil başına tokenizer ölçümü | [TURKIC.md](TURKIC.md) |
| 9 | **Telefonda çalışan Toprak** | HF Llama dışa aktarımı (logit eşitliği test edildi), GGUF ve MLX yolu, int8/int4, damıtma | [EDGE.md](EDGE.md) |
| 10 | **Morfoloji Mikroskobu** | Probe, kontrol görevi, TopK SAE, aktivasyon yaması ve etkileşimli HTML raporları | [INTERPRETABILITY.md](INTERPRETABILITY.md) |

Hızlı örnekler:

```bash
# MTP + MoE ile eğitim
python3 training/train.py --model-size medium --mtp-heads 3 --num-experts 8 --experts-top-k 2

# Spekülatif çözümleme (greedy ile hız ve çıktı karşılaştırması)
python3 inference/speculative.py --checkpoint checkpoints/toprak_best.pt --compare

# Talimat eğitimi (LoRA) → tercih hizalaması
python3 training/sft.py --base-checkpoint checkpoints/toprak_best.pt \
  --data alignment/examples/sft_sample.jsonl --output checkpoints/sft.pt --lora-r 16
python3 training/dpo.py --base-checkpoint checkpoints/sft.pt \
  --data alignment/examples/dpo_sample.jsonl --output checkpoints/dpo.pt

# Kaynak gösteren mevzuat asistanı
python3 -m rag.cli index --docs rag/examples --out rag_index.json
python3 -m rag.cli ask --index rag_index.json --checkpoint checkpoints/dpo.pt \
  --query "Bir üye aynı anda en fazla kaç materyal ödünç alabilir?"

# Hugging Face / llama.cpp / MLX için dışa aktarım
python3 -m export.hf_llama --checkpoint checkpoints/dpo.pt --out export/toprak-hf
```

Bu yükseltmelerle birlikte düzeltilen hatalar (morfolojik başlığın kök
tokenlarını öğrenmemesi, KV cache'li çok-token attention maskesi, `.half()`
ile bozulan RoPE tablosu, uzun sohbette çökme, eksik checkpoint config'i)
[ARCHITECTURE_UPGRADES.md](ARCHITECTURE_UPGRADES.md#düzeltilen-hatalar)
içinde listelenmiştir.

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
│   ├── moe.py                    #    Morfoloji yönlendirmeli uzman karışımı (MoE)
│   ├── chat_template.py          #    Sohbet şablonu + kayıp maskesi
│   ├── archiphoneme.py           #    Arşifonemik ek kodeki (+lAr, +DA, ...)
│   ├── vowel_harmony.py          #    Ünlü Uyumu Auxiliary Loss (Türkçe'ye özel)
│   ├── consonant_harmony.py      #    Ünsüz Benzeşmesi Auxiliary Loss (Türkçe'ye özel)
│   ├── morph_weighting.py        #    Morfolojik Ağırlıklı CE Loss (dünyada ilk)
│   └── syllable_rhyme.py         #    Hece ve Kafiye Auxiliary Loss (dünyada ilk)
│
├── data/                         # Veri Toplama & İşleme
│   ├── sources.py                #    Türkçe kaynak URL'leri ve yapılandırma
│   ├── crawler.py                #    asyncio + aiohttp web crawler
│   ├── cleaner.py                #    Kalite, PII, dedup ve contamination pipeline
│   ├── governance.py             #    Provenance, lisans ve audit metadata şeması
│   ├── mixture.py                #    Müfredatlı veri karışımı örnekleyicisi
│   ├── turkic.py                 #    Türk dilleri: kayıt, betik, transliterasyon
│   ├── synthetic_math.py         #    Lisanssız sentetik Türkçe matematik verisi
│   └── dataset.py                #    PyTorch Dataset + DataLoader
│
├── training/                     # Eğitim
│   ├── train.py                  #    CLI — Ana eğitim entry point
│   ├── trainer.py                #    Eğitim döngüsü, checkpoint, logging
│   ├── scheduler.py              #    Cosine warmup LR scheduler
│   ├── sft.py                    #    Talimat eğitimi (SFT) + LoRA
│   ├── dpo.py                    #    DPO / ORPO tercih hizalaması
│   ├── grpo.py                   #    GRPO pekiştirmeli öğrenme
│   ├── rewards.py                #    Doğrulanabilir ödül fonksiyonları
│   └── distill.py                #    Öğretmen→öğrenci damıtma
│
├── inference/                    # Çıkarım & Sohbet
│   ├── generate.py               #    Metin üretimi (top-k, top-p, repetition penalty)
│   ├── chat.py                   #    Şablonlu interaktif sohbet
│   ├── speculative.py            #    Kendi kendine spekülatif çözümleme
│   └── grammar_guard.py          #    Ünlü uyumu korumalı üretim
│
├── alignment/                    # Toprak Anayasası + öz-düzeltme (Constitutional AI)
├── rag/                          # Kaynak gösteren Türkçe RAG (BM25, alıntı doğrulama)
├── export/                       # HF Llama dışa aktarımı + int8/int4 kuantizasyon
├── interpret/                    # Morfoloji Mikroskobu: probe, SAE, HTML rapor
│
├── evaluation/                   # Değerlendirme
│   ├── run_lm_eval.py            #    lm-evaluation-harness ile standart Türkçe benchmarklar
│   ├── eval.py                   #    Perplexity hesaplama
│   ├── suite.py                  #    Çok boyutlu deterministik eval motoru
│   ├── evaluate_suite.py         #    Checkpoint karşılaştırma CLI
│   ├── ablation.py               #    Eşlenik delta + bootstrap analizi
│   ├── compare_ablation.py       #    Ablation raporu CLI
│   ├── tokenizer_analysis.py     #    Fertility, kapsam ve morfoloji metrikleri
│   ├── analyze_tokenizer.py      #    Çoklu tokenizer karşılaştırma CLI
│   ├── tokenizer_seed.json       #    Sürümlü Türkçe tokenizer seed seti
│   ├── harmony_check.py          #    Üretimde ünlü uyumu ihlal oranı
│   ├── turkic_tokenizer_report.py#    Türk dilleri tokenizer verimi
│   └── benchmarks/               #    Sürümlü Türkçe seed benchmarklar
│
├── upload/                       # HuggingFace Entegrasyonu
│   └── push_to_hub.py            #    Model + tokenizer yükleme
│
├── scripts/                      # Yardımcı Araçlar
│   ├── prepare_data.py           #    Uçtan uca veri pipeline
│   ├── run_ablation.py           #    Kontrollü auxiliary-loss deney matrisi
│   ├── passkey_eval.py           #    Uzun bağlam passkey değerlendirmesi
│   ├── archiphoneme_corpus.py    #    Korpusu arşifonemik biçime çevirme
│   └── train_turkic_tokenizer.py #    Dengeli Türk dilleri tokenizer eğitimi
│
├── tests/                        # 🧪 Birim ve entegrasyon testleri (python -m pytest -q tests)
│
├── DATA_GOVERNANCE.md            #    Veri lisansı, kalite ve izlenebilirlik rehberi
├── EVALUATION.md                 #    Eval görevleri, metrikler ve rapor şeması
├── ABLATION.md                   #    Yardımcı loss katkı ölçüm protokolü
├── TOKENIZER_ANALYSIS.md         #    Tokenizer ölçüm ve karşılaştırma rehberi
├── REPRODUCIBILITY.md            #    Seed, manifest ve exact-resume protokolü
├── ARCHITECTURE_UPGRADES.md      #    MTP, MoE, YaRN, spekülatif çözümleme, hata düzeltmeleri
├── MORPHOPHONOLOGY.md            #    Arşifonemik ekler ve dilbilgisi koruması
├── ALIGNMENT.md                  #    SFT, DPO/ORPO, Toprak Anayasası, sohbet
├── REASONING.md                  #    GRPO ile akıl yürütme
├── LONG_CONTEXT_RAG.md           #    32K bağlam ve kaynak gösteren RAG
├── TURKIC.md                     #    Türk dilleri desteği
├── EDGE.md                       #    Cihazda çalıştırma (GGUF/MLX, kuantizasyon)
├── INTERPRETABILITY.md           #    Morfoloji Mikroskobu
├── requirements.txt              #    Python bağımlılıkları
└── LICENSE                       #    Apache License 2.0
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

# Hızlı test (örnek veri ile — vocab_size otomatik olarak 3000'e düşürülür)
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

Sohbet artık ortak sohbet şablonunu kullanır ve geçmişi modelin bağlamına
sığdırır. `--system "..."` ile sistem mesajı, `--grammar-guard mask` ile
ünlü uyumu koruması eklenebilir. En iyi sonuç için önce SFT ile talimat
eğitimi yapın (bkz. [ALIGNMENT.md](ALIGNMENT.md)).

### 4️⃣ Metin Üretimi

```bash
# Varsayılan checkpoint ile (checkpoints/toprak_last.pt)
python3 inference/generate.py \
  --prompt "Yapay zekanın geleceği" \
  --temperature 0.8 \
  --num-samples 3

# En iyi model ile
python3 inference/generate.py \
  --checkpoint checkpoints/toprak_best.pt \
  --prompt "Yapay zekanın geleceği" \
  --temperature 0.8 \
  --num-samples 3
```

### Değerlendirme

```bash
python3 evaluation/evaluate_suite.py \
  --checkpoint checkpoints/toprak_best.pt \
  --tokenizer toprak_tokenizer.model \
  --perplexity-data data_cache/clean/eval \
  --output evaluation/reports/toprak_best.json
```

Bu komut perplexity ile birlikte Türkçe dilbilgisi, morfoloji, okuduğunu
anlama, genel kültür, akıl yürütme, uzun bağlam, güvenlik ve ezberleme
metriklerini üretir. Perplexity yalnız aynı tokenizer ve sabit eval setiyle
üretilen raporlar arasında karşılaştırılmalıdır. Ayrıntılar:
[EVALUATION.md](EVALUATION.md).

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
python3 training/train.py --resume checkpoints/toprak_last.pt \
  --data-dir data_cache/clean/train

# 4. Çok boyutlu değerlendir
python3 evaluation/evaluate_suite.py --checkpoint checkpoints/toprak_best.pt \
  --perplexity-data data_cache/clean/eval

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
| **v2.x altyapısı** | MTP, MoE, YaRN, SFT/DPO/GRPO, RAG, Türk dilleri, cihazda çalıştırma, yorumlanabilirlik | ✅ Kod ve testler hazır; eğitilmiş checkpoint'lerle ölçüm bekliyor |

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
- **Ünsüz Benzeşmesi Loss**: Sert ünsüz sonrası yumuşak ünsüzlü ek tahminlerini cezalandırır (dünyada ilk)
- **Morfolojik Ağırlıklı Kayıp**: Ek tokenlerine yüksek ağırlık → morfoloji farkındalığı (dünyada ilk)
- **Morfolojik Sınır & Sentaks Başlığı**: Kök, ek ve özel token çoklu görev sınıflandırması (dünyada ilk)
- **Hece ve Kafiye Kaybı**: Dinamik hece ölçüsü (hece vezni) taşma ve satır sonu kontrolleri ile son-2 ses uyumu kafiye kısıtlamalarını eğitir (dünyada ilk)

</details>

<details>
<summary><strong>Veri Pipeline</strong></summary>

- **Crawler**: asyncio + aiohttp, robots.txt uyumlu, 1s rate limit
- **Temizleme**: HTML/Unicode, PII redaction, açıklanabilir kalite skoru, SHA-256 exact + SimHash near-dedup ve benchmark contamination kontrolü
- **Kaynaklar**: Wikipedia (~2GB), Haber siteleri (~5GB), Kamu kurumları (~1GB), Edebiyat (~500MB), Akademik (~2GB)
- **İzlenebilirlik**: Kaynak, dataset revision, lisans durumu, indirme zamanı, içerik hash'i ve kalite sinyalleri her belgede tutulur
- **Format**: JSONL — `toprak-document-v1`; ayrıntılar için [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md)

</details>

<details>
<summary><strong>Tokenizer</strong></summary>

- **Algoritma**: BPE (Byte Pair Encoding) — SentencePiece
- **Vocab**: 32,000 token
- **Karakter kapsama hedefi**: SentencePiece eğitim parametresi %99.99 + byte fallback
- **Sürümlü seed ölçümü**: 2.037 token/kelime, %0 UNK, %100 normalize round-trip; yeniden ölçmek için `evaluation/analyze_tokenizer.py`
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
- **Exact Resume**: RNG, DataLoader epoch/cursor, optimizer ve scheduler durumu checkpoint'te
- **Deney Manifesti**: Git/ortam sürümleri, tokenizer ve veri SHA-256 parmak izi
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
    <td align="center">
      <a href="https://github.com/byerlikaya">
        <img src="https://github.com/byerlikaya.png" width="80px;" alt="Barış Yerlikaya"/><br />
        <sub><b>Barış Yerlikaya</b></sub>
      </a><br />
      <sub>🐛 Bug Fix · 🔧 Tokenizer · 📦 Pipeline</sub>
    </td>
    <!-- Yeni katkıda bulunanlar buraya eklenecek -->
  </tr>
</table>

> 💡 **Sen de bu listeye girebilirsin!** Her kabul edilen PR ile katkıda bulunanlar listesine ekliyoruz. [Nasıl katkıda bulunabileceğini öğren →](CONTRIBUTING.md)

---

## Lisans

Bu proje [Apache License 2.0](LICENSE) altında yayınlanmıştır.

---

<p align="center">
  <strong>🌱 Her büyük ağaç, küçük bir tohumla başlar.</strong><br>
  <em>Toprak — Türk milletinin yapay zeka toprağı.</em>
</p>

<p align="center">
  <sub>Made with ❤️ by <a href="https://github.com/yabasi">Abbas Kandemir</a> in Türkiye</sub>
</p>
