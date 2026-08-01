# 🚀 Toprak — RunPod Eğitim Rehberi

A100 80GB üzerinde sıfırdan **Large (342M)** modelini Wikipedia TR + FineWeb-2 + CulturaX üçlüsüyle ~$120 maliyetle eğitmek için uçtan uca operasyonel rehber.

---

## Tahmini Maliyet & Süre

| Adım | Pod | Süre | $/sa | Maliyet |
|---|---|---|---|---|
| 1. Veri indirme + temizleme + tokenize | CPU pod | ~1 gün | $0.15 | ~$4 |
| 2. Sanity (Small, 2K step) | A100 80GB | ~1 saat | $1.52 | ~$1.5 |
| 3. **Large eğitim (~7B token, 55K step)** | A100 80GB | ~55-75 saat | $1.52 | **~$85-115** |
| 4. Eval + Upload | A100 (kısa) | ~2 saat | $1.52 | ~$3 |
| **TOPLAM** | | **~4-5 gün** | | **~$95-125** |

> İpucu: **Spot/Community Cloud** A100 SXM ~$1.20-1.30/saat'e iner. Checkpoint her 2500 step alındığı için preemption riski kabul edilebilir.

---

## 0. Ön Hazırlık (yerelde, 10 dk)

```bash
# 1. Repo'yu GitHub'a push et (RunPod'da clone için)
cd ~/Desktop/works/toprak
git add . && git commit -m "RunPod pipeline ready"
git push origin main

# 2. HuggingFace token al (https://huggingface.co/settings/tokens — read scope yeter)
# Bu token CulturaX / FineWeb-2 indirmek için gerekli olabilir
```

---

## 1. CPU Pod — Veri Hazırlama (~1 gün, ~$4)

### 1a) Pod oluştur

**RunPod Console → Deploy → CPU Pod:**
- Template: **runpod/pytorch:2.4.0** (CPU varyantı yoksa GPU template seç ama GPU=None bırak)
- vCPU: 4-8
- RAM: 32 GB
- Disk: **Network Volume 200 GB** (mutlaka persistent! pod silinse bile veri kalır)
- Mount point: `/workspace`

### 1b) Pod içinde kurulum

```bash
cd /workspace
git clone https://github.com/yabasi/toprak.git
cd toprak

pip install -r requirements.txt

# HuggingFace login (CulturaX/FineWeb için)
huggingface-cli login
```

### 1c) 3 dataset'i indir

```bash
# Wikipedia TR (~30 dk, ~1 GB JSONL)
# FineWeb-2 TR (~3-5 saat, hedef 30 GB JSONL)
# CulturaX TR (~3-5 saat, hedef 15 GB JSONL)
python scripts/download_all_corpora.py \
  --output-dir /workspace/data_raw \
  --wikipedia --fineweb2 --culturax \
  --fineweb-target-gb 30 \
  --culturax-target-gb 15
```

> Bu komut indirilen veriler hedef GB'a ulaşınca otomatik durur. CulturaX büyük; bant genişliğine bağlı olarak 2-6 saat sürer.

### 1d) Temizleme

```bash
# Kalite, PII, exact/near-dedup ve provenance pipeline'ı
python data/cleaner.py \
  --input /workspace/data_raw \
  --output /workspace/data_raw/clean

# Metadata, hash ve duplicate denetimi
python scripts/audit_corpus.py /workspace/data_raw/clean --strict
```

### 1e) Tokenizer'ı sıfırdan eğit

```bash
# Birleşik temiz korpus üzerinde SentencePiece BPE 32K
# Önce tokenizer eğitim verisi düz metin olarak hazırlanır
python -c "
from data.cleaner import ToprakCleaner
ToprakCleaner().prepare_tokenizer_data(
    '/workspace/data_raw/clean',
    '/workspace/data_raw/clean/tokenizer_train.txt'
)"

# Sonra tokenizer eğit (~30-60 dk, CPU)
python -c "
from model.tokenizer import train_tokenizer
train_tokenizer(
    '/workspace/data_raw/clean/tokenizer_train.txt',
    model_prefix='/workspace/toprak/toprak_tokenizer',
    vocab_size=32000,
)"
```

### 1f) Pre-tokenize → .bin shard'lar

```bash
# JSONL → uint16 .bin shard'ları (curriculum: Wiki son shard'lara)
python scripts/pretokenize.py \
  --input-dir /workspace/data_raw/clean \
  --tokenizer /workspace/toprak/toprak_tokenizer.model \
  --output-dir /workspace/data_bin \
  --shard-size 100000000 \
  --eval-ratio 0.005 \
  --curriculum --hq-sources wiki
```

**Çıktı:** `/workspace/data_bin/manifest.json` + `train_*.bin` (~14-20 GB) + `eval_*.bin`

### 1g) CPU pod'u DURDUR

RunPod Console → pod → Stop. Volume kalır, pod ücreti durur.

---

## 2. A100 Pod — Sanity Run (~1 saat, ~$1.5)

### 2a) A100 pod oluştur

**RunPod Console → Deploy → GPU Pod:**
- Template: **runpod/pytorch:2.4.0-py3.11-cuda12.4**
- GPU: **1x A100 SXM 80GB** (Community Cloud → ~$1.20-1.30/saat tasarrufu)
- vCPU: 16, RAM: 188 GB (default)
- Disk container: 50 GB
- **Aynı Network Volume'u mount et** → `/workspace`

### 2b) Sanity testi

```bash
cd /workspace/toprak
pip install -r requirements.txt
# Opsiyonel: pip install flash-attn --no-build-isolation

# Small modelle 2000 step — pipeline'ı doğrula
python training/train.py \
  --model-size small \
  --data-dir /workspace/data_bin \
  --tokenizer /workspace/toprak/toprak_tokenizer.model \
  --bin-mode --bf16 \
  --max-steps 2000 \
  --batch-size 16 --grad-accum 2 \
  --save-every 1000 \
  --checkpoint-dir /workspace/checkpoints_sanity \
  --log-dir /workspace/logs_sanity
```

**Beklenen:**
- Loss 10.4 → < 5.0 düşmeli
- NaN yok
- Throughput ~80-150K token/sn (small için)

Sorun yoksa devam, varsa burada düzelt. Bu adım GPU saatini "boşa harcamamak" için kritik.

---

## 3. A100 Pod — Large Tam Eğitim (~55-75 saat, ~$85-115)

```bash
python training/train.py \
  --model-size large \
  --data-dir /workspace/data_bin \
  --tokenizer /workspace/toprak/toprak_tokenizer.model \
  --bin-mode --bf16 \
  --batch-size 8 --grad-accum 8 \
  --max-steps 55000 --warmup-steps 2000 \
  --lr 3e-4 \
  --save-every 2500 \
  --checkpoint-dir /workspace/checkpoints \
  --log-dir /workspace/logs \
  --num-workers 4
```

**Hedefler:**
- 55K step × 64 efektif batch × 2048 seq ≈ **7.2B token**
- Throughput ~30-40K token/sn beklenir (A100 bf16, FlashAttn SDPA)
- Val perplexity < 18 (Wikipedia eval üzerinde)
- Her 2500 step checkpoint (`keep_last_n=3`)

### TensorBoard izleme (ikinci terminal / yerel)

```bash
# Pod içinde:
tensorboard --logdir /workspace/logs --port 6006 --bind_all

# RunPod Console → pod → "Connect" → "HTTP Service: 6006" → tarayıcıda aç
```

### Checkpoint yedekleme (önerilir, PRIVATE repo)

Her gece bir kez Hugging Face Hub'a push et (pod kayıp riski).
**Önemli: Repo PRIVATE olarak oluşturulur — sadece sen görürsün, kimse erişemez.**

```bash
# 1. Bir kerelik kurulum: PRIVATE repo oluştur
huggingface-cli login   # HF token gir (https://huggingface.co/settings/tokens)
huggingface-cli repo create toprak-large-v0.1-ckpt --type model --private

# 2. Her gece: best checkpoint'i private repoya yükle
huggingface-cli upload yabasi/toprak-large-v0.1-ckpt \
  /workspace/checkpoints/toprak_best.pt \
  toprak_best.pt
```

> Repo URL'inde 🔒 (kilit) ikonunu kontrol et: `https://huggingface.co/yabasi/toprak-large-v0.1-ckpt`
> Eğitim bitince ya bu repoyu public'e çevirirsin ya da yeni temiz public repo açıp sadece final modeli yüklersin (önerilir — bkz. Aşama 4).

---

## 4. Eval + Upload (~2 saat, ~$3)

### 4a) Final değerlendirme

```bash
python evaluation/eval.py \
  --checkpoint /workspace/checkpoints/toprak_best.pt \
  --eval-data /workspace/data_bin \
  --tokenizer /workspace/toprak/toprak_tokenizer.model
```

Hedef perplexity: **<18 (Wiki eval)**, kabul edilebilir <25.

### 4b) Hızlı sohbet testi (sanity)

```bash
python inference/chat.py \
  --checkpoint /workspace/checkpoints/toprak_best.pt \
  --tokenizer /workspace/toprak/toprak_tokenizer.model
```

Türkçe akıcılığı manuel kontrol et. Garip / bozuk üretim varsa public yayını ertele.

### 4c) HuggingFace'e PUBLIC yayın (launch)

> Önerilen: **temiz, ayrı bir public repo** oluştur — private ckpt repo'sunu olduğu gibi bırak.

```bash
# 1. Yeni PUBLIC repo (--private flag YOK = public)
huggingface-cli repo create toprak-large-v0.1 --type model

# 2. Final modeli, tokenizer'ı ve config'i yükle
huggingface-cli upload yabasi/toprak-large-v0.1 \
  /workspace/checkpoints/toprak_best.pt toprak_best.pt
huggingface-cli upload yabasi/toprak-large-v0.1 \
  /workspace/toprak/toprak_tokenizer.model toprak_tokenizer.model

# 3. Model kartı (README.md) — Apache 2.0 lisansı, atıflar, kullanım örneği
#    Bu dosyayı yerel olarak hazırla, sonra yükle:
huggingface-cli upload yabasi/toprak-large-v0.1 \
  /workspace/toprak/HF_README.md README.md
```

**Alternatif:** Mevcut script ile (özelleştirilmiş push):
```bash
python upload/push_to_hub.py \
  --checkpoint /workspace/checkpoints/toprak_best.pt \
  --repo yabasi/toprak-large-v0.1
```

### 4d) Pod'u durdur

A100 pod'u **DURDUR**. Volume'da ckpt + model + veri kalır (~$0.67/gün storage).

---

## ⚠️ Yaygın Sorunlar

| Sorun | Çözüm |
|---|---|
| `manifest.json bulunamadı` | `pretokenize.py` adımını çalıştır |
| `huggingface_hub` 401 (CulturaX) | `huggingface-cli login`, dataset usage policy'sini onayla |
| OOM (out of memory) | `--batch-size 4 --grad-accum 16` (efektif batch 64 kalır) |
| Yavaş throughput (<20K t/s) | `--num-workers 4`, `pin_memory=True` (otomatik bin-mode'da) |
| Spot preemption | Otomatik checkpoint'ten devam: `--resume /workspace/checkpoints/toprak_last.pt` |
| Loss plateauı | LR'yi yarıya indir veya warmup'u uzat |
| NaN loss | `--no-compile` dene; `--lr 1e-4`'e düşür |

---

## 💰 Maliyet Tasarruf Checklist

- [x] Veri hazırlık **CPU pod**'da (A100 değil) — ~$50 tasarruf
- [x] **Network Volume** — tekrar tekrar indirme yok
- [x] **Community/Spot A100** — saatlik %20-30 ucuz
- [x] **`bf16`** — A100 native, fp16'dan stabil, hız aynı
- [x] **`torch.compile()` AÇIK** — %15-25 hız
- [x] **Pre-tokenized .bin** — I/O bottleneck yok
- [x] Sanity run önce — A100'de hata ayıklama maliyeti yok
- [x] Pod'ları **kullanmadığında DURDUR** — saatlik faturalama

---

## 📋 Kısa Komut Özeti

```bash
# CPU pod (1 kere, ~$4)
python scripts/download_all_corpora.py --output-dir /workspace/data_raw --all
python data/cleaner.py --input /workspace/data_raw --output /workspace/data_raw/clean
# tokenizer eğit (yukarıdaki Python snippet)
python scripts/pretokenize.py --input-dir /workspace/data_raw/clean \
  --tokenizer /workspace/toprak/toprak_tokenizer.model \
  --output-dir /workspace/data_bin --curriculum

# A100 pod sanity (~$1.5)
python training/train.py --model-size small --data-dir /workspace/data_bin \
  --tokenizer /workspace/toprak/toprak_tokenizer.model --bin-mode --bf16 \
  --max-steps 2000 --batch-size 16 --grad-accum 2

# A100 pod Large eğitim (~$100)
python training/train.py --model-size large --data-dir /workspace/data_bin \
  --tokenizer /workspace/toprak/toprak_tokenizer.model --bin-mode --bf16 \
  --batch-size 8 --grad-accum 8 --max-steps 55000 --warmup-steps 2000 \
  --lr 3e-4 --save-every 2500 --num-workers 4
```

İyi şanslar! 🌱
