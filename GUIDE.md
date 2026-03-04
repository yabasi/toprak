# 🌱 Toprak — Kullanım Rehberi

Sıfırdan Türkçe dil modeli geliştirme rehberi.

---

## 📋 İçindekiler

1. [Kurulum](#1-kurulum)
2. [Veri Hazırlama](#2-veri-hazırlama)
3. [Eğitim](#3-eğitim)
4. [Eğitime Devam Etme](#4-eğitime-devam-etme)
5. [TensorBoard ile Takip](#5-tensorboard-ile-takip)
6. [Metin Üretimi](#6-metin-üretimi)
7. [Sohbet Modu](#7-sohbet-modu)
8. [Model Değerlendirme](#8-model-değerlendirme)
9. [HuggingFace'e Yükleme](#9-huggingfacee-yükleme)
10. [Model Boyutları](#10-model-boyutları)
11. [Eğitim Parametreleri](#11-eğitim-parametreleri)
12. [RTX 4090 Sunucuda Eğitim](#12-rtx-4090-sunucuda-eğitim)
13. [Sık Karşılaşılan Sorunlar](#13-sık-karşılaşılan-sorunlar)

---

## 1. Kurulum

```bash
# Projeyi klonla
git clone https://github.com/yabasi/toprak.git
cd toprak

# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## 2. Veri Hazırlama

Eğitimden önce veriyi indirip temizlemen gerekiyor.

```bash
# Tüm pipeline'ı çalıştır (indirme → temizleme → bölme)
python3 scripts/prepare_data.py

# Sadece test amaçlı küçük bir örnek veri ile
python3 scripts/prepare_data.py --use-sample
```

Veri hazırlandıktan sonra şu yapı oluşur:

```
data_cache/
├── clean/
│   ├── train/          ← Eğitim verisi (.jsonl dosyaları)
│   └── eval/           ← Değerlendirme verisi
└── wikipedia_tr.jsonl  ← Ham veri
```

---

## 3. Eğitim

### Temel kullanım

```bash
python3 training/train.py --model-size small --max-steps 1000
```

### Tüm parametrelerle

```bash
python3 training/train.py \
  --model-size small \
  --data-dir data_cache/clean/train \
  --eval-data-dir data_cache/clean/eval \
  --tokenizer toprak_tokenizer.model \
  --batch-size 16 \
  --lr 1e-4 \
  --max-steps 100000 \
  --warmup-steps 5000 \
  --grad-accum 4 \
  --save-every 5000 \
  --checkpoint-dir checkpoints \
  --log-dir logs
```

### Ünlü Uyumu Loss ile eğitim

Türkçe'ye özel auxiliary loss — ünlü uyumuna aykırı token tahminlerini cezalandırır. Mevcut eğitime checkpoint'ten devam ederken eklenebilir.

```bash
# Ünlü uyumu loss aktif
python3 training/train.py --model-size medium --vowel-harmony

# Lambda ağırlığı ve warmup ayarla
python3 training/train.py --model-size medium \
  --vowel-harmony --vh-lambda 0.1 --vh-warmup-steps 1000

# Mevcut eğitime ekleyerek devam et
python3 training/train.py --model-size medium \
  --resume checkpoints/toprak_step_17500.pt \
  --vowel-harmony --vh-lambda 0.1 --vh-warmup-steps 1000
```

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--vowel-harmony` | Kapalı | Ünlü uyumu auxiliary loss'u aktifleştir |
| `--vh-lambda` | 0.1 | Loss ağırlığı (0.05–0.3 arası önerilir) |
| `--vh-warmup-steps` | 1000 | Lambda warmup adım sayısı (ani spike önleme) |

### Morfolojik Ağırlıklı Kayıp ile eğitim

Türkçe'ye özel loss ağırlıklandırması — ek (suffix) tokenlerine daha yüksek cross-entropy ağırlığı vererek modelin Türkçe morfolojisini daha iyi öğrenmesini sağlar. Dünyada bir ilktir. Mevcut eğitime checkpoint'ten devam ederken eklenebilir.

```bash
# Morfolojik ağırlıklı kayıp aktif
python3 training/train.py --model-size medium --morph-weight

# Ek ağırlığı ve warmup ayarla
python3 training/train.py --model-size medium \
  --morph-weight --morph-suffix-weight 1.3 --morph-warmup-steps 500

# Mevcut eğitime ekleyerek devam et
python3 training/train.py --model-size medium \
  --resume checkpoints/toprak_step_21052.pt \
  --morph-weight --morph-suffix-weight 1.3 --morph-warmup-steps 500

# Ünlü uyumu ile birlikte kullan (her iki dünya ilki birden)
python3 training/train.py --model-size medium \
  --resume checkpoints/toprak_step_21052.pt \
  --morph-weight --morph-suffix-weight 1.3 --morph-warmup-steps 500 \
  --vowel-harmony --vh-lambda 0.1 --vh-warmup-steps 1000
```

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--morph-weight` | Kapalı | Morfolojik ağırlıklı CE loss'u aktifleştir |
| `--morph-suffix-weight` | 1.3 | Ek token ağırlığı (1.0–2.0 arası önerilir) |
| `--morph-warmup-steps` | 500 | Ağırlık warmup adım sayısı (ani bozulma önleme) |

> 💡 **Nasıl çalışır?** SentencePiece tokenizer'da `▁` prefix'li tokenler kök, prefix'siz olanlar ek (suffix) olarak kabul edilir. Vocab'daki ~%29 ek token'a daha yüksek ağırlık verilerek model Türkçe ekleme yapısına odaklanır.

### Optimizasyonları kapatma

```bash
# torch.compile kapatmak istersen (MPS'te uyarı veriyorsa)
python3 training/train.py --model-size small --no-compile

# Gradient checkpointing kapatmak istersen
python3 training/train.py --model-size small --no-grad-checkpoint

# İkisini de kapat
python3 training/train.py --model-size small --no-compile --no-grad-checkpoint
```

### Cihaz seçimi

```bash
# Otomatik algılama (varsayılan — CUDA > MPS > CPU sırasıyla seçer)
python3 training/train.py --model-size small

# Manuel seçim
python3 training/train.py --model-size small --device mps
python3 training/train.py --model-size small --device cuda
python3 training/train.py --model-size small --device cpu
```

---

## 4. Eğitime Devam Etme

Eğitim yarıda kaldıysa veya daha fazla adım eklemek istersen:

```bash
# Son checkpoint'ten devam et
python3 training/train.py \
  --model-size small \
  --resume checkpoints/toprak_step_5000.pt \
  --max-steps 200000

# En iyi modelden devam et
python3 training/train.py \
  --model-size small \
  --resume checkpoints/toprak_best.pt \
  --max-steps 200000
```

### Mevcut checkpoint'leri görmek için

```bash
ls -lh checkpoints/
```

> ⚠️ **Önemli:** `--model-size` parametresi checkpoint'teki model boyutuyla aynı olmalı. Small ile eğitilen model, small ile devam ettirilmeli.

---

## 5. TensorBoard ile Takip

Eğitim sırasında loss, learning rate ve diğer metrikleri görsel olarak takip et.

```bash
# Yeni bir terminal sekmesi aç ve çalıştır:
cd ~/Desktop/works/toprak
source venv/bin/activate
tensorboard --logdir logs
```

Tarayıcıda aç: **http://localhost:6006**

### Göreceğin grafikler:
- **train/loss** — Eğitim kaybı (düşmesi beklenir)
- **train/learning_rate** — Öğrenme oranı (warmup → cosine decay)
- **train/tokens_per_sec** — İşleme hızı
- **train/grad_norm** — Gradient büyüklüğü
- **train/vh_loss** — Ünlü uyumu auxiliary loss (`--vowel-harmony` aktifse)
- **train/root_loss** — Kök tokenlerinin CE loss'u (`--morph-weight` aktifse)
- **train/suffix_loss** — Ek tokenlerinin CE loss'u (`--morph-weight` aktifse)
- **train/suffix_weight_effective** — Efektif ek ağırlığı (warmup takibi)
- **eval/loss** — Değerlendirme kaybı
- **eval/perplexity** — Perplexity (düşük = daha iyi)

---

## 6. Metin Üretimi

Eğitilmiş modelle metin üret.

```bash
# Temel kullanım
python3 inference/generate.py \
  --checkpoint checkpoints/toprak_best.pt \
  --prompt "Türkiye'nin başkenti"

# Tüm parametrelerle
python3 inference/generate.py \
  --checkpoint checkpoints/toprak_best.pt \
  --tokenizer toprak_tokenizer.model \
  --prompt "Yapay zekanın geleceği" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9 \
  --num-samples 3
```

### Parametreler ne yapar?

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--temperature` | 0.8 | Yaratıcılık. 0.1 = tutarlı, 1.5 = yaratıcı/kaotik |
| `--top-k` | 50 | Her adımda en olası K token arasından seç |
| `--top-p` | 0.9 | Kümülatif olasılık eşiği (nucleus sampling) |
| `--max-tokens` | 200 | Üretilecek maksimum token sayısı |
| `--num-samples` | 1 | Kaç farklı çıktı üretilecek |

> 💡 **İpucu:** Daha tutarlı çıktı için `--temperature 0.3`, daha yaratıcı çıktı için `--temperature 1.2` dene.

---

## 7. Sohbet Modu

Terminal üzerinden modelle sohbet et.

```bash
python3 inference/chat.py \
  --checkpoint checkpoints/toprak_best.pt \
  --tokenizer toprak_tokenizer.model
```

### Sohbet komutları:
- Mesajını yaz, Enter'a bas
- `çık` veya `exit` → Çıkış
- `temizle` → Sohbet geçmişini sıfırla
- `ayar` → Temperature, top-k, top-p değiştir

---

## 8. Model Değerlendirme

Modelin kalitesini ölç (perplexity).

```bash
python3 evaluation/eval.py \
  --checkpoint checkpoints/toprak_best.pt \
  --eval-data data_cache/clean/eval \
  --tokenizer toprak_tokenizer.model
```

### Perplexity ne demek?

| Perplexity | Anlam |
|---|---|
| < 50 | ✅ Çok iyi |
| 50 - 100 | 🟡 İyi, iyi yolda |
| 100 - 500 | 🟠 Orta, daha fazla eğitim gerekli |
| > 500 | 🔴 Yetersiz, daha fazla veri ve eğitim gerekli |

---

## 9. HuggingFace'e Yükleme

Modeli HuggingFace'e yükleyip başkalarıyla paylaş.

```bash
# Önce HuggingFace'e giriş yap
huggingface-cli login

# Modeli yükle
python3 upload/push_to_hub.py \
  --checkpoint checkpoints/toprak_best.pt \
  --tokenizer toprak_tokenizer.model \
  --repo-name yabasi/toprak-small
```

---

## 10. Model Boyutları

| Boyut | Parametre | Eğitim Süresi (M4 Pro) | Eğitim Süresi (RTX 4090) | Komut |
|---|---|---|---|---|
| **small** | ~80M | ~1-2 gün | ~30 dk | `--model-size small` |
| **medium** | ~125M | ~4-6 gün | ~2-3 saat | `--model-size medium` |
| **large** | ~342M | ⚠️ RTX 4090 önerilir | ~6-8 saat | `--model-size large` |
| **xl** | ~941M | ⚠️ RTX 4090 gerekli | ~18-24 saat | `--model-size xl` |

> 💡 **Öneri:** M4 Pro'da `small` veya `medium` ile başla. `large` ve `xl` için RTX 4090 sunucu kirala.

---

## 11. Eğitim Parametreleri

### Komut satırı parametreleri

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--model-size` | small | Model boyutu: small, medium, large, xl |
| `--data-dir` | data_cache/clean/train | Eğitim verisi dizini |
| `--eval-data-dir` | *(yok)* | Eval verisi dizini (opsiyonel) |
| `--tokenizer` | toprak_tokenizer.model | Tokenizer dosyası |
| `--batch-size` | Modele göre | Batch büyüklüğü |
| `--lr` | Modele göre | Learning rate (öğrenme oranı) |
| `--max-steps` | Modele göre | Toplam eğitim adımı |
| `--warmup-steps` | Modele göre | LR warmup adımı |
| `--grad-accum` | Modele göre | Gradient accumulation adım sayısı |
| `--save-every` | 5000 | Kaç adımda bir checkpoint kaydet |
| `--resume` | *(yok)* | Devam edilecek checkpoint |
| `--checkpoint-dir` | checkpoints | Checkpoint kayıt dizini |
| `--device` | otomatik | mps, cuda veya cpu |
| `--no-compile` | *(kapalı)* | torch.compile devre dışı bırak |
| `--no-grad-checkpoint` | *(kapalı)* | Gradient checkpointing kapat |
| `--log-dir` | logs | TensorBoard log dizini |
| `--vowel-harmony` | *(kapalı)* | Ünlü uyumu auxiliary loss aktifleştir |
| `--vh-lambda` | 0.1 | Ünlü uyumu loss ağırlığı |
| `--vh-warmup-steps` | 1000 | Ünlü uyumu loss warmup adım sayısı |
| `--morph-weight` | *(kapalı)* | Morfolojik ağırlıklı CE loss aktifleştir |
| `--morph-suffix-weight` | 1.3 | Ek token ağırlık çarpanı |
| `--morph-warmup-steps` | 500 | Morfolojik ağırlık warmup adım sayısı |

### Modellerin varsayılan parametreleri

| Parametre | Small | Medium | Large | XL |
|---|---|---|---|---|
| d_model | 640 | 768 | 1024 | 1536 |
| layers | 14 | 16 | 28 | 36 |
| heads | 10 | 12 | 16 | 16 |
| kv_heads | 2 | 4 | 4 | 4 |
| max_seq_len | 512 | 512 | 2048 | 2048 |
| batch_size | 16 | 8 | 4 | 2 |
| grad_accum | 4 | 4 | 16 | 32 |
| efektif batch | 64 | 32 | 64 | 64 |
| learning_rate | 1e-4 | 1e-4 | 3e-4 | 3e-4 |
| max_steps | 100K | 100K | 300K | 500K |

---

## 12. RTX 4090 Sunucuda Eğitim

### Adım 1: Projeyi sunucuya yükle

```bash
# Sunucuya bağlan
ssh yabasi@sunucu-ip

# Projeyi klonla veya SCP ile gönder
scp -r ~/Desktop/works/toprak yabasi@sunucu-ip:~/toprak

# Sadece veri dosyalarını gönder (model kodu daha küçük)
scp -r data_cache/ toprak_tokenizer.model yabasi@sunucu-ip:~/toprak/
```

### Adım 2: Sunucuda ortamı kur

```bash
cd ~/toprak
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Adım 3: Eğitimi başlat

```bash
# CUDA otomatik algılanır
python3 training/train.py \
  --model-size large \
  --max-steps 300000

# Arka planda çalıştırmak için (SSH kapansa da devam eder):
nohup python3 training/train.py \
  --model-size large \
  --max-steps 300000 \
  > training.log 2>&1 &

# Log'u takip et:
tail -f training.log
```

### Adım 4: Checkpoint'i M4 Pro'ya indir

```bash
# Sunucudan en iyi modeli indir
scp yabasi@sunucu-ip:~/toprak/checkpoints/toprak_best.pt \
    ~/Desktop/works/toprak/checkpoints/

# M4 Pro'da test et
python3 inference/generate.py \
  --checkpoint checkpoints/toprak_best.pt \
  --prompt "Merhaba dünya"
```

---

## 13. Sık Karşılaşılan Sorunlar

### ❌ "Veri dosyası bulunamadı"
```
💡 Doğru veri dizinini belirtin:
   --data-dir data_cache/clean/train
```

### ❌ "Checkpoint bulunamadı"
```bash
# Mevcut checkpoint'leri görmek için:
ls checkpoints/

# Henüz eğitim yapılmadıysa, önce eğitin:
python3 training/train.py --model-size small --max-steps 1000
```

### ❌ "MPS kullanılamıyor"
Apple M serisi çip gerekli. Intel Mac'te `--device cpu` kullan.

### ❌ "CUDA out of memory"
```bash
# Batch size küçült
python3 training/train.py --model-size large --batch-size 2

# Veya gradient accumulation artır
python3 training/train.py --model-size large --batch-size 1 --grad-accum 64
```

### ❌ "torch.compile uyarıları"
MPS'te `torch.compile` tam desteklenmediği için uyarılar normal:
```bash
# Uyarıları sustur:
python3 -W ignore training/train.py --model-size small

# Veya compile'ı kapat:
python3 training/train.py --model-size small --no-compile
```

### ❌ "TensorBoard çalışmıyor"
```bash
pip install tensorboard "setuptools<82"
```

---

## 🗺️ Tipik Geliştirme Akışı

```
1. Veri hazırla
   └── python3 scripts/prepare_data.py

2. Small model ile hızlı test (1000 adım)
   └── python3 training/train.py --model-size small --max-steps 1000

3. TensorBoard'dan loss düşüyor mu kontrol et
   └── tensorboard --logdir logs → http://localhost:6006

4. Tam eğitim (small, 100K adım)
   └── python3 training/train.py --model-size small

5. Değerlendirme
   └── python3 evaluation/eval.py --checkpoint checkpoints/toprak_best.pt --eval-data data_cache/clean/eval

6. Metin üretimi ile test et
   └── python3 inference/generate.py --checkpoint checkpoints/toprak_best.pt --prompt "test"

7. Memnunsan → Büyük modele geç (RTX 4090 ile)
   └── python3 training/train.py --model-size large

8. HuggingFace'e yükle
   └── python3 upload/push_to_hub.py --checkpoint checkpoints/toprak_best.pt
```

---

> 🌱 **Toprak** — Sıfırdan, Türkçe, Açık Kaynak.
