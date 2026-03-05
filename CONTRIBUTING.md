# 🤝 Toprak'a Katkıda Bulunma Rehberi

Toprak'a ilgi gösterdiğiniz için teşekkürler! Bu proje, Türkçe için sıfırdan inşa edilen açık kaynak bir dil modelidir ve topluluk katkılarına açıktır. Bu rehber, katkı sürecini herkes için kolay ve verimli hâle getirmek amacıyla hazırlanmıştır.

---

## 📋 İçindekiler

1. [Davranış Kuralları](#-davranış-kuralları)
2. [Nasıl Katkıda Bulunabilirim?](#-nasıl-katkıda-bulunabilirim)
3. [Geliştirme Ortamının Kurulumu](#-geliştirme-ortamının-kurulumu)
4. [Proje Yapısı](#-proje-yapısı)
5. [Kod Yazım Kuralları](#-kod-yazım-kuralları)
6. [Git İş Akışı](#-git-iş-akışı)
7. [Pull Request Süreci](#-pull-request-süreci)
8. [Issue Açma Rehberi](#-issue-açma-rehberi)
9. [Katkı Alanları](#-katkı-alanları)
10. [Sık Sorulan Sorular](#-sık-sorulan-sorular)

---

## 📜 Davranış Kuralları

Bu proje, Türk yapay zeka topluluğunun gelişimine katkıda bulunmak amacıyla doğmuştur. Katılımcılardan beklentimiz:

- **Saygılı ve yapıcı iletişim** — Farklı deneyim seviyelerine hoşgörü gösterin.
- **Teknik tartışmalarda fikir odaklı olun** — Kişisel eleştirilerden kaçının.
- **Türkçe öncelikli** — Issue ve PR açıklamalarında Türkçe tercih edilir, ancak İngilizce de kabul edilir.
- **Açık kaynak ruhu** — Bilgi paylaşımı ve şeffaflık esastır.

Kabul edilemez davranışlar: hakaret, ayrımcılık, spam, kasıtlı sabotaj. Bu tür davranışlar projenin maintainer'ları tarafından uyarılır veya engellenir.

---

## 🚀 Nasıl Katkıda Bulunabilirim?

Her seviyeden katkı değerlidir:

| Seviye | Katkı Türü |
|---|---|
| 🟢 **Başlangıç** | Dokümantasyon düzeltmeleri, typo fix'ler, README iyileştirmeleri |
| 🟡 **Orta** | Bug fix'ler, yeni veri kaynakları ekleme, test yazma |
| 🔴 **İleri** | Mimari iyileştirmeler, yeni loss fonksiyonları, performans optimizasyonları |
| 🟣 **Araştırma** | Türkçe'ye özel NLP deneyleri, benchmark tasarımı, paper yazma |

> **İlk katkınız mı?** `good first issue` etiketli issue'ları kontrol edin. Herhangi bir konuda takılırsanız, issue açarak soru sormaktan çekinmeyin.

---

## 🛠 Geliştirme Ortamının Kurulumu

### Gereksinimler

- **Python 3.11+**
- **PyTorch 2.x** (`torch.compile()` desteği gerekli)
- **Git**
- macOS (MPS), Linux/Windows (CUDA) veya CPU

### Kurulum Adımları

```bash
# 1. Repoyu fork'layın ve klonlayın
git clone https://github.com/<KULLANICI_ADINIZ>/toprak.git
cd toprak

# 2. Upstream remote ekleyin
git remote add upstream https://github.com/yabasi/toprak.git

# 3. Virtual environment oluşturun
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 4. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Kurulumu Doğrulama

```bash
# Veri hazırlama (küçük örnek veri ile)
python3 scripts/prepare_data.py --use-sample

# Kısa eğitim testi
python3 training/train.py --model-size small --max-steps 100

# Metin üretimi testi
python3 inference/generate.py --checkpoint checkpoints/toprak_best.pt --prompt "Türkiye"
```

Her şey hatasız çalışıyorsa geliştirmeye hazırsınız!

---

## 📁 Proje Yapısı

Katkıda bulunmadan önce projenin yapısını anlamak önemlidir:

```
toprak/
├── model/                  # 🧠 Model mimarisi (Transformer, Attention, RoPE, vb.)
│   ├── config.py           #    ModelConfig dataclass + preset boyutlar (S/M/L/XL)
│   ├── transformer.py      #    ToprakLM ana modeli, SwiGLU FFN, TransformerBlock
│   ├── attention.py        #    Grouped Query Attention (GQA) + KV Cache
│   ├── norms.py            #    RMSNorm
│   ├── rope.py             #    Rotary Position Embedding
│   ├── tokenizer.py        #    SentencePiece BPE tokenizer wrapper
│   ├── vowel_harmony.py    #    Ünlü Uyumu Auxiliary Loss (🇹🇷 özgün)
│   └── morph_weighting.py  #    Morfolojik Ağırlıklı CE Loss (🇹🇷 özgün)
├── data/                   # 📊 Veri toplama ve işleme
│   ├── sources.py          #    Veri kaynakları ve URL tanımları
│   ├── crawler.py          #    asyncio+aiohttp web crawler
│   ├── cleaner.py          #    7 adımlı veri temizleme pipeline
│   └── dataset.py          #    ToprakDataset (PyTorch Dataset + DataLoader)
├── training/               # 🏋️ Eğitim
│   ├── train.py            #    CLI giriş noktası (argparse)
│   ├── trainer.py          #    ToprakTrainer — eğitim döngüsü, checkpoint, logging
│   └── scheduler.py        #    Cosine warmup LR scheduler
├── inference/              # 💬 Çıkarım
│   ├── generate.py         #    Metin üretimi (top-k, top-p, repetition penalty)
│   └── chat.py             #    Terminal tabanlı interaktif sohbet
├── evaluation/             # 📈 Değerlendirme
│   └── eval.py             #    Perplexity hesaplama ve model değerlendirme
├── upload/                 # ☁️ HuggingFace Hub entegrasyonu
│   └── push_to_hub.py      #    Model ve tokenizer yükleme
├── scripts/                # 🔧 Yardımcı scriptler
│   └── prepare_data.py     #    Uçtan uca veri hazırlama pipeline
├── utils/                  # 🧰 Ortak yardımcılar
│   └── validation.py       #    Hata yönetimi ve dosya doğrulama
├── requirements.txt        # Bağımlılıklar
├── toprak_tokenizer.model  # Eğitilmiş tokenizer (repo'da takip edilir)
├── toprak_tokenizer.vocab  # Tokenizer kelime listesi
├── GUIDE.md                # Kapsamlı kullanım rehberi
└── README.md               # Proje tanıtımı
```

### Veri Akışı

```
Wikipedia/Web → crawler.py → .jsonl → cleaner.py → clean/train/*.jsonl
                                                          ↓
toprak_tokenizer.model ← prepare_data.py         dataset.py → DataLoader
                                                          ↓
                                                   trainer.py → checkpoints/
                                                          ↓
                                              generate.py / chat.py / eval.py
```

---

## ✍️ Kod Yazım Kuralları

Projenin tutarlılığını korumak için aşağıdaki kurallara uyulması beklenir:

### Genel Kurallar

- **Python 3.11+** özellikleri kullanılabilir.
- **f-string** tercih edilir (`.format()` veya `%` yerine).
- **Type hint** kullanın — en azından fonksiyon imzalarında.
- Tüm `nn.Linear` katmanlarında **`bias=False`** — mimari kararıdır.

### Adlandırma Kuralları

| Öğe | Kural | Örnek |
|---|---|---|
| Sınıflar | PascalCase (İngilizce) | `ToprakLM`, `RMSNorm`, `VowelHarmonyLoss` |
| Fonksiyonlar | snake_case (İngilizce) | `precompute_freqs_cis`, `apply_rotary_emb` |
| Sabitler | UPPER_SNAKE_CASE | `BACK_VOWELS`, `CRAWL_DELAY` |
| Private | Alt çizgi ön eki | `_init_weights`, `_classify_last_vowel` |
| CLI argümanları | kebab-case | `--model-size`, `--data-dir` |

### Dil Kullanımı (Türkçe-İngilizce Karışımı)

Bu proje bilinçli bir Türkçe-İngilizce denge izler:

- **İngilizce**: Değişken isimleri, sınıf isimleri, fonksiyon isimleri, teknik terimler
- **Türkçe**: Docstring'ler, yorum satırları, kullanıcıya yönelik mesajlar (print/CLI), README/GUIDE

```python
# ✅ Doğru
class VowelHarmonyLoss(nn.Module):
    """Türkçe ünlü uyumu auxiliary loss fonksiyonu."""
    
    def _classify_last_vowel(self, token: str) -> str:
        """Token'ın son ünlüsünü kalın/ince olarak sınıflandır."""
        # Kalın ünlüler: a, ı, o, u
        # İnce ünlüler: e, i, ö, ü
        ...

# ❌ Yanlış — Türkçe değişken isimleri
class UnluUyumuKaybi(nn.Module):
    def son_unluyu_siniflandir(self, token):
        ...
```

### Docstring Formatı

**Google-style docstring** kullanılır. Modül seviyesinde Türkçe açıklama, parametrelerde teknik terimler İngilizce:

```python
"""
Toprak — Modül Başlığı
Modülün ne yaptığının kısa Türkçe açıklaması.
"""

def fonksiyon(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Fonksiyonun kısa açıklaması.

    Args:
        x: (batch_size, seq_len, d_model) — giriş tensörü
        mask: Opsiyonel attention mask

    Returns:
        output: (batch_size, seq_len, d_model) — çıkış tensörü
    """
```

### Import Sıralaması

```python
# 1. Standart kütüphane
import os
import math
from typing import List, Optional

# 2. Üçüncü parti kütüphaneler
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3. Proje içi modüller
from model.config import ModelConfig
from model.attention import GroupedQueryAttention
```

### Copyright Header

Her yeni `.py` dosyasının başına:

```python
# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.
```

---

## 🌿 Git İş Akışı

### Branch Adlandırma

```
feature/kisa-aciklama     # Yeni özellik
fix/hata-aciklamasi       # Bug fix
docs/dokueman-konusu      # Dokümantasyon
refactor/ne-degisti       # Kod yeniden yapılandırma
data/yeni-kaynak-adi      # Yeni veri kaynağı
test/test-konusu          # Test ekleme
perf/optimizasyon-konusu  # Performans iyileştirme
```

### Commit Mesajları

Açık ve anlamlı commit mesajları yazın. Türkçe veya İngilizce kabul edilir:

```bash
# ✅ İyi örnekler
git commit -m "feat: GQA attention'a sliding window desteği eklendi"
git commit -m "fix: KV cache'te bellek sızıntısı giderildi"
git commit -m "docs: GUIDE.md'ye fine-tuning bölümü eklendi"
git commit -m "data: TRT Haber veri kaynağı eklendi"
git commit -m "perf: tokenizer encode hızı %40 artırıldı"
git commit -m "test: VowelHarmonyLoss için unit test eklendi"

# ❌ Kötü örnekler
git commit -m "güncelleme"
git commit -m "fix"
git commit -m "asdfasdf"
```

### Genel Akış

```bash
# 1. Ana dalı güncelleyin
git checkout main
git pull upstream main

# 2. Feature branch oluşturun
git checkout -b feature/yeni-ozellik

# 3. Değişikliklerinizi yapın ve commit edin
git add .
git commit -m "feat: açıklayıcı mesaj"

# 4. Push edin
git push origin feature/yeni-ozellik

# 5. GitHub'da Pull Request açın
```

---

## 🔄 Pull Request Süreci

### PR Açmadan Önce Kontrol Listesi

- [ ] Kod hatasız çalışıyor (`python3 training/train.py --model-size small --max-steps 100`)
- [ ] Yeni dosyalara copyright header eklendi
- [ ] Docstring'ler yazıldı (Google-style, Türkçe açıklama)
- [ ] Type hint'ler eklendi (en azından fonksiyon imzalarında)
- [ ] Mevcut fonksiyonelliği bozmadığınızdan emin oldunuz
- [ ] Büyük değişiklikler için önce issue açıldı ve tartışıldı

### PR Şablonu

PR açarken aşağıdaki bilgileri eklemeye çalışın:

```markdown
## Ne Değişti?
Kısa ve net açıklama.

## Neden?
Bu değişikliğin motivasyonu / çözdüğü problem.

## Nasıl Test Ettim?
- [ ] Eğitim testi (`--max-steps 100`)
- [ ] Inference testi
- [ ] Yeni birim testleri (varsa)

## İlgili Issue
Fixes #123 (varsa)
```

### Review Süreci

1. PR açıldığında maintainer otomatik olarak bilgilendirilir.
2. Kod review yapılır — yapıcı geri bildirim verilir.
3. Gerekli düzeltmeler yapıldıktan sonra merge edilir.
4. Büyük mimari değişiklikler için en az 1 onay gereklidir.

---

## 🐛 Issue Açma Rehberi

### Bug Raporu

```markdown
**Ortam:**
- OS: macOS 15.x / Ubuntu 22.04 / Windows 11
- Python: 3.11.x
- PyTorch: 2.x.x
- Cihaz: MPS / CUDA (GPU modeli) / CPU

**Hata Açıklaması:**
Ne olması gerekiyordu vs. ne oldu.

**Tekrar Etme Adımları:**
1. Şu komutu çalıştırdım: `...`
2. Şu hatayı aldım: `...`

**Hata Mesajı:**
```
Tam hata mesajı buraya
```

**Ek Bilgi:**
Ekran görüntüsü, log dosyası vb.
```

### Özellik İsteği

```markdown
**Özellik:**
Ne eklenmeli / ne değişmeli?

**Motivasyon:**
Bu neden faydalı olur?

**Önerilen Çözüm:**
Nasıl uygulanabilir? (Opsiyonel)
```

---

## 🎯 Katkı Alanları

Yardıma en çok ihtiyaç duyulan alanlar:

### 📊 Veri

- Yeni Türkçe veri kaynakları ekleme (haber siteleri, kamu verileri, edebiyat)
- Veri temizleme pipeline iyileştirmeleri
- Daha büyük ve çeşitli eğitim verisi derleme
- Veri kalite metrikleri geliştirme

### 🧪 Test ve Kalite

- `pytest` ile birim testleri yazma (şu an test altyapısı yok — harika bir katkı fırsatı!)
- Her modül için temel testler: `model/`, `data/`, `training/`
- CI/CD pipeline kurulumu (GitHub Actions)
- Linter/formatter yapılandırması (ruff, black, isort)

### 📈 Benchmark ve Değerlendirme

- Türkçe NLP benchmark'ları entegrasyonu
- Farklı model boyutlarında karşılaştırmalı sonuçlar
- Inference hız benchmark'ları

### 🧠 Model ve Mimari

- Yeni attention mekanizmaları denemeleri
- Bellek optimizasyonları
- Quantization desteği (INT8, INT4)
- Flash Attention entegrasyonu
- Daha verimli KV cache stratejileri

### 🇹🇷 Türkçe'ye Özel İyileştirmeler

Bu projede dünyada ilk olan iki Türkçe'ye özel özellik var. Bu alanlardaki katkılar özellikle değerlidir:

- **Ünlü Uyumu Loss** (`model/vowel_harmony.py`) — Büyük ve küçük ünlü uyumunu kapsayacak iyileştirmeler
- **Morfolojik Ağırlıklı Loss** (`model/morph_weighting.py`) — Yeni ek tanıma stratejileri
- Türkçe'ye özgü yeni auxiliary loss fonksiyonları önerme

### 📝 Dokümantasyon

- Mevcut dokümantasyondaki eksikliklerin giderilmesi
- Türkçe NLP kavramlarının açıklanması
- API referans dokümantasyonu
- Eğitim sonuçları ve deneysel notlar

---

## ❓ Sık Sorulan Sorular

**S: Eğitime katılmak için güçlü bir GPU'm olması gerekir mi?**
H: Hayır. Küçük model (`small`, ~80M parametre) Apple Silicon MPS veya CPU üzerinde bile eğitilebilir. Kod katkıları, veri ekleme ve dokümantasyon için GPU gerekmez.

**S: Türkçe bilmem gerekir mi?**
H: Proje Türkçe odaklı olsa da, mimari ve altyapı katkıları dil bağımsızdır. Issue ve PR'lar İngilizce de açılabilir.

**S: Hangi PyTorch sürümünü kullanmalıyım?**
H: PyTorch 2.x (`torch.compile()` ve `F.scaled_dot_product_attention` desteği için). 2.0'dan düşük sürümler desteklenmez.

**S: Büyük bir değişiklik yapmak istiyorum, nasıl başlamalıyım?**
H: Önce bir issue açarak fikrini paylaş. Değişikliğin kapsamını tartıştıktan sonra kodlamaya başla. Bu sayede boşa emek harcanmaz.

**S: Fork'umu güncel tutmak için ne yapmalıyım?**
H:
```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## 🙏 Teşekkürler

Her katkı — ister bir typo düzeltmesi, ister yeni bir model özelliği — Türkçe yapay zeka ekosisteminin gelişmesine katkı sağlar. 

Bu projeye zaman ayıran herkese minnettarız. 🌱

---

<p align="center">
  <em>Toprak — Türk milletinin yapay zeka toprağı.</em><br>
  <sub>Maintained by <a href="https://github.com/yabasi">Abbas Kandemir</a></sub>
</p>
