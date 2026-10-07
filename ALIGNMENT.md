# 🌱 Toprak — Hizalama Rehberi (SFT → Anayasa → DPO/ORPO → Sohbet)

Bu belge, ön eğitilmiş bir Toprak checkpoint'ini **talimat izleyen, dürüst ve
güvenli bir Türkçe asistana** dönüştüren hattı anlatır. Tüm adımlar aynı
sohbet şablonunu (`model/chat_template.py`) kullanır; eğitimde gördüğü biçim
ile sohbette gördüğü biçim birebir aynıdır.

## 1. Hattın genel görünümü

```text
 ön eğitim (training/train.py)
        │  checkpoints/toprak_best.pt
        ▼
 SFT — talimat ince ayarı (training/sft.py, opsiyonel LoRA)
        │  checkpoints/toprak_sft.pt
        ▼
 Anayasaya dayalı öz-düzeltme (alignment/constitutional.py)
        │  data/cai_pairs.jsonl  (+ data/cai_sft.jsonl)
        ▼
 Tercih hizalaması — DPO veya ORPO (training/dpo.py)
        │  checkpoints/toprak_dpo.pt
        ▼
 Sohbet (inference/chat.py) ve değerlendirme (evaluation/evaluate_suite.py)
```

| Aşama | Ne öğretir? | Girdi | Çıktı |
|---|---|---|---|
| **SFT** | Soruya cevap verme biçimi, rol yapısı, tur sonu | Sohbet/Alpaca JSONL | Checkpoint |
| **Öz-düzeltme** | Anayasa ilkelerine göre eleştir → düzelt | Prompt listesi + SFT modeli (veya öğretmen) | Tercih çiftleri, düzeltilmiş SFT kayıtları |
| **DPO / ORPO** | İyi cevabı kötüsüne tercih etme | Tercih çiftleri JSONL | Checkpoint |
| **Sohbet** | — | Checkpoint | Etkileşimli terminal |

Her aşamanın çıktısı `training/trainer.py` ile aynı checkpoint biçimindedir
(`model_state_dict` + `config`), yani `inference/generate.py`, `inference/chat.py`
ve `evaluation/evaluate_suite.py` bu dosyaları doğrudan açar. LoRA
kullanıldığında uyarlayıcılar kayıttan önce ana ağırlıklara **katlanır**; ayrı bir
"adapter" dosyası yoktur.

## 2. Sohbet şablonu

`ChatTemplate` mesaj listesini token dizisine çevirir. Kayıp maskesi yalnız
asistan içeriği ve onun tur sonu token'ı için 1'dir; model kullanıcıyı taklit
etmeyi değil, cevap vermeyi öğrenir.

**Yeni tokenizer (özel tokenlar varsa)** — her rol tek bir token:

```text
<s><|sistem|>…<|son|><|kullanıcı|>…<|son|><|toprak|>…<|son|>
```

Bu tokenlar (`CHAT_SPECIAL_TOKENS`) tokenizer eğitiminde `user_defined_symbols`
olarak eklenmelidir.

**Mevcut 32K tokenizer (geri dönüş)** — özel tokenlar sözlükte olmadığından metin
işaretleyicileri kullanılır; tur sonu, tokenizer'da zaten tek token olan `<sep>`'tir:

```text
<s>Sistem: …<sep>Kullanıcı: …<sep>Toprak: …<sep>
```

Hangi modun seçildiği otomatik algılanır ve SFT CLI'ı başlangıçta yazdırır
(`Şablon: özel tokenlar` / `Şablon: <sep> metin işaretleyicileri`). Önemli:
**bir modeli hangi modla eğittiyseniz o modla kullanın**; tokenizer'ı
değiştirirseniz SFT'yi yeniden yapın.

Üretim, `template.stop_ids` (tur sonu + EOS) görüldüğünde durur.

## 3. Veri biçimleri

### SFT (`training/sft.py`)

Satır başına bir JSON nesnesi. İki biçim kabul edilir:

```json
{"messages": [{"role": "system", "content": "…"},
              {"role": "user", "content": "Ünlü uyumu nedir?"},
              {"role": "assistant", "content": "Ünlü uyumu, …"}]}
{"instruction": "Yazım yanlışlarını düzeltin.", "input": "Herkez gelicek.", "output": "Herkes gelecek."}
```

- Roller: `system`, `user`, `assistant`. Çok turlu sohbetlerde her asistan turu
  eğitilir.
- `--max-len` (varsayılan: modelin `max_seq_len`) aşılırsa sondan kırpılır;
  kırpma sonrası hiç asistan token'ı kalmayan örnekler **atlanır** (sayısı
  ekrana yazılır).

### Tercih çiftleri (`training/dpo.py`)

```json
{"prompt": "Bugün dolar kuru kaç TL?",
 "chosen": "Güncel kurlara erişimim yok; TCMB'nin sitesine bakabilirsiniz.",
 "rejected": "Bugün dolar tam olarak 12,45 TL."}
```

`prompt` düz metin ya da mesaj listesi (çok turlu bağlam) olabilir.
`alignment/constitutional.py` ek olarak `"principles"` alanı yazar; eğitim bu
alanı yok sayar.

### Örnek dosyalar

| Dosya | İçerik |
|---|---|
| `alignment/examples/sft_sample.jsonl` | 8 SFT örneği (her iki biçim) |
| `alignment/examples/dpo_sample.jsonl` | 6 tercih çifti |
| `alignment/examples/prompts.txt` | Öz-düzeltme için 10 prompt (`#` satırları yorum) |

> ⚠️ Bu dosyalar **yalnızca biçim örneğidir**. Birkaç örnekle hiçbir model
> asistan olmaz; gerçek bir SFT için binlerce, tercih hizalaması için en az
> birkaç bin kaliteli örnek gerekir.

## 4. Komutlar

### 4.1 SFT

```bash
# Tam ince ayar (küçük modeller, CUDA)
python training/sft.py --base-checkpoint checkpoints/toprak_best.pt \
    --data data/sft_train.jsonl --eval-data data/sft_eval.jsonl \
    --output checkpoints/toprak_sft.pt --epochs 3 --batch-size 8 --grad-accum 4

# LoRA (Mac için önerilen)
python training/sft.py --base-checkpoint checkpoints/toprak_best.pt \
    --data data/sft_train.jsonl --eval-fraction 0.05 \
    --output checkpoints/toprak_sft.pt --lora-r 16 --lora-alpha 32 --epochs 3
```

Önemli seçenekler: `--lr` (varsayılan tam ayarda `2e-5`, LoRA'da `2e-4`),
`--warmup-steps`, `--max-steps`, `--grad-clip`, `--lora-targets`
(varsayılan `q_proj,k_proj,v_proj,out_proj`; FFN için `gate_proj,up_proj,down_proj`
eklenebilir), `--system` (kayıtta sistem mesajı yoksa eklenecek varsayılan),
`--save-every`. En iyi değerlendirme kaybı `*_best.pt` olarak ayrıca kaydedilir.

### 4.2 Anayasaya dayalı öz-düzeltme

```bash
python alignment/constitutional.py --checkpoint checkpoints/toprak_sft.pt \
    --prompts alignment/examples/prompts.txt \
    --output data/cai_pairs.jsonl --sft-output data/cai_sft.jsonl \
    --trajectories data/cai_trajectories.jsonl --num-principles 2
```

Her prompt için: ilk cevap → rastgele `N` ilke → her ilke için eleştiri ve
düzeltme. Son düzeltme `chosen`, ilk cevap `rejected` olur; değişmeyen cevaplar
atlanır (`--keep-unchanged` ile tutulur). `--trajectories` tüm ara adımları
inceleme için yazar — **çıktıları mutlaka gözle örnekleyin.**

Python'dan harici bir öğretmenle kullanım (lisansı ve kullanım koşulları
uygun bir model olmalı):

```python
import random
from alignment import ConstitutionalReviser, build_preference_pairs, load_principles

def teacher(messages):          # messages: [{"role", "content"}, ...] → str
    return my_client.chat(messages)

reviser = ConstitutionalReviser(teacher, load_principles(), rng=random.Random(0), num_principles=2)
pairs = build_preference_pairs(prompts, reviser)
```

### 4.3 DPO / ORPO

```bash
# DPO — referans model = başlangıç checkpoint'i (dondurulmuş kopya)
python training/dpo.py --base-checkpoint checkpoints/toprak_sft.pt \
    --data data/cai_pairs.jsonl --output checkpoints/toprak_dpo.pt --beta 0.1

# DPO + LoRA — referans, LoRA'sı kapalı temel modeldir (ikinci kopya yok, bellek dostu)
python training/dpo.py --base-checkpoint checkpoints/toprak_sft.pt \
    --data data/cai_pairs.jsonl --output checkpoints/toprak_dpo.pt --lora-r 16

# ORPO — referanssız; SFT kaybı + olasılık oranı terimi
python training/dpo.py --method orpo --base-checkpoint checkpoints/toprak_sft.pt \
    --data data/cai_pairs.jsonl --output checkpoints/toprak_orpo.pt --orpo-lambda 0.1
```

- `--beta`: Referanstan uzaklaşmaya karşı fren (0.05–0.5). Küçük β daha agresif.
- `--label-smoothing ε`: Tercih etiketleri gürültülüyse (ör. otomatik üretilmiş
  çiftler) 0.05–0.1 deneyin (cDPO).
- `--ref-checkpoint`: Referansı başka bir checkpoint'ten almak için.
- Loglanan **marj** (β·Δchosen − β·Δrejected) ve **doğruluk** (marjı pozitif
  çiftlerin oranı) artmalı; kayıp `log 2 ≈ 0.693`'ten başlar.
- ORPO, SFT yapılmamış bir modelde de kullanılabilir; DPO önce SFT ister.

### 4.4 Sohbet

```bash
python inference/chat.py --checkpoint checkpoints/toprak_dpo.pt \
    --system "Sen Toprak'sın; Türkçe konuşan, dürüst ve yardımsever bir yapay zekâ asistanısın."
```

Komutlar: `çık`/`exit`, `ayar` (sıcaklık, top-k, top-p, maks token),
`temizle` (geçmişi sil). Geçmiş artık bir mesaj listesidir ve her turda
`max pozisyon − max_new_tokens` bütçesine sığacak şekilde en eski turlardan
kırpılır (sistem mesajı ve son kullanıcı mesajı korunur). Böylece eski
sürümdeki "uzun sohbette RoPE tablosu aşıldı" çökmesi giderildi.
Anayasanın kısa bir özetini sistem mesajı olarak vermek için
`alignment/constitution.md` sonundaki öneriyi kullanabilirsiniz.

## 5. Apple Silicon'da LoRA

- MPS ve CPU'da eğitim **float32** yapılır (MPS'te bf16 autocast kararsız
  olabilir); CUDA'da otomatik **bf16 autocast** kullanılır.
- LoRA yalnız uyarlayıcı parametrelerini eğitir: r=16 ile attention
  projeksiyonlarında parametrelerin yalnız ~%1'i eğitilir; AdamW durumu buna
  göre küçülür. Kaba bellek tahmini (float32, ağırlık + aktivasyon, 512 token):

| Model | Tam ince ayar | LoRA r=16 |
|---|---|---|
| ~80M (small) | ~2 GB | ~1 GB |
| ~342M (large) | ~6–8 GB | ~3 GB |
| ~941M (xl) | ~16+ GB | ~6–8 GB |

- Bellek yetmezse önce `--batch-size`'ı düşürüp `--grad-accum`'ı artırın,
  sonra `--max-len`'i kısaltın.
- DPO'da LoRA ayrıca referans modelin ikinci kopyasını gereksiz kılar.
- Kayıtta LoRA katlanır; çıkan checkpoint tam bir modeldir.

## 6. Toprak Anayasası

`alignment/constitution.md` (insan için) ve `alignment/constitution.json`
(makine için) aynı 15 ilkeyi içerir; testler ikisinin eşit olduğunu denetler.
Başlıklar: doğruluk ve kaynak, bilmediğini söyleme, zararlı içerikten kaçınma,
tarafsızlık ve siyasi denge, saygılı dil ve uygun hitap (siz/sen), Türkçe dil
kalitesi, mahremiyet ve KVKK, hukuki/tıbbi/finansal konularda uzmana
yönlendirme, çocuk güvenliği, kültürel duyarlılık, şeffaflık (yapay zekâ
olduğunu söyleme), gerçek yardımseverlik, açıklık ve öz anlatım,
manipülasyondan kaçınma, kriz ve acil durumlarda 112'ye yönlendirme.

Her ilkenin bir **eleştiri sorusu** ve bir **düzeltme talimatı** vardır.
İlke eklerken iki dosyayı birlikte güncelleyin ve `python -m pytest -q
tests/test_constitutional.py` çalıştırın.

## 7. Değerlendirme

Hizalamanın işe yarayıp yaramadığını ölçmeden ilerlemeyin:

```bash
# Önce: SFT/DPO öncesi temel model
python evaluation/evaluate_suite.py --checkpoint checkpoints/toprak_best.pt --output reports/base.json
# Sonra: hizalanmış model, temel rapora göre
python evaluation/evaluate_suite.py --checkpoint checkpoints/toprak_dpo.pt \
    --output reports/dpo.json --baseline reports/base.json --max-regression 0.05
```

- **Güvenlik kategorisi** (`evaluation/benchmarks/safety.jsonl`) hizalamanın
  ana göstergesidir: zararlı isteklerde ret oranı artmalı, zararsız isteklerde
  (ör. "yapıcı bir mesaj öner") gereksiz ret artmamalı.
- Bilgi, okuma ve Türkçe dilbilim kategorilerinde belirgin düşüş
  ("hizalama vergisi") varsa β'yı büyütün, öğrenme oranını ya da adım sayısını
  azaltın.
- Otomatik metrikler yetmez: her aşamadan sonra 30–50 cevabı elle okuyun.

## 8. Sınırlamalar ve dürüst notlar

- **Küçük modeller iyi veri ister.** 80M–340M parametreli bir model, ne kadar
  iyi hizalanırsa hizalansın, bilgi ve akıl yürütmede sınırlı kalır; SFT
  biçimi öğretir, bilgi eklemez. Hizalama kalitesi neredeyse tamamen veri
  kalitesine bağlıdır.
- **Öz-düzeltme, modelin kendisi kadar iyidir.** Zayıf bir SFT modeli anlamlı
  eleştiri ve düzeltme üretemez; bu durumda üretilen çiftler gürültüdür. Küçük
  modellerde, kullanım koşulları buna izin veren bir öğretmen modelle
  (`generate_fn`) veri üretip sonuçları elle süzmek daha verimlidir.
- **Örnek dosyalar yalnızca biçim örneğidir**; eğitim verisi değildir.
- **Lisans.** Hazır talimat ve tercih veri setlerinin (ve öğretmen model
  çıktılarının) lisansını ve kullanım koşullarını tek tek kontrol edin;
  birçoğu ticari kullanımı ya da başka modellerin eğitimini kısıtlar. Kaynak,
  lisans ve kararlarınızı [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md)'deki
  kayıt düzeniyle belgeleyin.
- **Kişisel veri.** Sohbet kayıtlarından veri üretiyorsanız KVKK kapsamında
  açık rıza ve anonimleştirme gerekir.
- DPO aşırı uygulanırsa model uzun, kaçamak ya da aşırı temkinli cevaplara
  kayabilir; değerlendirme ve elle okuma ile izleyin.
- Mevcut 32K tokenizer'da şablon metin işaretleyicileriyle çalışır; özel
  tokenlı yeni tokenizer daha kısa ve daha sağlam diziler verir.

## 9. Testler

```bash
python -m pytest -q tests/test_sft.py tests/test_dpo.py tests/test_constitutional.py tests/test_chat_cli.py
```

Testler CPU'da birkaç saniyede biter; DPO kaybının elle hesaplanmış değerlerini,
politika = referans iken kaybın `log 2` olduğunu, birkaç optimizasyon adımında
tercih marjının arttığını, LoRA katlamanın modeli değiştirmediğini, anayasa
dosyalarının eşitliğini ve uzun sohbette geçmiş kırpmanın RoPE sınırını
aşmadığını doğrular.
