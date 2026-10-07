# 📱 Telefonda Çalışan Toprak — Cihaz Üstü (Edge) Rehberi

Bu belge, Toprak modelini **bulut olmadan**, doğrudan dizüstü bilgisayarda,
Mac'te ve telefonda çalıştırmak için uçtan uca yolu anlatır:

```
 Büyük öğretmen (Large/XL)
        │  training/distill.py      (bilgi damıtma)
        ▼
 Küçük öğrenci (Small/Medium)
        │  export/quantize.py       (int8/int4 boyut-kalite ölçümü, PyTorch)
        │  export/hf_llama.py       (HF LlamaForCausalLM klasörü)
        ▼
 HF Llama klasörü ──► llama.cpp: GGUF (Q4_K_M) ──► llama-cli / Ollama / mobil uygulamalar
                  └─► MLX: mlx_lm.convert -q   ──► Mac / iPhone (MLX Swift)
```

> **Durum özeti:** Damıtma, saf PyTorch kuantizasyon ve HF Llama dışa aktarımı
> bu depoda **testlerle doğrulanmıştır**. GGUF / Ollama / MLX / iPhone adımları
> bu ortamda **çalıştırılmamıştır**; standart araçların belgelenmiş
> kullanımına dayanan **talimatlardır** (ayrıntı: [Doğrulananlar](#doğrulananlar-ve-yalnız-belgelenenler)).

---

## Neden cihaz üstü?

- **Mahremiyet ve KVKK:** Metin cihazdan hiç çıkmaz. Kişisel veri işleyen
  kurumlar (sağlık, hukuk, kamu, eğitim) için 6698 sayılı KVKK kapsamında yurt
  dışına veri aktarımı ve üçüncü taraf işleyici sorunları baştan ortadan kalkar.
- **Veri egemenliği:** Model ağırlıkları, tokenizer ve çıkarım tamamen sizin
  donanımınızda; hiçbir yabancı API'ye, hesap anahtarına veya kullanım
  politikasına bağımlılık yoktur.
- **Çevrimdışı çalışma:** Sahada, uçakta, internetin kısıtlı olduğu yerlerde
  ve afet senaryolarında da çalışır.
- **Maliyet ve gecikme:** Sunucu ücreti yok; ağ gidiş-dönüşü olmadığı için
  ilk token gecikmesi düşüktür.

---

## 1. Damıtma: Large → Small

Küçük model, büyük modelin olasılık dağılımını taklit ederek aynı veriyle
sıfırdan eğitilmesinden daha iyi sonuç verir.

```
L = α · T² · KL(p_öğretmen^T ‖ p_öğrenci^T) + (1 − α) · CE(öğrenci, etiket)
```

- `T` (sıcaklık): yumuşak hedefleri düzleştirir; `T²` gradyan ölçeğini korur.
- `α`: KL ağırlığı (0 = saf CE, 1 = saf damıtma). Pad tokenları her iki
  terimden de çıkarılır.
- `--top-k`: öğretmenin yalnız en yüksek k logit'i tutulur ve k üzerinde
  yeniden normalize edilir (öğretmen çıktısını saklarken/aktarırken bellek:
  32.000 yerine 2k değer). Öğrenci log-olasılıkları tam sözlükten alınır.
- Öğretmen ve öğrenci **aynı tokenizer'ı** kullanmalı (sözlük uyuşmazlığında
  hata verilir).

```bash
# Pre-tokenize shard'larla (önerilen)
python training/distill.py \
    --teacher checkpoints/toprak_large_best.pt \
    --student-size small \
    --data-dir data_cache/bin --bin-mode \
    --max-steps 20000 --batch-size 8 --grad-accum 4 \
    --temperature 2.0 --alpha 0.5 --top-k 64 \
    --checkpoint-dir checkpoints/distill

# JSONL dizini ile
python training/distill.py --teacher checkpoints/toprak_large_best.pt \
    --student-size small --data-dir data_cache/clean/train
```

Çıktı (`checkpoints/distill/toprak_distill_last.pt`) standart Toprak
checkpoint biçimindedir (`model_state_dict`, `config`, `global_step`) — yani
`inference/generate.py`, `export/quantize.py` ve `export/hf_llama.py` ile
doğrudan kullanılabilir.

Python API:

```python
from training.distill import distillation_loss, sparsify_teacher_logits, DistillTrainer
loss, parts = distillation_loss(student_logits, teacher_logits, labels,
                                temperature=2.0, alpha=0.5, top_k=64)
```

---

## 2. Kuantizasyon (saf PyTorch, boyut/kalite ölçümü)

```bash
python -m export.quantize --checkpoint checkpoints/distill/toprak_distill_last.pt \
    --bits 4 --group-size 64 --out checkpoints/toprak_small_int4.pt \
    --eval-text ornek_metin.txt        # opsiyonel kalite raporu
```

- **int8:** çıkış kanalı başına simetrik ölçek.
- **int4:** giriş boyutunda grup başına ölçek (32/64/128), iki nibble bir
  bayta paketlenir; `--zero-point` ile asimetrik.
- Yalnız transformer bloklarındaki lineer katmanlar (attention + FFN)
  kuantize edilir. **Bağlı embedding/lm_head float kalır ve bağlı kalır.**
  MoE yönlendiricisi (`router`) varsayılan olarak atlanır.

```python
from export.quantize import (quantize_model, load_quantized, save_quantized,
                             model_size_bytes, quantization_error_report)
q = quantize_model(copy.deepcopy(model), bits=4, group_size=64)
print(quantization_error_report(model, q, sample_ids))   # MSE, top-1 uyum, PPL farkı
save_quantized(q, "toprak_int4.pt"); q2 = load_quantized("toprak_int4.pt")
q2.generate(ids, max_new_tokens=50)                       # KV cache ile çalışır
```

> ⚠️ Bu Python int4 çekirdeği **hız için değil**, boyut ve kalite kontrolü
> içindir: her ileri geçişte ağırlıklar float'a açılır, bu yüzden fp32'den
> yavaştır. Hızlı int4 çıkarımı için aşağıdaki GGUF veya MLX yolunu kullanın.
> Buradaki ölçümler, GGUF/MLX'te hangi bit/grup ayarının kabul edilebilir
> kalite vereceğine karar vermek için kullanışlıdır.

---

## 3. HuggingFace Llama biçimine dışa aktarım

Toprak mimarisi (RMSNorm, SwiGLU, RoPE, GQA, bağlı embedding, bias yok)
Llama ile birebir eşlenir. Dönüştürücü:

```bash
python -m export.hf_llama \
    --checkpoint checkpoints/distill/toprak_distill_last.pt \
    --out exports/toprak-small-hf --dtype float16
```

Çıktı klasörü:

| Dosya | İçerik |
|---|---|
| `config.json` | `LlamaForCausalLM`, `tie_word_embeddings=true`, `num_key_value_heads`, `rms_norm_eps`, `rope_theta`, `rope_scaling` (+ transformers 5 için `rope_parameters`), `hidden_act=silu`, bias'sız |
| `model.safetensors` | Bağlı ağırlık tek kez (`model.embed_tokens.weight`), `lm_head` yazılmaz |
| `generation_config.json` | bos/eos/pad, `temperature=0.8`, `top_k=50`, `top_p=0.9`, `repetition_penalty=1.3` (generate.py varsayılanları) |
| `tokenizer.model` | SentencePiece (llama.cpp bunu okur) |
| `tokenizer.json` + `tokenizer_config.json` | `AutoTokenizer` için (transformers, MLX) |

Eşleme ayrıntıları:

- **RoPE düzeni:** Toprak ardışık çiftleri döndürür (`view_as_complex`),
  HF Llama `rotate_half` (yarım-yarım) kullanır. `q_proj` ve `k_proj`
  satırları her head içinde permüte edilir (Meta→HF permütasyonu); skorlar
  değişmez.
- **RMSNorm:** `weight · x / rms(x)` — HF LlamaRMSNorm ile aynı.
- **rope_scaling:** `linear` → `{"rope_type": "linear"}`; Toprak `ntk`
  (statik taban değişimi) → `rope_theta = θ · factor^(d/(d−2))`, ölçekleme yok;
  `yarn` → `{"rope_type": "yarn", factor, original_max_position_embeddings,
  beta_fast, beta_slow, attention_factor}`. YaRN ters frekansları
  transformers formülüyle karşılaştırılır; Toprak düzeltme aralığını
  `d/2−1` ile, transformers `d−1` ile kırptığı için aşırı uç ayarlarda fark
  olursa uyarı verilir.
- **Atılanlar:** MTP başlıkları (uyarı ile) ve morfoloji başlığı.

Python'dan kullanım:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("exports/toprak-small-hf")
model = AutoModelForCausalLM.from_pretrained("exports/toprak-small-hf")
ids = tok("Türkiye'nin başkenti", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=40)[0], skip_special_tokens=True))
```

### Tokenizer eşliği

`tokenizer.json`, SentencePiece modelinden doğrudan kurulur (BPE + byte
fallback + NFKC `precompiled_charsmap` + SentencePiece boşluk kuralları).
`AutoTokenizer(text)` → `ToprakTokenizer.encode(text, add_bos=True, add_eos=False)`
ile **aynı ID'leri** üretir. Doğrulama: test cümleleri (Türkçe harfler,
NFKC ligatür/tam genişlik, baştaki/sondaki/çoklu boşluk, satır sonu, sekme,
rakamlar, emoji ve CJK byte fallback), 3.000 rastgele parça dizisinin
tamamı ve README/GUIDE'ın 1.138 satırından 1.137'si birebir eşleşir (tek
farklı satır, aşağıdaki 1. maddedeki `<cls>`/`<mask>` sembollerini içerir).

Bilinen, kesin farklar:

1. **Kullanıcı tanımlı semboller** (`<sep>`, `<cls>`, `<mask>` ve yeni
   tokenizer'larda `<|sistem|>` gibi sohbet tokenları) metnin içinde yazılırsa:
   SentencePiece sembolün yanındaki boşluğu ayrı bir `▁` tokenı yapar
   (`"a <mask>"` → `▁a ▁ <mask>`), HF tarafında bu boşluk düşer
   (`▁a <mask>`); sembolden hemen sonra boşluksuz gelen metne ise HF `▁`
   ekler. Toprak'ın sohbet şablonu bu tokenları metinden değil ID'den eklediği
   için (`model/chat_template.py`) normal kullanım etkilenmez.
2. `<s>`, `</s>`, `<pad>`, `<unk>` düz metin olarak yazılırsa HF bunları özel
   token olarak eşler; SentencePiece harf harf kodlar.
3. HF varsayılanı **sona EOS eklemez** (Llama geleneği); Toprak
   `encode()` varsayılanı ekler.

---

## 4. GGUF (llama.cpp) → llama-cli / Ollama

> Bu ortamda çalıştırılmadı — standart llama.cpp akışıdır.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
pip install -r requirements.txt
cmake -B build && cmake --build build --config Release -j

# HF klasörü → GGUF (f16)
python convert_hf_to_gguf.py ../exports/toprak-small-hf \
    --outfile ../exports/toprak-small-f16.gguf --outtype f16

# 4-bit kuantizasyon (önerilen: Q4_K_M; kalite önemliyse Q5_K_M / Q8_0)
./build/bin/llama-quantize ../exports/toprak-small-f16.gguf \
    ../exports/toprak-small-Q4_K_M.gguf Q4_K_M

# Çalıştır
./build/bin/llama-cli -m ../exports/toprak-small-Q4_K_M.gguf \
    -p "Türkiye'nin başkenti" -n 128 --temp 0.8 --top-k 50 --top-p 0.9 --repeat-penalty 1.3
```

Notlar:
- llama.cpp, Llama mimarisinde `tokenizer.model` (SentencePiece) dosyasını
  okur. llama.cpp'nin SPM tokenizer'ı NFKC `precompiled_charsmap`
  normalizasyonunu ve çoklu boşluk sıkıştırmasını uygulamayabilir; NFKC dışı
  girdilerde (ligatür, tam genişlik karakterler, ardışık boşluklar) ID'ler
  Python tarafından farklı olabilir (doğrulanmadı).
- YaRN'lı modellerde `attention_factor` alanının llama.cpp tarafından nasıl
  yorumlandığı doğrulanmadı; uzun bağlam kalitesini ayrıca ölçün.

### Ollama

`Modelfile`:

```
FROM ./toprak-small-Q4_K_M.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.8
PARAMETER top_k 50
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.3
PARAMETER stop "</s>"
```

```bash
ollama create toprak-small -f Modelfile
ollama run toprak-small "Türkiye'nin başkenti"
```

Sohbet için ince ayarlı bir model ve sohbet tokenlarını içeren tokenizer
kullanıyorsanız şablonu `model/chat_template.py` biçimine uyarlayın
(`<s><|kullanıcı|>…<|son|><|toprak|>`) ve `PARAMETER stop "<|son|>"` ekleyin.

---

## 5. MLX (Apple Silicon Mac) ve iPhone

> Bu ortamda çalıştırılmadı (Linux) — standart `mlx-lm` akışıdır.

```bash
pip install mlx-lm
mlx_lm.convert --hf-path exports/toprak-small-hf -q --q-bits 4 --q-group-size 64 \
    --mlx-path exports/toprak-small-mlx-q4
mlx_lm.generate --model exports/toprak-small-mlx-q4 \
    --prompt "Türkiye'nin başkenti" --max-tokens 128 --temp 0.8
```

**iPhone / iPad:**
- **MLX Swift:** `mlx-swift-examples` deposundaki LLM örnek uygulaması (LLMEval)
  yerel bir model klasörünü yükleyebilir; `exports/toprak-small-mlx-q4`
  klasörünü uygulama paketine ekleyip model yapılandırmasında bu yolu
  gösterin. Llama mimarisi MLX Swift'te desteklenir.
- **llama.cpp tabanlı uygulamalar:** GGUF dosyası içe aktarabilen iOS/Android
  uygulamaları (llama.cpp'yi gömen açık kaynak uygulamalar) Q4_K_M dosyasını
  doğrudan çalıştırabilir. Kendi uygulamanız için llama.cpp'nin
  `examples/llama.swiftui` (iOS) ve `examples/llama.android` örneklerine bakın.

Telefon için öneri: **Small (Q4_K_M ≈ 46 MiB)** veya **Medium (≈ 72 MiB)**
rahatlıkla sığar; **Large (≈ 0,2 GiB)** modern telefonlarda çalışır; **XL
(≈ 0,5 GiB)** üst segment cihazlar içindir.

---

## Bellek / boyut tablosu

Değerler `model/config.py` preset'lerinden kodla hesaplandı (vocab 32.000,
bağlı embedding bir kez sayılır, MTP/morph başlıkları hariç). Small için
analitik sayım, gerçek `ToprakLM` örneğinin `model_size_bytes` ölçümüyle
birebir doğrulandı (fp16 153,1 MiB; int8 96,3 MiB; int4-g64 69,5 MiB).

| Preset | Parametre | Embedding | Lineer (blok) | fp16 | int8 (Toprak) | int4 g64 (Toprak) | GGUF Q4_K_M (≈) | KV cache fp16 @ bağlam |
|---|---|---|---|---|---|---|---|---|
| Small | 80,3M | 20,5M | 59,8M | 153 MiB | 96 MiB | 69 MiB | ~46 MiB | 3,5 MiB @ 512 |
| Medium | 125,3M | 24,6M | 100,7M | 239 MiB | 143 MiB | 98 MiB | ~72 MiB | 8,0 MiB @ 512 |
| Large | 341,6M | 32,8M | 308,7M | 651 MiB | 358 MiB | 219 MiB | ~197 MiB | 56 MiB @ 2048 |
| XL | 941,1M | 49,2M | 891,8M | 1.795 MiB | 945 MiB | 546 MiB | ~544 MiB | 108 MiB @ 2048 |

- **int8 / int4 (Toprak):** `export/quantize.py` biçimi — blok lineerleri
  8 bit (+ satır başına fp16 ölçek) veya 4 bit (+ 64 ağırlıkta bir fp16 ölçek
  = 4,25 bit/ağırlık); embedding fp16 bırakılır. Küçük modellerde embedding
  payı büyük olduğundan (Small'da %25) toplam küçülme sınırlıdır.
- **GGUF Q4_K_M:** tüm tensörler dahil ortalama ~4,85 bit/ağırlık varsayımıyla
  **tahmin**; gerçek dosya boyutu llama.cpp sürümüne göre birkaç MiB değişir.
- **KV cache:** `2 × katman × kv_head × head_dim × bağlam × 2 bayt` (GQA
  sayesinde küçük). Çalışma belleği ≈ ağırlık + KV cache + birkaç on MiB
  çalışma tamponu.

---

## Doğrulananlar ve yalnız belgelenenler

**Testlerle doğrulanan** (`python -m pytest -q tests/test_export_hf.py tests/test_quantize.py tests/test_distill.py`):

| Konu | Test | Sonuç |
|---|---|---|
| HF Llama logit eşliği | küçük rastgele ToprakLM → `LlamaForCausalLM` (float32) | maks. mutlak fark < 1e-4 (gözlenen ~3e-5, logit ölçeği ~16): RoPE yok, linear, ntk, yarn |
| Açgözlü üretim | Toprak (KV cache) vs `hf.generate(do_sample=False)` | 10 token birebir aynı |
| fp16 dışa aktarım | `--dtype float16` CLI, safetensors'ta `lm_head` yok | logit farkı < 0.1 |
| MoE | `num_experts > 0` | açık `NotImplementedError` (Mixtral önerisiyle) |
| MTP | `num_mtp_heads > 0` | başlıklar atılır, `UserWarning`, logitler yine eşleşir |
| Tokenizer | `AutoTokenizer.from_pretrained(out)` | 6 zor Türkçe cümlede `ToprakTokenizer` ile aynı ID ve decode |
| int4 paketleme | pack/unpack | birebir geri dönüş |
| int8 | görev üzerinde eğitilmiş küçük model | top-1 uyum 1,000, PPL farkı −%0,03 |
| int4 | aynı model, g32/g64, simetrik/sıfır noktalı | top-1 uyum 0,98–0,996, PPL farkı %0,5–1,8 |
| Boyut | blok lineerleri fp32 → int4 g32 | ×7,1 küçülme (int4 g64: ×7,5) |
| Kaydet/yükle | `save_quantized` → `load_quantized` | logitler birebir aynı, embedding bağlı |
| KV cache | kuantize model `generate` | cache'li üretim = cache'siz açgözlü üretim |
| Damıtma kaybı | KL=0 (eşit), T² ölçeği, α=0 → CE, pad maskesi, top-k | hepsi geçer |
| Damıtma eğitimi | küçük öğretmen → küçük öğrenci, 40 adım | KL 3,57 → 0,76 (görülmemiş veride 0,81) |

**Yalnız belgelenen (bu ortamda çalıştırılmadı):** llama.cpp GGUF dönüşümü ve
`llama-quantize`, `llama-cli`, Ollama, MLX (`mlx_lm.convert/generate`),
MLX Swift ve mobil uygulamalar. Bunlar standart Llama klasörü bekleyen
araçlardır ve dışa aktarılan klasör transformers ile birebir doğrulandığı için
çalışması beklenir; yine de ilk kullanımda kısa bir metinle Python çıktısıyla
karşılaştırma yapmanız önerilir.

---

## Sınırlamalar

- **MoE desteklenmiyor:** `num_experts > 0` modeller Llama'ya çevrilemez.
  Gelecek çalışma: uzmanları Mixtral eşlemesine aktarmak
  (`block_sparse_moe.experts.{i}.w1/w2/w3` + `gate`); morfolojik yönlendirme
  ipucunun Mixtral'de karşılığı olmadığı için davranış birebir korunamaz.
- **MTP başlıkları atılır:** Llama biçiminde karşılığı yoktur; MTP tabanlı
  kendi kendine spekülatif çözümleme yalnız PyTorch tarafında kullanılabilir
  (llama.cpp/MLX'te ayrı bir taslak model ile spekülatif çözümleme mümkündür).
- **Python int4 çekirdeği hız için değildir:** boyut ve kalite ölçümü
  içindir; dequantize-on-the-fly nedeniyle fp32'den yavaştır.
- **Tokenizer:** kullanıcı tanımlı sembollerin yanındaki boşluk ve düz metinde
  yazılmış kontrol tokenları için yukarıdaki farklar; llama.cpp'nin NFKC
  normalizasyonu doğrulanmadı.
- **YaRN:** transformers ile frekanslar ve `attention_factor` eşleşir
  (testte doğrulandı); llama.cpp/MLX'teki yorumlanışı doğrulanmadı.
- **Kuantize checkpoint'ler HF'ye çevrilmez:** HF/GGUF/MLX dönüşümünü her
  zaman tam hassasiyetli checkpoint'ten yapın; kuantizasyonu hedef araçta
  (`llama-quantize`, `mlx_lm.convert -q`) uygulayın.
