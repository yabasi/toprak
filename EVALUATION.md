# Toprak Değerlendirme Paketi

Toprak Eval v1, checkpointleri yalnız perplexity ile değil Türkçe yetenek ve
güvenlik boyutlarıyla karşılaştırmak için deterministik bir değerlendirme
paketidir.

Tokenizerın model eğitiminden önceki kapsam ve verim analizi ayrı olarak
[TOKENIZER_ANALYSIS.md](TOKENIZER_ANALYSIS.md) protokolüyle yapılır.

## Kapsam

Sürümlü seed set şu kategorileri içerir:

- Türkçe morfoloji, ünlü uyumu, ünsüz benzeşmesi ve dilbilgisi;
- okuduğunu anlama;
- genel kültür;
- mantık ve matematik;
- bağlamın başı, ortası ve sonundaki bilgiyi bulma;
- toksik çıktı için anahtar sözcük taraması;
- sentetik canary continuation ile ezberleme sızıntısı kontrolü.

Bu küçük seed set model seçimi için bir regresyon göstergesidir; kapsamlı
akademik benchmark veya güvenlik sertifikası değildir. Yeni ve lisansı uygun
benchmarklar aynı JSONL şemasında eklenebilir.

## Çalıştırma

```bash
python evaluation/evaluate_suite.py \
  --checkpoint checkpoints/toprak_last.pt \
  --tokenizer toprak_tokenizer.model \
  --output evaluation/reports/toprak_last.json
```

Perplexity'yi aynı rapora eklemek için:

```bash
python evaluation/evaluate_suite.py \
  --checkpoint checkpoints/toprak_last.pt \
  --perplexity-data data_cache/clean/eval
```

Manifest tabanlı pretokenized eval shard'ları için `--perplexity-bin-mode`
eklenir. Eval DataLoader son eksik batch'i atmaz; küçük eval setleri de rapora
dahil edilir.

İki checkpoint raporunu karşılaştırmak ve kabul eşiği uygulamak için:

```bash
python evaluation/evaluate_suite.py \
  --checkpoint checkpoints/toprak_candidate.pt \
  --baseline evaluation/reports/toprak_last.json \
  --max-regression 0.02 \
  --fail-below 0.40 \
  --output evaluation/reports/toprak_candidate.json
```

`--max-regression 0.02`, macro skor 0.02'den fazla düşerse komutu hata koduyla
sonlandırır. Rapor checkpoint, tokenizer ve her benchmark dosyasının SHA-256
değerini içerir. Suite sürümü, tokenizer hash'i veya benchmark hash'leri
uyuşmayan iki rapor karşılaştırma sırasında reddedilir.

## Ölçüm yöntemi

Multiple-choice ve minimal-pair örnekleri continuation log-olasılığıyla
puanlanır. Varsayılan sıralama token başına ortalama log-olasılığı kullanır;
örnekte `"length_normalize": false` verilirse toplam log-olasılığı kullanılır.

Üretim, güvenlik ve ezberleme görevleri sampling kullanmayan greedy decoding ile
çalışır. Bu sayede aynı checkpoint ve cihazdaki rapor tekrarlanabilir olur.

Uzun bağlam örnekleri dolgu metnini modelin `max_seq_len` değerine göre dinamik
olarak genişletir ve hedef bilgiyi bağlamın farklı konumlarına yerleştirir.

## JSONL şemaları

Ortak alanlar:

```json
{"id": "benzersiz-id", "type": "pairwise", "category": "morphology"}
```

Desteklenen görev tipleri:

- `multiple_choice`: `prompt`, `choices`, `answer`;
- `pairwise`: `prompt`, `chosen`, `rejected`;
- `generation`: `prompt`, `references`, isteğe bağlı `match`;
- `long_context`: `filler`, `needle`, `question`, `choices`, `answer`;
- `safety`: `prompt`, `unsafe_keywords`, isteğe bağlı `refusal_keywords`;
- `memorization`: `prompt`, `reference`, isteğe bağlı `leak_threshold`.

Tüm ID'ler benchmark dizini genelinde benzersiz olmalıdır. Loader eksik alanları,
geçersiz seçenek indekslerini ve desteklenmeyen görev tiplerini çalıştırmadan
önce reddeder.

## Standart benchmarklar (lm-evaluation-harness)

Seed set regresyon göstergesidir; Toprak'ı diğer modellerle kıyaslamak için
EleutherAI [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
entegrasyonu kullanılır. `evaluation/lm_eval_adapter.py`, `ToprakLM`'i lm-eval
`LM` arayüzüne (`loglikelihood`, `loglikelihood_rolling`, `generate_until`)
bağlar ve `toprak` adıyla kaydeder.

```bash
python evaluation/run_lm_eval.py \
  --checkpoint checkpoints/toprak_last.pt \
  --preset tr_core \
  --output evaluation/reports/lm_eval_toprak_last.json
```

Hazır görev grupları (`--list-presets` ile listelenir):

| Preset | Görevler | Not |
|---|---|---|
| `tr_core` | `xcopa_tr`, `xnli_tr`, `belebele_tur_Latn`, `global_piqa_nonparallel_cloze_tur_latn` | Log-olasılık tabanlı; küçük modellerde de sinyal verir |
| `tr_knowledge` | `turkishmmlu`, `global_mmlu_full_tr`, `include_base_44_turkish` | Bilgi yoğun; <1B modellerde şans seviyesine yakın olabilir |
| `tr_generative` | `xquad_tr` | Açgözlü üretim + F1 / exact match |
| `tr_all` | Hepsi | |

Ek görevler `--tasks gorev1,gorev2` ile eklenir. Hızlı duman testi için
`--limit 20` kullanılabilir; bu durumda skorlar karşılaştırma için geçerli
değildir. Konsol tablosu her görevin şans seviyesini de gösterir: şansın
standart hatanın üzerinde olmayan skorlar anlamlı kabul edilmemelidir.

Rapor; checkpoint ve tokenizer SHA-256 değerlerini, git commit'ini, lm-eval
sürümünü, görev sürümlerini, few-shot ayarlarını ve seed'i içerir. Farklı
lm-eval veya görev sürümleriyle üretilmiş skorlar doğrudan kıyaslanmamalıdır.

Notlar:

- Bağlam penceresi varsayılan olarak `config.max_seq_len`'dir; daha uzun
  girdiler soldan kırpılır. Belebele pasajları 512 tokenlık Small/Medium
  modellerde kırpılabilir.
- CUDA'da `--dtype bfloat16` ile karışık hassasiyet kullanılabilir; MPS ve CPU
  float32 çalışır.
- Belebele, FLORES-200 üzerinden Wikipedia kaynaklıdır. Türkçe Wikipedia
  eğitim verisinde olduğu için bu görevde contamination riski vardır; skorlar
  bu notla birlikte raporlanmalıdır.

### Baseline karşılaştırması

Toprak skorları ancak aynı görev, sürüm ve ayarlarla ölçülmüş başka
modellerle yan yana anlam kazanır. Baseline listesi
[configs/lm_eval_baselines.json](configs/lm_eval_baselines.json) dosyasındadır:

- `small` (≤1B, erişimi açık): YTÜ Cosmos Turkish GPT-2 (124M / 355M / 774M),
  XGLM-564M, Qwen2.5-0.5B, Qwen3-0.6B-Base;
- `large` (CUDA önerilir): TURNA (1.1B, encoder-decoder), Kumru-2B-Base,
  Qwen3-1.7B-Base;
- `gated` (HF lisans onayı + `HF_TOKEN`): Llama-3.2-1B, Gemma-3-1B.

```bash
# Baseline raporları (var olanlar atlanır, yarıda kalan koşu sürdürülebilir)
python scripts/run_baselines.py --tier small

# Toprak raporu
python evaluation/run_lm_eval.py \
  --checkpoint checkpoints/toprak_last.pt --preset tr_core \
  --output evaluation/reports/lm_eval_toprak_last.json

# Karşılaştırma tablosu
python evaluation/compare_lm_eval.py \
  evaluation/reports/lm_eval_toprak_last.json \
  evaluation/reports/baselines/*.json \
  --output BASELINES.md
```

`run_lm_eval.py --hf-model <id>` herhangi bir HF modelini aynı rapor şemasıyla
ölçer; model revizyon SHA'sı rapora sabitlenir. Karşılaştırma scripti
`--limit` ile üretilmiş raporları reddeder ve lm-eval sürümü, few-shot, görev
sürümü veya bağlam uzunluğu farklarını uyarı olarak listeler. Toprak Small/Medium
(512 token) ile adil kıyas için gerekirse `run_baselines.py --max-length 512`
kullanılır.

## Veri contamination

`evaluation/benchmarks/` dizini eğitim verisi hazırlanırken contamination
referansı olarak verilmelidir:

```bash
python data/cleaner.py \
  --input data_raw \
  --output data_clean \
  --benchmark-path evaluation/benchmarks \
  --contamination-action reject
```

Benchmark değiştirilirse eski ve yeni raporlar doğrudan kıyaslanmamalıdır;
rapordaki benchmark SHA-256 değerleri bu durumu görünür kılar.
