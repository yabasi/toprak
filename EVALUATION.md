# Toprak Değerlendirme Paketi

Toprak Eval v1, checkpointleri yalnız perplexity ile değil Türkçe yetenek ve
güvenlik boyutlarıyla karşılaştırmak için deterministik bir değerlendirme
paketidir.

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
