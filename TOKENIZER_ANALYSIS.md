# Tokenizer Analizi

Toprak tokenizerları aynı held-out metinlerde kapsam, sıkıştırma ve Türkçe
morfolojik sınır davranışıyla karşılaştırılır. Araç bir kalite iddiası üretmez;
ölçülebilir regresyon sinyalleri üretir.

## Hızlı analiz

Sürümlü 24 belgeli seed seti ve 36 morfoloji formuyla mevcut tokenizerı ölçün:

```bash
python evaluation/analyze_tokenizer.py \
  --tokenizer current=toprak_tokenizer.model \
  --output evaluation/reports/tokenizer-current.json \
  --markdown evaluation/reports/tokenizer-current.md
```

İki tokenizerı aynı girdide karşılaştırmak için `--tokenizer` tekrarlanır:

```bash
python evaluation/analyze_tokenizer.py \
  --tokenizer bpe32k=toprak_tokenizer.model \
  --tokenizer unigram32k=experiments/unigram.model \
  --input data_cache/tokenizer_heldout \
  --max-documents 10000 \
  --output evaluation/reports/tokenizer-comparison.json \
  --markdown evaluation/reports/tokenizer-comparison.md
```

`--input` JSON, JSONL, TXT dosyası veya bunları içeren bir dizin kabul eder.
JSON/JSONL kayıtlarında `text` zorunlu, `domain` isteğe bağlıdır. Aynı rapordaki
tüm tokenizerlar aynı sıralı belge listesi ve morfoloji probları üzerinde
çalışır; corpus ve prob SHA-256 değerleri rapora yazılır.

## Metrikler

- `tokens_per_word`: düşük değer daha kompakt tokenizasyonu gösterir;
- `unknown_rate`: byte fallback açıkken normalde sıfır olmalıdır;
- `byte_token_rate`: kapsanan ama byte parçalarına düşen içeriği görünür kılar;
- `roundtrip_exact_rate`: NFKC ve whitespace normalizasyonundan sonra geri dönüş;
- `exact_suffix_boundary_rate`: bilinen Türkçe ekin ayrı bir piece olması;
- `lemma_prefix_reuse_rate`: lemma piece dizisinin çekimli biçimde korunma oranı;
- domain bazlı p50/p95 belge uzunluğu ve karakter/byte başına token verimi;
- Türkçeye özgü küçük/büyük karakterlerin tek tek UNK/piece dökümü;
- vocab içindeki word-start, continuation, byte ve tek-karakter piece sayıları.

CI veya release kapısı için eşikler uygulanabilir:

```bash
python evaluation/analyze_tokenizer.py \
  --tokenizer candidate.model \
  --input data_cache/tokenizer_heldout \
  --max-tokens-per-word 2.20 \
  --max-unknown-rate 0 \
  --max-byte-token-rate 0.02 \
  --min-roundtrip-rate 1.0 \
  --output evaluation/reports/tokenizer-gate.json
```

Sürümlü seed set küçük ve proje tarafından Apache-2.0 altında hazırlanmıştır.
Tokenizer seçimi en azından haber, konuşma, teknik, hukuk, sosyal medya,
code-mixed ve gürültülü metinlerden oluşan eğitim-dışı büyük bir corpus üzerinde
tekrarlanmalıdır.
