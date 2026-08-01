# Curriculum ve Veri Karışımı

Toprak, kaynakları ayrı shard gruplarında tutar ve eğitim sırasında bu
gruplardan yapılandırılabilir oranlarla örnek çeker. Oranlar eğitim adımına
göre başlangıç değerinden bitiş değerine lineer değişir. Böylece veri dengesi
dosya sırasına veya korpusların ham büyüklüğüne bağlı kalmaz.

## Yapılandırma

[`configs/data_mixture.json`](configs/data_mixture.json) üç grup içeren örnek
bir tariftir. Her kaynak yalnız bir grupta bulunabilir ve tam bir grup
`default: true` olmalıdır. Eşleşmeyen kaynaklar bu varsayılan gruba gider.
Ağırlıkların toplamının 1 olması gerekmez; sampler her adımda normalize eder.

```json
{
  "version": "toprak-mixture-v1",
  "curriculum_steps": 20000,
  "groups": {
    "curated": {
      "sources": ["wiki"],
      "initial_weight": 0.65,
      "final_weight": 0.35
    },
    "other": {
      "default": true,
      "initial_weight": 0.35,
      "final_weight": 0.65
    }
  }
}
```

## Kullanım

Önce grup bilgili shard'ları üretin:

```bash
python scripts/pretokenize.py \
  --input-dir data_cache/clean \
  --tokenizer toprak_tokenizer.model \
  --output-dir data_bin \
  --mixture-config configs/data_mixture.json
```

Ardından normal bin eğitimini başlatın. Manifest içinde bir mixture tarifi
varsa sampler otomatik etkinleşir:

```bash
python training/train.py \
  --bin-mode \
  --data-dir data_bin \
  --tokenizer toprak_tokenizer.model
```

Basit iki gruplu geriye uyumlu tarif için `--mixture-config` yerine
`--curriculum --hq-sources wiki` kullanılabilir. Mixture manifestte mevcut olsa
bile ham shard davranışını karşılaştırmak için eğitime `--no-mixture-sampling`
eklenebilir.

## Davranış ve izlenebilirlik

- Sampler grubu ağırlığa göre, grup içindeki bloğu ise eşit olasılıkla ve geri
  koyarak seçer. Bir epoch içinde bazı blokların tekrarlanması veya hiç
  seçilmemesi beklenen davranıştır.
- Geçerli ağırlıklar TensorBoard'da `data_mixture/<grup>_weight` altında yazılır.
- Tam mixture tarifi deney manifestine ve checkpoint eğitim tarifine eklenir.
- Sampler epoch ve schedule durumu checkpointte saklanır; veri cursor'ı ile
  birlikte exact resume aynı örnek dizisini sürdürür.
- Yapılandırılmış bir grupta hiç eğitim dokümanı yoksa pre-tokenization erken
  hata verir. Kaynak etiketini ve `eval_ratio` değerini kontrol edin.
