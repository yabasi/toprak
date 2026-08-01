# Auxiliary Loss Ablation

Toprak'ın Türkçeye özel yardımcı kayıpları varsayılan olarak kapalıdır. Bir
özelliğin yararlı olduğu iddiası, aynı başlangıç checkpoint'i ve aynı eğitim
tarifiyle üretilen baseline karşısında ölçülmelidir.

## Deney matrisi

`scripts/run_ablation.py` şu varyantları aynı checkpoint'ten başlatır:

- `baseline`: yalnız standart dil modeli kaybı;
- `vowel_harmony`: ünlü uyumu kaybı;
- `consonant_harmony`: ünsüz benzeşmesi kaybı;
- `morph_weight`: morfolojik ağırlıklı CE;
- `morph_head`: morfolojik çoklu görev başlığı;
- `syllable_rhyme`: hece ve kafiye kaybı;
- `all_aux`: etkileşimleri görmek için tüm yardımcı kayıplar.

Önce komutları ve yolları doğrulayan dry-run çalıştırın:

```bash
python scripts/run_ablation.py \
  --base-checkpoint checkpoints/toprak_step_5000.pt \
  --data-dir data_cache/bin \
  --bin-mode \
  --model-size medium \
  --target-step 10000 \
  --seed 42 \
  --device cuda \
  --output-dir ablation_runs/run_001
```

Gerçek eğitimi başlatmak için aynı komuta `--execute` ekleyin. Her varyant ayrı
checkpoint/log dizinine yazılır, Toprak Eval v1 ile ölçülür ve sonunda
`ablation.json` ile `ablation.md` oluşturulur.

Resume sırasında başlangıç checkpoint'inin optimizer ve scheduler durumu aynen
korunur. Bu nedenle başlangıç checkpoint'i, hedef adıma uygun ortak LR planından
gelmelidir; sürücü tüm varyantlarda aynı scheduler durumunu doğrular.

Tek tek hazırlanmış Eval v1 raporları da karşılaştırılabilir:

```bash
python evaluation/compare_ablation.py \
  --baseline ablation_runs/run_001/reports/baseline.json \
  --candidate vowel=ablation_runs/run_001/reports/vowel_harmony.json \
  --candidate morph=ablation_runs/run_001/reports/morph_weight.json \
  --output ablation_runs/run_001/ablation.json \
  --markdown ablation_runs/run_001/ablation.md
```

Karşılaştırıcı suite, tokenizer, benchmark hash'i, model konfigürasyonu, global
step, başlangıç checkpoint SHA-256'sı, scheduler durumu ve yardımcı loss'lar
dışındaki eğitim tarifini doğrular. Örnek bazlı
candidate-minus-baseline farkı, kazanım/beraberlik/kayıp sayısı ve eşlenik
bootstrap %95 güven aralığı raporlanır.

## Yorumlama sınırları

Seed benchmark küçük bir regresyon setidir. Güven aralığının sıfırı dışlaması,
yalnız bu örneklerdeki eşlenik farkı destekler. Sağlam bir sonuç için deney
birden fazla eğitim seed'iyle tekrarlanmalı; ortalama, standart sapma, eğitim
maliyeti ve kategori bazlı sonuçlar birlikte raporlanmalıdır. Ablation kanıtı
olmadan README'deki yöntem açıklamaları kalite artışı iddiası sayılmamalıdır.
