# Tekrarlanabilir Deneyler

Toprak eğitimleri seed, veri sırası ve ortam bilgisiyle izlenir. Amaç aynı
yazılım/donanım koşullarında kesintisiz eğitim ile checkpoint'ten devam eden
eğitimin aynı örnek sırasını ve RNG akışını kullanmasıdır.

## Önerilen çalışma

```bash
python training/train.py \
  --model-size medium \
  --data-dir data_bin \
  --bin-mode \
  --seed 42 \
  --deterministic \
  --data-fingerprint manifest \
  --verify-data-hashes \
  --experiment-name medium-seed42
```

`--deterministic`, PyTorch deterministik algoritmalarını zorunlu kılar ve CUDA
için cuBLAS workspace ayarını yapar. Desteklenmeyen nondeterministic bir işlem
sessizce devam etmek yerine hata verir. Bu seçenek hız veya bellek maliyetini
artırabilir.

## Kaydedilen durum

Her çalışma `checkpoint_dir` ve `log_dir` içinde `experiment_manifest.json`
oluşturur. Checkpoint ayrıca aynı manifesti ve aşağıdaki durumları taşır:

- Python, NumPy, PyTorch CPU ve varsa tüm CUDA RNG durumları;
- DataLoader epoch başlangıç generator durumu ve epoch içi batch cursor'u;
- optimizer, GradScaler ve LR scheduler durumu;
- seed, tüm eğitim parametreleri ve yardımcı loss tarifi;
- tokenizer SHA-256 ve veri parmak izi;
- Git commit/branch/dirty durumu, komut satırı, işletim sistemi ve cihaz;
- Python ile kurulu paketlerin tam sürüm listesi.

Eğitim optimizer adımının ortasında Ctrl+C ile kesilirse model, RNG ve veri
cursor'u son tamamlanmış optimizer adımına geri sarılarak kaydedilir.

## Veri parmak izi modları

- `auto`: `manifest.json`/`corpus_manifest.json` varsa manifest, yoksa tam içerik;
- `manifest`: yalnız sürümlü manifest dosyaları; büyük shard setleri için hızlı;
- `full`: desteklenen tüm veri dosyalarının içerik SHA-256 değerleri;
- `metadata`: yalnız göreli yol ve byte boyutu; hızlı fakat içerik garantisi zayıf;
- `off`: yalnız açıkça izlenebilirliğin istenmediği tanılama çalışmaları.

Yeni `scripts/pretokenize.py` çıktıları `toprak-shards-v2` formatında tokenizer,
girdi corpus ve her shard için SHA-256 içerir. `--verify-data-hashes` bu shardları
eğitimden önce doğrular. Eski manifestlerde shard hash'i yoksa bu seçenek erken
hata verir; shardları yeniden üretin veya tam veri parmak izi kullanın.

## Sınırlar

Farklı GPU mimarileri, CUDA/cuDNN/PyTorch sürümleri veya distributed reduction
sıraları bit düzeyinde aynı sonucu garanti etmeyebilir. Manifest bu farkları
görünür kılar. Bilimsel raporlama için birden fazla bağımsız seed çalıştırılmalı;
tek seed'in birebir yeniden üretilebilmesi varyans ölçümünün yerine geçmez.
