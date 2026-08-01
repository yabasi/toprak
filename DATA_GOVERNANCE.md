# Toprak Veri Yönetişimi

Toprak eğitim korpusu, her belgenin kaynağını ve uygulanan kalite işlemlerini
izlenebilir tutan `toprak-document-v1` şemasını kullanır.

> Bu belge hukuki görüş değildir. `review_required` veya
> `inherited_review_required` durumundaki kaynaklar korpusa alınmadan önce
> kullanım amacı ve ilgili ülke bakımından ayrıca incelenmelidir.

## Kaynak ve lisans kayıtları

| Kaynak | Dataset | Lisans kaydı | Durum |
|---|---|---|---|
| Wikipedia | `wikimedia/wikipedia` (`20231101.tr`) | GFDL + CC BY-SA 3.0 | Veri kartından doğrulandı |
| FineWeb-2 | `HuggingFaceFW/fineweb-2` | ODC-By 1.0 + Common Crawl kullanım şartları | Veri kartından doğrulandı |
| CulturaX | `uonlp/CulturaX` | mC4 ve OSCAR üst kaynak koşulları | Ayrı inceleme gerekli |
| Web crawler | İlgili alan adı | Otomatik lisans varsayılmaz | Ayrı inceleme gerekli |

Makine tarafından kullanılan kayıtlar
[`data/governance.py`](data/governance.py) içindeki `SOURCE_REGISTRY`
tablosundadır.

## Belge şeması

Temizlenmiş her JSONL satırında en az şu yönetişim alanları bulunur:

```json
{
  "schema_version": "toprak-document-v1",
  "source": "fineweb2",
  "dataset_id": "HuggingFaceFW/fineweb-2",
  "dataset_revision": "main",
  "licenses": ["ODC-By-1.0"],
  "license_status": "verified_dataset_card",
  "license_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
  "source_url": "https://example.org/page",
  "source_record_id": "...",
  "downloaded_at": "2026-08-01T12:00:00+00:00",
  "ingested_at": "2026-08-01T12:05:00+00:00",
  "content_sha256": "...",
  "simhash64": "...",
  "quality_score": 0.91,
  "quality_signals": {},
  "pii_redactions": {"EMAIL": 1},
  "contamination_matches": []
}
```

Eski verilerde gerçek indirme zamanı bilinmiyorsa `downloaded_at` değeri
`null` kalır; temizleme/ingestion zamanı bunun yerine kullanılmaz.

## Temizleme

```bash
python data/cleaner.py \
  --input /workspace/data_raw \
  --output /workspace/data_clean \
  --quality-threshold 0.50 \
  --benchmark-path evaluation/contamination_reference \
  --contamination-action reject
```

Pipeline sırasıyla HTML/boilerplate temizliği, Unicode normalizasyonu,
yüksek kesinlikli PII redaction, kalite skorlama, benchmark contamination
kontrolü, SHA-256 exact dedup ve SimHash near-dedup uygular.

PII katmanı e-posta, IPv4, telefon ve checksum'u geçerli T.C. kimlik
numarası örüntülerini yer tutucularla değiştirir. Bu katman tüm kişisel
verilerin temizlendiğine dair garanti değildir; ad ve serbest biçimli adres
gibi bağlama bağlı alanlar ayrıca örneklenip denetlenmelidir.

Benchmark girdisi `.jsonl` veya `.txt` olabilir. JSONL kayıtlarında `text`,
`prompt`, `question`, `chosen`, `rejected`, `needle`, `reference`, `filler`,
`choices` ve `references` alanları taranır; isteğe bağlı `id` eşleşme kaydına
eklenir. Varsayılan davranış eşleşen belgeleri eğitim verisinden çıkarmaktır;
`flag` seçeneği yalnız metadata'ya işler.

Her temizleme çalışması çıktı dizinine `corpus_manifest.json` yazar. Manifest,
ayarları, sayaçları, kaynak/lisans durum dağılımını ve çıktı dosyalarının
SHA-256 değerlerini içerir.

## Audit

```bash
python scripts/audit_corpus.py /workspace/data_clean \
  --output /workspace/data_clean/audit_report.json \
  --strict
```

Strict audit şu durumlarda başarısız olur:

- geçersiz JSON;
- eksik yönetişim metadata'sı;
- şema veya içerik hash uyuşmazlığı;
- temiz korpusta tekrarlanan SHA-256 değeri.

Lisans incelemesi gereken belgeler raporda ayrıca sayılır; veri silme kararı
otomatik verilmez.
