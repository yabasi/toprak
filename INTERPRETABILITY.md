# Morfoloji Mikroskobu — Yorumlanabilirlik Araç Takımı

`interpret/` paketi, Toprak'ın iç temsillerinde Türkçe morfolojinin **nerede ve
ne kadar** kodlandığını ölçer: kök/ek ayrımı, büyük ünlü uyumu, ünsüz
benzeşmesi (fıstıkçı şahap) ve yaygın ek türleri. Saf PyTorch + standart
kütüphane; raporlar dış kaynak yüklemeyen tek dosyalık HTML'dir.

## Motivasyon

Toprak'ın yardımcı kayıpları (ünlü uyumu, ünsüz benzeşmesi, morfolojik
ağırlıklı CE, morfoloji başlığı, hece/kafiye) "Türkçe'ye özgü" ve kimi açık
kaynak modellerde ilk kez denenen fikirlerdir. Ancak bir kaybın eklenmesi ve
toplam kaybın düşmesi, modelin **içeride farklı bir şey öğrendiğini**
kanıtlamaz:

- Kayıp eğrisi, modelin yalnız çıktı dağılımını cezadan kaçacak biçimde
  kaydırdığını da gösterebilir (yüzeysel çözüm).
- Eval v1 benchmark'ı (bkz. [ABLATION.md](ABLATION.md)) davranışı ölçer, iç
  temsili değil.

Morfoloji Mikroskobu şu soruya kanıt toplar: *"Ünlü uyumu kaybıyla eğitilen
model, sıradaki ekin hangi ünlü sınıfını alacağını baseline'a göre daha erken
katmanda / daha seçici biçimde temsil ediyor mu?"*

## Yöntemler

### 1. Aktivasyon kaydı (`interpret/activations.py`)

`ActivationRecorder(model, sites=("resid_post","attn_out","ffn_out"), layers=None)`
her `TransformerBlock`'a ileri besleme kancası takar:

| nokta | anlamı |
|---|---|
| `resid_pre` | bloğa giren artık akış (0. blokta gömme çıktısı) |
| `resid_post` | bloktan çıkan artık akış |
| `attn_out` | dikkatin artık akışa eklediği katkı |
| `ffn_out` | FFN'in (SwiGLU ya da MoE) eklediği katkı |

MoE bloklarında `MorphRoutedMoE` `(out, aux)` demeti döndürür; kanca demetin
ilk elemanını alır ve `last_routing` (B, T, k) uzman indekslerini ayrıca
kaydeder. `resid_post = resid_pre + attn_out + ffn_out` özdeşliği testlerle
doğrulanır. Kaydedici bir bağlam yöneticisidir; çıkışta tüm kancalar kalkar.

`collect(model, tokenizer, texts, max_len)` cümleleri tek tek (dolgusuz)
çalıştırır ve token'larla hizalı, düzleştirilmiş (N satır) aktivasyonlar,
token ID/metinleri, cümle indeksleri, model'in kök/ek/özel sınıfları ve MoE
yönlendirmesini döndürür. `layer_dict()` sondalar için `emb, L1..LN` sıralı
sözlüğü verir (`Lk` = k. bloğun çıktısı).

### 2. Dilbilimsel etiketler (`interpret/features.py`)

`model/vowel_harmony.py` ve `model/consonant_harmony.py` tablolarını yeniden
kullanır. Etiket tanımsızsa `-1` (sondada yok sayılır).

| özellik | sınıflar | not |
|---|---|---|
| `kelime_basi` | devam / kelime başı | `▁` öneki |
| `morf_sinifi` | kök / ek / özel | `ToprakLM.token_morph_classes` ile aynı kural |
| `son_unlu` | kalın / ince / yok | kelimenin bu token dahil son ünlüsü |
| `sonraki_ek` | hayır / evet | **ileriye dönük**: sıradaki token bu kelimeye ek mi |
| `beklenen_uyum` | kalın / ince | sıradaki ek ünlülü ise, uyumun gerektirdiği sınıf |
| `sonraki_ek_unlusu` | kalın / ince | sıradaki ekin *gerçek* ilk ünlüsü (istisnalar dahil: saat+ten) |
| `sert_unsuz` | hayır / evet | kelime şimdiye dek f s t k ç ş h p ile mi bitiyor |
| `ek_turu` | diğer / çoğul / bulunma / ayrılma / geçmiş -DI / -mIş | yalnız ek parçalarında |
| `sonraki_ek_turu` | (aynı) | ileriye dönük |

**Uyarı:** Bunlar BPE parçaları üzerinde *sezgisel* kurallardır, morfolojik
çözümleyici değildir. "larda" parçası yalnız "çoğul" sayılır; "da" bağlacı
kelime başı olduğu için ek sayılmaz ama "▁kitapla"+"rın" gibi morfem sınırına
denk gelmeyen bölmeler gürültü üretir.

### 3. Doğrusal sondalar ve kontrol görevleri (`interpret/probes.py`)

`train_linear_probe(X, y, num_classes, epochs, lr, weight_decay, val_split, seed, class_balanced=True, groups=None, val_mask=None)`
saf torch çok sınıflı lojistik regresyondur (özellikler eğitim istatistiğiyle
standartlaştırılır, tam-batch Adam + L2). Doğrulama doğruluğu, makro-F1 ve
çoğunluk sınıfı tabanını döndürür. CLI bölmeyi **cümle düzeyinde** yapar;
aynı cümlenin token'ları hem eğitim hem doğrulamada bulunmaz.

Aşırı iddiadan kaçınmak için **Hewitt & Liang (2019) kontrol görevi**:
`control_labels` her token türüne (ID) gerçek etiket dağılımından rastgele
ama sabit bir etiket atar. Sonda bu anlamsız görevi de yapabiliyorsa yüksek
doğruluk temsilin değil sondanın/ezberin kapasitesidir.

> **Seçicilik = doğruluk − kontrol doğruluğu.** Rapordaki "en seçici katman"
> buna göre seçilir.

`probe_all_layers(acts_by_layer, labels, token_ids=..., groups=..., seeds=(0,1,2))`
her katmanda sonda + kontrol çalıştırır; birden çok tohumda ortalama ± sapma
verir.

### 4. TopK seyrek otokodlayıcı (`interpret/sae.py`)

Gao ve ark. (2024) tarifi: `z = TopK(W_enc(x − b_pre) + b_enc)`,
`x̂ = W_dec z + b_pre`. Kod çözücü sütunları birim normda tutulur (gradyanın
paralel bileşeni atılır, her adımdan sonra yeniden normlanır), `b_pre` veri
ortalamasıyla, `W_enc` = `W_decᵀ` ile başlatılır; girdi ölçeği SAE içinde
saklanır. Ölü latentler izlenir; AuxK kaybı ölü latentlerle artığı modelleyerek
onları canlandırır. Kalite: **NMSE** = E‖x−x̂‖² / E‖x−x̄‖², **FVE** = 1 − NMSE.

- `train_sae(acts, d_hidden, k, epochs, lr, seed)` → `SAETrainResult`
  (`history`, `nmse`, `fve`, `dead_fraction`).
- `feature_top_tokens(sae, acts, token_strings, n)` → her latentin en güçlü
  bağlamları.
- `feature_label_association(sae, acts, labels, positive_class)` → latentleri
  ikili etiketle nokta-çift serili korelasyona (r) göre sıralar. Bir "ünlü uyumu
  özelliği" şöyle aranır: `labels = beklenen_uyum`, `positive_class = 1` (ince);
  en yüksek r'li latentin bağlamları ince ünlülü kelime gövdelerinin sonu mu?

### 5. Nedensel müdahale (`interpret/patching.py`)

Sondalar okunabilirliği gösterir, kullanımı değil. Bir sonraki adım:

- `ablate_feature(model, layer, direction=...)` — bir yönün (ör. sonda
  ağırlık farkı) bileşenini katman çıktısından siler (`scale=0`) ya da büyütür.
- `ablate_feature(model, layer, sae=..., feature=f)` — bir SAE latentinin
  katkısını çıkarır.
- `patch_activations(model, layer, source, positions)` — temiz çalıştırmanın
  aktivasyonunu bozuk çalıştırmaya yamalar.
- `next_token_logprob(model, ids, token_set)` — bir token kümesinin (ör. ince
  ünlülü ekler) toplam log-olasılığı.

Örnek: "ince uyum" latentini sildikten sonra ince ünlülü ek kümesinin
log-olasılığı düşüyor, kalın kümeninki artıyorsa latent nedensel olarak
kullanılıyordur.

```python
from interpret.patching import ablate_feature, next_token_logprob
base = next_token_logprob(model, ids, ince_ek_ids)
with ablate_feature(model, layer=5, sae=sae, feature=123):
    ablated = next_token_logprob(model, ids, ince_ek_ids)
etki = (ablated - base)[0, konum]
```

### 6. MoE yönlendirme analizi

MoE açık modellerde rapor, her MoE katmanı için **uzman × morfolojik sınıf**
tablosu gösterir (`model.moe.routing_by_morph_class`): bir sınıfın
yönlendirmelerinin hangi uzmana gittiği (sütun payı). Eşit dağılım ≈ 1/E;
belirgin yoğunlaşma, morfolojik yönlendirme ipucunun (`moe_morph_routing`)
uzman özelleşmesi ürettiğinin işaretidir.

## Komutlar

```bash
# Tüm özellikler × tüm katmanlar sondası → reports/probe/probe.{json,html}
python -m interpret.cli probe \
  --checkpoint checkpoints/toprak_last.pt \
  --tokenizer toprak_tokenizer.model \
  --texts interpret/examples/sentences.txt \
  --out reports/probe --seeds 0 1 2

# Bir katmanda TopK SAE → reports/sae/sae.{json,html} (+ --save-sae ile sae.pt)
python -m interpret.cli sae --checkpoint checkpoints/toprak_last.pt \
  --layer L6 --k 16 --d-hidden 4096 --epochs 30 --out reports/sae --save-sae

# İki checkpoint'in katman bazında farkı (B − A)
python -m interpret.cli compare \
  --checkpoint-a ablation_runs/run_001/checkpoints/baseline/toprak_last.pt \
  --checkpoint-b ablation_runs/run_001/checkpoints/vowel_harmony/toprak_last.pt \
  --name-a baseline --name-b vowel_harmony --seeds 0 1 2 --out reports/cmp_vowel

# Hazır probe.json raporlarından karşılaştırma (yeniden hesaplamadan)
python -m interpret.cli compare --report-a a/probe.json --report-b b/probe.json \
  --name-a baseline --name-b morph_head --out reports/cmp_morph
```

Ortak seçenekler: `--max-len`, `--device cpu|cuda|mps`, `--site resid_post|attn_out|ffn_out`,
`--features ...`, `--epochs`, `--lr`, `--weight-decay`, `--val-split`, `--view-sentences`.
Örnek metin: `interpret/examples/sentences.txt` (41 ek bakımından zengin cümle;
`#` satırları yorum). Güvenilir sonuç için birkaç bin cümlelik, eğitimde
görülmemiş bir metin kümesi kullanın.

## Raporu okumak

**probe.html**

- *Katman eğrileri*: her özellik için katman başına mavi çubuk = sonda
  doğruluğu (doğrulama cümleleri), turuncu = kontrol görevi doğruluğu,
  kesikli çizgi = çoğunluk tabanı. Üzerine gelince seçicilik ve makro-F1
  görünür; "Tablo" açılır kutusu sayıları verir.
- Beklenen desen: `kelime_basi`, `morf_sinifi` zaten `emb`'de yüksek (token
  kimliğinden okunur — seçicilik düşük olabilir); `son_unlu`, `beklenen_uyum`,
  `sert_unsuz` bağlam gerektirdiği için ilk katmanlarda yükselir; ileriye dönük
  `sonraki_ek*` özellikleri genelde orta/geç katmanlarda zirve yapar.
- *Token görünümü*: cümle / özellik / katman / sınıf seçin; token rengi
  sondanın o sınıfa verdiği olasılıktır. Kırmızı çerçeve = yanlış tahmin,
  kesikli = etiket tanımsız. "eğitim" etiketli cümlelerde olasılıklar
  örneklem-içidir; dürüst inceleme için "doğrulama" cümlelerine bakın.
- *MoE yönlendirme*: uzman × kök/ek/özel pay tablosu.

**sae.html** — eğitim özeti (NMSE başlangıç→son, FVE, ölü latent oranı), her
özellik/sınıf için en ilişkili latentler (r, sınıfta/dışında ateşlenme oranı)
ve en sık ateşlenen latentlerin bağlamları (işaretli token = latentin
ateşlendiği yer).

**compare.html** — her özellik için katman başına Δ doğruluk (B − A; mavi
artış, kırmızı azalış). `*` / kalın çerçeve: |Δ| > 2·SE (binom SE + tohum
sapması; kaba bir eşik). Metin parmak izi, tokenizer veya mimari farklıysa
uyarı gösterilir.

## Önerilen deney protokolü (ABLATION.md ile)

1. `scripts/run_ablation.py` ile aynı başlangıç checkpoint'i, aynı veri,
   aynı adım sayısı ve **aynı tohum**la `baseline` ve her yardımcı kayıp
   varyantını eğitin (bkz. [ABLATION.md](ABLATION.md)).
2. Eğitimde görülmemiş, sabit bir sonda metni seçin (ör. Eval dışı 2–5 bin
   cümle). Parmak izi raporun `texts_sha256` alanına yazılır.
3. Her varyant için `compare` çalıştırın: `--checkpoint-a` baseline,
   `--checkpoint-b` varyant, `--seeds 0 1 2`.
4. Ön-kayıtlı hipotezler (sonuçları görmeden yazın):
   - `vowel_harmony` → `beklenen_uyum` ve `sonraki_ek_unlusu` seçiciliği artar
     ve/veya zirve daha erken katmana kayar.
   - `consonant_harmony` → `sert_unsuz` seçiciliği artar.
   - `morph_head` / `morph_weight` → `sonraki_ek`, `morf_sinifi` artar.
   - Kontrol: ilgisiz özelliklerde (ör. `kelime_basi`) Δ ≈ 0 beklenir;
     her yerde artış "genel olarak daha iyi model" etkisidir, özel değil.
5. Belirgin farkları SAE + müdahale ile doğrulayın: varyantta ilgili latenti
   bulun (`sae`), `ablate_feature` ile silin ve Eval v1'deki ilgili
   kategoride (uyum, ek) davranış değişimini ölçün.
6. Sonuçları Eval v1 ablation raporuyla birlikte sunun: davranış farkı +
   temsil farkı + nedensel etki üçlüsü en güçlü kanıttır.

## Sınırlar ve uyarılar

- **Sonda ≠ nedensel kullanım.** Bir bilginin doğrusal olarak okunabilmesi,
  modelin onu tahminde kullandığı anlamına gelmez; `interpret/patching.py`
  ile müdahale deneyleri sonraki adımdır.
- Doğrusal sondalar doğrusal olmayan kodlamayı kaçırır; yüksek kapasiteli
  sondalar ise ezberler. Bu yüzden yalnız doğrusal sonda + kontrol görevi
  kullanılır ve seçicilik raporlanır.
- Etiketler sezgiseldir (BPE parçaları ≠ morfemler); özellikle `ek_turu` ve
  istisnalar (yabancı kökenli kelimeler, -yor, -ken gibi uyuma girmeyen ekler)
  gürültü kaynağıdır.
- Küçük metin kümelerinde doğrulama kümesi küçüktür; tek tohumlu farklar
  gürültülü olabilir. `compare`'deki 2·SE eşiği kaba bir sezgiseldir, çoklu
  karşılaştırma düzeltmesi içermez.
- Token görünümündeki olasılıklar ilk tohumun sondalarından gelir; eğitim
  cümlelerinde örneklem-içidir.
- SAE'ler küçük veride (birkaç bin token) kararsız olabilir; latent sayısını
  ve `k`'yi veriyle ölçekleyin, ölü latent oranını izleyin.
- Cümleler tek tek işlenir (dolgu yok); büyük metin kümelerinde CPU'da
  yavaş olabilir — `--device cuda` kullanın.

## Referanslar

- Hewitt & Liang, *Designing and Interpreting Probes with Control Tasks*, 2019.
- Gao ve ark., *Scaling and Evaluating Sparse Autoencoders*, 2024.
- Bricken ve ark., *Towards Monosemanticity*, 2023.
- Meng ve ark., *Locating and Editing Factual Associations in GPT* (causal tracing), 2022.
