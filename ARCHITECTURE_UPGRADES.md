# Mimari Yükseltmeler

Bu belge Toprak'a eklenen dört mimari yeteneği ve bunlarla birlikte düzeltilen
hataları anlatır. Tüm yükseltmeler **varsayılan olarak kapalıdır**: mevcut
checkpoint'ler aynen yüklenir, eski eğitim komutları aynı davranır.

| Yetenek | Dosya | Açma |
|---|---|---|
| Çoklu Token Tahmini (MTP) | `model/transformer.py` (`MTPHead`) | `--mtp-heads 3` |
| Kendi kendine spekülatif çözümleme | `inference/speculative.py` | `python inference/speculative.py ...` |
| Morfoloji yönlendirmeli MoE | `model/moe.py` | `--num-experts 8 --experts-top-k 2` |
| Uzun bağlam (linear / NTK / YaRN) | `model/rope.py` | `--max-seq-len 16384 --rope-scaling yarn` |
| Sohbet şablonu | `model/chat_template.py` | SFT/DPO/GRPO/sohbet tarafından otomatik |

---

## 1. Çoklu Token Tahmini (MTP)

Türkçede tek bir kelime çoğu zaman 3-6 BPE tokenına bölünür
(`▁Gör` `eme` `yecek` `leri` `nden`). Standart dil modeli yalnız bir sonraki
tokenı tahmin eder. MTP ile model, aynı gizli durumdan `t+2 … t+K+1`
tokenlarını da tahmin etmeyi öğrenir; yani pratikte **kelimenin geri kalan
eklerini önceden planlar**.

Tasarım:

```
h_t ──► lm_head ──► t+1                    (ana başlık)
h_t ──► MTPHead_k ──► lm_head ──► t+k+1     (k = 1..K, ortak lm_head)

MTPHead(h) = RMSNorm( h + W2 · SiLU(W1 · RMSNorm(h)) )
```

- `W2` sıfırla başlatılır, böylece başlıklar eğitimin başında ana başlığın
  gizli durumunu kullanır ve kararlı biçimde ayrışır.
- Kayıp: `L = L_LM + mtp_lambda · ortalama_k CE(MTP_k, hedef[t+k])`.
  TensorBoard'da `train/mtp_loss` olarak izlenir.
- Maliyet: Medium modelde 3 başlık ≈ +3,5M parametre (%2,8).

```bash
python training/train.py --model-size medium --mtp-heads 3 --mtp-lambda 0.3 ...
```

## 2. Kendi Kendine Spekülatif Çözümleme

`inference/speculative.py` ayrı bir taslak modeline ihtiyaç duymaz:

1. **Taslak**: MTP başlıkları (`--draft-source mtp`) ya da MTP'siz
   checkpoint'lerde *prompt lookup* (`--draft-source ngram`): dizinin sonundaki
   n-gram daha önce geçtiyse onu izleyen tokenlar önerilir. RAG ve özetleme
   gibi kaynaktan kopyalayan görevlerde bu yöntem özellikle etkilidir.
2. **Doğrulama**: `[kesin token + K taslak]` KV cache ile **tek ileri geçişte**
   işlenir; ana modelin greedy tahminleriyle örtüşen en uzun önek kabul edilir,
   KV cache reddedilen kısımdan kırpılır.

**Garanti:** greedy modda çıktı, standart greedy çözümlemeyle token token
aynıdır. Bu `tests/test_architecture_upgrades.py` içinde hem MTP hem n-gram
taslakları için doğrulanır. Taslakların hepsi kabul edildiğinde ileri geçiş
sayısı `≈ N / (K+1)` düzeyine iner (bu da testte doğrulanır).

```bash
python inference/speculative.py --checkpoint checkpoints/toprak_last.pt \
  --prompt "Türkiye'nin başkenti" --max-tokens 200 --compare
```

`--compare` standart greedy ile süreyi ve çıktı eşitliğini raporlar. Gerçek
hızlanma, eğitilmiş bir modelde taslak kabul oranına bağlıdır. Rapordaki
`acceptance_rate` ve `tokens_per_forward` değerlerini izleyin.

Sınırlama: şimdilik yalnız greedy (temperature=0) çözümleme desteklenir.
Örneklemeli spekülatif çözümleme (rejection sampling) sonraki adım olabilir.

## 3. Morfoloji Yönlendirmeli Uzman Karışımı (MoE)

Yoğun SwiGLU FFN yerine `N` küçük SwiGLU uzmanı vardır ve yönlendirici her
token için `top-k` uzman seçer. **Türkçeye özgü fark:** yönlendiricinin
girdisine giriş tokenının morfolojik sınıfının (kök / ek / özel) öğrenilebilir
bir gömmesi eklenir. Bu sınıf giriş tokenından bilindiği için çıkarımda ek
maliyeti yoktur. Gömme sıfırla başlar; ipucunun ne kadar kullanılacağını model
öğrenir.

- Yük dengeleme: Switch Transformer kaybı `N · Σ f_i · P_i`
  (`moe_aux_loss_coef`, varsayılan 0.01; TensorBoard'da `train/moe_aux_loss`).
- `--moe-layer-freq 2` ile yalnız her ikinci blok MoE olur.
- `moe_d_ff` verilmezse uzman boyutu `d_ff / top_k` olur, böylece token başına
  aktif hesap yoğun modelle aynı kalır.
- `model.moe.routing_by_morph_class()` hangi uzmanın kök, ek ve özel tokenlara
  baktığını gösteren `uzman × sınıf` tablosunu üretir. Bu tablo
  `interpret/` araçlarında görselleştirilir.

Gerçek parametre sayıları (bu depodaki presetlerle ölçüldü, MoE tüm bloklarda):

| Preset | Yoğun | MoE 8 uzman, top-2 (varsayılan uzman boyu) | MoE 8×, top-2, uzman = `d_ff` |
|---|---|---|---|
| Small | 80,3M | 218,4M toplam / 80,4M aktif | 402,5M toplam / 126,4M aktif |
| Medium | 125,3M | 351,9M toplam / 125,4M aktif | 653,9M toplam / 200,9M aktif |
| Large | 341,6M | 1,05B toplam / 341,9M aktif | 1,99B toplam / 577,2M aktif |

```bash
python training/train.py --model-size medium --num-experts 8 --experts-top-k 2 ...
```

Notlar: MoE blokları gradient checkpointing'i atlar. HF/Llama dışa aktarımı MoE
modellerini desteklemez (bkz. EDGE.md). MoE'nin Türkçe için yoğun modele göre
kazancı henüz ölçülmedi. `ABLATION.md` protokolüyle aynı aktif hesapta
karşılaştırılmalıdır.

## 4. Uzun Bağlam: RoPE Ölçekleme

`ModelConfig.rope_scaling` üç yöntemi destekler:

- `linear`: pozisyonlar `factor`'e bölünür (Position Interpolation).
- `ntk`: RoPE tabanı `θ · s^(d/(d-2))` olarak büyütülür.
- `yarn`: yüksek frekanslı boyutlar korunur, düşük frekanslı boyutlar
  interpolasyonla uzatılır, aradaki bant rampa ile karıştırılır. Ayrıca attention
  sıcaklığı `mscale = 0.1·ln(s) + 1` ile düzeltilir.

```bash
# 2K ile ön eğitilmiş checkpoint'i 16K'ya genişletme (devam eğitimi)
python training/train.py --model-size large --resume checkpoints/toprak_last.pt \
  --max-seq-len 16384 --rope-scaling yarn --rope-original-max-seq-len 2048 ...
```

Ayrıntılı tarif, passkey değerlendirmesi ve kaynak gösteren RAG için
[LONG_CONTEXT_RAG.md](LONG_CONTEXT_RAG.md) dosyasına bakın.

## 5. Sohbet Şablonu

`model/chat_template.py` SFT, DPO, GRPO ve sohbet arayüzünün ortak formatıdır.
Bu sayede eğitim ile çıkarım arasında format kayması olmaz:

```
<s><|sistem|>…<|son|><|kullanıcı|>…<|son|><|toprak|>…<|son|>
```

Mevcut 32K tokenizer bu özel tokenları içermediği için geriye uyumlu biçime
düşülür (`Kullanıcı: …<sep>Toprak: …<sep>`); `<sep>` tokenizer'da zaten tek
tokendır. `train_tokenizer()` artık sohbet tokenlarını varsayılan olarak
`user_defined_symbols`'a ekler. Kayıp maskesi yalnız asistan cevabı ve onun tur
sonu tokenı için 1'dir. Ayrıntılar: [ALIGNMENT.md](ALIGNMENT.md).

---

## Düzeltilen Hatalar

1. **Morfolojik başlık kök tokenlarını öğrenmiyordu.** Sınıf etiketi `0 = kök`,
   `ignore_index` ise `pad_token_id = 0` idi. Sonuç olarak tüm kök hedefleri
   kayıptan düşüyor, pad tokenları ise "özel" sınıfı olarak eğitiliyordu. Artık
   maske hedef token ID'sinden kuruluyor.
2. **Cache'li çok-token attention maskesizdi.** KV cache varken `T > 1` token
   işlendiğinde (spekülatif doğrulama, parça parça prefill) causal maske
   uygulanmıyordu. Artık sağ-alt hizalı açık bir maske kullanılıyor; parça
   parça prefill tam prefill ile aynı logit'leri veriyor (test edildi).
3. **`model.half()` / `.to(dtype)` RoPE tablosunu bozuyordu.** Complex
   `freqs_cis` buffer'ı gerçel sayıya çevrilip sanal kısım siliniyordu. Artık
   dtype dönüşümlerinde tablo korunuyor.
4. **Uzun sohbette çökme.** Bağlam RoPE tablosunu aştığında anlaşılmaz bir
   şekil hatası çıkıyordu. Artık açık bir hata mesajı veriliyor, `generate_text`
   üretim uzunluğunu pozisyon sınırına göre kırpıyor ve sohbet arayüzü geçmişi
   bütçeye sığdırıyor.
5. **Checkpoint config'i eksikti.** Yalnız 9 alan yazılıyordu. Artık
   `ModelConfig.architecture_dict()` tüm mimari alanları (MTP, MoE,
   `rope_scaling`, özel token ID'leri) kaydediyor. Eski checkpoint'ler
   varsayılanlarla yüklenmeye devam ediyor.

## Testler

```bash
python -m pytest -q tests/test_architecture_upgrades.py tests/test_chat_template.py tests/test_trainer_upgrades.py
```

`tests/test_trainer_upgrades.py`, MTP + MoE + YaRN + morfolojik başlık açıkken
`ToprakTrainer`'ı uçtan uca çalıştırır. Checkpoint'in inference yükleyicisiyle
birebir kurulduğunu ve kaldığı yerden devamın bit düzeyinde aynı sonucu
verdiğini doğrular.
