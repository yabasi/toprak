# 🧠 Düşünen Toprak — GRPO ile Akıl Yürütme

Bu belge, Toprak'a **doğrulanabilir ödüllü pekiştirmeli öğrenme** (RLVR) ile
adım adım akıl yürütmeyi öğretme hattını anlatır: lisanssız sentetik Türkçe
matematik verisi → SFT ısınması → GRPO.

| Dosya | İçerik |
|---|---|
| `data/synthetic_math.py` | Tohumlu, deterministik Türkçe sözel problem üreticisi (11 şablon × 3 seviye) |
| `training/rewards.py` | Cevap çıkarma, sayı/kesir/yüzde eşleştirme, ödül fonksiyonları |
| `training/grpo.py` | GRPO eğiticisi + CLI |
| `tests/test_synthetic_math.py`, `tests/test_rewards.py`, `tests/test_grpo.py` | Birim testleri ve oyuncak RL testi |

---

## 1. GRPO nedir?

**GRPO (Group Relative Policy Optimization)**, PPO'nun değer (critic) ağı
gerektirmeyen bir türevidir (DeepSeekMath / DeepSeek-R1). Her soru için
modelden **G tamamlama** örneklenir, her biri kurallarla puanlanır ve taban
çizgisi olarak **grubun ortalama ödülü** kullanılır:

```
A_i  = (r_i − mean(r)) / (std(r) + eps)          # grup içi avantaj; std = 0 → A = 0
ρ_t  = π_θ(o_t) / π_old(o_t)                     # örnekleme anındaki politikaya oran
L_pg = −min(ρ_t·A_i, clip(ρ_t, 1−ε, 1+ε_high)·A_i)
KL_t = exp(ref − pol) − (ref − pol) − 1          # k3 tahmincisi, her zaman ≥ 0
L    = Σ mask·(L_pg + β·KL_t) / Σ mask           # yalnız tamamlama tokenları
```

- **Avantaj**: Grubun hepsi doğru ya da hepsi yanlışsa (std = 0) o soru
  gradyan üretmez. `zero_std_frac` metriği bunun oranını gösterir; çok
  yüksekse sorular ya çok kolay ya çok zordur.
- **Kırpma**: `num_iterations > 1` iken aynı rollout'larla birden çok
  güncelleme yapılır; oran kırpması politikanın örnekleme anından fazla
  uzaklaşmasını engeller. `--clip-eps-high` (ör. 0.28) DAPO'daki
  "clip-higher" fikridir; düşük olasılıklı doğru tokenların büyümesine izin
  verir.
- **KL cezası**: Politika, dondurulmuş başlangıç modeline (referans) k3
  tahmincisiyle bağlanır; dil bozulmasını ve ödül sömürüsünü yavaşlatır.
  `--beta-kl 0` referans modeli tamamen kapatır (bellek tasarrufu).
- **Maskeler**: Prompt tokenları ve durdurma sonrası dolgu (pad) kayba
  girmez; durdurma tokenı (`<|son|>`/`<sep>`/EOS) kayba **dahildir**, böylece
  model durmayı da öğrenir. Ortalama token düzeyindedir (uzun cevaplar
  örnek başına seyreltilmez).
- **Örnekleme**: Kendi KV-cache'li döngümüz G tamamlamayı paralel üretir;
  sıcaklık ve isteğe bağlı top-k desteklenir. π_old log-olasılıkları
  rollout'tan hemen sonra aynı ağırlıklarla tam diziler üzerinde yeniden
  hesaplanır (sayısal tutarlılık için; ilk güncellemede ρ = 1).

## 2. `<düşünce>` biçimi

Model şu çıktıyı üretmeye yönlendirilir (sistem mesajı:
`training/rewards.py: REASONING_SYSTEM_PROMPT`):

```
<düşünce>
Birinci ürünler: 6 × 10 = 60 TL.
İkinci ürünler: 1 × 112 = 112 TL.
Toplam: 60 + 112 = 172 TL.
Para üstü: 200 − 172 = 28 TL.
Sonuç: 28
</düşünce>
Cevap: 28
```

Etiketler mevcut 32K tokenizer'da birkaç token olarak kodlanır; özel token
gerekmez. Çıkarımda kullanıcıya yalnız `Cevap:` satırı gösterilebilir.

## 3. Ödül tasarımı (`training/rewards.py`)

| Bileşen | Değer aralığı | Varsayılan ağırlık | Açıklama |
|---|---|---|---|
| `correctness_reward` | 0 / 1 | 1.0 | `extract_answer` + `answers_match` |
| `format_reward` | 0 / 0.25 / 0.5 / 1 | 0.2 | Tam biçim 1; etiketler + Cevap ama hatalı sıra 0.5; yalnız `Cevap:` 0.25 |
| `language_reward` | 0 … 1 | 0.1 | Harflerin Türkçe alfabeden olma oranı (q/w/x tolere edilir) |
| `length_penalty` | −1 … 0 | 0.1 | `soft_limit` sonrası lineer ceza, `hard_limit`'te −1 |

`combine_rewards({"correctness": 1.0, "format": 0.2, ...})` ağırlıklı toplam
döndürür; `.detailed()` bileşenleri ayrı verir. Yeni bileşen için
`functions={"ad": fn}` geçin (`fn(completion, gold=None, **kw) -> float`).

**Cevap çıkarma** Türkçe sayı yazımını bilir: `Cevap: 42`, `cevap 3/4`,
`Sonuç: 12,5` (ondalık virgül), `1.250` (binlik nokta), `1.250,75`, `-7`,
`%25`, `25%`, `yüzde 25`, `\boxed{…}`. Önce son "Cevap/Sonuç/Yanıt"
işaretinden sonraki ilk sayıya, yoksa düşünce bloğundan sonraki son sayıya
bakılır. Not: `3.5` gibi 3'lü grup olmayan noktalar ondalık sayılır,
`1.500` ise binliktir (Türkçe kural).

**Eşleştirme** kesin kesirlerle yapılır: `0,75 == 3/4 == 6/8`; altın cevap
`%25` ise `25`, `%25` ve `0,25` kabul edilir. `tol` göreli toleranstır
(varsayılan 1e-6; `0,33 ≠ 1/3`).

## 4. Sentetik veri üreticisi (`data/synthetic_math.py`)

> **Lisans notu:** Tüm problemler bu dosyadaki şablonlardan rastgele sayılarla
> üretilir. **ÖSYM, MEB, yayınevi ya da başka bir telifli sınav/kitap
> kaynağından hiçbir soru kopyalanmamıştır.** Üretilen veri projenin
> Apache-2.0 lisansıyla serbestçe kullanılabilir. Bu belgeye ya da veri
> kümesine telifli sınav sorusu eklemeyin.

Şablonlar (her biri seviye 1–3): `alisveris`, `yas`, `hiz`, `isci_havuz`,
`yuzde` (indirim, zam, ters yüzde, yüzde oranı), `kesir`, `bolunebilme`
(içerme-dışlama dahil), `sayi_dizisi` (aritmetik/geometrik, toplam),
`denklem`, `olasilik` (kesir cevaplı), `birim_cevirme`.

Kayıt biçimi:

```json
{"id": "train-42-000017", "question": "...", "answer": "49,5", "answer_value": 49.5,
 "level": 3, "template": "isci_havuz", "solution": "adım adım çözüm...\nSonuç: 49,5"}
```

- Cevaplar `fractions.Fraction` ile kesin hesaplanır; kanonik dize Türkçe
  yazımdır (`1.250`, `12,5`, `5/12`, `%25`). `answer_value` tam sayı, ondalık
  ya da `"a/b"` dizesidir.
- **Dilbilgisi:** Sayıdan sonra ad tekil kalır ("5 elma"). Rakamla yazılan
  sayılara ekler okunuşa göre getirilir: `number_suffix(n, hâl)` →
  `5'e, 6'ya, 10'a, 40'a, 100'e, 3'ü, 2'yi, 9'u, 60'ı, 4'te, 24'tür, 12'si`;
  kesirlerde payın okunuşu esas alınır (`3/5'ü` = beşte üçü, `1/4'ini`).
  Hâller: `acc, dat, loc, abl, gen, ins, poss3, poss3_acc, cop`.
  Özel adlar için `name_suffix("Mehmet", "abl")` → `Mehmet'ten`.
- Her örnek kendi `(seed, i)` tohumlu RNG'siyle üretilir → tamamen
  deterministik. Test kümesi ayrı tohumla üretilir ve eğitimde birebir geçen
  sorular testten atılır.

### Şablon ekleme

```python
@template("faiz")
def _faiz(rng, level):
    ana_para = rng.choice(range(1000, 10001, 500))
    oran = rng.choice([5, 10, 20])
    faiz = Fraction(ana_para * oran, 100)
    soru = f"{N(ana_para)} TL yıllık %{oran} basit faizle 1 yıl bekletilirse kaç TL faiz alınır?"
    adimlar = [f"Faiz = {N(ana_para)} × {oran} / 100 = {N(faiz)} TL."]
    return soru, faiz, adimlar, "auto"   # tür: "auto" | "fraction" | "percent"
```

Kurallar: cevabı **kesin** hesaplayın (Fraction), sayıya gelen ekleri
`number_suffix`/`name_suffix` ile üretin, sayıdan sonra çoğul ad
kullanmayın ve `tests/test_synthetic_math.py`'ye bağımsız yeniden hesaplama
testi ekleyin.

## 5. Önerilen hat

1. **Ön eğitim** (`training/train.py`) — dil yetkinliği.
2. **SFT ısınması** — sentetik çözümlerle `<düşünce>` biçimini öğretin.
   GRPO, biçimi hiç üretemeyen bir modelden başlarsa ödül hep 0 olur ve
   öğrenme olmaz.
   ```bash
   python data/synthetic_math.py --output-dir data/reasoning_sft --format sft --train 50000 --test 1000
   # → {"messages": [sistem, kullanıcı, asistan]} kayıtları; ChatTemplate ile SFT eğitiminde kullanın
   ```
3. **GRPO** — doğruluk ödülüyle akıl yürütmeyi pekiştirin.
   ```bash
   python data/synthetic_math.py --output-dir data/reasoning --train 20000 --test 500 --seed 42
   python training/grpo.py --checkpoint checkpoints/toprak_sft.pt \
       --data data/reasoning/train.jsonl --output checkpoints/grpo \
       --steps 500 --group-size 8 --prompts-per-step 4 --lr 1e-6 --beta-kl 0.04 \
       --max-new-tokens 256 --temperature 1.0
   ```
   Kayıtlar (`toprak_grpo_step_N.pt`, `toprak_grpo_last.pt`)
   `training/trainer.py` biçimindedir; `inference/generate.py --checkpoint`
   ile doğrudan açılır. Metrikler `grpo_log.jsonl`'a yazılır.
4. **Değerlendirme** — ayrık test kümesinde (`data/reasoning/test.jsonl`)
   doğruluk ve `evaluation/benchmarks/reasoning.jsonl`.

Kolay başlangıç için `--levels 1` ile veri üretip müfredat (curriculum)
uygulayın; doğruluk %70'i aşınca seviye 2–3'ü ekleyin.

## 6. İzlenecek metrikler

| Metrik | Beklenen davranış |
|---|---|
| `reward_mean`, `accuracy` | Yavaş ama düzenli artış |
| `zero_std_frac` | 0.2–0.6 arası sağlıklı; ~1 ise veri çok kolay/zor (seviye değiştirin, G'yi artırın) |
| `kl` | Küçük ve yavaş büyüyen (≈0.01–0.1); patlıyorsa lr'yi düşürün / β'yı artırın |
| `completion_length` | Düşünce uzayabilir; aniden çökme ya da `max_new_tokens`'a yapışma kötü işaret |
| `truncated_frac` | Yüksekse `max_new_tokens`'ı artırın ya da `--w-length`'i yükseltin |
| `clip_frac` | `num_iterations=1`'de ~0; >0.2 ise adımlar çok büyük |

## 7. Hesaplama beklentileri

Her adım `prompts_per_step × group_size` tamamlama üretir (varsayılan
4 × 8 = 32) ve politika + referans için ileri geçiş yapar. Bellek kabaca
politika + AdamW durumları + referans model kopyası kadardır (`--beta-kl 0`
referansı kaldırır). Örnekleme, adım süresinin çoğunu alır.

- Small (~80M) modelde tek bir tüketici GPU'sunda/M-serisi Mac'te adım
  başına saniyeler mertebesi; 500 adım birkaç saat.
- CPU yalnız testler ve küçük denemeler içindir.

## 8. Dürüst sınırlamalar

- **Küçük modeller zayıf akıl yürütür.** 80–340M parametreli bir model
  çok adımlı aritmetikte sınırlıdır; GRPO var olan yeteneği keskinleştirir,
  yoktan yaratmaz. Gerçekçi hedef: seviye 1–2 şablonlarda belirgin artış.
- **RL için yetkin bir SFT modeli şart.** Ödül seyrek (0/1) olduğundan,
  başlangıç modeli gruplardan en az birini ara sıra doğru yapmalıdır;
  aksi halde `zero_std_frac ≈ 1` olur ve gradyan çıkmaz.
- **Ödül sömürüsü (reward hacking):** Model doğru cevabı tahmin edip düşünce
  bloğunu boş/anlamsız doldurabilir, son sayıyı tekrarlayabilir ya da biçim
  ödülünü cevapsız toplamaya çalışabilir. Biçim ağırlığını doğruluktan
  küçük tutun, örnek çıktıları düzenli okuyun ve KL cezasını kapatmayın.
  Düşünce içeriğinin doğruluğu ödüllendirilmez; yalnız son cevap doğrulanır.
- **Sentetik dağılım dar:** Şablonlar sınırlı dilsel çeşitlilik içerir;
  model şablon kalıplarını ezberleyebilir. Ayrık tohumlu test kümesi aynı
  şablon dağılımından gelir, gerçek dünya genellemesini tam ölçmez.
- **Cevap çıkarma sezgiseldir:** Biçim dışı çıktılarda yanlış sayı
  yakalanabilir; `3.5`/`1.500` gibi nokta belirsizlikleri Türkçe kurala göre
  çözülür.
- Uzunluk cezası karakter tabanlıdır (yaklaşık 4 karakter/token varsayımı).
