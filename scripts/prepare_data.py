# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Veri Hazırlama Pipeline
Wikipedia verisi indirme, tokenizer eğitimi ve eğitim verisi hazırlama.

Kullanım:
    # Tüm pipeline'ı çalıştır
    python scripts/prepare_data.py

    # Sadece Wikipedia indir
    python scripts/prepare_data.py --step download

    # Sadece tokenizer eğit
    python scripts/prepare_data.py --step tokenizer

    # Sadece eğitim verisi hazırla
    python scripts/prepare_data.py --step prepare
"""

import argparse
import json
import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm


# ─────────────────────────────────────────────────
# 1. Wikipedia Verisi İndirme
# ─────────────────────────────────────────────────

def download_wikipedia(output_dir: str = "data_cache", max_articles: int = None):
    """
    Türkçe Wikipedia'yı HuggingFace datasets ile indir.

    Args:
        output_dir: Çıktı dizini
        max_articles: Maksimum makale sayısı (None = hepsi)
    """
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "wikipedia_tr.jsonl")

    print("📥 Türkçe Wikipedia indiriliyor (HuggingFace datasets)...")
    print("   Bu ilk seferde biraz zaman alabilir...")

    # Türkçe Wikipedia dataset'ini yükle
    dataset = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train")

    total = len(dataset)
    if max_articles:
        total = min(total, max_articles)

    print(f"   Toplam makale: {len(dataset):,}")
    print(f"   İşlenecek: {total:,}")

    count = 0
    total_words = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for i, article in enumerate(tqdm(dataset, total=total, desc="Wikipedia")):
            if max_articles and i >= max_articles:
                break

            text = article.get("text", "").strip()
            title = article.get("title", "")

            # Çok kısa makaleleri atla
            words = text.split()
            if len(words) < 50:
                continue

            doc = {
                "text": text,
                "title": title,
                "source": "wikipedia_tr",
                "word_count": len(words),
            }

            json.dump(doc, f, ensure_ascii=False)
            f.write("\n")
            count += 1
            total_words += len(words)

    print(f"\n✅ Wikipedia verisi kaydedildi:")
    print(f"   Dosya: {output_file}")
    print(f"   Makale: {count:,}")
    print(f"   Kelime: {total_words:,}")
    print(f"   Tahmini boyut: {os.path.getsize(output_file) / 1e9:.2f} GB")

    return output_file


# ─────────────────────────────────────────────────
# 2. Alternatif: Örnek Türkçe Veri Oluşturma
# ─────────────────────────────────────────────────

def create_sample_data(output_dir: str = "data_cache", num_samples: int = 1000):
    """
    Geliştirme ve test amaçlı örnek Türkçe veri oluştur.
    Wikipedia indirme tamamlanana kadar bununla tokenizer test edilebilir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Türkçe örnek paragraflar
    sample_texts = [
        "Türkiye Cumhuriyeti, Güneydoğu Avrupa ve Güneybatı Asya'da yer alan bir ülkedir. Başkenti Ankara olan bu ülke, kuzeyinde Karadeniz, batısında Ege Denizi ve güneyinde Akdeniz bulunur. Türkiye, zengin tarihi ve kültürel mirasıyla bilinir. Anadolu toprakları birçok medeniyete ev sahipliği yapmıştır. Hitit, Frig, Lidya, Pers, Roma ve Bizans medeniyetleri bu topraklarda kurulmuş ve gelişmiştir.",
        "İstanbul, Türkiye'nin en büyük şehri ve ülkenin ekonomik, kültürel ve tarihi merkezidir. Boğaziçi ile ikiye ayrılan şehir, hem Avrupa hem de Asya kıtasında yer alır. Tarihi yarımadada bulunan Sultanahmet Camii, Ayasofya ve Topkapı Sarayı dünyaca ünlü tarihi eserlerdir. İstanbul, aynı zamanda önemli bir liman şehri olup, dünya ticaretinde stratejik bir konuma sahiptir.",
        "Türk mutfağı, dünya mutfakları arasında önemli bir yere sahiptir. Kebap çeşitleri, baklavalar, pideler ve çeşitli meze kültürü Türk mutfağının temel unsurlarıdır. Osmanlı saray mutfağından gelen zengin lezzetler, Anadolu'nun farklı bölgelerinin yerel tatlarıyla birleşerek eşsiz bir mutfak geleneği oluşturmuştur. Türk kahvesi ve çay kültürü de Türk sosyal yaşamının ayrılmaz bir parçasıdır.",
        "Bilgisayar bilimi, bilgisayarların ve hesaplama süreçlerinin teori, tasarım, geliştirme ve uygulamalarıyla ilgilenen bir bilim dalıdır. Algoritmalar, veri yapıları, programlama dilleri ve yapay zeka bu alanın temel konularındandır. Günümüzde derin öğrenme ve büyük dil modelleri gibi yapay zeka teknolojileri hızla gelişmektedir.",
        "Matematik, sayılar, yapılar, uzay ve değişim gibi soyut kavramları inceleyen temel bir bilim dalıdır. Cebir, geometri, analiz ve olasılık teorisi matematiğin ana dalları arasında yer alır. Matematik, fizikten mühendisliğe, ekonomiden bilgisayar bilimine kadar birçok alanda temel araç olarak kullanılmaktadır.",
        "Osmanlı İmparatorluğu, 1299 yılında Osman Gazi tarafından kurulan ve altı yüz yılı aşkın süre hüküm süren büyük bir devlettir. İmparatorluk, en geniş sınırlarına Sultan Süleyman döneminde ulaşmıştır. Üç kıtaya yayılan devlet, farklı din ve kültürden insanları bir arada barındırmıştır.",
        "Yapay zeka, makinelerin insan zekasını taklit etmesini sağlayan bilgisayar bilimi dalıdır. Doğal dil işleme, makine öğrenimi, derin öğrenme ve sinir ağları yapay zekanın temel alt alanlarıdır. Büyük dil modelleri, milyarlarca parametre ile eğitilerek insan benzeri metin üretebilmektedir.",
        "Atatürk, Türkiye Cumhuriyeti'nin kurucusu ve ilk cumhurbaşkanıdır. 1881 yılında Selanik'te doğmuştur. Kurtuluş Savaşı'nı başarıyla yöneterek modern Türkiye'nin temellerini atmıştır. Eğitim, hukuk ve toplumsal alanda gerçekleştirdiği devrimlerle ülkeyi çağdaş medeniyetler seviyesine taşımayı hedeflemiştir.",
        "Python, yüksek seviyeli ve genel amaçlı bir programlama dilidir. Okunabilirliği ve basit sözdizimi ile bilinir. Veri bilimi, yapay zeka, web geliştirme ve otomasyon gibi birçok alanda yaygın olarak kullanılmaktadır. Python'un zengin kütüphane ekosistemi, geliştiricilere güçlü araçlar sunar.",
        "Karadeniz Bölgesi, Türkiye'nin kuzeyinde yer alan ve yoğun yeşil örtüsüyle bilinen bir coğrafi bölgedir. Bölge, bol yağış alması nedeniyle çay ve fındık tarımı için uygundur. Trabzon, Rize, Artvin ve Sinop bölgenin önemli şehirleri arasındadır. Bölgenin kendine özgü mutfağı, müziği ve kültürel gelenekleri vardır.",
        "Doğal dil işleme, bilgisayarların insan dilini anlamasını ve üretmesini sağlayan bir yapay zeka dalıdır. Metin sınıflandırma, duygu analizi, makine çevirisi ve soru cevaplama bu alanın temel uygulamalarıdır. Transformer mimarisi, doğal dil işlemede devrim yaratmış ve modern dil modellerinin temelini oluşturmuştur.",
        "Kapadokya, Türkiye'nin Nevşehir ili sınırları içinde yer alan doğal ve tarihi bir bölgedir. Peri bacaları olarak bilinen ilginç kaya oluşumları, yeraltı şehirleri ve kaya kiliseleriyle ünlüdür. Bölge, sıcak hava balonu turları ve eşsiz manzaralarıyla dünya çapında turistleri kendine çekmektedir.",
        "Edebiyat, duygu ve düşüncelerin yazılı veya sözlü olarak estetik biçimde ifade edilmesidir. Roman, hikâye, şiir, deneme ve tiyatro edebiyatın ana türleridir. Türk edebiyatı, Divan edebiyatından modern Türk edebiyatına kadar zengin bir geleneğe sahiptir. Nazım Hikmet, Orhan Pamuk ve Yaşar Kemal Türk edebiyatının önemli temsilcilerindendir.",
        "Fizik, madde ve enerjinin doğasını, hareketini ve davranışını inceleyen temel bir doğa bilimidir. Mekanik, elektromanyetizma, termodinamik, optik ve kuantum fiziği bu bilimin başlıca dallarıdır. Einstein'ın görelilik teorisi ve kuantum mekaniği modern fiziğin iki temel direğidir.",
        "Ege Bölgesi, Türkiye'nin batısında yer alır ve Ege Denizi kıyılarıyla bilinir. İzmir, bölgenin en büyük şehri ve Türkiye'nin üçüncü büyük kentidir. Zeytinyağı, üzüm ve incir bölgenin önemli tarım ürünleridir. Efes antik kenti, Pamukkale travertenleri ve Bergama akropolü bölgenin önemli turistik merkezleridir.",
    ]

    output_file = os.path.join(output_dir, "sample_tr.jsonl")
    txt_file = os.path.join(output_dir, "tokenizer_train.txt")

    with open(output_file, "w", encoding="utf-8") as f_jsonl, \
         open(txt_file, "w", encoding="utf-8") as f_txt:
        for i in range(num_samples):
            text = sample_texts[i % len(sample_texts)]
            # Biraz çeşitlilik ekle
            if i > len(sample_texts):
                text = text + " " + sample_texts[random.randint(0, len(sample_texts)-1)]

            doc = {
                "text": text,
                "source": "sample",
                "word_count": len(text.split()),
            }
            json.dump(doc, f_jsonl, ensure_ascii=False)
            f_jsonl.write("\n")
            f_txt.write(text + "\n")

    print(f"✅ Örnek veri oluşturuldu:")
    print(f"   {output_file} ({num_samples} döküman)")
    print(f"   {txt_file} (tokenizer eğitim verisi)")

    return output_file, txt_file


# ─────────────────────────────────────────────────
# 3. Tokenizer Eğitimi
# ─────────────────────────────────────────────────

def train_tokenizer_from_data(
    input_file: str,
    model_prefix: str = "toprak_tokenizer",
    vocab_size: int = 32_000,
):
    """
    Hazırlanan veri üzerinde SentencePiece tokenizer eğit.

    Args:
        input_file: Düz metin dosyası (satır satır) veya JSONL
        model_prefix: Çıktı model prefix'i
        vocab_size: Kelime dağarcığı büyüklüğü
    """
    import sentencepiece as spm

    # Eğer JSONL ise önce düz metne dönüştür
    if input_file.endswith(".jsonl"):
        txt_file = input_file.replace(".jsonl", "_tokenizer.txt")
        print(f"   JSONL → TXT dönüştürülüyor...")
        with open(input_file, "r", encoding="utf-8") as fin, \
             open(txt_file, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "").strip()
                    if text:
                        fout.write(text + "\n")
                except json.JSONDecodeError:
                    continue
        input_file = txt_file

    print(f"\n📝 Tokenizer eğitiliyor...")
    print(f"   Girdi: {input_file}")
    print(f"   Vocab size: {vocab_size:,}")
    print(f"   Model type: BPE")

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9999,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<sep>", "<cls>", "<mask>"],
        normalization_rule_name="nfkc",
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_digits=True,
        byte_fallback=True,
        num_threads=os.cpu_count() or 4,
        input_sentence_size=5_000_000,  # Büyük veri setleri için
        shuffle_input_sentence=True,
    )

    print(f"\n✅ Tokenizer eğitildi:")
    print(f"   Model: {model_prefix}.model")
    print(f"   Vocab: {model_prefix}.vocab")

    # Doğrulama
    verify_tokenizer(f"{model_prefix}.model")

    return f"{model_prefix}.model"


def verify_tokenizer(model_path: str):
    """Tokenizer'ı Türkçe örnek cümlelerle doğrula."""
    from model.tokenizer import ToprakTokenizer

    tokenizer = ToprakTokenizer(model_path)

    test_sentences = [
        "Merhaba dünya!",
        "Türkiye'nin başkenti Ankara'dır.",
        "Yapay zeka ile doğal dil işleme çalışıyoruz.",
        "İstanbul Boğazı Avrupa ve Asya'yı birbirinden ayırır.",
        "Çiçeklerin güzelliği ve öğretmenliğin şefkati.",  # ç, ö, ğ, ş testi
    ]

    print(f"\n🔍 Tokenizer Doğrulama (vocab={tokenizer.vocab_size:,}):")
    for sent in test_sentences:
        ids = tokenizer.encode(sent, add_bos=False, add_eos=False)
        decoded = tokenizer.decode(ids)
        tokens = [tokenizer.id_to_token(i) for i in ids]
        print(f"\n   Girdi:   \"{sent}\"")
        print(f"   Tokenlar: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print(f"   ID sayısı: {len(ids)}")
        print(f"   Decode:  \"{decoded}\"")
        match = "✅" if decoded.strip() == sent.strip() else "⚠️"
        print(f"   Eşleşme: {match}")


# ─────────────────────────────────────────────────
# 4. Eğitim Verisi Hazırlama
# ─────────────────────────────────────────────────

def prepare_training_data(
    data_dir: str = "data_cache",
    output_dir: str = "data_cache/clean",
    tokenizer_model: str = "toprak_tokenizer.model",
    train_ratio: float = 0.95,
):
    """
    Verileri temizle ve train/eval olarak böl.
    """
    from data.cleaner import ToprakCleaner

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    cleaner = ToprakCleaner(min_words=30)

    print(f"\n🧹 Veri temizleniyor...")

    # JSONL dosyalarını temizle ve böl
    for filename in os.listdir(data_dir):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(data_dir, filename)
        train_file = os.path.join(train_dir, filename)
        eval_file = os.path.join(eval_dir, filename)

        print(f"   İşleniyor: {filename}")

        with open(filepath, "r", encoding="utf-8") as fin, \
             open(train_file, "w", encoding="utf-8") as f_train, \
             open(eval_file, "w", encoding="utf-8") as f_eval:

            for line in fin:
                try:
                    doc = json.loads(line)
                    cleaned = cleaner.clean_text(doc.get("text", ""))
                    if cleaned:
                        doc["text"] = cleaned
                        doc["word_count"] = len(cleaned.split())
                        out = json.dumps(doc, ensure_ascii=False) + "\n"

                        # Train/eval split
                        if random.random() < train_ratio:
                            f_train.write(out)
                        else:
                            f_eval.write(out)
                except json.JSONDecodeError:
                    continue

    cleaner.print_stats()

    # Boyut bilgisi
    train_size = sum(
        os.path.getsize(os.path.join(train_dir, f))
        for f in os.listdir(train_dir) if f.endswith(".jsonl")
    )
    eval_size = sum(
        os.path.getsize(os.path.join(eval_dir, f))
        for f in os.listdir(eval_dir) if f.endswith(".jsonl")
    )
    print(f"\n✅ Veri hazır:")
    print(f"   Train: {train_dir} ({train_size/1e6:.1f} MB)")
    print(f"   Eval:  {eval_dir} ({eval_size/1e6:.1f} MB)")


# ─────────────────────────────────────────────────
# 5. Ana Pipeline
# ─────────────────────────────────────────────────

def run_full_pipeline(args):
    """Tüm veri hazırlama pipeline'ını çalıştır."""

    print("🌱 Toprak — Veri Hazırlama Pipeline")
    print("=" * 60)

    step = args.step

    # Adım 1: Veri toplama
    if step in ("all", "download"):
        if args.use_sample:
            print("\n📦 Örnek veri oluşturuluyor (test amaçlı)...")
            create_sample_data(args.data_dir, num_samples=args.sample_count)
        else:
            print("\n📦 Wikipedia verisi indiriliyor...")
            download_wikipedia(args.data_dir, max_articles=args.max_articles)

    # Adım 2: Tokenizer eğitimi
    if step in ("all", "tokenizer"):
        # Tokenizer eğitim verisini hazırla
        data_file = None

        # Önce tokenizer_train.txt var mı kontrol et
        txt_path = os.path.join(args.data_dir, "tokenizer_train.txt")
        if os.path.exists(txt_path):
            data_file = txt_path
        else:
            # JSONL dosyalarından oluştur
            jsonl_files = [f for f in os.listdir(args.data_dir) if f.endswith(".jsonl")]
            if jsonl_files:
                data_file = os.path.join(args.data_dir, jsonl_files[0])
            else:
                print("❌ Tokenizer için eğitim verisi bulunamadı!")
                print("   Önce 'download' adımını çalıştırın.")
                return

        train_tokenizer_from_data(
            data_file,
            model_prefix=args.tokenizer_prefix,
            vocab_size=args.vocab_size,
        )

    # Adım 3: Eğitim verisi hazırlama
    if step in ("all", "prepare"):
        prepare_training_data(
            data_dir=args.data_dir,
            output_dir=os.path.join(args.data_dir, "clean"),
            tokenizer_model=f"{args.tokenizer_prefix}.model",
        )

    print("\n" + "=" * 60)
    print("✅ Pipeline tamamlandı!")

    if step == "all":
        print(f"\n📋 Sonraki adım — eğitimi başlat:")
        print(f"   python training/train.py \\")
        print(f"     --model-size small \\")
        print(f"     --data-dir {args.data_dir}/clean/train \\")
        print(f"     --eval-data-dir {args.data_dir}/clean/eval \\")
        print(f"     --tokenizer {args.tokenizer_prefix}.model")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🌱 Toprak — Veri Hazırlama Pipeline"
    )

    parser.add_argument(
        "--step", type=str, default="all",
        choices=["all", "download", "tokenizer", "prepare"],
        help="Çalıştırılacak adım"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data_cache",
        help="Veri dizini"
    )
    parser.add_argument(
        "--use-sample", action="store_true",
        help="Wikipedia yerine örnek veri kullan (hızlı test için)"
    )
    parser.add_argument(
        "--sample-count", type=int, default=5000,
        help="Örnek veri miktarı"
    )
    parser.add_argument(
        "--max-articles", type=int, default=None,
        help="İndirilecek maks makale sayısı"
    )
    parser.add_argument(
        "--tokenizer-prefix", type=str, default="toprak_tokenizer",
        help="Tokenizer model prefix"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32000,
        help="Tokenizer vocab size"
    )

    args = parser.parse_args()
    run_full_pipeline(args)
