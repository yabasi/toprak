# Toprak Eval v1 Seed Set

Bu dizindeki JSONL örnekleri Toprak projesi için özgün olarak hazırlanmıştır ve
projenin kökündeki Apache-2.0 lisansı altında yayımlanır. Haricî bir benchmarktan
kopyalanmış soru veya yanıt içermez.

Seed set, eğitim sırasında hızlı regresyon tespiti içindir. Örnek sayısı ve konu
çeşitliliği akademik sonuç, genel yetenek iddiası veya güvenlik sertifikası için
yeterli değildir. Raporlar yalnız aynı suite sürümü, tokenizer ve benchmark
SHA-256 değerleriyle karşılaştırılır.

Yeni örnek eklerken:

- tüm dizinde benzersiz bir `id` kullanın;
- cevabı ve görev tipini `EVALUATION.md` şemasına göre doğrulayın;
- örneği eğitim corpusundan uzak tutmak için contamination taramasını çalıştırın;
- haricî veri kullanılıyorsa kaynak, lisans ve kullanım koşullarını burada belgeleyin.
