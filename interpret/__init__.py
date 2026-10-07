# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Morfoloji Mikroskobu (yorumlanabilirlik araç takımı)

Modelin iç temsillerinde Türkçe morfolojinin (kök/ek ayrımı, ünlü uyumu,
ünsüz benzeşmesi, ek türleri) ne kadar ve hangi katmanda kodlandığını
ölçmek için araçlar:

- activations: ileri besleme kancalarıyla (forward hook) katman aktivasyonu
  ve MoE yönlendirme kaydı
- features: BPE parçaları üzerinden token düzeyinde dilbilimsel etiketler
- probes: doğrusal sondalar (probe) + kontrol görevi / seçicilik
- sae: TopK seyrek otokodlayıcı (SAE) ve özellik-etiket ilişkisi
- patching: yön/özellik silme (ablation) ve aktivasyon yamalama
- report / cli: JSON + bağımsız (self-contained) HTML raporlar

Ayrıntılar için INTERPRETABILITY.md dosyasına bakın.
"""
