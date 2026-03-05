# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Türkçe Veri Kaynakları
Web crawler için hedef URL'ler ve kaynak tanımları.
"""

# Kaynak kategorileri ve URL'ler
SOURCES = {
    "wikipedia": {
        "name": "Türkçe Wikipedia",
        "type": "ansiklopedi",
        "priority": "yüksek",
        "urls": [
            "https://tr.wikipedia.org/wiki/Özel:Rastgele",
        ],
        "notes": "Wikipedia dump tercih edilir: https://dumps.wikimedia.org/trwiki/",
        "estimated_size": "~2GB",
    },
    "haber": {
        "name": "Türkçe Haber Siteleri",
        "type": "haber",
        "priority": "yüksek",
        "urls": [
            "https://www.hurriyet.com.tr",
            "https://www.milliyet.com.tr",
            "https://www.haberturk.com",
            "https://www.ntv.com.tr",
            "https://www.bbc.com/turkce",
            "https://www.dw.com/tr",
            "https://tr.euronews.com",
            "https://www.trthaber.com",
        ],
        "estimated_size": "~5GB",
    },
    "kamu": {
        "name": "Kamu Kurumları",
        "type": "resmi",
        "priority": "orta",
        "urls": [
            "https://www.resmigazete.gov.tr",
            "https://www.tbmm.gov.tr",
            "https://www.meb.gov.tr",
        ],
        "estimated_size": "~1GB",
    },
    "edebiyat": {
        "name": "Türkçe Edebiyat",
        "type": "edebiyat",
        "priority": "orta",
        "urls": [
            "https://www.antoloji.com",
            "https://www.siir.gen.tr",
        ],
        "estimated_size": "~500MB",
    },
    "akademik": {
        "name": "Akademik Kaynaklar",
        "type": "akademik",
        "priority": "orta",
        "urls": [
            "https://dergipark.org.tr",
            "https://tez.yok.gov.tr",
        ],
        "estimated_size": "~2GB",
    },
}

# Wikipedia dump URL (en büyük ve en temiz kaynak)
WIKIPEDIA_DUMP_URL = "https://dumps.wikimedia.org/trwiki/latest/trwiki-latest-pages-articles.xml.bz2"

# Crawler'ın atlayacağı URL pattern'leri
SKIP_PATTERNS = [
    r".*\.(jpg|jpeg|png|gif|svg|ico|css|js|woff|ttf|pdf|zip|rar)$",
    r".*/login.*",
    r".*/register.*",
    r".*/admin.*",
    r".*\?.*page=\d+$",
]

# Minimum metin uzunluğu (kelime sayısı)
MIN_WORD_COUNT = 50

# Crawler bekleme süresi (saniye)
CRAWL_DELAY = 1.0

# Maksimum eşzamanlı bağlantı
MAX_CONCURRENT = 5
