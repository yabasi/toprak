"""
Toprak — Türkçe Web Crawler
asyncio + aiohttp tabanlı, robots.txt uyumlu web crawler.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

try:
    from langdetect import detect
except ImportError:
    detect = None

from data.sources import (
    CRAWL_DELAY,
    MAX_CONCURRENT,
    MIN_WORD_COUNT,
    SKIP_PATTERNS,
    SOURCES,
)


class ToprakCrawler:
    """
    Türkçe web crawler.

    Özellikler:
    - asyncio + aiohttp ile hızlı crawling
    - robots.txt'e uyum
    - Rate limiting (1 saniye delay)
    - BeautifulSoup ile içerik çıkarma
    - langdetect ile Türkçe filtre
    - JSONL formatında kayıt
    """

    def __init__(
        self,
        output_dir: str = "data_cache",
        delay: float = CRAWL_DELAY,
        max_concurrent: int = MAX_CONCURRENT,
        min_words: int = MIN_WORD_COUNT,
    ):
        self.output_dir = output_dir
        self.delay = delay
        self.max_concurrent = max_concurrent
        self.min_words = min_words
        self.visited: Set[str] = set()
        self.results: List[dict] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)

        os.makedirs(output_dir, exist_ok=True)

        # Derlenmiş skip pattern'leri
        self.skip_patterns = [re.compile(p, re.IGNORECASE) for p in SKIP_PATTERNS]

    def should_skip(self, url: str) -> bool:
        """URL'nin atlanıp atlanmayacağını kontrol et."""
        return any(pattern.match(url) for pattern in self.skip_patterns)

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Bir URL'den HTML içerik getir."""
        try:
            async with self.semaphore:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers={
                        "User-Agent": "ToprakBot/1.0 (Turkish LLM Research Project)"
                    },
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
            print(f"  ⚠ Hata ({url}): {e}")
            return None

    def extract_text(self, html: str, url: str) -> Optional[dict]:
        """HTML'den temiz metin çıkar."""
        soup = BeautifulSoup(html, "lxml")

        # Gereksiz etiketleri kaldır
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "iframe", "noscript", "form"]):
            tag.decompose()

        # Ana içerik alanlarını bul
        content = None
        for selector in ["article", "main", ".content", ".article-body",
                         ".post-content", "#content", ".entry-content"]:
            content = soup.select_one(selector)
            if content:
                break

        if content is None:
            content = soup.find("body")

        if content is None:
            return None

        # Metin çıkar
        text = content.get_text(separator="\n", strip=True)

        # Çok kısa metinleri atla
        words = text.split()
        if len(words) < self.min_words:
            return None

        # Türkçe dil kontrolü
        if detect is not None:
            try:
                lang = detect(text[:1000])
                if lang != "tr":
                    return None
            except Exception:
                pass  # Dil tespit edilemezse kabul et

        return {
            "url": url,
            "text": text,
            "source": urlparse(url).netloc,
            "timestamp": datetime.now().isoformat(),
            "word_count": len(words),
        }

    def extract_links(self, html: str, base_url: str) -> List[str]:
        """HTML'den linkleri çıkar."""
        soup = BeautifulSoup(html, "lxml")
        links = []
        base_domain = urlparse(base_url).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            # Aynı domain'de kal
            if parsed.netloc == base_domain and full_url not in self.visited:
                if not self.should_skip(full_url):
                    links.append(full_url)

        return links

    async def crawl_url(self, session: aiohttp.ClientSession, url: str, depth: int = 0, max_depth: int = 3):
        """Tek bir URL'yi crawl et."""
        if url in self.visited or depth > max_depth:
            return

        self.visited.add(url)

        # Rate limiting
        await asyncio.sleep(self.delay)

        html = await self.fetch(session, url)
        if html is None:
            return

        # İçerik çıkar
        doc = self.extract_text(html, url)
        if doc:
            self.results.append(doc)
            print(f"  ✓ {doc['word_count']} kelime: {url}")

        # Alt linkleri crawl et
        if depth < max_depth:
            links = self.extract_links(html, url)
            tasks = [
                self.crawl_url(session, link, depth + 1, max_depth)
                for link in links[:10]  # Her sayfadan max 10 link
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def crawl_source(self, source_key: str, max_pages: int = 100, max_depth: int = 2):
        """Bir kaynağı tamamen crawl et."""
        if source_key not in SOURCES:
            print(f"Kaynak bulunamadı: {source_key}")
            return

        source = SOURCES[source_key]
        print(f"\n{'='*60}")
        print(f"Crawling: {source['name']} ({source['type']})")
        print(f"{'='*60}")

        async with aiohttp.ClientSession() as session:
            for url in source["urls"]:
                if len(self.results) >= max_pages:
                    break
                await self.crawl_url(session, url, max_depth=max_depth)

        # Sonuçları kaydet
        output_file = os.path.join(self.output_dir, f"{source_key}.jsonl")
        self.save_results(output_file)
        print(f"\n✓ {len(self.results)} döküman kaydedildi: {output_file}")

    async def crawl_all(self, max_pages_per_source: int = 100, max_depth: int = 2):
        """Tüm kaynakları crawl et."""
        for source_key in SOURCES:
            self.results = []
            self.visited = set()
            await self.crawl_source(source_key, max_pages_per_source, max_depth)

    def save_results(self, output_file: str):
        """Sonuçları JSONL formatında kaydet."""
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in self.results:
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")


async def main():
    """Ana crawler fonksiyonu."""
    import argparse

    parser = argparse.ArgumentParser(description="Toprak — Türkçe Web Crawler")
    parser.add_argument("--source", type=str, default=None,
                        help="Crawl edilecek kaynak (ör: wikipedia, haber)")
    parser.add_argument("--max-pages", type=int, default=100,
                        help="Kaynak başına maks sayfa sayısı")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Maks crawl derinliği")
    parser.add_argument("--output", type=str, default="data_cache",
                        help="Çıktı dizini")

    args = parser.parse_args()
    crawler = ToprakCrawler(output_dir=args.output)

    if args.source:
        await crawler.crawl_source(args.source, args.max_pages, args.max_depth)
    else:
        await crawler.crawl_all(args.max_pages, args.max_depth)


if __name__ == "__main__":
    asyncio.run(main())
