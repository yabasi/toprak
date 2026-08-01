# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import json
import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ToprakDataset(Dataset):
    """
    Türkçe metin dataset'i.

    Tokenize edilmiş veriyi sabit uzunluklu bloklara böler
    ve (input, target) çiftleri olarak sunar.

    Döküman seviyesinde karıştırma desteği:
    - Metinler önce döküman olarak yüklenir
    - Karıştırılır (shuffle)
    - Ardından tek token dizisine birleştirilir
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_seq_len: int = 512,
        split: str = "train",
        shuffle_docs: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Temizlenmiş JSONL dosyalarının bulunduğu dizin
            tokenizer: ToprakTokenizer instance
            max_seq_len: Maksimum sequence uzunluğu
            split: 'train' veya 'eval'
            shuffle_docs: Dökümanları karıştır (train için önerilir)
            seed: Rastgelelik seed'i
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split

        # Tüm metinleri yükle ve tokenize et
        print(f"Veri yükleniyor ({split})...")
        self.tokens = self._load_and_tokenize(data_dir, shuffle_docs, seed)
        print(f"  Toplam token: {len(self.tokens):,}")
        print(f"  Toplam blok: {len(self):,}")

    def _load_and_tokenize(self, data_dir: str, shuffle_docs: bool, seed: int) -> List[int]:
        """Tüm metinleri yükle, karıştır ve tokenize et."""
        # 1. Dökümanları ayrı ayrı yükle
        documents = []

        for filename in sorted(os.listdir(data_dir)):
            filepath = os.path.join(data_dir, filename)

            if filename.endswith(".jsonl"):
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            doc = json.loads(line)
                            text = doc.get("text", "")
                            if text:
                                documents.append(text)
                        except json.JSONDecodeError:
                            continue

            elif filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        if text:
                            documents.append(text)

        print(f"  Döküman sayısı: {len(documents):,}")

        # 2. Dökümanları karıştır
        if shuffle_docs:
            rng = random.Random(seed)
            rng.shuffle(documents)
            print(f"  ✓ Dökümanlar karıştırıldı (seed={seed})")

        # 3. Tokenize et ve birleştir
        all_tokens = []
        total_docs = len(documents)
        for i, text in enumerate(documents, 1):
            if i % 100 == 0 or i == total_docs:
                print(f"  Tokenize ediliyor: {i}/{total_docs}", end="\r")

            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            all_tokens.extend(tokens)
        print()

        return all_tokens

    def __len__(self) -> int:
        """Kullanılabilir blok sayısı."""
        if len(self.tokens) <= self.max_seq_len:
            return 0
        return (len(self.tokens) - 1) // self.max_seq_len

    def __getitem__(self, idx: int) -> dict:
        """
        Bir blok döndür.

        Returns:
            dict: {
                'input_ids': tensor (max_seq_len,),
                'labels': tensor (max_seq_len,)
            }
        """
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1  # +1 çünkü labels input'tan bir kaydırılmış

        chunk = self.tokens[start:end]

        # Yeterli token yoksa pad'le
        if len(chunk) < self.max_seq_len + 1:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_seq_len + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)  # input
        y = torch.tensor(chunk[1:], dtype=torch.long)    # target (bir kaydırılmış)

        return {"input_ids": x, "labels": y}


class ToprakPreTokenizedDataset(Dataset):
    """
    Önceden tokenize edilmiş binary veri için Dataset.

    Büyük veri setleri için: veriyi önceden tokenize edip .bin dosyasına kaydet,
    ardından memory-mapped olarak yükle.

    Eski versiyon (geriye uyumlu) — int32, tek dosya.
    Yeni multi-shard + uint16 desteği için: ToprakShardDataset
    """

    def __init__(self, bin_file: str, max_seq_len: int = 512, dtype=np.int32):
        self.max_seq_len = max_seq_len
        self.data = np.memmap(bin_file, dtype=dtype, mode="r")
        print(f"Pre-tokenized veri yüklendi: {len(self.data):,} token")

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.max_seq_len

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1

        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return {"input_ids": x, "labels": y}


class ToprakShardDataset(Dataset):
    """
    Manifest tabanlı, çoklu .bin shard'ı tek bir lojik dataset olarak sunar.

    `scripts/pretokenize.py` tarafından üretilen format:
        data_bin/
            manifest.json
            train_00000.bin
            train_00001.bin
            ...
            eval_00000.bin

    Tüm shard'lar memory-mapped (numpy memmap) → RAM tüketmez.
    Global index → (shard, lokal offset) mapping ile O(1) blok erişimi.

    Args:
        bin_dir: shard dizini (manifest.json içermeli)
        split: "train" veya "eval"
        max_seq_len: blok uzunluğu (eğitim seq_len ile eşleşmeli)
        dtype: Opsiyonel numpy dtype doğrulaması; asıl dtype manifestten okunur
        shuffle_shards: epoch başında shard'ları karıştır (curriculum'u bozar; default False)
        seed: rng seed
    """

    def __init__(
        self,
        bin_dir: str,
        split: str = "train",
        max_seq_len: int = 2048,
        dtype=None,
        shuffle_shards: bool = False,
        seed: int = 42,
        expected_vocab_size: Optional[int] = None,
    ):
        manifest_path = os.path.join(bin_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest bulunamadı: {manifest_path}. "
                f"Önce scripts/pretokenize.py çalıştırın."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if split not in ("train", "eval"):
            raise ValueError(f"split 'train' veya 'eval' olmalı, '{split}' verildi")

        if split not in manifest:
            raise ValueError(f"Manifest '{split}' bölümü içermiyor: {manifest_path}")

        dtype_map = {
            "uint16": np.dtype(np.uint16),
            "int32": np.dtype(np.int32),
            "int64": np.dtype(np.int64),
        }
        manifest_dtype_name = manifest.get("dtype")
        if manifest_dtype_name not in dtype_map:
            raise ValueError(
                f"Manifest dtype desteklenmiyor: {manifest_dtype_name!r}. "
                f"Desteklenenler: {', '.join(dtype_map)}"
            )
        manifest_dtype = dtype_map[manifest_dtype_name]
        if dtype is not None and np.dtype(dtype) != manifest_dtype:
            raise ValueError(
                f"dtype uyuşmazlığı: manifest={manifest_dtype}, istenen={np.dtype(dtype)}"
            )

        manifest_vocab_size = manifest.get("tokenizer_vocab_size")
        if expected_vocab_size is not None and manifest_vocab_size != expected_vocab_size:
            raise ValueError(
                "Tokenizer vocab uyuşmazlığı: "
                f"manifest={manifest_vocab_size}, model/tokenizer={expected_vocab_size}"
            )

        meta = manifest[split]
        self.split = split
        self.max_seq_len = max_seq_len
        self.dtype = manifest_dtype
        self.bin_dir = os.path.realpath(bin_dir)
        self.curriculum = bool(manifest.get("curriculum", False))

        # Shard'ları yükle (memmap)
        shard_entries = list(meta.get("shards", []))
        if shuffle_shards and split == "train":
            rng = random.Random(seed)
            rng.shuffle(shard_entries)

        self.shards: List[np.memmap] = []
        self.shard_paths: List[str] = []
        # Her shard'ın "kullanılabilir blok sayısı" (son artık atılır)
        self.shard_blocks: List[int] = []
        # Kümülatif blok sayısı — binary search ile global→lokal index
        self.cum_blocks: List[int] = []
        running = 0
        skipped_shards = 0
        file_token_total = 0

        for entry in shard_entries:
            relative_path = entry.get("path")
            if not relative_path:
                raise ValueError(f"Geçersiz shard girdisi: {entry!r}")
            path = os.path.realpath(os.path.join(self.bin_dir, relative_path))
            if os.path.commonpath([self.bin_dir, path]) != self.bin_dir:
                raise ValueError(f"Shard yolu bin dizini dışına çıkıyor: {relative_path}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Manifestteki shard bulunamadı: {path}")
            if os.path.getsize(path) % manifest_dtype.itemsize != 0:
                raise ValueError(
                    f"Shard byte boyutu dtype ile uyumsuz: {path} "
                    f"({os.path.getsize(path)} byte, dtype={manifest_dtype})"
                )
            arr = np.memmap(path, dtype=manifest_dtype, mode="r")
            n_tokens = len(arr)
            file_token_total += n_tokens
            declared_tokens = entry.get("tokens")
            if declared_tokens != n_tokens:
                raise ValueError(
                    f"Shard token sayısı uyuşmuyor: {relative_path}; "
                    f"manifest={declared_tokens}, dosya={n_tokens}"
                )
            # Bir blok = max_seq_len + 1 token (input + 1 kaydırılmış label)
            n_blocks = max(0, (n_tokens - 1) // max_seq_len)
            if n_blocks == 0:
                skipped_shards += 1
                continue
            self.shards.append(arr)
            self.shard_paths.append(path)
            self.shard_blocks.append(n_blocks)
            running += n_blocks
            self.cum_blocks.append(running)

        if not self.shards:
            raise RuntimeError(
                f"Hiç kullanılabilir shard bulunamadı ({bin_dir}, split={split})"
            )

        self.total_tokens = int(file_token_total)
        self.total_blocks = running

        declared_total = meta.get("total_tokens")
        if declared_total != self.total_tokens:
            raise ValueError(
                f"{split} toplam token sayısı uyuşmuyor: "
                f"manifest={declared_total}, dosyalar={self.total_tokens}"
            )

        print(
            f"📦 ToprakShardDataset[{split}] "
            f"hazır: {len(self.shards)} shard, "
            f"{self.total_tokens:,} token, {self.total_blocks:,} blok "
            f"(seq_len={max_seq_len})"
        )
        if skipped_shards:
            print(f"   ⚠ {skipped_shards} shard atlandı")

    def __len__(self) -> int:
        return self.total_blocks

    def _locate(self, global_idx: int) -> tuple:
        """Global block index → (shard_idx, local_block_idx)"""
        # cum_blocks artan; bisect_right uygun.
        import bisect
        shard_idx = bisect.bisect_right(self.cum_blocks, global_idx)
        if shard_idx == 0:
            local_block = global_idx
        else:
            local_block = global_idx - self.cum_blocks[shard_idx - 1]
        return shard_idx, local_block

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.total_blocks:
            raise IndexError(idx)
        shard_idx, local_block = self._locate(idx)
        arr = self.shards[shard_idx]
        start = local_block * self.max_seq_len
        end = start + self.max_seq_len + 1
        chunk = np.asarray(arr[start:end], dtype=np.int64)
        # Edge case: son blok shard sonunda 1 eksik kalabilir
        if len(chunk) < self.max_seq_len + 1:
            pad_n = (self.max_seq_len + 1) - len(chunk)
            chunk = np.concatenate([chunk, np.zeros(pad_n, dtype=np.int64)])
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return {"input_ids": x, "labels": y}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    seed: Optional[int] = None,
) -> DataLoader:
    """DataLoader oluştur."""
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )


def tokenize_and_save(
    data_dir: str,
    tokenizer,
    output_file: str,
    max_files: Optional[int] = None,
):
    """
    Verileri tokenize edip binary dosyaya kaydet.
    Büyük veri setleri için önerilir.
    """
    import numpy as np

    all_tokens = []
    file_count = 0

    for filename in sorted(os.listdir(data_dir)):
        if max_files and file_count >= max_files:
            break

        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "")
                        if text:
                            tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
                            all_tokens.extend(tokens)
                    except json.JSONDecodeError:
                        continue
            file_count += 1

    # NumPy array olarak kaydet
    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(output_file)
    print(f"✓ {len(arr):,} token kaydedildi: {output_file}")
    return output_file
