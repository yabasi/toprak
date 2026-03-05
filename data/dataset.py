# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

import json
import os
import random
from typing import List, Optional

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
        for text in documents:
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            all_tokens.extend(tokens)

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
    """

    def __init__(self, bin_file: str, max_seq_len: int = 512):
        """
        Args:
            bin_file: Tokenize edilmiş veri (.bin dosyası, int32 numpy array)
            max_seq_len: Maksimum sequence uzunluğu
        """
        import numpy as np

        self.max_seq_len = max_seq_len
        self.data = np.memmap(bin_file, dtype=np.int32, mode="r")
        print(f"Pre-tokenized veri yüklendi: {len(self.data):,} token")

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.max_seq_len

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1

        chunk = self.data[start:end].astype(int)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return {"input_ids": x, "labels": y}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """DataLoader oluştur."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
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
