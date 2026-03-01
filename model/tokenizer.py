"""
Toprak — BPE Tokenizer
SentencePiece tabanlı Türkçe BPE tokenizer eğitimi ve HuggingFace uyumlu wrapper.
"""

import os
import sentencepiece as spm
from typing import List, Optional, Union


def train_tokenizer(
    input_file: str,
    model_prefix: str = "toprak_tokenizer",
    vocab_size: int = 32_000,
    model_type: str = "bpe",
    character_coverage: float = 0.9999,
):
    """
    SentencePiece BPE tokenizer eğitimi.

    Args:
        input_file: Eğitim verisi dosyası (düz metin, satır satır)
        model_prefix: Çıktı model dosya adı prefix'i
        vocab_size: Kelime dağarcığı büyüklüğü
        model_type: 'bpe' veya 'unigram'
        character_coverage: Karakter kapsama oranı (Türkçe için yüksek tutulmalı)
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<sep>", "<cls>", "<mask>"],
        normalization_rule_name="nfkc",
        # Türkçe morfolojisi için
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_digits=True,
        byte_fallback=True,
        num_threads=os.cpu_count() or 4,
    )
    print(f"Tokenizer eğitildi: {model_prefix}.model ({vocab_size} vocab)")


class ToprakTokenizer:
    """
    Toprak BPE Tokenizer — SentencePiece wrapper.

    Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) doğru şekilde işler.
    HuggingFace PreTrainedTokenizer ile uyumlu arayüz.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: SentencePiece .model dosyasının yolu
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model bulunamadı: {model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # Özel token ID'leri
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        self.vocab_size = self.sp.get_piece_size()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Metni token ID listesine dönüştür.

        Args:
            text: Girdi metni
            add_bos: Başlangıç tokeni ekle
            add_eos: Bitiş tokeni ekle
        Returns:
            Token ID listesi
        """
        ids = self.sp.encode(text, out_type=int)

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Token ID listesini metne dönüştür.

        Args:
            ids: Token ID listesi
        Returns:
            Decoded metin
        """
        # Özel tokenleri filtrele
        filtered = [
            i for i in ids
            if i not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        ]
        return self.sp.decode(filtered)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> dict:
        """
        Birden fazla metni batch olarak encode et.

        Args:
            texts: Metin listesi
            max_length: Maksimum token uzunluğu
            padding: Kısa sequence'ları pad'le
            add_bos: Başlangıç tokeni ekle
            add_eos: Bitiş tokeni ekle

        Returns:
            dict: {'input_ids': List[List[int]], 'attention_mask': List[List[int]]}
        """
        encoded = [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

        # Truncation
        if max_length:
            encoded = [ids[:max_length] for ids in encoded]

        # Padding
        if padding:
            max_len = max(len(ids) for ids in encoded)
            attention_mask = []
            for i, ids in enumerate(encoded):
                pad_len = max_len - len(ids)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
                encoded[i] = ids + [self.pad_token_id] * pad_len
        else:
            attention_mask = [[1] * len(ids) for ids in encoded]

        return {
            "input_ids": encoded,
            "attention_mask": attention_mask,
        }

    def get_vocab_size(self) -> int:
        """Kelime dağarcığı büyüklüğünü döndür."""
        return self.vocab_size

    def id_to_token(self, token_id: int) -> str:
        """Token ID'den token string'ine dönüştür."""
        return self.sp.id_to_piece(token_id)

    def token_to_id(self, token: str) -> int:
        """Token string'inden token ID'ye dönüştür."""
        return self.sp.piece_to_id(token)

    def __len__(self) -> int:
        return self.vocab_size
