# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Doğrulama ve Hata Yönetimi
Kullanıcı dostu hata mesajları ile dosya, dizin ve bağımlılık kontrolü.
"""

import os
import sys


class ToprakError(Exception):
    """Toprak projesi için kullanıcı dostu hata sınıfı."""
    pass


def validate_file_exists(filepath: str, description: str, hint: str = "") -> str:
    """
    Dosyanın var olduğunu doğrula.

    Args:
        filepath: Kontrol edilecek dosya yolu
        description: Dosyanın ne olduğu (ör. "Tokenizer model dosyası")
        hint: Dosya yoksa ne yapılması gerektiği

    Returns:
        Mutlak dosya yolu

    Raises:
        ToprakError: Dosya bulunamazsa
    """
    abs_path = os.path.abspath(filepath)
    if not os.path.isfile(abs_path):
        msg = f"\n❌ {description} bulunamadı!\n"
        msg += f"   Aranan yol: {abs_path}\n"
        if hint:
            msg += f"\n💡 Çözüm: {hint}\n"
        raise ToprakError(msg)
    return abs_path


def validate_dir_exists(dirpath: str, description: str, hint: str = "") -> str:
    """
    Dizinin var olduğunu doğrula.

    Args:
        dirpath: Kontrol edilecek dizin yolu
        description: Dizinin ne olduğu
        hint: Dizin yoksa ne yapılması gerektiği

    Returns:
        Mutlak dizin yolu

    Raises:
        ToprakError: Dizin bulunamazsa
    """
    abs_path = os.path.abspath(dirpath)
    if not os.path.isdir(abs_path):
        msg = f"\n❌ {description} bulunamadı!\n"
        msg += f"   Aranan yol: {abs_path}\n"
        if hint:
            msg += f"\n💡 Çözüm: {hint}\n"
        raise ToprakError(msg)
    return abs_path


def validate_dir_has_data(dirpath: str, extensions: list = None, description: str = "Veri dizini") -> list:
    """
    Dizinde belirtilen uzantılara sahip dosya olduğunu doğrula.

    Args:
        dirpath: Kontrol edilecek dizin
        extensions: Kabul edilen uzantılar (ör. [".jsonl", ".txt"])
        description: Dizinin ne olduğu

    Returns:
        Bulunan dosya listesi

    Raises:
        ToprakError: Dizinde uygun dosya yoksa
    """
    if extensions is None:
        extensions = [".jsonl", ".txt"]

    abs_path = validate_dir_exists(dirpath, description,
        hint="Önce veri hazırlama scriptini çalıştırın:\n"
             "   python3 scripts/prepare_data.py --use-sample")

    found_files = []
    for f in os.listdir(abs_path):
        if any(f.endswith(ext) for ext in extensions):
            found_files.append(os.path.join(abs_path, f))

    if not found_files:
        # Alt dizinlere bak ve öneri sun
        subdirs = [d for d in os.listdir(abs_path)
                    if os.path.isdir(os.path.join(abs_path, d))]
        data_in_subdirs = {}
        for subdir in subdirs:
            subpath = os.path.join(abs_path, subdir)
            count = sum(1 for f in os.listdir(subpath)
                        if any(f.endswith(ext) for ext in extensions))
            if count > 0:
                data_in_subdirs[subdir] = count

        msg = f"\n❌ {description} içinde veri dosyası bulunamadı!\n"
        msg += f"   Aranan yol: {abs_path}\n"
        msg += f"   Aranan uzantılar: {', '.join(extensions)}\n"

        if data_in_subdirs:
            msg += f"\n💡 Veri dosyaları alt dizinlerde bulundu:\n"
            for subdir, count in data_in_subdirs.items():
                msg += f"   📁 {subdir}/ → {count} dosya\n"
            msg += f"\n   Doğru dizini belirtin, örneğin:\n"
            first_subdir = list(data_in_subdirs.keys())[0]
            msg += f"   --data-dir {os.path.join(dirpath, first_subdir)}\n"
        else:
            msg += f"\n💡 Çözüm: Önce veri hazırlayın:\n"
            msg += f"   python3 scripts/prepare_data.py --use-sample\n"

        raise ToprakError(msg)

    return found_files


def validate_checkpoint(filepath: str) -> str:
    """Checkpoint dosyasını doğrula."""
    return validate_file_exists(
        filepath,
        description="Model checkpoint dosyası",
        hint="Mevcut checkpoint'leri görmek için:\n"
             "   ls checkpoints/\n"
             "   Eğitilmiş bir model yoksa, önce eğitin:\n"
             "   python3 training/train.py --model-size small"
    )


def validate_tokenizer(filepath: str) -> str:
    """Tokenizer dosyasını doğrula."""
    return validate_file_exists(
        filepath,
        description="Tokenizer model dosyası",
        hint="Tokenizer eğitmek için:\n"
             "   python3 scripts/prepare_data.py --step tokenizer"
    )


def validate_dataset_size(dataset, min_blocks: int = 1, description: str = "Eğitim verisi"):
    """Dataset'in yeterli veri içerdiğini doğrula."""
    if len(dataset) < min_blocks:
        msg = f"\n❌ {description} çok az veri içeriyor!\n"
        msg += f"   Toplam blok: {len(dataset)} (minimum: {min_blocks})\n"
        msg += f"   Toplam token: {len(dataset.tokens):,}\n"
        msg += f"\n💡 Çözüm:\n"
        msg += f"   1. Doğru veri dizinini belirttiğinizden emin olun\n"
        msg += f"   2. Daha fazla veri ekleyin veya daha kısa max_seq_len kullanın\n"
        msg += f"   3. Test için: python3 scripts/prepare_data.py --use-sample\n"
        raise ToprakError(msg)


def setup_error_handler():
    """
    ToprakError için güzel formatlanmış hata çıktısı.
    Script'lerin başında çağrılmalı.
    """
    original_excepthook = sys.excepthook

    def toprak_excepthook(exc_type, exc_value, exc_tb):
        if exc_type == ToprakError:
            print(str(exc_value), file=sys.stderr)
            sys.exit(1)
        else:
            original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = toprak_excepthook
