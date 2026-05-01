import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = r"E:\Studia\Grzyby\data_folder\MO_94" 

DEST_DIR = r"E:\Studia\Grzyby\dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

def split_dataset(source, dest, train_r, val_r, test_r, seed):
    random.seed(seed)
    source_path = Path(source)
    dest_path = Path(dest)

    print(f"--- Rozpoczęcie stratyfikowanego podziału danych (Seed: {seed}) ---")
    print(f"Źródło: {source_path}")
    print(f"Cel: {dest_path}")
    print(f"Proporcje: Train {train_r*100}%, Val {val_r*100}%, Test {test_r*100}%")

    if not source_path.exists():
        print(f"BŁĄD: Folder źródłowy {source_path} nie istnieje!")
        return

    for split in ['train', 'val', 'test']:
        split_dir = dest_path / split
        if split_dir.exists():
            print(f"Usuwanie starego folderu {split_dir}...")
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    classes = [d for d in source_path.iterdir() if d.is_dir()]
    print(f"Znaleziono klas: {len(classes)}")

    total_train, total_val, total_test = 0, 0, 0

    for cls_dir in classes:
        cls_name = cls_dir.name
        (dest_path / 'train' / cls_name).mkdir(parents=True, exist_ok=True)
        (dest_path / 'val' / cls_name).mkdir(parents=True, exist_ok=True)
        (dest_path / 'test' / cls_name).mkdir(parents=True, exist_ok=True)

        files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.shuffle(files)
        total_files = len(files)

        if total_files == 0:
            print(f"  [UWAGA] Pusty folder: {cls_name}")
            continue

        train_end = int(total_files * train_r)
        val_end = train_end + int(total_files * val_r)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        for f in train_files:
            shutil.copy2(f, dest_path / 'train' / cls_name / f.name)
        for f in val_files:
            shutil.copy2(f, dest_path / 'val' / cls_name / f.name)
        for f in test_files:
            shutil.copy2(f, dest_path / 'test' / cls_name / f.name)

        print(f"  Klasa: {cls_name:<30} | Razem: {total_files:<4} | Train: {len(train_files):<3} | Val: {len(val_files):<3} | Test: {len(test_files):<3}")

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

    print("\n--- PODSUMOWANIE ---")
    print(f"Rozdzielono łącznie {total_train + total_val + total_test} plików.")
    print(f"Train: {total_train}")
    print(f"Val:   {total_val}")
    print(f"Test:  {total_test}")
    print("Podział zakończony pomyślnie. Zbiór danych gotowy do nauki.")

if __name__ == "__main__":
    split_dataset(SOURCE_DIR, DEST_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)