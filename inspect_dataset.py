import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


TEXT_EXTS = {".txt", ".tsv", ".csv", ".jsonl"}
TRAIN_HINTS = ("train", "training")
TEST_HINTS = ("test", "eval", "evaluation", "dev", "valid", "validation")
ENCODINGS = ("utf-8", "gb18030", "big5")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def iter_text_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT_EXTS:
            yield p


def has_any_keyword(path: Path, keywords: Sequence[str]) -> bool:
    full = str(path).lower()
    return any(k in full for k in keywords)


def detect_splits(files: Sequence[Path]) -> Tuple[List[Path], List[Path]]:
    train_files = [f for f in files if has_any_keyword(f, TRAIN_HINTS)]
    test_files = [f for f in files if has_any_keyword(f, TEST_HINTS)]

    if train_files and test_files:
        return train_files, test_files

    # Fallback: if only one type is found, treat the rest as the other split.
    if train_files and not test_files:
        others = [f for f in files if f not in train_files]
        if others:
            test_files = others
    elif test_files and not train_files:
        others = [f for f in files if f not in test_files]
        if others:
            train_files = others

    # Final fallback: all files as train, no test files.
    if not train_files and not test_files:
        train_files = list(files)

    return train_files, test_files


def count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as f:
        for _ in f:
            total += 1
    return total


def decode_bytes(raw: bytes) -> str:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_line(line: str) -> Tuple[str, str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 4:
        return parts[-2], parts[-1]  # common format: source, score, zh, ja
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0] if parts else "", ""


def sample_lines(path: Path, sample_size: int, seed: int) -> List[str]:
    random.seed(seed)
    sample: List[str] = []
    seen = 0
    with path.open("rb") as f:
        for raw in f:
            line = decode_bytes(raw)
            line = line.strip()
            if not line:
                continue
            seen += 1
            if len(sample) < sample_size:
                sample.append(line)
            else:
                j = random.randint(1, seen)
                if j <= sample_size:
                    sample[j - 1] = line
    return sample


def inspect_split(name: str, files: Sequence[Path], sample_size: int, seed: int) -> None:
    print(f"\n=== {name} ===")
    if not files:
        print("No files found.")
        print("Total samples: 0")
        return

    total = 0
    for f in files:
        n = count_lines(f)
        total += n
        print(f"- {f} | lines: {n}")

    print(f"Total samples: {total}")
    print(f"\n{name} preview (up to {sample_size} rows):")

    shown = 0
    for file_idx, f in enumerate(files):
        rows = sample_lines(f, sample_size=max(1, sample_size // max(1, len(files))), seed=seed + file_idx)
        for row in rows:
            zh, ja = parse_line(row)
            shown += 1
            print(f"[{shown}] zh: {zh[:120]}")
            print(f"    ja: {ja[:120]}")
            if shown >= sample_size:
                return


def fallback_virtual_test(train_files: Sequence[Path], test_ratio: float) -> None:
    total = sum(count_lines(f) for f in train_files)
    test_n = int(total * test_ratio)
    train_n = total - test_n
    print("\n[Notice] No explicit test split files detected.")
    print(f"[Notice] Virtual split by ratio only: train={train_n}, test={test_n} (ratio={test_ratio})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect zh-ja dataset without opening huge files.")
    parser.add_argument("--data-dir", default="dataset", help="Dataset root directory.")
    parser.add_argument("--sample-size", type=int, default=6, help="Number of preview samples per split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Virtual test ratio if no test files exist.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    files = sorted(iter_text_files(data_dir))
    if not files:
        print("No supported text files found.")
        return

    train_files, test_files = detect_splits(files)
    inspect_split("Train", train_files, args.sample_size, args.seed)
    inspect_split("Test", test_files, args.sample_size, args.seed + 999)

    if not test_files:
        fallback_virtual_test(train_files, args.test_ratio)


if __name__ == "__main__":
    main()
