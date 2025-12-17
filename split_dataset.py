#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split


def collect_image_paths(class_dir: Path) -> List[Path]:
	allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
	return [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext]


def stratified_split(files: List[Path], train_size: float, val_size: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
	train_files, temp_files = train_test_split(files, train_size=train_size, random_state=seed)
	relative_val_size = val_size / (1.0 - train_size)
	val_files, test_files = train_test_split(temp_files, train_size=relative_val_size, random_state=seed)
	return train_files, val_files, test_files


def copy_files(files: List[Path], target_dir: Path):
	target_dir.mkdir(parents=True, exist_ok=True)
	for src in files:
		dst = target_dir / src.name
		shutil.copy2(src, dst)


def main():
	parser = argparse.ArgumentParser(description="Split brain tumor dataset into train/val/test for YOLOv8 classification.")
	parser.add_argument("--input_dir", type=str, default="brain_tumor_dataset", help="Path with subfolders yes/ and no/")
	parser.add_argument("--output_dir", type=str, default="datasets/brain-tumor", help="Output dataset root")
	parser.add_argument("--train", type=float, default=0.7, help="Train split fraction")
	parser.add_argument("--val", type=float, default=0.15, help="Validation split fraction")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	args = parser.parse_args()

	input_root = Path(args.input_dir)
	output_root = Path(args.output_dir)

	if not input_root.exists():
		raise FileNotFoundError(f"Input directory not found: {input_root}")

	class_names = ["no", "yes"]

	random.seed(args.seed)
	output_root.mkdir(parents=True, exist_ok=True)

	for class_name in class_names:
		class_dir = input_root / class_name
		if not class_dir.exists():
			raise FileNotFoundError(f"Missing class folder: {class_dir}")

		files = collect_image_paths(class_dir)
		if len(files) == 0:
			raise RuntimeError(f"No images found in {class_dir}")

		train_files, val_files, test_files = stratified_split(files, train_size=args.train, val_size=args.val, seed=args.seed)

		copy_files(train_files, output_root / "train" / class_name)
		copy_files(val_files, output_root / "val" / class_name)
		copy_files(test_files, output_root / "test" / class_name)

	print(f"Created dataset at {output_root.resolve()}")
	print("Folder structure:")
	for split in ["train", "val", "test"]:
		for class_name in class_names:
			folder = output_root / split / class_name
			count = len(list(folder.glob('*')))
			print(f" - {folder}: {count} files")


if __name__ == "__main__":
	main()
