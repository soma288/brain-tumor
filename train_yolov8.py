#!/usr/bin/env python3
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
	parser = argparse.ArgumentParser(description="Train YOLOv8 classification model for brain tumor detection")
	parser.add_argument("--data", type=str, default="datasets/brain-tumor", help="Path to classification dataset root (train/val/test)")
	parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="Base classification model checkpoint")
	parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
	parser.add_argument("--imgsz", type=int, default=224, help="Image size for training")
	parser.add_argument("--batch", type=int, default=32, help="Batch size")
	parser.add_argument("--project", type=str, default="runs/classify", help="Project directory for runs")
	parser.add_argument("--name", type=str, default="brain_tumor", help="Run name")
	args = parser.parse_args()

	data_path = Path(args.data)
	if not data_path.exists():
		raise FileNotFoundError(f"Dataset not found: {data_path}")

	model = YOLO(args.model)
	results = model.train(
		data=str(data_path),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		project=args.project,
		name=args.name,
	)
	print(results)


if __name__ == "__main__":
	main()
