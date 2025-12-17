## Brain Tumor Detection with YOLOv8 (Classification)

- Uses YOLOv8 classification (not custom CNN) to predict tumor presence in brain MRI images.
- Streamlit app provides upload, prediction, and a single color‑coded YES/NO with confidence.

### Why YOLOv8 Classification (not CNN)

You required YOLOv8 and no CNN. We use Ultralytics' YOLOv8 classification head (`yolov8n-cls.pt`). This repurposes the YOLO backbone with a built‑in classifier head for two classes: `no` and `yes`. It avoids building a separate CNN while leveraging YOLO's training/inference pipeline.

### Methodology (What we did)

1. Data preparation
   - Input folder: `brain_tumor_dataset/` with `no/` and `yes/` images.
   - We created a train/val/test split in `datasets/brain-tumor/` preserving class folders using `src/split_dataset.py`.
2. Training
   - We trained YOLOv8 classification with `yolov8n-cls.pt` backbone on the split dataset (`src/train_yolov8.py`).
   - Parameters: `imgsz=224`, batch/epochs configurable.
   - Outputs saved to `runs/classify/brain_tumor/weights/best.pt`.
3. Inference app
   - `app.py` loads the classification weights and predicts class probabilities.
   - UI displays a single line: YES (red) or NO (green) with confidence percentage.

### Setup

```bash
cd /Users/akash/code/Brain-Tumor-Yolo
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Prepare dataset

Expected input: `brain_tumor_dataset/` containing `yes/` and `no/` folders.

Split into train/val/test:

```bash
python src/split_dataset.py --input_dir brain_tumor_dataset --output_dir datasets/brain-tumor --train 0.7 --val 0.15
```

Resulting structure:

```
datasets/brain-tumor/
  train/
    no/
    yes/
  val/
    no/
    yes/
  test/
    no/
    yes/
```

### Train

```bash
python src/train_yolov8.py --data datasets/brain-tumor --epochs 20 --imgsz 224 --batch 16 --model yolov8n-cls.pt
```

Best weights: `runs/classify/brain_tumor/weights/best.pt`.

### Run the app

```bash
streamlit run app.py
```

Use the sidebar to set a different weights path if needed.

### How confidence is computed

- The model returns class probabilities for `no` and `yes`.
- We show only one line:
  - If `yes` probability ≥ `no`, we display YES with that probability.
  - Otherwise, we display NO with its probability.
- Colors: YES in red (alert), NO in green (ok).

### Files of interest

- `src/split_dataset.py`: Splits raw dataset into YOLOv8 classification layout.
- `src/train_yolov8.py`: Trains YOLOv8 classification model.
- `app.py`: Streamlit inference UI and logic.
- `.gitignore`: Ignores venv, caches, data splits, runs, large artifacts.

### Tips

- If you re-run the split, delete `datasets/brain-tumor/` first to avoid mixing old files.
- For higher accuracy, increase `--epochs` and consider `yolov8s-cls.pt` or `yolov8m-cls.pt`.
- Ensure balanced splits between `no` and `yes` for reliable evaluation.
