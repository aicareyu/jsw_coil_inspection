# Coil Surface Inspection

Customer demo / POC for detecting **FOREIGN EDGE**, **EDGE RUB**, and **LOOSE COIL** defects on steel coils.

This is **not** a production inspection system. The current defective-coil set is small, so the application does **not** assume a reliably trained YOLO model. It runs a real image-analysis pipeline today and can switch to YOLO11 later by dropping in `models/best.pt`.

## Installation

```bash
python -m venv venv
```

Activate the environment:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app is local and offline. No cloud API is required.

On first launch the app extracts sample images from:

`Defective Coils images.xlsx`

into:

```
samples/foreign_edge/
samples/edge_rub/
samples/loose_coil/
```

You can also extract them manually:

```bash
python -m scripts.extract_samples
```

## Customer demo flow

1. Open the dashboard.
2. Upload a coil image, start the camera, or click a demo sample.
3. The same detector inspects the frame.
4. The left pane shows the localized defect.
5. The right pane shows **FAIL** plus the defect name, or **PASS**.

Accepted upload types: JPG, JPEG, PNG, MP4, AVI.

## DEMO MODE vs YOLO MODEL MODE

### DEMO MODE

Shown when `models/best.pt` is missing or cannot be loaded.

- Backend: **Demo CV**
- Status: **DEMO MODE — TRAINED MODEL NOT INSTALLED**
- Detector: classical computer vision on coil-edge texture
- The detector looks for the coil surface, winding-edge structure, and irregular elongated edge damage
- It is **not** filename-based and does **not** hard-code FAIL or a fixed box

This fallback exists because there are not enough labeled good/bad samples to train a trustworthy YOLO model yet.

### YOLO MODEL MODE

Copy a trained Ultralytics YOLO11 weights file to:

```
models/best.pt
```

Restart the app. It will load YOLO automatically.

- Backend: **YOLO**
- Status: **LOADED**
- The UI does not change
- Classes must be `0 foreign_edge`, `1 edge_rub`, `2 loose_coil`

If CUDA / GPU is unavailable, inference falls back to CPU.

## Future YOLO training

Prepared dataset folders:

```
dataset/
  images/train
  images/val
  labels/train
  labels/val
```

Training config: `data.yaml`

Classes:

```
0 foreign_edge
1 edge_rub
2 loose_coil
```

Typical later workflow:

1. Upload new defect images
2. Use **ANNOTATE** to export YOLO labels
3. Or bootstrap review labels: `python -m scripts.bootstrap_labels`
4. Train YOLO11:

```bash
python -m scripts.train_yolo
```

or:

```bash
yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640
```

5. Copy the resulting `best.pt` into `models/`
6. Restart the inspection app

The annotate page does **not** fake a training result.

## Project layout

```
app.py
config.yaml
data.yaml
inspection/          # detector facade, YOLO, demo CV, pre/postprocess
camera/              # OpenCV USB / laptop camera manager
video/               # frame-interval video inspection
database/            # SQLite history
ui/                  # industrial Streamlit dashboard
samples/foreign_edge
dataset/
models/
tests/
```

## Tests

```bash
pip install pytest
pytest -q
```

Coverage includes detector PASS/FAIL, missing YOLO weights, invalid images, unavailable camera, video processing, and inspection history.

## Notes

- Red circles on some source Excel images are **manual marks**, not the defect. The demo detector ignores them as much as possible and looks for the actual edge irregularity.
- Other coil defects in the Excel (telescopicity, waviness, etc.) are not the target class of this demo.
- Live camera uses OpenCV index `0` by default (`config.yaml`). If no USB/laptop camera is present, the UI shows **CAMERA NOT AVAILABLE** and offers a browser-camera fallback.
