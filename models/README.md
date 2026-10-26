# Models

## YOLO (preferred later)

Place a trained YOLO11 weights file here:

```
models/best.pt
```

Classes:

```
0 foreign_edge
1 edge_rub
2 loose_coil
```

If this file is missing, the application uses the demo detector.

## Demo classifier

`demo_classifier.npz` is a small nearest-neighbor classifier trained from the Excel samples.

Retrain it after adding samples:

```bash
python -m scripts.train_demo_classifier
```

Train YOLO after annotating:

```bash
python -m scripts.bootstrap_labels
python -m scripts.train_yolo
```
