# Coil defect dataset

Use this folder for YOLO11 training.

```
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

Label format:

```
class_id x_center y_center width height
```

Classes in `data.yaml`:

```
0 foreign_edge
1 edge_rub
2 loose_coil
```

Workflow:

1. Extract samples: `python -m scripts.extract_samples`
2. Annotate in the app, or bootstrap review labels: `python -m scripts.bootstrap_labels`
3. Train: `python -m scripts.train_yolo`
4. The trainer copies `best.pt` into `models/` when training finishes.
