from __future__ import annotations

from typing import Iterable

DEFECT_CLASSES: dict[str, str] = {
    "foreign_edge": "FOREIGN EDGE",
    "edge_rub": "EDGE RUB",
    "loose_coil": "LOOSE COIL",
}

CLASS_IDS: dict[str, int] = {
    "foreign_edge": 0,
    "edge_rub": 1,
    "loose_coil": 2,
}

ID_TO_CLASS: dict[int, str] = {idx: name for name, idx in CLASS_IDS.items()}

SAMPLE_FOLDERS: dict[str, str] = {
    "foreign_edge": "foreign_edge",
    "edge_rub": "edge_rub",
    "loose_coil": "loose_coil",
}

EXCEL_LABELS: dict[str, tuple[str, ...]] = {
    "foreign_edge": ("Edge cuts", "Edge Foreign material"),
    "edge_rub": ("Edge Rubs",),
    "loose_coil": ("Loose coil",),
}

YOLO_ALIASES: dict[str, str] = {
    "foreign_edge": "foreign_edge",
    "edge_cut": "foreign_edge",
    "edge_cuts": "foreign_edge",
    "edge_foreign_material": "foreign_edge",
    "edge_rub": "edge_rub",
    "edge_rubs": "edge_rub",
    "loose_coil": "loose_coil",
    "loose": "loose_coil",
}


def display_name(class_name: str) -> str:
    return DEFECT_CLASSES.get(class_name, class_name.replace("_", " ").upper())


def normalize_class_name(raw: str) -> str:
    key = str(raw).strip().lower().replace(" ", "_")
    return YOLO_ALIASES.get(key, key if key in DEFECT_CLASSES else "foreign_edge")


def class_id(class_name: str) -> int:
    return CLASS_IDS.get(normalize_class_name(class_name), 0)


def iter_classes() -> Iterable[tuple[str, str, int]]:
    for name, label in DEFECT_CLASSES.items():
        yield name, label, CLASS_IDS[name]
