from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from inspection.classes import EXCEL_LABELS, SAMPLE_FOLDERS
from inspection.config import project_root

LABEL_ROWS = {
    1: "Edge cuts",
    18: "Edge Rubs",
    35: "Edge Foreign material",
    53: "Edgescoop",
    72: "Foreign material Rolled",
    89: "OD Telescopicity",
    111: "ID Telescopicity",
    133: "Conical Telescopicity",
    151: "Staggered Telescopicity",
    171: "Loose coil",
    192: "Edge Burst",
    211: "Elliptical",
    226: "Shift In middle",
    243: "Waviness",
}


def _label_for_row(row: int) -> str:
    current = "unknown"
    for start, name in sorted(LABEL_ROWS.items()):
        if row >= start:
            current = name
        else:
            break
    return current


def find_excel(root: Path | None = None) -> Path | None:
    root = root or project_root()
    candidates = [
        root / "JSW Defective Coils images (1).xlsx",
        root / "JSW Defective Coils images.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.glob("*Defective*Coil*.xlsx"))
    return matches[0] if matches else None


def _media_by_excel_label(excel_path: Path) -> dict[str, list[str]]:
    ns = {"xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"}
    grouped: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(excel_path) as archive:
        rels = ET.fromstring(archive.read("xl/drawings/_rels/drawing1.xml.rels"))
        rid_to_media = {}
        for rel in rels:
            rid = rel.attrib.get("Id")
            target = rel.attrib.get("Target", "")
            rid_to_media[rid] = Path(target).name
        drawing = ET.fromstring(archive.read("xl/drawings/drawing1.xml"))
        for anchor in list(drawing.findall("xdr:twoCellAnchor", ns)) + list(
            drawing.findall("xdr:oneCellAnchor", ns)
        ):
            from_el = anchor.find("xdr:from", ns)
            if from_el is None:
                continue
            row = int(from_el.find("xdr:row", ns).text) + 1
            blip = None
            for element in anchor.iter():
                if element.tag.endswith("blip"):
                    blip = element
                    break
            if blip is None:
                continue
            rid = blip.attrib.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
            )
            media = rid_to_media.get(rid)
            if media:
                grouped[_label_for_row(row)].append(media)
    return grouped


def extract_class_samples(
    class_name: str,
    excel_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    root = project_root()
    excel_path = Path(excel_path) if excel_path else find_excel(root)
    folder = SAMPLE_FOLDERS.get(class_name, class_name)
    output_dir = Path(output_dir) if output_dir else root / "samples" / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = (
        sorted(output_dir.glob("*.png"))
        + sorted(output_dir.glob("*.jpg"))
        + sorted(output_dir.glob("*.jpeg"))
    )
    if excel_path is None or not excel_path.exists():
        return existing

    grouped = _media_by_excel_label(excel_path)
    written: list[Path] = []
    index = 1
    with zipfile.ZipFile(excel_path) as archive:
        for excel_label in EXCEL_LABELS.get(class_name, ()):
            for media in grouped.get(excel_label, []):
                suffix = Path(media).suffix.lower() or ".png"
                dest = output_dir / f"{class_name}_{index:02d}{suffix}"
                dest.write_bytes(archive.read(f"xl/media/{media}"))
                written.append(dest)
                index += 1
    return written


def extract_foreign_edge_samples(
    excel_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    return extract_class_samples("foreign_edge", excel_path=excel_path, output_dir=output_dir)


def extract_all_samples(excel_path: Path | None = None) -> dict[str, list[Path]]:
    extracted: dict[str, list[Path]] = {}
    for class_name in EXCEL_LABELS:
        extracted[class_name] = extract_class_samples(class_name, excel_path=excel_path)
    return extracted


if __name__ == "__main__":
    results = extract_all_samples()
    for name, paths in results.items():
        print(f"{name}: {len(paths)} samples")
