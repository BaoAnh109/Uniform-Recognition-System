from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class YoloBBox:
    cls: int
    x: float
    y: float
    w: float
    h: float

def parse_yolo_label_file(label_path: Path) -> List[YoloBBox]:
    bboxes: List[YoloBBox] = []
    if not label_path.exists():
        return bboxes

    for ln, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path} line {ln}: expected 5 fields, got {len(parts)} -> '{line}'")

        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        bboxes.append(YoloBBox(cls=cls, x=x, y=y, w=w, h=h))
    return bboxes

def is_in_01(v: float) -> bool:
    return 0.0 <= v <= 1.0

def validate_dataset_yolo(
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Dict[int, int]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    class_counts: Dict[int, int] = {i: 0 for i in range(num_classes)}
    errors: List[str] = []

    image_paths = [p for p in images_dir.rglob("*") if p.suffix.lower() in image_exts]
    if not image_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    for img_path in image_paths:
        rel = img_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")
        try:
            bboxes = parse_yolo_label_file(label_path)
        except Exception as e:
            errors.append(f"[LABEL_PARSE] {label_path}: {e}")
            continue

        for bb in bboxes:
            if not (0 <= bb.cls < num_classes):
                errors.append(f"[CLASS_RANGE] {label_path}: class_id={bb.cls} out of range [0,{num_classes-1}]")
                continue

            if not all(map(is_in_01, [bb.x, bb.y, bb.w, bb.h])):
                errors.append(
                    f"[COORD_RANGE] {label_path}: "
                    f"(x,y,w,h)=({bb.x:.4f},{bb.y:.4f},{bb.w:.4f},{bb.h:.4f}) not all in [0,1]"
                )
                continue

            if bb.w <= 0 or bb.h <= 0:
                errors.append(f"[NONPOS_SIZE] {label_path}: w/h must be > 0, got w={bb.w}, h={bb.h}")
                continue

            class_counts[bb.cls] += 1

    if errors:
        preview = "\n".join(errors[:30])
        raise RuntimeError(
            f"Dataset validation failed with {len(errors)} issues. First issues:\n{preview}"
        )

    return class_counts

if __name__ == "__main__":
    # Example usage:
    dataset_root = Path(r"D:\NCKH_AI\Crop_img\output")
    counts = validate_dataset_yolo(
        images_dir=dataset_root / "images",
        labels_dir=dataset_root / "labels",
        num_classes=18,
    )
    print("OK. BBox counts per class:", counts)
