#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor


# Label mapping theo model card:
# 0 Background, 1 Hat, 2 Hair, 3 Sunglasses, 4 Upper-clothes, 5 Skirt, 6 Pants,
# 7 Dress, 8 Belt, 9 Left-shoe, 10 Right-shoe, 11 Face, 12 Left-leg, 13 Right-leg,
# 14 Left-arm, 15 Right-arm, 16 Bag, 17 Scarf
LABELS: Dict[int, str] = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}


@dataclass(frozen=True)
class BuildConfig:
    input_dir: Path
    output_dir: Path
    model_name: str
    device: str
    include_labels: Tuple[int, ...]
    padding: float
    min_area_ratio: float
    morph_kernel_ratio: float
    keep_largest_component: bool
    save_cutout: bool
    seed: int
    split_train: float
    split_val: float
    split_test: float


class ClothesSegmenter:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = torch.device(device)
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_label_map(self, image: Image.Image) -> np.ndarray:
        """
        Returns:
            pred (H, W) uint8 label map.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, C, h, w)

        # Upsample về kích thước ảnh gốc
        w, h = image.size
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return pred


def list_images(input_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
    paths: List[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)


def ensure_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)


def postprocess_mask(
    mask_255: np.ndarray,
    morph_kernel_ratio: float,
    keep_largest: bool,
) -> np.ndarray:
    """
    mask_255: uint8 in {0,255}
    """
    h, w = mask_255.shape[:2]
    k = max(3, int(min(h, w) * morph_kernel_ratio))
    k = ensure_odd(k)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Close để lấp lỗ, Open để bỏ nhiễu nhỏ
    m = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

    if keep_largest:
        # connected components cần ảnh 0/1
        m01 = (m > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m01, connectivity=8)
        if num > 1:
            # stats[0] là background
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = 1 + int(np.argmax(areas))
            out = np.zeros_like(m01)
            out[labels == largest_idx] = 1
            m = (out * 255).astype(np.uint8)

    return m


def mask_to_bbox(mask_255: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_255 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def pad_bbox(
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    padding: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    px = int(bw * padding)
    py = int(bh * padding)

    x1p = max(0, x1 - px)
    y1p = max(0, y1 - py)
    x2p = min(img_w - 1, x2 + px)
    y2p = min(img_h - 1, y2 + py)
    return x1p, y1p, x2p, y2p


def save_png_mask(path: Path, mask_255: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask_255)


def save_cutout_rgba(path: Path, crop_img_rgb: np.ndarray, crop_mask_255: np.ndarray) -> None:
    """
    crop_img_rgb: (H,W,3) uint8
    crop_mask_255: (H,W) uint8
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha = crop_mask_255
    rgba = np.dstack([crop_img_rgb, alpha]).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)


def write_splits(output_dir: Path, ids: List[str], train: float, val: float, test: float, seed: int) -> None:
    assert abs((train + val + test) - 1.0) < 1e-6, "Tổng split phải = 1.0"
    rng = random.Random(seed)
    ids2 = ids[:]
    rng.shuffle(ids2)

    n = len(ids2)
    n_train = int(n * train)
    n_val = int(n * val)
    train_ids = ids2[:n_train]
    val_ids = ids2[n_train:n_train + n_val]
    test_ids = ids2[n_train + n_val:]

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""), encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""), encoding="utf-8")
    (split_dir / "test.txt").write_text("\n".join(test_ids) + ("\n" if test_ids else ""), encoding="utf-8")


def build_dataset(cfg: BuildConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "images").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    if cfg.save_cutout:
        (cfg.output_dir / "cutouts").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("builder")
    logger.info("Input: %s", cfg.input_dir)
    logger.info("Output: %s", cfg.output_dir)
    logger.info("Model: %s", cfg.model_name)
    logger.info("Device: %s", cfg.device)
    logger.info("Include labels: %s", cfg.include_labels)
    logger.info("Label names: %s", [LABELS.get(i, str(i)) for i in cfg.include_labels])

    segmenter = ClothesSegmenter(model_name=cfg.model_name, device=cfg.device)

    images = list_images(cfg.input_dir)
    logger.info("Found %d images", len(images))

    meta_path = cfg.output_dir / "meta.jsonl"
    ok_ids: List[str] = []
    fail_count = 0

    with meta_path.open("w", encoding="utf-8") as fmeta:
        for in_path in tqdm(images, desc="Processing", unit="img"):
            rel_id = in_path.relative_to(cfg.input_dir).as_posix()
            out_stem = rel_id.replace("/", "__")  # tránh thư mục lồng nhau
            out_img_path = cfg.output_dir / "images" / (Path(out_stem).stem + ".png")
            out_mask_path = cfg.output_dir / "masks" / (Path(out_stem).stem + ".png")
            out_cutout_path = cfg.output_dir / "cutouts" / (Path(out_stem).stem + ".png")

            record = {
                "id": Path(out_stem).stem,
                "input_rel": rel_id,
                "input_abs": str(in_path),
                "output_image": str(out_img_path),
                "output_mask": str(out_mask_path),
                "include_labels": list(cfg.include_labels),
                "status": "fail",
                "reason": "",
                "bbox_xyxy": None,
                "area_ratio": None,
            }

            try:
                image = Image.open(in_path).convert("RGB")
                w, h = image.size

                pred = segmenter.predict_label_map(image)
                # mask theo include_labels
                m = np.isin(pred, np.array(cfg.include_labels, dtype=np.uint8)).astype(np.uint8) * 255
                m = postprocess_mask(m, morph_kernel_ratio=cfg.morph_kernel_ratio, keep_largest=cfg.keep_largest_component)

                area_ratio = float((m > 0).sum() / (h * w))
                record["area_ratio"] = area_ratio

                if area_ratio < cfg.min_area_ratio:
                    record["reason"] = f"mask too small (area_ratio={area_ratio:.6f})"
                    fail_count += 1
                    fmeta.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                bbox = mask_to_bbox(m)
                if bbox is None:
                    record["reason"] = "no mask pixels after postprocess"
                    fail_count += 1
                    fmeta.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                bbox_padded = pad_bbox(bbox, img_w=w, img_h=h, padding=cfg.padding)
                x1, y1, x2, y2 = bbox_padded
                record["bbox_xyxy"] = [x1, y1, x2, y2]

                # Crop ảnh và mask (PIL crop dùng right/lower exclusive)
                crop_img = image.crop((x1, y1, x2 + 1, y2 + 1))
                crop_mask = m[y1:y2 + 1, x1:x2 + 1]

                # Save
                out_img_path.parent.mkdir(parents=True, exist_ok=True)
                crop_img.save(out_img_path)

                save_png_mask(out_mask_path, crop_mask)

                if cfg.save_cutout:
                    crop_np = np.array(crop_img)  # RGB
                    save_cutout_rgba(out_cutout_path, crop_np, crop_mask)

                record["status"] = "ok"
                ok_ids.append(record["id"])
                fmeta.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                record["reason"] = f"exception: {type(e).__name__}: {e}"
                fail_count += 1
                fmeta.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Done. OK=%d, FAIL=%d", len(ok_ids), fail_count)
    if ok_ids:
        write_splits(
            output_dir=cfg.output_dir,
            ids=ok_ids,
            train=cfg.split_train,
            val=cfg.split_val,
            test=cfg.split_test,
            seed=cfg.seed,
        )
        logger.info("Wrote splits to %s/splits/", cfg.output_dir)


def parse_args() -> BuildConfig:
    p = argparse.ArgumentParser(
        description="Build shirt (upper-clothes) dataset: crop + mask from input images using SegFormer clothes segmentation.",
    )
    p.add_argument("--input_dir", type=str, default="input", help="Thư mục ảnh đầu vào (mặc định: input)")
    p.add_argument("--output_dir", type=str, default="output", help="Thư mục output dataset (mặc định: output)")

    p.add_argument(
        "--model_name",
        type=str,
        default="mattmdjaga/segformer_b2_clothes",
        help="HuggingFace model name",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda | cpu | cuda:0 ...",
    )
    p.add_argument(
        "--include_labels",
        type=str,
        default="4",
        help="Các label cần lấy, phân tách bởi dấu phẩy. VD: '4' (Upper-clothes) hoặc '4,7' (Upper-clothes + Dress)",
    )
    p.add_argument("--padding", type=float, default=0.15, help="Nới bbox theo tỷ lệ (0.0 - 0.5 khuyến nghị)")
    p.add_argument("--min_area_ratio", type=float, default=0.003, help="Bỏ qua ảnh nếu mask quá nhỏ (tỷ lệ diện tích)")
    p.add_argument("--morph_kernel_ratio", type=float, default=0.01, help="Kernel morphology theo tỷ lệ cạnh ngắn")
    p.add_argument("--no_keep_largest_component", action="store_true", help="Không giữ component lớn nhất (mặc định có)")
    p.add_argument("--save_cutout", action="store_true", help="Xuất thêm PNG RGBA nền trong suốt (cutout)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_train", type=float, default=0.90)
    p.add_argument("--split_val", type=float, default=0.05)
    p.add_argument("--split_test", type=float, default=0.05)

    args = p.parse_args()

    include_labels = tuple(int(x.strip()) for x in args.include_labels.split(",") if x.strip() != "")

    return BuildConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        device=args.device,
        include_labels=include_labels,
        padding=float(args.padding),
        min_area_ratio=float(args.min_area_ratio),
        morph_kernel_ratio=float(args.morph_kernel_ratio),
        keep_largest_component=not bool(args.no_keep_largest_component),
        save_cutout=bool(args.save_cutout),
        seed=int(args.seed),
        split_train=float(args.split_train),
        split_val=float(args.split_val),
        split_test=float(args.split_test),
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    setup_logging()
    cfg = parse_args()
    build_dataset(cfg)


if __name__ == "__main__":
    main()
