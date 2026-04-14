from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List
import os

import cv2
import numpy as np


@dataclass(frozen=True)
class CutConfig:
    pad: int = 10
    min_area: int = 5000
    keep_largest: bool = True
    morph_ksize: int = 7
    morph_iters: int = 1
    output_rgba: bool = True


def _ensure_binary_mask(mask_gray: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    return m


def _clean_mask(mask_bin: np.ndarray, cfg: CutConfig) -> np.ndarray:
    k = max(3, cfg.morph_ksize | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    m = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_iters)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_iters)

    if not cfg.keep_largest:
        return m

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    best_area = int(stats[best_idx, cv2.CC_STAT_AREA])
    if best_area < cfg.min_area:
        return m

    out = np.zeros_like(m)
    out[labels == best_idx] = 255
    return out


def _bbox_from_mask(mask_bin: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask không có vùng trắng.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def cut_shirt(image_bgr: np.ndarray, mask_gray: np.ndarray, cfg: CutConfig):
    if image_bgr.shape[:2] != mask_gray.shape[:2]:
        raise ValueError(f"Image và mask khác kích thước: {image_bgr.shape[:2]} vs {mask_gray.shape[:2]}")

    mask_bin = _ensure_binary_mask(mask_gray)
    mask_bin = _clean_mask(mask_bin, cfg)

    x1, y1, x2, y2 = _bbox_from_mask(mask_bin)

    h, w = mask_bin.shape[:2]
    pad = max(0, int(cfg.pad))
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w - 1, x2 + pad)
    y2p = min(h - 1, y2 + pad)

    crop_img = image_bgr[y1p:y2p + 1, x1p:x2p + 1]
    crop_m   = mask_bin[y1p:y2p + 1, x1p:x2p + 1]

    if cfg.output_rgba:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([crop_rgb, crop_m])
        return rgba, crop_m, (x1p, y1p, x2p, y2p)

    cut = cv2.bitwise_and(crop_img, crop_img, mask=crop_m)
    return cut, crop_m, (x1p, y1p, x2p, y2p)


def iter_images(images_dir: Path, recursive: bool, exts: List[str]) -> Iterable[Path]:
    exts = [e.lower().lstrip(".") for e in exts]
    if recursive:
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                yield p
    else:
        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                yield p


def find_mask(
    img_path: Path,
    masks_dir: Path,
    mask_ext: str,
    mask_suffix: str,
) -> Optional[Path]:
    """
    Ưu tiên:
    1) masks_dir / (stem + mask_suffix + .mask_ext)
    2) masks_dir / (stem + .mask_ext)  (nếu suffix không tìm thấy)
    """
    mask_ext = mask_ext.lower().lstrip(".")
    cand1 = masks_dir / f"{img_path.stem}{mask_suffix}.{mask_ext}"
    if cand1.exists():
        return cand1

    cand2 = masks_dir / f"{img_path.stem}.{mask_ext}"
    if cand2.exists():
        return cand2

    return None


def save_output(out_path: Path, out_img: np.ndarray, out_mask: np.ndarray, rgba: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if rgba:
        # out_img = RGB + A, lưu PNG cần BGRA
        bgr = cv2.cvtColor(out_img[:, :, :3], cv2.COLOR_RGB2BGR)
        bgra = np.dstack([bgr, out_img[:, :, 3]])
        cv2.imwrite(str(out_path.with_suffix(".png")), bgra)
    else:
        cv2.imwrite(str(out_path), out_img)

    cv2.imwrite(str(out_path.with_suffix(".mask.png")), out_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default=r"D:\NCKH_AI\Crop_img\output\images", help="Thư mục ảnh (mặc định: output/images)")
    ap.add_argument("--masks_dir", default=r"D:\NCKH_AI\Crop_img\output\masks", help="Thư mục mask (mặc định: output/masks)")
    ap.add_argument("--out_dir", default=r"D:\NCKH_AI\Crop_img\output\cutouts_batch", help="Thư mục lưu kết quả (mặc định: output/cutouts_batch)")
    ap.add_argument("--recursive", action="store_true", help="Duyệt ảnh theo subfolder")
    ap.add_argument("--exts", default="jpg,jpeg,png", help="Đuôi ảnh, ví dụ: jpg,png")
    ap.add_argument("--mask_ext", default="png", help="Đuôi mask (thường png)")
    ap.add_argument("--mask_suffix", default="_mask", help="Hậu tố mask (vd _mask). Nếu mask cùng tên thì vẫn chạy OK.")
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--rgba", action="store_true", help="Xuất PNG RGBA nền trong suốt")
    ap.add_argument("--min_area", type=int, default=5000)
    ap.add_argument("--morph_ksize", type=int, default=7)
    ap.add_argument("--morph_iters", type=int, default=1)
    ap.add_argument("--keep_largest", action="store_true", help="Giữ component lớn nhất (khuyến nghị bật)")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)

    cfg = CutConfig(
        pad=args.pad,
        min_area=args.min_area,
        keep_largest=bool(args.keep_largest),
        morph_ksize=args.morph_ksize,
        morph_iters=args.morph_iters,
        output_rgba=bool(args.rgba),
    )

    exts = [x.strip() for x in args.exts.split(",") if x.strip()]
    img_paths = list(iter_images(images_dir, args.recursive, exts))

    if not img_paths:
        raise SystemExit("Không tìm thấy ảnh trong images_dir với exts đã chọn.")

    ok = 0
    miss = 0
    fail = 0

    for i, img_path in enumerate(img_paths, start=1):
        mask_path = find_mask(img_path, masks_dir, args.mask_ext, args.mask_suffix)
        if mask_path is None:
            miss += 1
            print(f"[MISS] {img_path.name} -> không tìm thấy mask.")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            fail += 1
            print(f"[FAIL] Không đọc được img/mask: {img_path.name} | {mask_path.name}")
            continue

        try:
            out_img, out_m, bbox = cut_shirt(img, msk, cfg)

            # giữ cấu trúc thư mục tương đối nếu recursive
            rel = img_path.relative_to(images_dir)
            out_base = (out_dir / rel).with_suffix(".png" if cfg.output_rgba else img_path.suffix)

            save_output(out_base, out_img, out_m, rgba=cfg.output_rgba)
            ok += 1
            print(f"[OK {i}/{len(img_paths)}] {rel.as_posix()} bbox={bbox}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {img_path.name}: {e}")

    print("\n=== SUMMARY ===")
    print(f"Total: {len(img_paths)} | OK: {ok} | MISS_MASK: {miss} | FAIL: {fail}")


if __name__ == "__main__":
    # giảm log OpenCV nếu cần
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    main()
