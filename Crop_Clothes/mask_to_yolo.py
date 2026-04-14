import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def mask_to_yolo_bbox(mask, class_id=0):
    """
    Chuyển đổi mask nhị phân sang YOLO bbox format.
    Trả về list các dòng string format: "class_id x_center y_center width height"
    """
    # Tìm contours hoặc bounding box của các vùng trắng (255)
    # Ở đây giả sử mask đã là binary (0 và 255)
    
    # Cách đơn giản nhất: tìm bounding box bao quanh toàn bộ vùng trắng
    # Tuy nhiên, nếu muốn detect từng đối tượng rời rạc thì cần findContours.
    # Với crop_img.py, thường chỉ có 1 đối tượng chính sau khi crop (hoặc đã được lọc keep_largest).
    # Nhưng để chắc chắn, ta dùng findContours để lấy bao đóng của tất cả.
    
    # Tuy nhiên, YOLO format cần (x_center, y_center, w, h) chuẩn hóa theo kích thước ảnh.
    
    h_img, w_img = mask.shape[:2]
    
    # Tìm các điểm khác 0
    ys, xs = np.where(mask > 0)
    
    if len(xs) == 0 or len(ys) == 0:
        return None

    # Tìm bbox bao quanh toàn bộ đối tượng (vì ảnh đã được crop tập trung vào đối tượng)
    x1, x2 = np.min(xs), np.max(xs)
    y1, y2 = np.min(ys), np.max(ys)
    
    # Tính toán theo format YOLO
    # Box width, height
    bw = x2 - x1
    bh = y2 - y1
    
    # Center x, y
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    
    # Normalize
    norm_cx = cx / w_img
    norm_cy = cy / h_img
    norm_w = bw / w_img
    norm_h = bh / h_img
    
    # Clip values to [0, 1] just in case
    norm_cx = min(max(norm_cx, 0.0), 1.0)
    norm_cy = min(max(norm_cy, 0.0), 1.0)
    norm_w = min(max(norm_w, 0.0), 1.0)
    norm_h = min(max(norm_h, 0.0), 1.0)
    
    return f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}"

def convert_masks_to_yolo(masks_dir, labels_dir, class_id=0):
    masks_dir = Path(masks_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    mask_files = list(masks_dir.glob("*.png"))
    print(f"Tìm thấy {len(mask_files)} masks trong {masks_dir}")
    
    count = 0
    for mask_path in tqdm(mask_files, desc="Converting"):
        # Đọc ảnh mask (grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Không đọc được {mask_path}")
            continue
            
        yolo_line = mask_to_yolo_bbox(mask, class_id)
        
        if yolo_line:
            # Tạo file .txt cùng tên
            label_path = labels_dir / (mask_path.stem + ".txt")
            label_path.write_text(yolo_line, encoding="utf-8")
            count += 1
            
    print(f"Đã tạo {count} file labels tại {labels_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo nhãn YOLO từ ảnh mask.")
    parser.add_argument("--masks_dir", type=str, default=r"D:\NCKH_AI\Crop_img\output\masks", help="Thư mục chứa ảnh mask")
    parser.add_argument("--labels_dir", type=str, default=r"D:\NCKH_AI\Crop_img\output\labels", help="Thư mục lưu nhãn YOLO")
    parser.add_argument("--class_id", type=int, default=0, help="Class ID cho YOLO (mặc định: 0)")
    
    args = parser.parse_args()
    
    convert_masks_to_yolo(args.masks_dir, args.labels_dir, args.class_id)
