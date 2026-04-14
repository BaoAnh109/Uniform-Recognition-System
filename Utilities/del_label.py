import os
from pathlib import Path

def sync_labels_with_images(images_dir, labels_dir, dry_run=True):
    """
    Xóa file label nếu không có ảnh tương ứng.
    
    Args:
        images_dir (str): Đường dẫn thư mục ảnh.
        labels_dir (str): Đường dẫn thư mục label.
        dry_run (bool): True = Chỉ log kiểm tra, False = Xóa thật.
    """
    img_path = Path(images_dir)
    lbl_path = Path(labels_dir)

    # 1. Định nghĩa các đuôi ảnh hợp lệ
    valid_img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif'}

    # 2. Lấy danh sách (tên file không đuôi)
    # Sử dụng set để tìm kiếm O(1) -> Tối ưu tốc độ cho dataset lớn
    images_stem = {f.stem for f in img_path.iterdir() if f.suffix.lower() in valid_img_extensions}
    labels_files = [f for f in lbl_path.iterdir() if f.suffix.lower() == '.txt']

    # Thông số thống kê
    total_labels = len(labels_files)
    deleted_count = 0
    remaining_labels = 0
    
    print(f"{'='*20} REPORT {'='*20}")
    print(f"📂 Image Dir: {images_dir} | Found: {len(images_stem)} images")
    print(f"📂 Label Dir: {labels_dir} | Found: {total_labels} labels")
    print(f"⚙️  Mode: {'[DRY RUN - Chỉ kiểm tra]' if dry_run else '[DELETE - Xóa thật]'}")
    print("-" * 50)

    # 3. Duyệt và xử lý
    for lbl_file in labels_files:
        # Nếu tên label (không đuôi) KHÔNG nằm trong set tên ảnh
        if lbl_file.stem not in images_stem:
            if not dry_run:
                try:
                    os.remove(lbl_file)
                    print(f"❌ [DELETED] {lbl_file.name}")
                except OSError as e:
                    print(f"⚠️  Lỗi xóa {lbl_file.name}: {e}")
            else:
                print(f"🔍 [WOULD DELETE] {lbl_file.name}")
            
            deleted_count += 1
        else:
            remaining_labels += 1

    # 4. Tính toán ảnh không có label (Background images)
    # (Ảnh có trong folder nhưng không có file txt tương ứng)
    labels_stem_remaining = {f.stem for f in lbl_path.iterdir() if f.suffix.lower() == '.txt'}
    background_images = len(images_stem - labels_stem_remaining)

    # 5. Tổng kết
    print("-" * 50)
    print(f"📊 KẾT QUẢ CUỐI CÙNG:")
    print(f"   - Tổng label ban đầu:  {total_labels}")
    print(f"   - Label rác (đã/sẽ xóa): {deleted_count}")
    print(f"   - Label hợp lệ còn lại:  {remaining_labels}")
    print(f"   - Số ảnh không có label: {background_images} (Sẽ được YOLO coi là Background/Negative sample)")
    print("=" * 48)

# --- CẤU HÌNH ---
# Thay đường dẫn của bạn vào đây
IMG_DIR = "datasets/train/images"
LBL_DIR = "datasets/train/labels"

# Chạy thử trước (True). Nếu OK thì sửa thành False để xóa thật.
sync_labels_with_images(IMG_DIR, LBL_DIR, dry_run=True)