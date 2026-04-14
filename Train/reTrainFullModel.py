from ultralytics import YOLO
import os

# Tắt check update và chuyển sang chế độ offline
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

def train_model():
    # Khởi tạo model mới hoàn toàn (ví dụ YOLOv8n hoặc YOLOv11n)
    # Thay vì dùng 'best.pt' cũ, ta dùng tên model chuẩn để train từ đầu
    model = YOLO("yolo11s.pt") 

    # Thực hiện huấn luyện với Augmentation chuẩn
    results = model.train(
        # --- Cấu hình cơ bản ---
        data="D:\\NCKH_AI\\dataset\\dataset_yolo\\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,          # RTX 4050 6GB nên để 16, nếu báo Out of Memory thì hạ xuống 8
        workers=4,
        device=0,          # Sử dụng GPU RTX 4050
        project='School_Uniform_Project',
        name='Retrain_Full_Model_100epochs',
        patience=20,
        
        # --- Huấn luyện từ đầu ---
        pretrained=False,   # Quan trọng: Không sử dụng trọng số đã huấn luyện trước đó
        
        # --- Data Augmentation (Tối ưu cho đồng phục) ---
        hsv_h=0.015,       # Thay đổi sắc độ (0.0 - 1.0)
        hsv_s=0.7,         # Thay đổi độ bão hòa
        hsv_v=0.4,         # Thay đổi giá trị độ sáng (quan trọng cho môi trường ánh sáng thay đổi)
        degrees=10.0,      # Xoay nhẹ ảnh (giúp nhận diện khi người nghiêng)
        translate=0.1,     # Dịch chuyển ảnh
        scale=0.5,         # Phóng to/thu nhỏ (giúp nhận diện xa/gần)
        shear=0.1,         # Cắt nghiêng
        perspective=0.0,   # Biến đổi phối cảnh
        flipud=0.0,        # Lật ngược (không cần thiết cho người)
        fliplr=0.5,        # Lật trái phải (rất cần thiết)
        mosaic=1.0,        # Ghép 4 ảnh thành 1 (giúp học vật thể nhỏ tốt hơn)
        mixup=0.1,         # Trộn 2 ảnh (tăng độ khó cho model)
        copy_paste=0.1,    # Copy vật thể dán sang ảnh khác
        
        # --- Optimizer & LR ---
        optimizer='SGD',   # SGD thường tốt hơn cho việc train từ đầu nếu có đủ epoch
        lr0=0.01,          # Trả về mức mặc định 0.01 vì ta đang train mới
        cache=True         # RTX 4050 có tốc độ truy xuất tốt, nếu RAM hệ thống > 16GB thì để True
    )

if __name__ == '__main__':
    train_model()