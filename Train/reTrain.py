from ultralytics import YOLO
import os

# Tắt check update để tránh treo như lỗi trước đó
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

def train_model():
    # Khởi tạo mô hình
    model = YOLO("D:\\NCKH_AI\\Test\\Vui\\best_agu.pt")  # Hoặc đường dẫn model của bạn

    # Thực hiện huấn luyện
    results = model.train(
        data="C:\\Users\\Anh\\Downloads\\dataset_yolo\\data.yaml",      # Đường dẫn tới file yaml mới của bạn
        epochs=200,             # Số vòng lặp (tùy chỉnh tùy độ lớn dữ liệu)
        imgsz=640,             # Kích thước ảnh đầu vào
        batch=16,  # Điều chỉnh theo dung lượng VRAM của GPU
        workers=4,            # Số lượng worker để tải dữ liệu (tùy chỉnh theo CPU)
        lr0=0.001,             # Học phí khởi tạo (thấp hơn mặc định 0.01 để tránh phá vỡ weights cũ)
        cache=False,            # Không cache dữ liệu để tiết kiệm RAM (tùy chỉnh nếu bạn có đủ RAM)
        freeze=10,             # Đóng băng 10 lớp đầu (Backbone) để giữ lại khả năng trích xuất đặc trưng cơ bản
        device=0,   # Chạy trên GPU (0) hoặc 'cpu'
        project='School_Uniform_Project',
        name='reTrainModel' # Tên thư mục lưu kết quả huấn luyện
    )

if __name__ == '__main__':
    
    train_model()