# Uniform Recognition System

Hệ thống nhận diện đồng phục sử dụng Computer Vision và Deep Learning, tập trung vào bài toán:

- Phát hiện người trong khung hình
- Tách vùng trang phục/phần thân trên
- Phân loại đúng / sai đồng phục
- Hỗ trợ pipeline xử lý dữ liệu, gán nhãn, crop ảnh, huấn luyện và kiểm tra mô hình

Dự án phù hợp cho các bài toán như:

- Giám sát tuân thủ đồng phục trong trường học / doanh nghiệp
- Tạo bộ dữ liệu nhận diện đồng phục
- Nghiên cứu pipeline detection + segmentation + classification trong môi trường thực tế

---

## Mục tiêu dự án

Dự án được xây dựng để giải quyết bài toán nhận diện đồng phục trong video / camera / ảnh đầu vào với pipeline tổng quát:

1. **Phát hiện người**
2. **Tách vùng thân trên / vùng quần áo**
3. **Nhận diện đúng hay sai đồng phục**
4. **Theo dõi đối tượng theo thời gian**
5. **Ổn định kết quả bằng voting**
6. **Cảnh báo khi phát hiện sai đồng phục**

Ngoài phần suy luận (inference), repo còn chứa các công cụ phục vụ:

- Tách frame từ video
- Sinh dữ liệu crop quần áo
- Chuyển mask sang YOLO bbox
- Làm sạch dataset / đồng bộ label
- Kiểm tra chất lượng dataset YOLO
- Huấn luyện / train lại mô hình YOLO

---

## Tính năng chính

- Nhận diện đồng phục theo thời gian thực
- Hỗ trợ **YOLO detection / segmentation**
- Hỗ trợ **tracking** để theo dõi từng người qua nhiều frame
- Cơ chế **vote theo lịch sử** để giảm nhiễu
- Có thể bật / tắt segmentation trong UI
- Có cảnh báo âm thanh khi phát hiện sai đồng phục
- Có GUI phục vụ xử lý dữ liệu và chạy mô hình
- Có sẵn các script huấn luyện và tiếp tục huấn luyện model
- Có bộ tiện ích tiền xử lý dữ liệu cho pipeline huấn luyện

---

## Cấu trúc thư mục

```text
Uniform-Recognition-System/
├── App/
│   ├── Uniform_Detect_Version6.py
│   ├── Uniform_Detect_Version6_vote_threshold_toggle.py
│   ├── best_final.pt
│   ├── custom_botsort.yaml
│   ├── voice_fail.mp3
│   └── yolo11s-seg.pt
│
├── Crop_Clothes/
│   ├── batch_cut_shirt.py
│   ├── crop_img.py
│   ├── gui_app.py
│   ├── mask_to_yolo.py
│   └── shirt_cutout_app.py
│
├── Models/
│   ├── best_agu.pt
│   ├── yolo11s-seg.pt
│   ├── yolo11s.pt
│   └── yolo26n.pt
│
├── Tests/
│   ├── test_dataset.py
│   └── ui_check_model.py
│
├── Train/
│   ├── reTrain.py
│   ├── reTrainFullModel.py
│   └── resume_Train.py
│
└── Utilities/
    ├── Labeling.py
    ├── check_id.py
    ├── compute_occlusion.py
    ├── del_label.py
    └── video_to_frames.py
