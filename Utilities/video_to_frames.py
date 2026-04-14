import cv2
import argparse
from pathlib import Path
import sys

def is_blurry(image, threshold=100.0):
    """
    Kiểm tra ảnh có bị mờ không sử dụng phương sai Laplacian.
    Trả về (True/False, score). Score càng thấp thì càng mờ.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, score

def extract_frames(video_path, output_dir, blur_threshold=100.0, frame_skip=5):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Không thể mở video {video_path}")
        return

    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Chỉ xử lý mỗi (frame_skip + 1) frame
        if frame_count % (frame_skip + 1) == 0:
            blurry, score = is_blurry(frame, blur_threshold)
            
            if not blurry:
                # Lưu frame
                out_name = f"{video_path.stem}_frame_{frame_count:06d}.png"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), frame)
                saved_count += 1
                # print(f"Saved {out_name} (Score: {score:.2f})")
            else:
                pass
                # print(f"Skipped frame {frame_count} (Blurry, Score: {score:.2f})")
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    print(f"Hoàn thành. Đã lưu {saved_count} ảnh vào thư mục '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cắt frame từ video và loại bỏ ảnh mờ.")
    parser.add_argument("--video", type=str, default=r"D:\NCKH_AI\Crop_img\input\20251224_121355.mp4", help="Đường dẫn đến file video")
    parser.add_argument("--output", type=str, default="input", help="Thư mục lưu ảnh (mặc định: input)")
    parser.add_argument("--blur_threshold", type=float, default=110.0, help="Ngưỡng mờ (mặc định: 100.0). Tăng lên nếu muốn lọc kỹ hơn.")
    parser.add_argument("--skip", type=int, default=5, help="Số frame bỏ qua giữa các lần lấy (mặc định: 5)")
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.blur_threshold, args.skip)
