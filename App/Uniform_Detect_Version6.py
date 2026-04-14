"""
AI Uniform Detection System v5 Optimized + Audio
=================================================
Các tối ưu chính:
  - Voting theo ratio-based thay cho vote_threshold cố định
  - Uniform decision dựa trên class/conf/area/score, không còn len(boxes)>0
  - Có thể bật/tắt segmentation tại UI
  - Khi tắt segmentation: ưu tiên detect model nhẹ hơn nếu có file tương ứng
  - Fallback bbox crop nếu segmentation thiếu mask
  - Có benchmark stage time để dễ đánh giá hiệu năng
  - [MỚI] Phát âm thanh cảnh báo khi phát hiện SAI ĐỒNG PHỤC
"""

import cv2
import numpy as np
import threading
import queue
import time
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont
from collections import deque, defaultdict

from ultralytics import YOLO
import supervision as sv
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
import torch
import sys
import playsound


# ===============================================================
#  CẤU HÌNH MẶC ĐỊNH
# ===============================================================
PERSON_MODEL_PATH  = r"yolo11s-seg.pt"
UNIFORM_MODEL_PATH = r"best_final.pt"

# [ÂM THANH] Đường dẫn file âm thanh cảnh báo sai đồng phục
AUDIO_FAIL_PATH = "voice_fail.mp3"
# Khoảng thời gian tối thiểu (giây) giữa hai lần phát âm thanh liên tiếp
AUDIO_COOLDOWN = 2.0

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720
MAX_UNIFORM_BATCH = 8

COLOR_PASS = sv.Color.GREEN
COLOR_FAIL = sv.Color.RED
COLOR_WAIT = sv.Color.YELLOW


def _filename_only(path_str: str) -> str:
    try:
        return Path(path_str).name
    except Exception:
        return str(path_str).replace("\\", "/").split("/")[-1]


def guess_detect_model_path(seg_path: str) -> str | None:
    """
    Từ yolo11s-seg.pt -> thử đoán yolo11s.pt.
    Nếu không đúng pattern thì trả None.
    """
    p = Path(seg_path)
    name = p.name
    if "-seg" not in name:
        return None
    detect_name = name.replace("-seg", "")
    return str(p.with_name(detect_name))


def crop_upper_body_bbox(frame, box, upper_ratio):
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    y2c = int(y1 + (y2 - y1) * upper_ratio)
    x1, y1, x2, y2c = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2c)
    if x2 <= x1 or y2c <= y1:
        return None
    crop = frame[y1:y2c, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def apply_mask_and_crop(frame, mask_tensor, box, upper_ratio):
    """
    Crop upper body theo segmentation mask, nhưng chỉ resize/cắt trong bbox.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    y2c = int(y1 + (y2 - y1) * upper_ratio)
    x1, y1, x2, y2c = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2c)

    bw = x2 - x1
    bh = y2c - y1
    if bw <= 0 or bh <= 0:
        return None

    crop = frame[y1:y2c, x1:x2]
    if crop.size == 0:
        return None

    try:
        mask_np = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (fw, fh), interpolation=cv2.INTER_LINEAR)
        local_mask = mask_resized[y1:y2c, x1:x2]
        masked = (crop * (local_mask > 0.5)[..., np.newaxis]).astype(np.uint8)
        return masked
    except Exception:
        return None


def resize_keep_aspect(image, max_w, max_h):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def cv2_to_pil(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# Font tiếng Việt dùng cho annotation (tải 1 lần, tái dùng toàn bộ chương trình)
def _load_unicode_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Thử lần lượt các font hệ thống hỗ trợ Unicode/tiếng Việt.
    Fallback về font mặc định của PIL nếu không tìm thấy.
    """
    candidates = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Fallback PIL default (không hỗ trợ Unicode nhưng không crash)
    return ImageFont.load_default()


_UNICODE_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int = 18):
    if size not in _UNICODE_FONT_CACHE:
        _UNICODE_FONT_CACHE[size] = _load_unicode_font(size)
    return _UNICODE_FONT_CACHE[size]


def draw_label_unicode(frame: np.ndarray, label: str,
                       x1: int, y1: int,
                       bg_bgr: tuple, font_size: int = 18) -> np.ndarray:
    """
    Vẽ label Unicode/tiếng Việt lên frame (numpy BGR) bằng PIL.
    Trả về frame đã vẽ (cùng array, in-place convert).
    """
    if not label.strip():
        return frame

    font      = _get_font(font_size)
    pil_img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw      = ImageDraw.Draw(pil_img)

    # Đo kích thước text
    try:
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
    except AttributeError:
        # PIL < 9.2 dùng textsize
        tw, th = draw.textsize(label, font=font)

    pad   = 4
    rx1   = x1
    ry1   = max(0, y1 - th - pad * 2)
    rx2   = x1 + tw + pad * 2
    ry2   = y1

    # Nền hình chữ nhật màu status
    r, g, b = bg_bgr[2], bg_bgr[1], bg_bgr[0]   # BGR -> RGB
    draw.rectangle([rx1, ry1, rx2, ry2], fill=(r, g, b))

    # Chữ trắng
    draw.text((rx1 + pad, ry1 + pad), label, font=font, fill=(255, 255, 255))

    # Chuyển lại BGR numpy
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def parse_positive_uniform_classes(model_names) -> set[int]:
    """
    Cố gắng suy ra lớp "đúng đồng phục" từ tên class của model uniform.
    Nếu không đoán được, fallback nhận tất cả class.
    """
    positive_keywords = {
        "uniform", "dongphuc", "đồng phục", "ao", "shirt", "top",
        "pants", "quan", "trouser", "blazer", "jacket"
    }
    positive_ids = set()

    if isinstance(model_names, dict):
        iterable = model_names.items()
    else:
        iterable = enumerate(model_names)

    for cid, name in iterable:
        text = str(name).strip().lower()
        if any(k in text for k in positive_keywords):
            positive_ids.add(int(cid))

    return positive_ids


def evaluate_uniform_result(res, positive_class_ids: set[int] | None,
                            min_conf: float, min_area_ratio: float,
                            min_score: float) -> tuple[bool, float, dict]:
    """
    Quyết định uniform dựa trên:
      - class hợp lệ
      - conf >= min_conf
      - area_ratio >= min_area_ratio
      - total_score >= min_score

    score = conf * sqrt(area_ratio) để box lớn + conf cao có trọng số tốt hơn.
    """
    info = {
        "valid_count": 0,
        "max_conf": 0.0,
        "best_area_ratio": 0.0,
        "total_score": 0.0,
    }

    if res is None or res.boxes is None or len(res.boxes) == 0:
        return False, 0.0, info

    try:
        xyxy    = res.boxes.xyxy.cpu().numpy()
        confs   = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        return False, 0.0, info

    try:
        img_h, img_w = res.orig_shape[:2]
        img_area = max(1.0, float(img_h * img_w))
    except Exception:
        img_area = 1.0

    for box, conf, cid in zip(xyxy, confs, cls_ids):
        if positive_class_ids is not None and len(positive_class_ids) > 0 and cid not in positive_class_ids:
            continue

        x1, y1, x2, y2 = box
        area       = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        area_ratio = area / img_area

        if conf < min_conf or area_ratio < min_area_ratio:
            continue

        score = float(conf) * float(np.sqrt(max(area_ratio, 1e-9)))
        info["valid_count"]      += 1
        info["max_conf"]          = max(info["max_conf"], float(conf))
        info["best_area_ratio"]   = max(info["best_area_ratio"], area_ratio)
        info["total_score"]      += score

    decision   = (info["valid_count"] > 0) and (info["total_score"] >= min_score)
    confidence = info["total_score"]
    return decision, confidence, info


def compute_ratio_status(history: list[int], pass_ratio: float, fail_ratio: float,
                         min_votes: int) -> tuple[str, sv.Color]:
    total = len(history)
    if total < max(1, min_votes):
        return "Verifying...", COLOR_WAIT

    ratio = (sum(history) / total) if total else 0.0
    if ratio >= pass_ratio:
        return "ĐÚNG ĐỒNG PHỤC", COLOR_PASS
    if ratio <= fail_ratio:
        return "SAI ĐỒNG PHỤC", COLOR_FAIL
    return "Verifying...", COLOR_WAIT


def blend_mask_overlay(frame, mask_tensor, color_bgr, alpha=0.35):
    """
    Phục vụ checkbox show_mask.
    """
    try:
        fh, fw = frame.shape[:2]
        mask_np      = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (fw, fh), interpolation=cv2.INTER_LINEAR)
        binary  = mask_resized > 0.5

        overlay = frame.copy()
        overlay[binary] = (
            overlay[binary] * (1.0 - alpha) + np.array(color_bgr, dtype=np.float32) * alpha
        ).astype(np.uint8)
        return overlay
    except Exception:
        return frame


# ===============================================================
#  [MỚI] AUDIO THREAD
#  Phát âm thanh trong thread riêng để không block UI / detection.
# ===============================================================
class AudioThread(threading.Thread):
    """
    Lắng nghe audio_q và phát từng file âm thanh tuần tự.
    Chạy daemon thread nên tự kết thúc khi chương trình đóng.
    """
    def __init__(self, audio_q: queue.Queue):
        super().__init__(daemon=True)
        self.q = audio_q

    def run(self):
        while True:
            sound_path = self.q.get()   # block đến khi có task
            try:
                playsound.playsound(sound_path)
            except Exception:
                pass   # bỏ qua lỗi âm thanh, không crash chương trình


# ===============================================================
#  SHARED STATE
# ===============================================================
class AppState:
    def __init__(self):
        self.conf_person    = 0.5
        self.conf_uniform   = 0.6
        self.check_interval = 1
        self.history_len    = 30
        self.upper_body     = 0.6

        # ratio-based voting
        self.pass_ratio = 0.65
        self.fail_ratio = 0.35
        self.min_votes  = 5

        # uniform decision
        self.uniform_min_area_ratio = 0.015
        self.uniform_min_score      = 0.08

        # toggles
        self.use_segmentation = True
        self.show_box         = True
        self.show_track_id    = True
        self.show_status      = True
        self.show_mask        = False

        # [MỚI] bật/tắt âm thanh từ UI
        self.audio_enabled = True

        self._lock = threading.Lock()
        self.track_history   = defaultdict(lambda: deque(maxlen=self.history_len))
        self.id_status_cache = {}

    def reset_tracking(self):
        with self._lock:
            self.track_history.clear()
            self.id_status_cache.clear()

    def get_status(self, track_id):
        with self._lock:
            return self.id_status_cache.get(track_id, ("Verifying...", COLOR_WAIT))

    def set_status(self, track_id, text, color):
        with self._lock:
            self.id_status_cache[track_id] = (text, color)

    def cleanup_stale_ids(self, active_ids: set):
        with self._lock:
            stale = set(self.track_history.keys()) - active_ids
            for tid in stale:
                self.track_history.pop(tid, None)
                self.id_status_cache.pop(tid, None)

    def append_history(self, track_id, value):
        with self._lock:
            hist = self.track_history[track_id]
            if hist.maxlen != self.history_len:
                trimmed = list(hist)[-self.history_len:]
                self.track_history[track_id] = deque(trimmed, maxlen=self.history_len)
            self.track_history[track_id].append(value)
            return list(self.track_history[track_id])


# ===============================================================
#  THREADED CAMERA
# ===============================================================
class ThreadedCamera:
    def __init__(self, source):
        self.cap     = None
        self.q       = queue.Queue(maxsize=5)
        self.stopped = False
        self._open_camera(source)

        if self.stopped:
            return

        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _open_camera(self, source):
        is_webcam = isinstance(source, int)

        if is_webcam:
            backends = []
            if sys.platform == "win32":
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            elif sys.platform == "linux":
                backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
            else:
                backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

            for backend in backends:
                for idx in range(3):
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        time.sleep(0.5)
                        ret, _ = cap.read()
                        if ret:
                            self.cap = cap
                            self._configure_cap()
                            return
                        cap.release()
            self.stopped = True
        else:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                self.cap = cap
                self._configure_cap()
            else:
                self.stopped = True

    def _configure_cap(self):
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _update(self):
        consecutive_failures = 0
        max_failures = 10

        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self.stopped = True
                        break
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0
                self.q.put(frame)
            else:
                time.sleep(0.001)

    def read(self):
        if self.stopped and self.q.empty():
            return False, None
        try:
            return True, self.q.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def is_opened(self):
        return not self.stopped and self.cap is not None

    def release(self):
        self.stopped = True
        if self.cap:
            self.cap.release()


# ===============================================================
#  DETECTION THREAD
# ===============================================================
class DetectionThread(threading.Thread):
    def __init__(self, source_type, source_path, cfg: AppState,
                 frame_q: queue.Queue, stats_q: queue.Queue,
                 log_q: queue.Queue, audio_q: queue.Queue):   # [MỚI] nhận audio_q
        super().__init__(daemon=True)
        self.source_type = source_type
        self.source_path = source_path
        self.cfg         = cfg
        self.frame_q     = frame_q
        self.stats_q     = stats_q
        self.log_q       = log_q
        self._stop_ev    = threading.Event()

        # [MỚI] audio
        self.audio_q          = audio_q
        self.last_announced   = {}     # tid -> text lần cuối đã thông báo
        self.last_audio_time  = 0.0   # timestamp lần phát âm thanh gần nhất

    def stop(self):
        self._stop_ev.set()

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_q.put(f"[{ts}] {msg}")

    def _emit_stats(self, stats):
        if not self.stats_q.full():
            self.stats_q.put(stats)

    # ----------------------------------------------------------
    #  [MỚI] Phát âm thanh cảnh báo (thread-safe, có cooldown)
    # ----------------------------------------------------------
    def _trigger_audio(self, tid: int, new_text: str):
        """
        Đưa file âm thanh vào audio_q khi:
          - audio_enabled = True
          - status mới là SAI ĐỒNG PHỤC
          - status vừa đổi so với lần thông báo trước của tid này
          - đã qua AUDIO_COOLDOWN giây kể từ lần phát trước
        """
        if not self.cfg.audio_enabled:
            return
        if new_text not in ("SAI ĐỒNG PHỤC",):
            return

        last_text = self.last_announced.get(tid)
        if new_text == last_text:
            return   # không phát lại nếu status không đổi

        now = time.time()
        if now - self.last_audio_time < AUDIO_COOLDOWN:
            return   # cooldown chưa hết

        audio_path = AUDIO_FAIL_PATH
        if Path(audio_path).exists():
            self.audio_q.put(audio_path)
            self.last_audio_time      = now
            self.last_announced[tid]  = new_text
        else:
            # Ghi log một lần nếu file không tồn tại
            if not getattr(self, "_audio_missing_warned", False):
                self._log(f"! File âm thanh không tồn tại: {audio_path}")
                self._audio_missing_warned = True

    def _load_models(self, device):
        self._log("Đang tải model...")
        use_seg = self.cfg.use_segmentation

        person_seg_model = YOLO(PERSON_MODEL_PATH).to(device)
        uniform_model    = YOLO(UNIFORM_MODEL_PATH).to(device)

        person_active_model = person_seg_model
        person_mode         = "seg"

        detect_guess = guess_detect_model_path(PERSON_MODEL_PATH)
        if not use_seg and detect_guess and Path(detect_guess).exists():
            try:
                person_active_model = YOLO(detect_guess).to(device)
                person_mode         = "detect"
                self._log(f"✓ Detect model: {_filename_only(detect_guess)}")
            except Exception as e:
                self._log(f"! Không tải được detect model nhẹ hơn ({e}), fallback sang seg model.")
                person_active_model = person_seg_model
                person_mode         = "seg"
        elif not use_seg:
            self._log("! Không tìm thấy detect model tương ứng, sẽ dùng seg model nhưng crop theo bbox.")

        self._log(f"✓ Person model: {_filename_only(PERSON_MODEL_PATH)}")
        self._log(f"✓ Uniform model: {_filename_only(UNIFORM_MODEL_PATH)}")
        self._log(f"Mode person pipeline: {'Segmentation' if use_seg else 'BBox/Detect'}")

        return person_active_model, person_seg_model, uniform_model, person_mode

    def run(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        use_half = device in ("cuda", "mps")
        self._log(f"Device: {device.upper()} | Half precision: {use_half}")
        self.log_q.put(f"__DEVICE__{device.upper()}")

        try:
            model_person, model_person_seg, model_uniform, person_mode = self._load_models(device)
            positive_class_ids = parse_positive_uniform_classes(model_uniform.names)
            if positive_class_ids:
                cls_info = ", ".join(str(c) for c in sorted(positive_class_ids))
                self._log(f"Uniform positive class IDs: {cls_info}")
            else:
                self._log("! Không suy ra được class dương rõ ràng từ tên class, sẽ chấp nhận mọi class hợp lệ.")
        except Exception as e:
            self._log(f"✗ Lỗi tải model: {e}")
            return

        single_image = (self.source_type == "image")
        cap          = None
        frame_img    = None

        if self.source_type == "webcam":
            cap = ThreadedCamera(0)
            if not cap.is_opened():
                self._log("✗ Không thể mở webcam. Kiểm tra kết nối hoặc quyền truy cập.")
                return
            self._log("Camera đã khởi động.")
        elif self.source_type == "video":
            cap = ThreadedCamera(self.source_path)
            self._log(f"Video: {_filename_only(self.source_path)}")
        elif single_image:
            frame_img = cv2.imread(self.source_path)
            if frame_img is None:
                self._log("✗ Không đọc được ảnh.")
                return

        frame_count = 0
        fps_counter = 0
        fps_start   = time.time()
        fps_display = 0.0

        # benchmark trung bình trượt
        bench = {
            "person_ms":  deque(maxlen=30),
            "uniform_ms": deque(maxlen=30),
            "crop_ms":    deque(maxlen=30),
        }

        while not self._stop_ev.is_set():
            if single_image:
                frame = frame_img.copy()
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    self._log("Hết nguồn video / mất kết nối camera.")
                    break

            frame_count += 1
            fps_counter += 1

            now = time.time()
            if now - fps_start >= 1.0:
                fps_display = fps_counter / max(1e-6, (now - fps_start))
                fps_counter = 0
                fps_start   = now

            st = self.cfg
            if self._stop_ev.is_set():
                break

            person_t0 = time.perf_counter()
            results   = model_person.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                conf=st.conf_person,
                verbose=False,
                retina_masks=False,
                half=use_half
            )
            bench["person_ms"].append((time.perf_counter() - person_t0) * 1000.0)
            result = results[0]

            cnt_pass  = 0
            cnt_fail  = 0
            tids_list  = []
            boxes_list = []
            masks_map  = {}
            detections = None

            if result.boxes is not None and result.boxes.id is not None:
                try:
                    detections = sv.Detections.from_ultralytics(result)
                    valid_idx  = detections.tracker_id != None
                    detections = detections[valid_idx]
                except Exception:
                    detections = None

                if detections is not None and len(detections.xyxy) > 0:
                    raw_masks = None
                    if st.use_segmentation and getattr(result, "masks", None) is not None and result.masks is not None:
                        try:
                            raw_masks = result.masks.data[valid_idx]
                        except Exception:
                            raw_masks = None

                    crop_t0    = time.perf_counter()
                    crops_list = []

                    for i, (box, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                        if tid is None:
                            continue

                        mask_tensor = None
                        if raw_masks is not None and i < len(raw_masks):
                            mask_tensor = raw_masks[i]

                        if st.use_segmentation and mask_tensor is not None:
                            crop = apply_mask_and_crop(frame, mask_tensor, box, st.upper_body)
                        else:
                            crop = crop_upper_body_bbox(frame, box, st.upper_body)

                        if crop is None or crop.size == 0:
                            continue

                        tid = int(tid)
                        crops_list.append(crop)
                        tids_list.append(tid)
                        boxes_list.append(box)
                        masks_map[tid] = mask_tensor

                    bench["crop_ms"].append((time.perf_counter() - crop_t0) * 1000.0)

                    st.cleanup_stale_ids(set(tids_list))

                    should_check = single_image or (frame_count % max(1, st.check_interval) == 0)
                    if should_check and crops_list:
                        if self._stop_ev.is_set():
                            break

                        uniform_t0 = time.perf_counter()
                        uni_results = []
                        for i in range(0, len(crops_list), MAX_UNIFORM_BATCH):
                            chunk = crops_list[i:i + MAX_UNIFORM_BATCH]
                            uni_results.extend(
                                model_uniform.predict(
                                    chunk,
                                    conf=max(0.05, st.conf_uniform * 0.5),
                                    verbose=False,
                                    half=use_half
                                )
                            )
                        bench["uniform_ms"].append((time.perf_counter() - uniform_t0) * 1000.0)

                        for res, tid in zip(uni_results, tids_list):
                            is_uniform, _, info = evaluate_uniform_result(
                                res=res,
                                positive_class_ids=positive_class_ids if positive_class_ids else None,
                                min_conf=st.conf_uniform,
                                min_area_ratio=st.uniform_min_area_ratio,
                                min_score=st.uniform_min_score,
                            )
                            history    = st.append_history(tid, 1 if is_uniform else 0)
                            text, color = compute_ratio_status(
                                history=history,
                                pass_ratio=st.pass_ratio,
                                fail_ratio=st.fail_ratio,
                                min_votes=1 if single_image else st.min_votes,
                            )
                            if single_image:
                                if is_uniform:
                                    text, color = "ĐÚNG ĐỒNG PHỤC", COLOR_PASS
                                else:
                                    text, color = "SAI ĐỒNG PHỤC", COLOR_FAIL
                            st.set_status(tid, text, color)

                            # [MỚI] Kích hoạt âm thanh khi status đổi sang SAI ĐỒNG PHỤC
                            self._trigger_audio(tid, text)

                            # debug info nhẹ vào log mỗi khi đủ dữ liệu và trạng thái rõ ràng
                            if len(history) == 1 or len(history) == st.min_votes:
                                ratio = sum(history) / max(1, len(history))
                                self._log(
                                    f"TID {tid}: valid={info['valid_count']} max_conf={info['max_conf']:.2f} "
                                    f"area={info['best_area_ratio']:.3f} score={info['total_score']:.3f} ratio={ratio:.2f}"
                                )

            # annotation
            if detections is not None and len(boxes_list) > 0:
                if st.show_mask and st.use_segmentation:
                    for tid in tids_list:
                        mask_tensor = masks_map.get(tid)
                        if mask_tensor is None:
                            continue
                        status_text, col = st.get_status(tid)
                        bgr   = (col.b, col.g, col.r)
                        frame = blend_mask_overlay(frame, mask_tensor, bgr)

                for box, tid in zip(boxes_list, tids_list):
                    status_text, col = st.get_status(tid)
                    bgr = (col.b, col.g, col.r)
                    x1, y1, x2, y2 = map(int, box)

                    if status_text == "ĐÚNG ĐỒNG PHỤC":
                        cnt_pass += 1
                    elif status_text == "SAI ĐỒNG PHỤC":
                        cnt_fail += 1

                    if st.show_box:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

                    parts = []
                    if st.show_track_id:
                        parts.append(f"ID:{tid}")
                    if st.show_status:
                        parts.append(status_text)
                    label = " ".join(parts)

                    if label.strip():
                        # Dùng PIL để vẽ text Unicode/tiếng Việt (cv2.putText không hỗ trợ)
                        frame = draw_label_unicode(frame, label, x1, y1, bgr, font_size=18)

            display = resize_keep_aspect(frame, MAX_DISPLAY_W, MAX_DISPLAY_H)
            if self.frame_q.full():
                try:
                    self.frame_q.get_nowait()
                except queue.Empty:
                    pass
            self.frame_q.put_nowait(display)

            total_persons = len(tids_list)
            stats = {
                "fps":        fps_display,
                "persons":    total_persons,
                "pass":       cnt_pass,
                "fail":       cnt_fail,
                "person_ms":  (sum(bench["person_ms"])  / len(bench["person_ms"]))  if bench["person_ms"]  else 0.0,
                "uniform_ms": (sum(bench["uniform_ms"]) / len(bench["uniform_ms"])) if bench["uniform_ms"] else 0.0,
                "crop_ms":    (sum(bench["crop_ms"])    / len(bench["crop_ms"]))    if bench["crop_ms"]    else 0.0,
                "seg_mode":   "ON" if st.use_segmentation else "OFF",
                "person_mode": person_mode,
            }
            self._emit_stats(stats)

            if single_image:
                break

        if cap:
            cap.release()
        self._log("Detection đã dừng.")
        self.log_q.put("__DONE__")


# ===============================================================
#  UI CHÍNH
# ===============================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Uniform Detection System v5 Optimized")
        self.geometry("1450x920")
        self.minsize(1150, 720)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.cfg = AppState()
        self.det_thread: DetectionThread | None = None
        self.frame_q = queue.Queue(maxsize=2)
        self.stats_q = queue.Queue(maxsize=5)
        self.log_q   = queue.Queue(maxsize=200)

        # [MỚI] Khởi tạo audio queue và thread
        self.audio_q = queue.Queue()
        self.audio_thread = AudioThread(self.audio_q)
        self.audio_thread.start()

        self._build_ui()
        self._poll()

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_control_panel()
        self._build_video_panel()
        self._build_stats_panel()
        self._build_log_console()

    def _build_header(self):
        hdr = ctk.CTkFrame(self, height=60, corner_radius=0)
        hdr.grid(row=0, column=0, columnspan=3, sticky="ew")
        hdr.grid_propagate(False)

        ctk.CTkLabel(
            hdr,
            text="AI SMART UNIFORM DETECTION SYSTEM",
            font=("Arial", 22, "bold")
        ).pack(side="left", padx=20, pady=10)

        ctk.CTkLabel(
            hdr,
            text="Voting ratio-based | Uniform score/class/conf | Toggle segmentation",
            font=("Arial", 12)
        ).pack(side="left", padx=20)

        self.lbl_device = ctk.CTkLabel(hdr, text="Device: –", font=("Arial", 12))
        self.lbl_device.pack(side="left", padx=10)

        self.lbl_sys_status = ctk.CTkLabel(
            hdr, text="● Ready", font=("Arial", 12, "bold"), text_color="green"
        )
        self.lbl_sys_status.pack(side="right", padx=20)

    def _build_control_panel(self):
        ctrl = ctk.CTkScrollableFrame(self, width=320, corner_radius=8)
        ctrl.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(8, 4), pady=8)

        self._section(ctrl, "📥  Nguồn đầu vào")
        ctk.CTkButton(
            ctrl, text="📸  Webcam Realtime", height=38,
            fg_color="#2E86C1", command=lambda: self._start("webcam")
        ).pack(fill="x", pady=3)
        ctk.CTkButton(
            ctrl, text="🎥  Chọn Video File", height=38,
            fg_color="#28B463", command=lambda: self._start("video")
        ).pack(fill="x", pady=3)
        ctk.CTkButton(
            ctrl, text="🖼️  Chọn Image File", height=38,
            fg_color="#D35400", command=lambda: self._start("image")
        ).pack(fill="x", pady=3)

        self._section(ctrl, "🧠  Model")
        ctk.CTkLabel(ctrl, text="Person model:", anchor="w").pack(fill="x")
        ctk.CTkLabel(
            ctrl, text=_filename_only(PERSON_MODEL_PATH),
            font=("Arial", 10), text_color="gray", anchor="w"
        ).pack(fill="x")
        detect_guess = guess_detect_model_path(PERSON_MODEL_PATH)
        ctk.CTkLabel(ctrl, text="Detect fallback:", anchor="w").pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(
            ctrl,
            text=_filename_only(detect_guess) if detect_guess else "Không suy ra được",
            font=("Arial", 10), text_color="gray", anchor="w"
        ).pack(fill="x")
        ctk.CTkLabel(ctrl, text="Uniform model:", anchor="w").pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(
            ctrl, text=_filename_only(UNIFORM_MODEL_PATH),
            font=("Arial", 10), text_color="gray", anchor="w"
        ).pack(fill="x")

        self._section(ctrl, "⚙️  Detection Settings")
        self._slider(
            ctrl, "Person Confidence", 0.1, 1.0, self.cfg.conf_person,
            lambda v: setattr(self.cfg, "conf_person", round(v, 2))
        )
        self._slider(
            ctrl, "Uniform Confidence", 0.1, 1.0, self.cfg.conf_uniform,
            lambda v: setattr(self.cfg, "conf_uniform", round(v, 2))
        )
        self._slider(
            ctrl, "Frame Skip Interval", 1, 10, self.cfg.check_interval,
            lambda v: setattr(self.cfg, "check_interval", int(v)), integer=True
        )
        self._slider(
            ctrl, "Tracking History", 10, 60, self.cfg.history_len,
            lambda v: setattr(self.cfg, "history_len", int(v)), integer=True
        )
        self._slider(
            ctrl, "Upper-body Ratio", 0.3, 0.9, self.cfg.upper_body,
            lambda v: setattr(self.cfg, "upper_body", round(v, 2))
        )

        self._section(ctrl, "🧮  Ratio Voting")
        self._slider(
            ctrl, "Pass Ratio", 0.50, 0.95, self.cfg.pass_ratio,
            lambda v: setattr(self.cfg, "pass_ratio", round(v, 2))
        )
        self._slider(
            ctrl, "Fail Ratio", 0.05, 0.50, self.cfg.fail_ratio,
            lambda v: setattr(self.cfg, "fail_ratio", round(v, 2))
        )
        self._slider(
            ctrl, "Min Votes", 1, 20, self.cfg.min_votes,
            lambda v: setattr(self.cfg, "min_votes", int(v)), integer=True
        )

        self._section(ctrl, "🎯  Uniform Scoring")
        self._slider(
            ctrl, "Min Area Ratio", 0.001, 0.050, self.cfg.uniform_min_area_ratio,
            lambda v: setattr(self.cfg, "uniform_min_area_ratio", round(v, 3)),
            steps=49
        )
        self._slider(
            ctrl, "Min Score", 0.01, 0.30, self.cfg.uniform_min_score,
            lambda v: setattr(self.cfg, "uniform_min_score", round(v, 3)),
            steps=29
        )

        self._section(ctrl, "🖥️  Display / Pipeline")
        self._checkbox(
            ctrl, "Dùng segmentation", self.cfg.use_segmentation,
            lambda v: setattr(self.cfg, "use_segmentation", bool(v))
        )
        self._checkbox(
            ctrl, "Hiện bounding box", self.cfg.show_box,
            lambda v: setattr(self.cfg, "show_box", bool(v))
        )
        self._checkbox(
            ctrl, "Hiện Tracking ID", self.cfg.show_track_id,
            lambda v: setattr(self.cfg, "show_track_id", bool(v))
        )
        self._checkbox(
            ctrl, "Hiện trạng thái", self.cfg.show_status,
            lambda v: setattr(self.cfg, "show_status", bool(v))
        )
        self._checkbox(
            ctrl, "Hiện Seg. Mask", self.cfg.show_mask,
            lambda v: setattr(self.cfg, "show_mask", bool(v))
        )

        # [MỚI] checkbox bật/tắt âm thanh cảnh báo
        self._section(ctrl, "🔊  Âm thanh cảnh báo")
        self._checkbox(
            ctrl, "Bật âm thanh SAI đồng phục", self.cfg.audio_enabled,
            lambda v: setattr(self.cfg, "audio_enabled", bool(v))
        )
        ctk.CTkLabel(
            ctrl, text=f"File: {_filename_only(AUDIO_FAIL_PATH)}",
            font=("Arial", 10), text_color="gray", anchor="w"
        ).pack(fill="x")
        ctk.CTkLabel(
            ctrl, text=f"Cooldown: {AUDIO_COOLDOWN:.0f}s",
            font=("Arial", 10), text_color="gray", anchor="w"
        ).pack(fill="x")

        self._section(ctrl, "🎛️  Điều khiển hệ thống")
        ctk.CTkButton(
            ctrl, text="▶  Start Detection", height=38,
            fg_color="#1A8F4C", command=self._btn_start
        ).pack(fill="x", pady=3)
        ctk.CTkButton(
            ctrl, text="⏹  Stop Detection", height=38,
            fg_color="#C0392B", command=self._btn_stop
        ).pack(fill="x", pady=3)
        ctk.CTkButton(
            ctrl, text="🔄  Reset Tracking", height=38,
            fg_color="#7D3C98", command=self._btn_reset
        ).pack(fill="x", pady=3)

    def _build_video_panel(self):
        vid_frame = ctk.CTkFrame(self, corner_radius=8)
        vid_frame.grid(row=1, column=1, sticky="nsew", padx=4, pady=8)
        vid_frame.grid_rowconfigure(0, weight=1)
        vid_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            vid_frame, text="Chưa có tín hiệu video",
            font=("Arial", 16), text_color="gray"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

    def _build_stats_panel(self):
        stats = ctk.CTkFrame(self, width=260, corner_radius=8)
        stats.grid(row=1, column=2, sticky="nsew", padx=(4, 8), pady=8)
        stats.grid_propagate(False)

        ctk.CTkLabel(stats, text="📊  Thống kê", font=("Arial", 14, "bold")).pack(pady=(12, 6))

        def metric_row(parent, label):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=4)
            ctk.CTkLabel(row, text=label, font=("Arial", 11), text_color="gray").pack(anchor="w")
            val = ctk.CTkLabel(row, text="—", font=("Arial", 20, "bold"))
            val.pack(anchor="w")
            return val

        self.lbl_fps        = metric_row(stats, "FPS")
        self.lbl_persons    = metric_row(stats, "Người phát hiện")
        self.lbl_pass       = metric_row(stats, "✅  Đúng đồng phục")
        self.lbl_fail       = metric_row(stats, "❌  Sai đồng phục")
        self.lbl_segmode    = metric_row(stats, "Segmentation")
        self.lbl_person_ms  = metric_row(stats, "Person stage (ms)")
        self.lbl_crop_ms    = metric_row(stats, "Crop stage (ms)")
        self.lbl_uniform_ms = metric_row(stats, "Uniform stage (ms)")

    def _build_log_console(self):
        log_frame = ctk.CTkFrame(self, height=180, corner_radius=8)
        log_frame.grid(row=2, column=1, columnspan=2, sticky="nsew", padx=(4, 8), pady=(0, 8))
        log_frame.grid_propagate(False)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(log_frame, text="📋  System Log", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(6, 0)
        )
        self.log_box = ctk.CTkTextbox(log_frame, font=("Courier", 11), activate_scrollbars=True)
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.log_box.configure(state="disabled")

    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title, font=("Arial", 12, "bold"), anchor="w").pack(fill="x", pady=(14, 2))
        ctk.CTkFrame(parent, height=1, fg_color="gray30").pack(fill="x", pady=(0, 6))

    def _slider(self, parent, label, mn, mx, default, callback, integer=False, steps=None):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        val_var = tk.DoubleVar(value=default)
        disp = ctk.CTkLabel(
            row, width=52,
            text=str(int(default)) if integer else (f"{default:.3f}" if default < 0.1 else f"{default:.2f}")
        )
        ctk.CTkLabel(row, text=label, anchor="w").pack(fill="x")

        if steps is None:
            steps = int((mx - mn) * 10) if not integer else int(mx - mn)

        slider = ctk.CTkSlider(
            row, from_=mn, to=mx, variable=val_var, number_of_steps=max(1, int(steps))
        )

        def _on_change(v):
            val = float(v)
            if integer:
                disp.configure(text=str(int(val)))
            else:
                disp.configure(text=f"{val:.3f}" if mx <= 0.3 else f"{val:.2f}")
            callback(val)

        slider.configure(command=_on_change)
        slider.pack(side="left", fill="x", expand=True, padx=(0, 4))
        disp.pack(side="right")

    def _checkbox(self, parent, label, default, callback):
        var = tk.BooleanVar(value=default)

        def _on():
            callback(var.get())

        ctk.CTkCheckBox(parent, text=label, variable=var, command=_on).pack(anchor="w", pady=2)

    def _resolve_source(self, source_type):
        if source_type == "webcam":
            return "webcam", None
        if source_type == "video":
            path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
            return ("video", path) if path else (None, None)
        if source_type == "image":
            path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp")])
            return ("image", path) if path else (None, None)
        return None, None

    def _start(self, source_type):
        stype, path = self._resolve_source(source_type)
        if not stype:
            return
        self._stop_thread()
        self._launch(stype, path)

    def _btn_start(self):
        if self.det_thread and self.det_thread.is_alive():
            self._log_ui("Đang chạy rồi. Nhấn Stop trước.")
            return
        self._log_ui("Nhấn nút Webcam / Video / Image để chọn nguồn.")

    def _btn_stop(self):
        self._stop_thread()

    def _btn_reset(self):
        self.cfg.reset_tracking()
        self._log_ui("Đã reset tracking data.")

    def _launch(self, stype, path):
        self.det_thread = DetectionThread(
            stype, path, self.cfg,
            self.frame_q, self.stats_q, self.log_q,
            self.audio_q   # [MỚI] truyền audio_q vào detection thread
        )
        self.det_thread.start()
        self.lbl_sys_status.configure(text="● Running", text_color="lime")
        self._log_ui(f"Khởi động chế độ: {stype.upper()}")

    def _stop_thread(self):
        if self.det_thread and self.det_thread.is_alive():
            self.det_thread.stop()
            self.det_thread.join(timeout=3)
        self.lbl_sys_status.configure(text="● Ready", text_color="green")

    def _poll(self):
        try:
            frame  = self.frame_q.get_nowait()
            img_tk = cv2_to_pil(frame)
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk
        except queue.Empty:
            pass

        try:
            s = self.stats_q.get_nowait()
            self.lbl_fps.configure(text=f"{s['fps']:.1f}")
            self.lbl_persons.configure(text=str(s["persons"]))
            self.lbl_pass.configure(text=str(s["pass"]), text_color="lime")
            self.lbl_fail.configure(
                text=str(s["fail"]),
                text_color="#FF6B6B" if s["fail"] > 0 else "white"
            )
            self.lbl_segmode.configure(text=f"{s['seg_mode']} ({s['person_mode']})")
            self.lbl_person_ms.configure(text=f"{s['person_ms']:.1f}")
            self.lbl_crop_ms.configure(text=f"{s['crop_ms']:.1f}")
            self.lbl_uniform_ms.configure(text=f"{s['uniform_ms']:.1f}")
        except queue.Empty:
            pass

        while True:
            try:
                msg = self.log_q.get_nowait()
                if msg.startswith("__DEVICE__"):
                    device_name = msg.replace("__DEVICE__", "")
                    self.lbl_device.configure(text=f"Device: {device_name}")
                elif msg == "__DONE__":
                    self.lbl_sys_status.configure(text="● Ready", text_color="green")
                else:
                    self._log_ui(msg)
            except queue.Empty:
                break

        try:
            if self.winfo_exists():
                self.after(16, self._poll)
        except Exception:
            pass

    def _log_ui(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def on_close(self):
        self._stop_thread()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()