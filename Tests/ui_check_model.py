from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO


# =========================
# Config
# =========================
WEIGHTS_PATH = r"D:\NCKH_AI\Crop_img\runs\detect\train\weights\best.pt"  # <-- SỬA LẠI

DEFAULT_CONF = 0.20
DEFAULT_IOU = 0.85
DEFAULT_IMGSZ = 832
DEFAULT_MAX_DET = 300

DEFAULT_LINE_WIDTH = 1
DEFAULT_FONT_SIZE = 0.6  # nhỏ hơn nữa: 0.5 / 0.4

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "0"
    return "cpu"


@dataclass
class InferConfig:
    weights_path: str
    conf: float
    iou: float
    imgsz: int
    max_det: int
    device: str
    class_id: Optional[int]
    line_width: int
    font_size: float
    frame_skip: int = 1
    max_seconds: int = 0  # 0 = unlimited


class YoloEngine:
    def __init__(self) -> None:
        self._model: Optional[YOLO] = None
        self._loaded: Optional[str] = None

    def load(self, weights_path: str) -> YOLO:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy weights: {weights_path}")
        if self._model is None or self._loaded != str(p):
            self._model = YOLO(str(p))
            self._loaded = str(p)
        return self._model

    @staticmethod
    def _predict(model: YOLO, img_bgr: np.ndarray, cfg: InferConfig):
        kwargs = dict(
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            max_det=cfg.max_det,
            device=cfg.device,
            verbose=False,
        )
        if cfg.class_id is not None:
            kwargs["classes"] = [cfg.class_id]
        return model.predict(img_bgr, **kwargs)[0]

    def infer_image(self, image_path: str, cfg: InferConfig) -> Tuple[bool, Image.Image, str]:
        model = self.load(cfg.weights_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise RuntimeError("Không đọc được ảnh.")

        r0 = self._predict(model, img_bgr, cfg)
        n = int(len(r0.boxes)) if r0.boxes is not None else 0
        detected = n > 0

        # Viền mỏng + chữ nhỏ
        pil_annotated = r0.plot(pil=True, line_width=cfg.line_width, font_size=cfg.font_size)

        msg = "CÓ NHẬN" if detected else "KHÔNG NHẬN"
        summary = (
            f"Kết quả: {msg} | detections={n}\n"
            f"conf={cfg.conf} | iou={cfg.iou} | imgsz={cfg.imgsz} | max_det={cfg.max_det} | "
            f"line_width={cfg.line_width} | font_size={cfg.font_size} | device={cfg.device}"
        )
        return detected, pil_annotated, summary

    def annotate_video_to_mp4(
        self,
        input_video: str,
        output_video: str,
        cfg: InferConfig,
        progress_cb=None,  # (done, total)
        status_cb=None,    # (text)
    ) -> Tuple[bool, int, int]:
        model = self.load(cfg.weights_path)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError("Không mở được video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            cap.release()
            raise RuntimeError("Không đọc được kích thước video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_frames = 0
        if cfg.max_seconds and cfg.max_seconds > 0:
            max_frames = int(cfg.max_seconds * fps)

        out_p = Path(output_video)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Không tạo được file output MP4 (VideoWriter).")

        kwargs = dict(
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            max_det=cfg.max_det,
            device=cfg.device,
            verbose=False,
        )
        if cfg.class_id is not None:
            kwargs["classes"] = [cfg.class_id]

        denom = total_frames if total_frames > 0 else 1
        if max_frames > 0 and denom > max_frames:
            denom = max_frames

        if status_cb:
            status_cb(
                f"Input: {input_video} | FPS: {fps:.2f} | Size: {w}x{h} | "
                f"conf={cfg.conf}, iou={cfg.iou}, imgsz={cfg.imgsz}, max_det={cfg.max_det}"
            )

        processed = 0
        written = 0
        detected_any = False
        frame_idx = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_idx += 1
            if cfg.frame_skip > 1 and (frame_idx % cfg.frame_skip != 0):
                continue

            processed += 1
            if max_frames > 0 and processed > max_frames:
                break

            r0 = model.predict(frame_bgr, **kwargs)[0]
            if r0.boxes is not None and len(r0.boxes) > 0:
                detected_any = True

            # plot PIL -> convert sang BGR để ghi video
            im_pil = r0.plot(pil=True, line_width=cfg.line_width, font_size=cfg.font_size)  # RGB
            annotated_bgr = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)
            written += 1

            if progress_cb:
                progress_cb(processed, denom)

        cap.release()
        writer.release()
        return detected_any, processed, written


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YOLO Tkinter UI (Ảnh/Video/Camera) - chỉnh imgsz để tách box")
        self.geometry("1080x760")
        self.minsize(980, 700)

        self.engine = YoloEngine()

        # Shared Vars
        self.var_weights = tk.StringVar(value=WEIGHTS_PATH)
        self.var_conf = tk.DoubleVar(value=DEFAULT_CONF)
        self.var_iou = tk.DoubleVar(value=DEFAULT_IOU)
        self.var_imgsz = tk.IntVar(value=DEFAULT_IMGSZ)
        self.var_max_det = tk.IntVar(value=DEFAULT_MAX_DET)
        self.var_line_width = tk.IntVar(value=DEFAULT_LINE_WIDTH)
        self.var_font_size = tk.DoubleVar(value=DEFAULT_FONT_SIZE)

        self.var_device = tk.StringVar(value=resolve_device())
        self.var_use_class = tk.BooleanVar(value=False)
        self.var_class_id = tk.IntVar(value=0)

        # Video Vars
        self.var_frame_skip = tk.IntVar(value=1)
        self.var_max_seconds = tk.IntVar(value=0)

        # Paths
        self.image_path: Optional[str] = None
        self.video_path: Optional[str] = None
        self.video_out_path: Optional[str] = None
        self.last_annotated_image: Optional[Image.Image] = None

        # Preview handles
        self.tk_preview: Optional[ImageTk.PhotoImage] = None
        self.tk_cam_preview: Optional[ImageTk.PhotoImage] = None

        # Camera state
        self.cam_stop_event = threading.Event()
        self.cam_thread: Optional[threading.Thread] = None
        self.cam_lock = threading.Lock()
        self.cam_latest_pil: Optional[Image.Image] = None
        self.cam_latest_info: str = "Camera: chưa chạy"
        self.cam_running = False

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        # Model config
        lf_model = ttk.LabelFrame(root, text="Model")
        lf_model.pack(fill=tk.X, **pad)

        ttk.Label(lf_model, text="Weights (.pt):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(lf_model, textvariable=self.var_weights, width=95).grid(row=0, column=1, sticky="we", padx=8, pady=6)
        ttk.Button(lf_model, text="Browse", command=self.browse_weights).grid(row=0, column=2, padx=8, pady=6)
        lf_model.columnconfigure(1, weight=1)

        # Params
        lf_params = ttk.LabelFrame(root, text="Inference Params (tách box: tăng imgsz, tăng iou, giảm conf)")
        lf_params.pack(fill=tk.X, **pad)

        ttk.Label(lf_params, text="conf:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(lf_params, textvariable=self.var_conf, width=10).grid(row=0, column=1, padx=8, pady=6)

        ttk.Label(lf_params, text="iou:").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(lf_params, textvariable=self.var_iou, width=10).grid(row=0, column=3, padx=8, pady=6)

        ttk.Label(lf_params, text="imgsz:").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(lf_params, textvariable=self.var_imgsz, width=10).grid(row=0, column=5, padx=8, pady=6)

        ttk.Label(lf_params, text="max_det:").grid(row=0, column=6, sticky="w", padx=8, pady=6)
        ttk.Entry(lf_params, textvariable=self.var_max_det, width=10).grid(row=0, column=7, padx=8, pady=6)

        ttk.Label(lf_params, text="line_width:").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(lf_params, textvariable=self.var_line_width, width=10).grid(row=1, column=1, padx=8, pady=6)

        ttk.Label(lf_params, text="font_size:").grid(row=1, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(lf_params, textvariable=self.var_font_size, width=10).grid(row=1, column=3, padx=8, pady=6)

        ttk.Label(lf_params, text="device:").grid(row=1, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(lf_params, textvariable=self.var_device, width=12).grid(row=1, column=5, padx=8, pady=6)

        chk = ttk.Checkbutton(lf_params, text="Chỉ 1 class_id", variable=self.var_use_class, command=self._toggle_class)
        chk.grid(row=1, column=6, sticky="w", padx=8, pady=6)
        self.ent_class = ttk.Entry(lf_params, textvariable=self.var_class_id, width=10, state="disabled")
        self.ent_class.grid(row=1, column=7, padx=8, pady=6)

        # Tabs
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_image = ttk.Frame(notebook)
        self.tab_video = ttk.Frame(notebook)
        self.tab_cam = ttk.Frame(notebook)

        notebook.add(self.tab_image, text="Ảnh")
        notebook.add(self.tab_video, text="Video")
        notebook.add(self.tab_cam, text="Camera")

        self._build_image_tab()
        self._build_video_tab()
        self._build_cam_tab()

    def _toggle_class(self) -> None:
        self.ent_class.configure(state=("normal" if self.var_use_class.get() else "disabled"))

    def browse_weights(self) -> None:
        p = filedialog.askopenfilename(title="Chọn weights .pt", filetypes=[("PyTorch weights", "*.pt"), ("All", "*.*")])
        if p:
            self.var_weights.set(p)

    def _cfg(self) -> InferConfig:
        conf = float(self.var_conf.get())
        iou = float(self.var_iou.get())
        imgsz = int(self.var_imgsz.get())
        max_det = int(self.var_max_det.get())
        line_width = int(self.var_line_width.get())
        font_size = float(self.var_font_size.get())

        if not (0.0 < conf <= 1.0):
            raise ValueError("conf phải trong (0, 1].")
        if not (0.0 < iou <= 1.0):
            raise ValueError("iou phải trong (0, 1].")
        if imgsz < 320 or imgsz > 1920:
            raise ValueError("imgsz nên trong khoảng 320..1920.")
        if max_det < 1 or max_det > 3000:
            raise ValueError("max_det nên trong khoảng 1..3000.")
        if line_width < 1 or line_width > 10:
            raise ValueError("line_width nên trong khoảng 1..10.")
        

        return InferConfig(
            weights_path=self.var_weights.get().strip(),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=self.var_device.get().strip(),
            class_id=(int(self.var_class_id.get()) if self.var_use_class.get() else None),
            line_width=line_width,
            font_size=font_size,
            frame_skip=max(1, int(self.var_frame_skip.get())),
            max_seconds=max(0, int(self.var_max_seconds.get())),
        )

    # =========================
    # Image Tab
    # =========================
    def _build_image_tab(self) -> None:
        pad = {"padx": 10, "pady": 6}

        lf = ttk.LabelFrame(self.tab_image, text="Input Ảnh")
        lf.pack(fill=tk.X, **pad)

        ttk.Button(lf, text="Thêm ảnh", command=self.pick_image).grid(row=0, column=0, padx=8, pady=10)
        self.btn_run_image = ttk.Button(lf, text="Chạy YOLO (Ảnh)", command=self.run_image_threaded, state="disabled")
        self.btn_run_image.grid(row=0, column=1, padx=8, pady=10)

        self.btn_save_image = ttk.Button(lf, text="Lưu ảnh kết quả", command=self.save_annotated_image, state="disabled")
        self.btn_save_image.grid(row=0, column=2, padx=8, pady=10)

        self.lbl_image_status = ttk.Label(lf, text="Chưa chọn ảnh.")
        self.lbl_image_status.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=6)

        lf2 = ttk.LabelFrame(self.tab_image, text="Preview / Kết quả")
        lf2.pack(fill=tk.BOTH, expand=True, **pad)

        self.preview_label = ttk.Label(lf2)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.var_image_result = tk.StringVar(value="Kết quả: -")
        ttk.Label(lf2, textvariable=self.var_image_result, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)

    def pick_image(self) -> None:
        p = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All", "*.*")],
        )
        if not p:
            return
        if Path(p).suffix.lower() not in IMAGE_EXTS:
            messagebox.showwarning("Cảnh báo", "File có thể không phải ảnh phổ biến.")
        self.image_path = p
        self.lbl_image_status.configure(text=f"Ảnh: {p}")
        self.btn_run_image.configure(state="normal")
        self.btn_save_image.configure(state="disabled")
        self.var_image_result.set("Kết quả: -")
        self._show_image(p)

    def _show_image(self, image_path: str) -> None:
        try:
            pil = Image.open(image_path).convert("RGB")
            self._render_preview(pil)
        except Exception:
            pass

    def _render_preview(self, pil_img: Image.Image) -> None:
        w = self.preview_label.winfo_width() or 980
        h = self.preview_label.winfo_height() or 520
        img = pil_img.copy()
        img.thumbnail((w - 20, h - 20))
        self.tk_preview = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.tk_preview)

    def run_image_threaded(self) -> None:
        if not self.image_path:
            messagebox.showinfo("Thiếu input", "Bạn chưa chọn ảnh.")
            return

        try:
            cfg = self._cfg()
        except Exception as e:
            messagebox.showerror("Sai cấu hình", str(e))
            return

        self.btn_run_image.configure(state="disabled")
        self.btn_save_image.configure(state="disabled")
        self.var_image_result.set("Kết quả: Đang xử lý...")

        def worker():
            try:
                _, pil_annotated, summary = self.engine.infer_image(self.image_path, cfg)
                self.last_annotated_image = pil_annotated
                self.after(0, lambda: self._render_preview(pil_annotated))
                self.after(0, lambda: self.var_image_result.set(summary))
                self.after(0, lambda: self.btn_save_image.configure(state="normal"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
                self.after(0, lambda: self.var_image_result.set("Kết quả: Lỗi"))
            finally:
                self.after(0, lambda: self.btn_run_image.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def save_annotated_image(self) -> None:
        if self.last_annotated_image is None:
            messagebox.showinfo("Thông báo", "Chưa có ảnh kết quả để lưu.")
            return

        p = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            defaultextension=".jpg",
            filetypes=[("JPG", "*.jpg"), ("PNG", "*.png"), ("All", "*.*")],
        )
        if not p:
            return

        try:
            ext = Path(p).suffix.lower()
            if ext in {".jpg", ".jpeg"}:
                self.last_annotated_image.save(p, quality=95)
            else:
                self.last_annotated_image.save(p)
            messagebox.showinfo("OK", f"Đã lưu: {p}")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    # =========================
    # Video Tab
    # =========================
    def _build_video_tab(self) -> None:
        pad = {"padx": 10, "pady": 6}

        lf = ttk.LabelFrame(self.tab_video, text="Input/Output Video")
        lf.pack(fill=tk.X, **pad)

        ttk.Button(lf, text="Chọn video", command=self.pick_video).grid(row=0, column=0, padx=8, pady=10)
        ttk.Button(lf, text="Chọn nơi lưu output (.mp4)", command=self.pick_video_output).grid(row=0, column=1, padx=8, pady=10)

        self.btn_run_video = ttk.Button(lf, text="Chạy & Xuất video", command=self.run_video_threaded, state="disabled")
        self.btn_run_video.grid(row=0, column=2, padx=8, pady=10)

        self.btn_open_video = ttk.Button(lf, text="Open Output", command=self.open_video_output, state="disabled")
        self.btn_open_video.grid(row=0, column=3, padx=8, pady=10)

        self.lbl_video_status = ttk.Label(lf, text="Chưa chọn video.")
        self.lbl_video_status.grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=6)

        lf2 = ttk.LabelFrame(self.tab_video, text="Tuỳ chọn video")
        lf2.pack(fill=tk.X, **pad)

        ttk.Label(lf2, text="frame_skip:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(lf2, textvariable=self.var_frame_skip, width=10).grid(row=0, column=1, padx=8, pady=6)

        ttk.Label(lf2, text="max_seconds (0=all):").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(lf2, textvariable=self.var_max_seconds, width=12).grid(row=0, column=3, padx=8, pady=6)

        lf3 = ttk.LabelFrame(self.tab_video, text="Tiến trình / Kết quả")
        lf3.pack(fill=tk.BOTH, expand=True, **pad)

        self.progress = ttk.Progressbar(lf3, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=10)

        self.var_video_result = tk.StringVar(value="Kết quả: -")
        ttk.Label(lf3, textvariable=self.var_video_result, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=6)

    def pick_video(self) -> None:
        p = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.wmv"), ("All", "*.*")],
        )
        if not p:
            return
        if Path(p).suffix.lower() not in VIDEO_EXTS:
            messagebox.showwarning("Cảnh báo", "File có thể không phải video phổ biến.")
        self.video_path = p
        self._refresh_video_buttons()
        self.lbl_video_status.configure(text=f"Video: {p} | Output: {self.video_out_path or '(chưa chọn)'}")

    def pick_video_output(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Lưu output video (MP4)",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
        )
        if not p:
            return
        if not p.lower().endswith(".mp4"):
            p += ".mp4"
        self.video_out_path = p
        self._refresh_video_buttons()
        self.lbl_video_status.configure(text=f"Video: {self.video_path or '(chưa chọn)'} | Output: {self.video_out_path}")

    def _refresh_video_buttons(self) -> None:
        can_run = bool(self.video_path) and bool(self.video_out_path)
        self.btn_run_video.configure(state=("normal" if can_run else "disabled"))
        self.btn_open_video.configure(state=("normal" if (self.video_out_path and Path(self.video_out_path).exists()) else "disabled"))

    def open_video_output(self) -> None:
        if self.video_out_path and Path(self.video_out_path).exists():
            os.startfile(self.video_out_path)
        else:
            messagebox.showinfo("Thông báo", "Chưa có output hoặc file không tồn tại.")

    def run_video_threaded(self) -> None:
        if not self.video_path:
            messagebox.showinfo("Thiếu input", "Bạn chưa chọn video.")
            return
        if not self.video_out_path:
            messagebox.showinfo("Thiếu output", "Bạn chưa chọn nơi lưu output. Hãy bấm 'Chọn nơi lưu output (.mp4)'.")
            return

        try:
            cfg = self._cfg()
        except Exception as e:
            messagebox.showerror("Sai cấu hình", str(e))
            return

        self.btn_run_video.configure(state="disabled")
        self.btn_open_video.configure(state="disabled")
        self.progress["value"] = 0
        self.var_video_result.set("Kết quả: Đang xử lý...")

        def worker():
            t0 = time.time()

            def progress_cb(done: int, total: int):
                pct = int(min(100, (done / total) * 100)) if total > 0 else 0
                self.after(0, lambda: self.progress.configure(value=pct))

            def status_cb(text: str):
                self.after(0, lambda: self.lbl_video_status.configure(text=text))

            try:
                detected, processed, written = self.engine.annotate_video_to_mp4(
                    input_video=self.video_path,
                    output_video=self.video_out_path,
                    cfg=cfg,
                    progress_cb=progress_cb,
                    status_cb=status_cb,
                )
                dt = time.time() - t0
                msg = "CÓ NHẬN" if detected else "KHÔNG NHẬN"
                self.after(
                    0,
                    lambda: self.var_video_result.set(
                        f"Kết quả: {msg} | processed={processed} | written={written} | time={dt:.1f}s\n"
                        f"Output: {self.video_out_path}\n"
                        f"conf={cfg.conf} | iou={cfg.iou} | imgsz={cfg.imgsz} | max_det={cfg.max_det}"
                    ),
                )
                self.after(0, lambda: self.progress.configure(value=100))
                self.after(0, self._refresh_video_buttons)

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
                self.after(0, lambda: self.var_video_result.set("Kết quả: Lỗi"))
            finally:
                self.after(0, lambda: self.btn_run_video.configure(state=("normal" if (self.video_path and self.video_out_path) else "disabled")))

        threading.Thread(target=worker, daemon=True).start()

    # =========================
    # Camera Tab
    # =========================
    def _build_cam_tab(self) -> None:
        pad = {"padx": 10, "pady": 6}

        lf = ttk.LabelFrame(self.tab_cam, text="Camera (Webcam)")
        lf.pack(fill=tk.X, **pad)

        self.btn_cam_start = ttk.Button(lf, text="Start Camera", command=self.start_camera)
        self.btn_cam_start.grid(row=0, column=0, padx=8, pady=10)

        self.btn_cam_stop = ttk.Button(lf, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_cam_stop.grid(row=0, column=1, padx=8, pady=10)

        self.lbl_cam_status = ttk.Label(lf, text="Camera: chưa chạy")
        self.lbl_cam_status.grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=6)

        lf2 = ttk.LabelFrame(self.tab_cam, text="Preview")
        lf2.pack(fill=tk.BOTH, expand=True, **pad)

        self.cam_preview_label = ttk.Label(lf2)
        self.cam_preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.var_cam_result = tk.StringVar(value="Kết quả: -")
        ttk.Label(lf2, textvariable=self.var_cam_result, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)

    def start_camera(self) -> None:
        if self.cam_running:
            return

        try:
            cfg = self._cfg()
        except Exception as e:
            messagebox.showerror("Sai cấu hình", str(e))
            return

        # test load model sớm để báo lỗi weights ngay
        try:
            self.engine.load(cfg.weights_path)
        except Exception as e:
            messagebox.showerror("Lỗi model", str(e))
            return

        self.cam_stop_event.clear()
        self.cam_running = True
        self.btn_cam_start.configure(state="disabled")
        self.btn_cam_stop.configure(state="normal")
        self.lbl_cam_status.configure(text="Camera: đang chạy...")

        def cam_worker():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.after(0, lambda: messagebox.showerror("Lỗi", "Không mở được webcam (VideoCapture(0))."))
                self.after(0, self.stop_camera)
                return

            model = self.engine.load(cfg.weights_path)

            frame_count = 0
            while not self.cam_stop_event.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                frame_count += 1
                # Thêm frame_skip để giảm tải realtime
                if cfg.frame_skip > 1 and (frame_count % cfg.frame_skip != 0):
                    continue

                r0 = model.predict(
                    frame_bgr,
                    conf=cfg.conf,
                    iou=cfg.iou,
                    imgsz=cfg.imgsz,
                    max_det=cfg.max_det,
                    device=cfg.device,
                    verbose=False,
                    classes=([cfg.class_id] if cfg.class_id is not None else None),
                )[0]

                n = int(len(r0.boxes)) if r0.boxes is not None else 0
                im_pil = r0.plot(pil=True, line_width=cfg.line_width, font_size=cfg.font_size)

                info = (
                    f"Camera: detections={n} | conf={cfg.conf} iou={cfg.iou} imgsz={cfg.imgsz} max_det={cfg.max_det} "
                    f"| line={cfg.line_width} font={cfg.font_size}"
                )

                with self.cam_lock:
                    self.cam_latest_pil = im_pil
                    self.cam_latest_info = info

            cap.release()

        self.cam_thread = threading.Thread(target=cam_worker, daemon=True)
        self.cam_thread.start()

        # bắt đầu loop update UI
        self.after(30, self._cam_ui_loop)

    def _cam_ui_loop(self) -> None:
        if not self.cam_running:
            return

        with self.cam_lock:
            pil_img = self.cam_latest_pil
            info = self.cam_latest_info

        if pil_img is not None:
            # resize phù hợp preview
            w = self.cam_preview_label.winfo_width() or 980
            h = self.cam_preview_label.winfo_height() or 520
            img = pil_img.copy()
            img.thumbnail((w - 20, h - 20))
            self.tk_cam_preview = ImageTk.PhotoImage(img)
            self.cam_preview_label.configure(image=self.tk_cam_preview)

        self.lbl_cam_status.configure(text=info)
        self.var_cam_result.set(info)

        self.after(30, self._cam_ui_loop)

    def stop_camera(self) -> None:
        if not self.cam_running:
            return

        self.cam_running = False
        self.cam_stop_event.set()

        self.btn_cam_start.configure(state="normal")
        self.btn_cam_stop.configure(state="disabled")
        self.lbl_cam_status.configure(text="Camera: đã dừng")

        # Không join lâu trong UI thread; thread là daemon nên sẽ tự dừng.
        with self.cam_lock:
            self.cam_latest_pil = None
            self.cam_latest_info = "Camera: đã dừng"


if __name__ == "__main__":
    app = App()
    app.mainloop()
