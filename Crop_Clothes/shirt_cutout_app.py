import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import queue
import torch
import numpy as np
import time
from pathlib import Path

# Import logic
try:
    from crop_img import ClothesSegmenter
    from video_to_frames import is_blurry
except ImportError as e:
    print(f"Import Error: {e}")
    pass

# --- Logic from batch_cut_shirt.py (Adapted) ---

def ensure_binary_mask(mask_gray: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    return m

def clean_mask(mask_bin: np.ndarray, min_area=5000, keep_largest=True) -> np.ndarray:
    # Morphological operations to remove noise and fill holes
    k = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    
    # Close to fill holes, Open to remove noise
    m = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    if not keep_largest:
        return m

    # Keep only the largest connected component
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    best_area = int(stats[best_idx, cv2.CC_STAT_AREA])
    
    if best_area < min_area:
        return np.zeros_like(m) # Too small, return empty

    out = np.zeros_like(m)
    out[labels == best_idx] = 255
    return out

def get_bbox_from_mask(mask_bin: np.ndarray):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def cut_shirt_rgba(image_rgb: np.ndarray, mask_bin: np.ndarray, pad=10):
    bbox = get_bbox_from_mask(mask_bin)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    h, w = mask_bin.shape[:2]
    
    # Add padding
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w - 1, x2 + pad)
    y2p = min(h - 1, y2 + pad)

    # Crop
    crop_img = image_rgb[y1p:y2p + 1, x1p:x2p + 1]
    crop_mask = mask_bin[y1p:y2p + 1, x1p:x2p + 1]

    # Create RGBA
    rgba = np.dstack([crop_img, crop_mask])
    return rgba

# --- Main Application ---

class ShirtCutoutApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shirt Cutout Tool (Video to Transparent PNG)")
        self.root.geometry("1000x700")
        
        self.setup_ui()
        
        self.segmenter = None
        self.processing = False
        self.queue = queue.Queue()
        
        # Load model in background
        self.log("Loading AI Model... Please wait.")
        threading.Thread(target=self.load_model, daemon=True).start()
        
        self.root.after(100, self.process_queue)

    def setup_ui(self):
        # 1. Control Panel
        control_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Video Selection
        frame_video = ttk.Frame(control_frame)
        frame_video.pack(fill=tk.X, pady=2)
        ttk.Label(frame_video, text="Video File:", width=12).pack(side=tk.LEFT)
        self.video_path_var = tk.StringVar()
        ttk.Entry(frame_video, textvariable=self.video_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(frame_video, text="Browse...", command=self.select_video).pack(side=tk.LEFT)

        # Output Selection
        frame_out = ttk.Frame(control_frame)
        frame_out.pack(fill=tk.X, pady=2)
        ttk.Label(frame_out, text="Output Dir:", width=12).pack(side=tk.LEFT)
        self.output_path_var = tk.StringVar()
        ttk.Entry(frame_out, textvariable=self.output_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(frame_out, text="Browse...", command=self.select_output_dir).pack(side=tk.LEFT)

        # Parameters
        frame_params = ttk.Frame(control_frame)
        frame_params.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame_params, text="Frame Skip:").pack(side=tk.LEFT)
        self.frame_skip_var = tk.IntVar(value=5)
        ttk.Entry(frame_params, textvariable=self.frame_skip_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame_params, text="Blur Threshold:").pack(side=tk.LEFT, padx=(15, 0))
        self.blur_threshold_var = tk.DoubleVar(value=100.0)
        ttk.Entry(frame_params, textvariable=self.blur_threshold_var, width=8).pack(side=tk.LEFT, padx=5)

        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        self.btn_start = ttk.Button(btn_frame, text="START PROCESSING", command=self.start_processing)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(btn_frame, text="STOP", command=self.stop_processing).pack(side=tk.LEFT, padx=5)

        # 2. Display Area
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left: Original, Right: Result
        self.panel_orig = self.create_image_panel(display_frame, "Original Frame", 0)
        self.panel_res = self.create_image_panel(display_frame, "Cutout Result (Transparent)", 1)

        # 3. Log/Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

    def create_image_panel(self, parent, title, col):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, sticky="nsew", padx=5)
        parent.columnconfigure(col, weight=1)
        
        ttk.Label(frame, text=title, font=("Arial", 10, "bold")).pack(pady=5)
        lbl = ttk.Label(frame, background="#333333") # Dark background to see transparency
        lbl.pack(expand=True, fill=tk.BOTH)
        return lbl

    def log(self, msg):
        self.queue.put(("status", msg))

    def load_model(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.segmenter = ClothesSegmenter("mattmdjaga/segformer_b2_clothes", device)
            self.log(f"Model loaded successfully on {device}")
        except Exception as e:
            self.log(f"Error loading model: {e}")

    def select_video(self):
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if p: self.video_path_var.set(p)

    def select_output_dir(self):
        p = filedialog.askdirectory()
        if p: self.output_path_var.set(p)

    def start_processing(self):
        if not self.segmenter:
            messagebox.showerror("Error", "Model is not loaded yet!")
            return
        
        video = self.video_path_var.get()
        out_dir = self.output_path_var.get()
        
        if not video or not out_dir:
            messagebox.showwarning("Missing Info", "Please select both Video and Output Directory.")
            return
            
        self.processing = True
        self.btn_start.config(state=tk.DISABLED)
        
        skip = self.frame_skip_var.get()
        blur = self.blur_threshold_var.get()
        
        threading.Thread(target=self.run_pipeline, args=(video, out_dir, skip, blur), daemon=True).start()

    def stop_processing(self):
        self.processing = False
        self.log("Stopping...")

    def run_pipeline(self, video_path, out_dir_str, frame_skip, blur_thresh):
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        count = 0
        saved_count = 0
        
        while self.processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            count += 1
            if count % (frame_skip + 1) != 0:
                continue
                
            # Check blur
            is_blur, score = is_blurry(frame, threshold=blur_thresh)
            if is_blur:
                self.log(f"Frame {count}: Skipped (Blurry score {score:.1f})")
                continue
            
            # Process
            try:
                # Convert to RGB for model
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                # Predict
                pred = self.segmenter.predict_label_map(pil_img)
                
                # Extract Upper-clothes (Label 4)
                # You can add more labels here if needed e.g. [4, 7] for dress
                mask_raw = np.isin(pred, [4]).astype(np.uint8) * 255
                
                # Clean mask
                mask_clean = clean_mask(mask_raw)
                
                # Cutout
                rgba_result = cut_shirt_rgba(img_rgb, mask_clean)
                
                if rgba_result is not None:
                    # Save
                    filename = f"shirt_{count:06d}.png"
                    save_path = out_dir / filename
                    
                    # Convert RGBA to PIL to save
                    res_pil = Image.fromarray(rgba_result)
                    res_pil.save(save_path)
                    
                    saved_count += 1
                    self.log(f"Saved: {filename}")
                    
                    # Update UI
                    self.queue.put(("update_img", (img_rgb, rgba_result)))
                else:
                    self.log(f"Frame {count}: No shirt found")
                    
            except Exception as e:
                print(f"Error: {e}")
            
            # Small delay to keep UI responsive
            # time.sleep(0.01)

        cap.release()
        self.processing = False
        self.queue.put(("done", saved_count))

    def process_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == "status":
                    self.status_var.set(data)
                    
                elif msg_type == "update_img":
                    orig, res = data
                    self.show_image(self.panel_orig, orig)
                    self.show_image(self.panel_res, res)
                    
                elif msg_type == "done":
                    self.status_var.set(f"Finished! Saved {data} images.")
                    self.btn_start.config(state=tk.NORMAL)
                    messagebox.showinfo("Done", f"Processing complete.\nSaved {data} images.")
                    
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def show_image(self, label_widget, img_array):
        # Resize to fit panel
        h, w = img_array.shape[:2]
        target_h = 400
        scale = target_h / h
        target_w = int(w * scale)
        
        img_pil = Image.fromarray(img_array)
        img_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ShirtCutoutApp(root)
    root.mainloop()
