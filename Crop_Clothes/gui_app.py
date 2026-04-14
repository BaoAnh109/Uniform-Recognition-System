import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import threading
import queue
import torch
import numpy as np
import time

# Import logic from existing files
try:
    from crop_img import ClothesSegmenter, postprocess_mask, mask_to_bbox, pad_bbox
    from video_to_frames import is_blurry
    from mask_to_yolo import mask_to_yolo_bbox
except ImportError as e:
    print(f"Import Error: {e}")
    # Mock classes/functions for testing if imports fail
    pass

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Processing Pipeline")
        self.root.geometry("1200x600")
        
        self.setup_ui()
        
        self.segmenter = None
        self.processing = False
        self.queue = queue.Queue()
        
        # Load model in background
        self.status_var.set("Loading model... Please wait.")
        threading.Thread(target=self.load_model, daemon=True).start()
        
        self.root.after(100, self.process_queue)

    def setup_ui(self):
        # Top frame for controls
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Button(top_frame, text="Select Video", command=self.select_video).pack(side=tk.LEFT, padx=5)
        self.video_path_var = tk.StringVar()
        ttk.Label(top_frame, textvariable=self.video_path_var).pack(side=tk.LEFT, padx=5)

        ttk.Button(top_frame, text="Select Output Dir", command=self.select_output_dir).pack(side=tk.LEFT, padx=5)
        self.output_path_var = tk.StringVar()
        ttk.Label(top_frame, textvariable=self.output_path_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(top_frame, text="Start Processing", command=self.start_processing).pack(side=tk.LEFT, padx=20)
        ttk.Button(top_frame, text="Stop", command=self.stop_processing).pack(side=tk.LEFT, padx=5)

        # Settings frame
        settings_frame = ttk.Frame(self.root, padding=5)
        settings_frame.pack(fill=tk.X)

        ttk.Label(settings_frame, text="Frame Skip:").pack(side=tk.LEFT, padx=5)
        self.frame_skip_var = tk.IntVar(value=5)
        ttk.Entry(settings_frame, textvariable=self.frame_skip_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(settings_frame, text="Blur Threshold:").pack(side=tk.LEFT, padx=5)
        self.blur_threshold_var = tk.DoubleVar(value=100.0)
        ttk.Entry(settings_frame, textvariable=self.blur_threshold_var, width=8).pack(side=tk.LEFT, padx=5)

        # Main content area
        content_frame = ttk.Frame(self.root, padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Grid layout for images
        # Col 0: Input Frame
        # Col 1: Mask
        # Col 2: Result (Cropped + BBox)
        
        self.create_image_panel(content_frame, "Input Frame", 0)
        self.create_image_panel(content_frame, "Segmentation Mask", 1)
        self.create_image_panel(content_frame, "Result (Cropped + BBox)", 2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)

    def create_image_panel(self, parent, title, col):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")
        parent.columnconfigure(col, weight=1)
        
        ttk.Label(frame, text=title, font=("Arial", 12, "bold")).pack()
        
        lbl = ttk.Label(frame)
        lbl.pack(expand=True)
        
        # Store reference to label to update it later
        if col == 0: self.lbl_input = lbl
        elif col == 1: self.lbl_mask = lbl
        elif col == 2: self.lbl_result = lbl

    def load_model(self):
        try:
            # Initialize with default settings from crop_img.py
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.segmenter = ClothesSegmenter("mattmdjaga/segformer_b2_clothes", device)
            self.queue.put(("status", f"Model loaded on {device}"))
        except Exception as e:
            self.queue.put(("status", f"Error loading model: {e}"))

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path_var.set(path)

    def select_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_path_var.set(path)

    def start_processing(self):
        if not self.segmenter:
            return
        if not self.video_path_var.get():
            return
        if not self.output_path_var.get():
            self.status_var.set("Please select an output directory first.")
            return
        
        try:
            skip = self.frame_skip_var.get()
            blur = self.blur_threshold_var.get()
        except ValueError:
            self.status_var.set("Invalid settings values")
            return
            
        self.processing = True
        threading.Thread(target=self.process_video, args=(self.video_path_var.get(), self.output_path_var.get(), skip, blur), daemon=True).start()

    def stop_processing(self):
        self.processing = False

    def process_video(self, video_path, output_dir_str, frame_skip, blur_threshold):
        from pathlib import Path
        output_dir = Path(output_dir_str)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "masks").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        count = 0
        
        while self.processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            count += 1
            if count % (frame_skip + 1) != 0:
                continue
                
            # 1. Check Blur
            is_blur, score = is_blurry(frame, threshold=blur_threshold)
            if is_blur:
                self.queue.put(("status", f"Frame {count}: Skipped (Blurry, score={score:.1f})"))
                continue
                
            # Convert to PIL for processing
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # 2. Segment
            try:
                pred = self.segmenter.predict_label_map(pil_img)
                
                # Filter labels (Upper-clothes = 4) - hardcoded for demo
                include_labels = [4] 
                mask = np.isin(pred, include_labels).astype(np.uint8) * 255
                mask = postprocess_mask(mask, morph_kernel_ratio=0.01, keep_largest=True)
                
                # 3. BBox & Crop
                bbox = mask_to_bbox(mask)
                if bbox is None:
                    self.queue.put(("status", f"Frame {count}: No object found"))
                    continue
                    
                w, h = pil_img.size
                bbox_padded = pad_bbox(bbox, w, h, padding=0.15)
                x1, y1, x2, y2 = bbox_padded
                
                crop_img = pil_img.crop((x1, y1, x2 + 1, y2 + 1))
                crop_mask = mask[y1:y2 + 1, x1:x2 + 1]
                
                # 4. Create Label
                yolo_line = mask_to_yolo_bbox(crop_mask, class_id=0)
                label_status = "Label Created: OK" if yolo_line else "Label Created: Failed"
                
                # Save to disk
                if yolo_line:
                    base_name = f"frame_{count:06d}"
                    
                    # Save Image
                    img_save_path = output_dir / "images" / f"{base_name}.png"
                    crop_img.save(img_save_path)
                    
                    # Save Mask
                    mask_save_path = output_dir / "masks" / f"{base_name}.png"
                    cv2.imwrite(str(mask_save_path), crop_mask)
                    
                    # Save Label
                    label_save_path = output_dir / "labels" / f"{base_name}.txt"
                    label_save_path.write_text(yolo_line, encoding="utf-8")
                    
                    label_status += " (Saved)"
                
                # 5. Visualize Result
                # Draw bbox on crop_img
                vis_img = np.array(crop_img)
                if yolo_line:
                    # Parse yolo line back to coords for drawing
                    _, ncx, ncy, nw, nh = map(float, yolo_line.split())
                    ch, cw = vis_img.shape[:2]
                    
                    cx, cy = ncx * cw, ncy * ch
                    bw, bh = nw * cw, nh * ch
                    
                    bx1 = int(cx - bw/2)
                    by1 = int(cy - bh/2)
                    bx2 = int(cx + bw/2)
                    by2 = int(cy + bh/2)
                    
                    cv2.rectangle(vis_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                
                # Send to UI
                self.queue.put(("update_ui", (img_rgb, mask, vis_img, f"Frame {count}: {label_status}")))
                
                # Sleep slightly to make it viewable
                time.sleep(0.2) 
            except Exception as e:
                print(f"Error processing frame: {e}")
                self.queue.put(("status", f"Error: {e}"))
            
        cap.release()
        self.queue.put(("status", "Processing finished"))
        self.processing = False

    def process_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                if msg_type == "status":
                    self.status_var.set(data)
                elif msg_type == "update_ui":
                    img_rgb, mask, vis_img, status = data
                    self.update_images(img_rgb, mask, vis_img)
                    self.status_var.set(status)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def update_images(self, img_rgb, mask, vis_img):
        # Resize for display
        display_size = (350, 350)
        
        # Input
        im1 = Image.fromarray(img_rgb)
        im1.thumbnail(display_size)
        ph1 = ImageTk.PhotoImage(im1)
        self.lbl_input.configure(image=ph1)
        self.lbl_input.image = ph1
        
        # Mask
        im2 = Image.fromarray(mask)
        im2.thumbnail(display_size)
        ph2 = ImageTk.PhotoImage(im2)
        self.lbl_mask.configure(image=ph2)
        self.lbl_mask.image = ph2
        
        # Result
        im3 = Image.fromarray(vis_img)
        im3.thumbnail(display_size)
        ph3 = ImageTk.PhotoImage(im3)
        self.lbl_result.configure(image=ph3)
        self.lbl_result.image = ph3

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
