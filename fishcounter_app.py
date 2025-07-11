import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import glob
import os

# Add scripts/ to sys.path to import YOLOv6Detector
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from detector import YOLOv6Detector

def list_video_devices():
    # List /dev/video* devices
    devices = sorted(glob.glob('/dev/video*'))
    return devices if devices else ["/dev/video0"]

class FishCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv6 Fish Counter")
        self.root.attributes('-zoomed', True)
        self.running = False
        self.frame = None
        self.fish_count = 0

        # Camera path selection with ComboBox
        cam_frame = tk.Frame(root)
        cam_frame.pack(pady=5)
        tk.Label(cam_frame, text="Camera Device:").pack(side=tk.LEFT)
        self.available_cams = list_video_devices()
        self.camera_var = tk.StringVar(value=self.available_cams[0])
        self.camera_combo = ttk.Combobox(cam_frame, textvariable=self.camera_var, values=self.available_cams, width=20, state="readonly")
        self.camera_combo.pack(side=tk.LEFT)

        # UI Elements
        self.video_label = tk.Label(root)
        self.video_label.pack(expand=True, fill=tk.BOTH)

        self.count_label = tk.Label(root, text="Fish Count: 0", font=("Arial", 16))
        self.count_label.pack(pady=10)

        self.start_button = ttk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.stop_button = ttk.Button(root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Detector
        model_path = str(Path(__file__).parent / "models" / "model1.pt")
        yaml_path = str(Path(__file__).parent / "models" / "dataset.yaml")
        self.detector = YOLOv6Detector(model_path, yaml_path, device="cpu")  # or "cuda:0"

        self.cap = None
        self.thread = None

        # Default black frame (for fallback)
        self.default_shape = (480, 640, 3)  # Height, Width, Channels
        self.black_frame = np.zeros(self.default_shape, dtype=np.uint8)

    def start_detection(self):
        if not self.running:
            cam_path = self.camera_var.get()
            self.cap = cv2.VideoCapture(cam_path)
            if not self.cap.isOpened():
                self.count_label.config(text=f"Error: Could not open camera at {cam_path}")
                # Still show black frame
                self.running = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.thread = threading.Thread(target=self.detection_loop, daemon=True)
                self.thread.start()
                return
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.thread.start()

    def stop_detection(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
            self.cap = None

    def detection_loop(self):
        while self.running:
            frame = None
            ret = False
            if self.cap:
                ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                # Use black frame if no valid frame
                frame = self.black_frame.copy()
                detections = []
                fish_count = 0
            else:
                detections = self.detector.detect(frame)
                fish_count = len(detections)
                frame = self.detector.visualize(frame, detections)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.root.after(0, self.update_ui, img_tk, fish_count)
            time.sleep(0.03)  # ~30 FPS

    def update_ui(self, img_tk, count):
        self.video_label.imgtk = img_tk  # Prevent garbage collection
        self.video_label.config(image=img_tk)
        self.count_label.config(text=f"Fish Count: {count}")

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FishCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()