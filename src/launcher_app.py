# gui_app.py

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import platform
from PIL import Image, ImageTk

# Import the main ObjectDetector class
from utils import *

class DetectorGUI:
    """
    A Tkinter-based GUI for the real-time fish detection and counting application.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Fish Detector and Counter")

        # --- State Variables ---
        self.video_thread = None
        self.stop_event = threading.Event()
        self.cap = None
        self.detector = None

        # --- Window Maximization ---
        if platform.system() == "Windows":
            self.root.state('zoomed')
        else: # For Linux/MacOS
            self.root.attributes('-zoomed', True)

        # --- Main Layout Frames ---
        # Top frame for controls
        self.top_frame = ttk.Frame(self.root, padding="10")
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        # Main content frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.main_frame.grid_columnconfigure(1, weight=1) # Make right panel expandable
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Left panel for counts
        self.left_panel = ttk.LabelFrame(self.main_frame, text="Live Counts", padding="10")
        self.left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        # Right panel for video
        self.right_panel = ttk.LabelFrame(self.main_frame, text="Video Feed", padding="10")
        self.right_panel.grid(row=0, column=1, sticky="nsew")

        # --- Populate Widgets ---
        self._create_control_widgets()
        self._create_count_widgets()
        self._create_video_widget()
        
        # --- Graceful Shutdown ---
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_control_widgets(self):
        """Creates and places the widgets in the top control frame."""
        # Camera selection dropdown
        ttk.Label(self.top_frame, text="Select Camera:").pack(side=tk.LEFT, padx=(0, 5))
        self.camera_options = self._find_cameras()
        self.camera_var = tk.StringVar(value=self.camera_options[0] if self.camera_options else "")
        self.camera_dropdown = ttk.Combobox(self.top_frame, textvariable=self.camera_var, values=self.camera_options, state="readonly")
        self.camera_dropdown.pack(side=tk.LEFT, padx=5)

        # Start button
        self.start_button = ttk.Button(self.top_frame, text="Start", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Stop button
        self.stop_button = ttk.Button(self.top_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def _create_count_widgets(self):
        """Creates and places the labels for displaying counts."""
        self.count_labels = {}
        count_names = ["Method 1 (Boundary Cross)", "Method 2 (In Region)", "Method 3 (Intersection)"]
        
        for i, name in enumerate(count_names):
            ttk.Label(self.left_panel, text=f"{name}:", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(10, 2))
            
            count_var = tk.StringVar(value="0")
            label = ttk.Label(self.left_panel, textvariable=count_var, font=("Helvetica", 16))
            label.pack(anchor="w", pady=(0, 10))
            self.count_labels[f"count_{i+1}"] = count_var

    def _create_video_widget(self):
        """Creates the label widget that will display the video feed."""
        self.video_label = ttk.Label(self.right_panel)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def start_processing(self):
        """Handles the 'Start' button click event."""
        if not self.camera_var.get():
            messagebox.showerror("Error", "No camera selected or available.")
            return

        # Disable start button and enable stop button
        self.start_button.config(state=tk.DISABLED)
        self.camera_dropdown.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Clear previous stop event and start a new thread
        self.stop_event.clear()
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_processing(self):
        """Handles the 'Stop' button click event."""
        self.stop_event.set() # Signal the thread to stop
        self.start_button.config(state=tk.NORMAL)
        self.camera_dropdown.config(state="readonly")
        self.stop_button.config(state=tk.DISABLED)
        
        # Clear the video label
        self.video_label.config(image='')
        self.video_label.image = None


    def _video_loop(self):
        """The main loop for video capture and processing, runs in a separate thread."""
        try:
            # --- Initialization ---
            camera_index = int(self.camera_var.get().split()[-1])
            model_path = "./utils/rknn_model_zoo-main/rknn_model_zoo-main/examples/yolov6/model/yolov6.rknn"
            self.detector = ObjectDetector(model_path=model_path, img_size=(640, 640), obj_thresh=0.1, nms_thresh=0.1)
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open camera {camera_index}")

            # --- Processing Loop ---
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Warning: Could not read frame from camera.")
                    time.sleep(0.1)
                    continue
                
                # The core processing call
                processed_frame, counts = self.detector.process_frame(frame)
                
                # Update GUI with the results
                self.root.after(0, self._update_gui, processed_frame, counts)

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred in the video thread:\n{e}")
        finally:
            if self.cap:
                self.cap.release()
            print("Video thread stopped and resources released.")

    def _update_gui(self, frame, counts):
        """Safely updates the Tkinter GUI from the main thread."""
        # Update video frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk # Keep a reference

        # Update count labels
        self.count_labels["count_1"].set(str(counts["count_1"]))
        self.count_labels["count_2"].set(str(counts["count_2"]))
        self.count_labels["count_3"].set(str(counts["count_3"]))

    def _find_cameras(self):
        """Scans for and returns a list of available camera devices."""
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr

    def _on_closing(self):
        """Handles the window close event to ensure clean shutdown."""
        if self.video_thread and self.video_thread.is_alive():
            self.stop_event.set()
            self.video_thread.join(timeout=1) # Wait for thread to finish
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorGUI(root)
    root.mainloop()
