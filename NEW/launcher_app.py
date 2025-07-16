import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import logging
import glob

# --- Import your actual engine ---
# Make sure 'program_counting_adapt.py' is in the same directory
# or in Python's path.
try:
    # Assuming the class is in 'program_counting_adapt.py'
    from program_counting_adapt import FishTrackingEngine
except ImportError:
    messagebox.showerror("Import Error", "Could not find 'program_counting_adapt.py'. Make sure it's in the same folder as this script.")
    exit()

# --- Setup Logging ---
# This will show INFO messages from your engine in the console.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FishCounterApp:
    """
    A simplified Tkinter GUI application for the FishTrackingEngine
    with only Start and Stop buttons and robust, automatic camera detection.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Fish Counter")
        self.root.geometry("800x650")

        # --- State Variables ---
        self.is_running = False
        self.video_thread = None
        self.cap = None
        self.engine = None

        # --- Style ---
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10)
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))

        # --- GUI Widgets ---
        header_label = ttk.Label(root, text="Fish Detection and Counting", style="Header.TLabel")
        header_label.pack(pady=10)

        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(padx=10, pady=10, expand=True, fill="both")

        controls_frame = ttk.Frame(root)
        controls_frame.pack(pady=10)

        self.start_button = ttk.Button(controls_frame, text="Start", command=self.start_counting_thread)
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = ttk.Button(controls_frame, text="Stop", command=self.stop_counting, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10)

        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def start_counting_thread(self):
        """Starts the video processing in a separate thread."""
        if self.is_running:
            return
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.is_running = True
        self.status_var.set("Initializing...")
        
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

    def find_and_open_camera(self):
        """
        Automatically detects and opens the first available camera.
        Tries higher index cameras first (e.g., /dev/video1 before /dev/video0).
        """
        # Scan for video devices
        camera_paths = sorted(glob.glob("/dev/video*"), reverse=True)
        logging.info(f"Found potential cameras: {camera_paths}")
        if not camera_paths:
            raise IOError("No cameras found. Please connect a camera.")

        for path in camera_paths:
            self.status_var.set(f"Trying to open camera: {path}")
            logging.info(f"Attempting to open {path}...")
            # Try to open the camera using the V4L2 backend
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if cap.isOpened():
                logging.info(f"Successfully opened camera: {path}")
                self.status_var.set(f"Using camera: {path}")
                return cap
        
        # If no camera could be opened after trying all paths
        raise IOError(f"Could not open any camera. Tried: {camera_paths}")


    def video_loop(self):
        """The main video processing loop that runs in the background."""
        try:
            # --- Camera Initialization ---
            self.cap = self.find_and_open_camera()
            
            # --- Engine Initialization ---
            self.status_var.set("Initializing Tracking Engine...")
            model_path = "./rknn_model_zoo-main/rknn_model_zoo-main/examples/yolov6/model/yolov6.rknn"
            self.engine = FishTrackingEngine(model_path=model_path)
            self.engine.reset()

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.status_var.set("Processing...")

            # --- Main Processing Loop ---
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to read frame. Stopping.")
                    break

                processed_frame, counts = self.engine.process_frame(frame)
                self.update_gui_frame(processed_frame)

        except Exception as e:
            logging.error(f"An error occurred in the video loop: {e}")
            messagebox.showerror("Runtime Error", f"An error occurred: {e}")
        finally:
            # Cleanup on thread exit
            if self.cap:
                self.cap.release()
            self.is_running = False
            # Schedule GUI update on the main thread
            self.root.after(0, self.on_stop)

    def update_gui_frame(self, frame):
        """Converts a CV2 frame to a Tkinter image and updates the GUI."""
        if frame is None:
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def stop_counting(self):
        """Signals the video loop to stop."""
        if self.is_running:
            self.is_running = False
            self.status_var.set("Stopping...")

    def on_stop(self):
        """Handles GUI updates after the video loop has stopped."""
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.status_var.set("Ready")
        self.video_label.config(image='')
        self.video_label.imgtk = None

    def quit_app(self):
        """Cleans up and closes the application."""
        self.stop_counting()
        # Give the thread a moment to stop gracefully before destroying the window
        self.root.after(100, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    app = FishCounterApp(root)
    root.mainloop()
