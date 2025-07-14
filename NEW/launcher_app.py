import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import platform
import glob

# Import the engine class from your other file
from tracker_engine import FishTrackingEngine

class FishCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Fish Counter")
        self.root.geometry("1280x800")
        self.root.configure(bg="#2c3e50")

        # --- State Variables ---
        self.running = False
        self.cap = None
        self.thread = None

        # --- Initialize the Engine ---
        # This can take a moment, so it's good to let the user know.
        print("INFO: Initializing tracking engine... Please wait.")
        self.engine = FishTrackingEngine()
        print("INFO: Engine ready.")

        # --- UI Layout ---
        # Top frame for controls
        control_frame = tk.Frame(root, bg="#34495e", padx=10, pady=10)
        control_frame.pack(fill=tk.X, side=tk.TOP)

        # Camera selection
        tk.Label(control_frame, text="Select Camera:", bg="#34495e", fg="white").pack(side=tk.LEFT, padx=(10, 5))
        self.available_cams = self.list_video_devices()
        self.camera_var = tk.StringVar(value=self.available_cams[0] if self.available_cams else "")
        self.camera_combo = ttk.Combobox(control_frame, textvariable=self.camera_var, values=self.available_cams, state="readonly", width=30)
        self.camera_combo.pack(side=tk.LEFT, padx=5)

        # Control Buttons
        self.start_button = ttk.Button(control_frame, text="▶ Start Counting", command=self.start_counting, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=20)
        self.stop_button = ttk.Button(control_frame, text="■ Stop Counting", command=self.stop_counting, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # --- Counts Display ---
        self.count_label = tk.Label(control_frame, text="Count: 0", font=("Arial", 16, "bold"), bg="#34495e", fg="#3498db")
        self.count_label.pack(side=tk.RIGHT, padx=20)

        # --- Video Display Label ---
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # --- Event Handlers ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def list_video_devices(self):
        """Finds available video devices for Linux."""
        if platform.system() != "Linux":
            messagebox.showwarning("Warning", "This application is optimized for Linux. Camera detection may be unreliable.")
            return ["0"] # Fallback for other systems
        
        devices = sorted(glob.glob("/dev/video*"))
        if not devices:
            messagebox.showerror("Error", "No cameras found at /dev/video*. Please check your connection.")
        return devices

    def start_counting(self):
        if self.running:
            return

        cam_path = self.camera_var.get()
        if not cam_path:
            messagebox.showerror("Error", "No camera selected or available.")
            return

        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.camera_combo.config(state=tk.DISABLED)

        # Reset the engine's state for a fresh start
        self.engine.reset()

        # Initialize the camera
        self.cap = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Failed to open camera: {cam_path}")
            self.stop_counting()
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Start the video processing in a background thread
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def stop_counting(self):
        self.running = False
        # Wait for the thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join()

        if self.cap:
            self.cap.release()
            self.cap = None

        # Show a black screen when stopped
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.display_frame(black_frame)

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.camera_combo.config(state=tk.NORMAL)

    def video_loop(self):
        """The main loop for reading frames and processing them."""
        while self.running:
            if not self.cap: break
            
            ret, frame = self.cap.read()
            if not ret:
                print("WARN: Could not read frame from camera. Stopping.")
                # Automatically stop if the camera feed is lost
                self.root.after(0, self.stop_counting)
                break
            
            # 🚀 Send the raw frame to the engine and get the processed result
            processed_frame, counts = self.engine.process_frame(frame)

            # Schedule the UI update on the main thread
            self.root.after(0, self.update_ui, processed_frame, counts)
            time.sleep(0.01) # Yield to other threads

        print("INFO: Video loop has terminated.")

    def update_ui(self, frame, counts):
        """Updates the video label and count text in the UI."""
        # Update the video feed
        self.display_frame(frame)
        
        # Update the count label
        # You can choose which count to display or show all of them
        main_count = counts.get("count_1", 0)
        self.count_label.config(text=f"Count: {main_count}")

    def display_frame(self, frame):
        """Converts a CV2 frame to a Tkinter image and displays it."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize to fit the label while maintaining aspect ratio (optional but recommended)
        label_w, label_h = self.video_label.winfo_width(), self.video_label.winfo_height()
        if label_w > 1 and label_h > 1: # Ensure the label has been rendered
            img_pil.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Display the image
        self.video_label.config(image=img_tk)
        # Keep a reference to prevent Python's garbage collector from deleting it
        self.video_label.image = img_tk

    def on_closing(self):
        """Handles the window close event."""
        print("INFO: Closing application...")
        if self.running:
            self.stop_counting()
        self.root.destroy()

if __name__ == "__main__":
    # Note: On some Linux systems, you might need to run with sudo
    # if you get permission errors for the camera or NPU.
    root = tk.Tk()
    
    # Optional: Add a simple style for the buttons
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="white", background="#2980b9", font=('Arial', 12, 'bold'))

    app = FishCounterApp(root)
    root.mainloop()