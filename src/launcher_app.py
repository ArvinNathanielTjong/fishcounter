# launcher_app.py

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import platform
from PIL import Image, ImageTk
import os
import glob
import numpy as np
import time

# Import the main ObjectDetector class
# Pastikan file object_detector.py dan detector_util.py ada di dalam folder 'utils'
try:
    from utils.object_detector import ObjectDetector
    from utils.hardware_controller import BuzzerController
except ImportError as e:
    messagebox.showerror("Import Error", f"Tidak dapat mengimpor ObjectDetector. Pastikan struktur folder sudah benar.\n\nError: {e}")
    import sys
    sys.exit()


# ============================================================================
# KELAS UTAMA APLIKASI (KONTROLER)
# ============================================================================
class FishCounterApp(tk.Tk):
    """
    Kelas kontroler utama yang mengatur semua halaman (frame) dalam aplikasi.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Fish Counter Pro")
        if platform.system() == "Windows":
            self.state('zoomed')
        else:
            self.attributes('-zoomed', True)

        # Kontainer utama tempat semua frame/halaman akan ditumpuk
        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.detector = None # Detector akan diinisialisasi saat dibutuhkan
        
        try:
            # Assuming BuzzerController class is in the same file
            self.buzzer = BuzzerController(pin=7) 
        except Exception as e:
            # Handle cases where GPIO setup fails (e.g., not run on Orange Pi)
            messagebox.showwarning("Buzzer Warning", f"Could not initialize buzzer.\n\nError: {e}")
            self.buzzer = None # Set to None so app doesn't crash

        # Siapkan semua halaman yang ada
        for F in (MainMenu, FreeCountPage, TargetCountPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenu")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_frame(self, page_name, target=None):
        """Menampilkan sebuah frame berdasarkan nama kelasnya."""
        frame = self.frames[page_name]
        if page_name == "TargetCountPage" and target is not None:
            frame.set_target(target) # Kirim data target ke halaman target
        frame.tkraise()

    def get_detector(self):
        """Membuat instance detector jika belum ada."""
        if self.detector is None:
            try:
                # Sesuaikan path ke model Anda
                model_path = "./utils/rknn_model_zoo-main/rknn_model_zoo-main/examples/yolov6/model/yolov6.rknn" 
                if not os.path.exists(model_path):
                     raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")
                self.detector = ObjectDetector(model_path=model_path, img_size=(640, 640), obj_thresh=0.048, nms_thresh=0.048)
            except Exception as e:
                messagebox.showerror("Model Error", f"Gagal memuat model AI:\n{e}")
                return None
        return self.detector

    def on_closing(self):
        """Menangani event penutupan aplikasi."""
        if self.buzzer:
            self.buzzer.cleanup()
        self.destroy()

# ============================================================================
# HALAMAN MENU UTAMA
# ============================================================================

class MainMenu(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # 1. Siapkan Style
        style = ttk.Style()
        style.configure("Menu.TButton", font=("Helvetica", 18, "bold"), padding=20)
        # Style untuk frame dengan warna latar
        style.configure("Left.TFrame", background= 'blue')  # Biru muda
        style.configure("Right.TFrame", background='green') # Hijau muda

        # 2. Konfigurasi grid agar membagi layar 50/50 dan bisa membesar
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # 3. BUAT KONTENAINER FRAME TERLEBIH DAHULU
        left_frame = ttk.Frame(self, style="Left.TFrame")
        right_frame = ttk.Frame(self, style="Right.TFrame")

        # 4. Tempatkan kontainer pada grid
        left_frame.grid(row=0, column=0, sticky="nsew")
        right_frame.grid(row=0, column=1, sticky="nsew")

        # 5. BARULAH BUAT WIDGET YANG AKAN MASUK KE DALAM FRAME
        # Widget untuk frame kiri
        label_left = ttk.Label(left_frame, text="Mode Bebas", font=("Helvetica", 24, "bold"), background='blue')
        btn1 = ttk.Button(left_frame, text="Perhitungan\nBebas", style="Menu.TButton",
                          command=lambda: controller.show_frame("FreeCountPage"))

        # Widget untuk frame kanan
        label_right = ttk.Label(right_frame, text="Mode Target", font=("Helvetica", 24, "bold"), background='green')
        btn2 = ttk.Button(right_frame, text="Perhitungan\n Sesuai \nJumlah", style="Menu.TButton",
                          command=lambda: controller.show_frame("TargetCountPage"))

        # 6. Atur layout widget di dalam framenya masing-masing
        label_left.pack(pady=(100, 20), padx=20)
        btn1.pack(pady=20, padx=50, fill="x")

        label_right.pack(pady=(100, 20), padx=20)
        btn2.pack(pady=20, padx=50, fill="x")

# ============================================================================
# KELAS DASAR UNTUK HALAMAN PENGHITUNGAN (AGAR TIDAK DUPLIKASI KODE)
# ============================================================================
class BaseCountingPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.video_thread = None
        self.stop_event = threading.Event()
        self.cap = None

        # --- Layout Utama ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Panel Kontrol Atas ---
        self.top_frame = ttk.Frame(self, padding="10")
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        # --- Panel Kiri (Data) ---
        self.left_panel = ttk.LabelFrame(self, text="Live Data", padding="10")
        self.left_panel.grid(row=1, column=0, sticky="ns", padx=10, pady=10)

        # --- Panel Kanan (Video) ---
        self.right_panel = ttk.LabelFrame(self, text="Video Feed", padding="10")
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        self.video_label = ttk.Label(self.right_panel, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self._create_control_widgets()
        self._create_count_widgets()

    def _create_control_widgets(self):
        # Tombol kembali
        self.back_button = ttk.Button(self.top_frame, text="< Kembali ke Menu",
                                      command=lambda: self.controller.show_frame("MainMenu"))
        self.back_button.pack(side=tk.LEFT, padx=5)

        # Dropdown Kamera
        ttk.Label(self.top_frame, text="Kamera:").pack(side=tk.LEFT, padx=(20, 5))
        self.camera_options = self._find_cameras()
        self.camera_var = tk.StringVar(value=self.camera_options[0] if self.camera_options else "")
        self.camera_dropdown = ttk.Combobox(self.top_frame, textvariable=self.camera_var, values=self.camera_options, state="readonly")
        self.camera_dropdown.pack(side=tk.LEFT, padx=5)

        # Tombol Start & Stop
        self.start_button = ttk.Button(self.top_frame, text="Start", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(self.top_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def _create_count_widgets(self):
        self.count_labels = {}
        count_names = ["Metode 1 (Garis Batas)", "Metode 2 (Zona)", "Metode 3 (Lintasan)"]
        for i, name in enumerate(count_names):
            ttk.Label(self.left_panel, text=f"{name}:", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(10, 2))
            count_var = tk.StringVar(value="0")
            label = ttk.Label(self.left_panel, textvariable=count_var, font=("Helvetica", 16))
            label.pack(anchor="w", pady=(0, 10))
            self.count_labels[f"count_{i+1}"] = count_var

    def start_processing(self):
        if "No camera" in self.camera_var.get() or not self.camera_var.get():
            messagebox.showerror("Error", "Kamera tidak dipilih atau tidak tersedia.")
            return

        # ====================================================================
        # TO RESET EVERTIME IT STARTS PROCESSING
        # ====================================================================
        detector = self.controller.get_detector()
        if detector:
            detector.reset()  # <--- THIS IS THE KEY!
        else:
            # get_detector already shows an error, so we can just stop.
            return

        # Also reset the labels on the screen to "0"
        self.count_labels["count_1"].set("0")
        self.count_labels["count_2"].set("0")
        self.count_labels["count_3"].set("0")
        # ====================================================================


        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.back_button.config(state=tk.DISABLED)
        self.camera_dropdown.config(state=tk.DISABLED)

        self.stop_event.clear()
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_processing(self):
        self.stop_event.set()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.back_button.config(state=tk.NORMAL)
        self.camera_dropdown.config(state="readonly")
        
        # Tampilkan frame hitam saat berhenti
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        img_pil = Image.fromarray(black_frame)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk

    def _video_loop(self):
        detector = self.controller.get_detector()
        if detector is None:
            self.stop_processing()
            return
            
        try:
            camera_path = self.camera_var.get()
            if platform.system() == "Linux":
                # self.cap = cv2.VideoCapture(camera_path, cv2.CAP_V4L2) #for camera
                self.cap = cv2.VideoCapture("../output.avi")
            else:
                self.cap = cv2.VideoCapture(int(camera_path))

            if not self.cap.isOpened():
                raise IOError(f"Tidak dapat membuka kamera {camera_path}")

            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Peringatan: Tidak dapat membaca frame.")
                    time.sleep(0.1)
                    continue
                
                processed_frame, counts = detector.process_frame(frame)
                
                self.controller.after(0, self._update_gui, processed_frame, counts)

    

        except Exception as e:
            messagebox.showerror("Error Proses Video", f"Terjadi kesalahan:\n{e}")
        finally:
            if self.cap:
                self.cap.release()
            self.controller.after(0, self.stop_processing) # Pastikan UI kembali ke state stop

    def _update_gui(self, frame, counts):
        if self.stop_event.is_set():
            return

         # --- AWAL PERUBAHAN ---
        # 1. Dapatkan ukuran widget label tempat video akan ditampilkan
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()

        # 2. Dapatkan ukuran frame video asli
        frame_h, frame_w, _ = frame.shape

        # 3. Hindari pembagian dengan nol jika label belum muncul di layar
        if label_w == 1 or label_h == 1:
            # Jika label belum punya ukuran, tunda update sebentar
            self.controller.after(20, lambda: self._update_gui(frame, counts))
            return

        # 4. Hitung rasio skala agar pas tanpa merusak aspek rasio
        scale_w = label_w / frame_w
        scale_h = label_h / frame_h
        scale = min(scale_w, scale_h)

        # 5. Hitung dimensi baru dan resize frame
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        # --- AKHIR PERUBAHAN ---
        # 3. Create black canvas the size of the label
        canvas = np.zeros((label_h, label_w, 3), dtype=np.uint8)

        # 4. Compute top-left coordinates to center the frame
        y_offset = (label_h - new_h) // 2
        x_offset = (label_w - new_w) // 2

        # 5. Paste resized frame onto canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

        # 6. Convert to PIL and display
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk

        self.count_labels["count_1"].set(str(counts["count_1"]))
        self.count_labels["count_2"].set(str(counts["count_2"]))
        self.count_labels["count_3"].set(str(counts["count_3"]))

    def _find_cameras(self):
        # Fungsi ini menemukan kamera yang tersedia
        system_platform = platform.system()
        available_devices = []
        if system_platform == "Linux":
            video_paths = sorted(glob.glob("/dev/video*"))
            for path in video_paths:
                cap = cv2.VideoCapture(path)
                if cap is not None and cap.isOpened():
                    available_devices.append(path)
                    cap.release()
        else: # Windows
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    available_devices.append(str(i))
                    cap.release()
        return available_devices if available_devices else ["No camera found"]


# ============================================================================
# HALAMAN PERHITUNGAN BEBAS
# ============================================================================
class FreeCountPage(BaseCountingPage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        # Tidak ada kustomisasi khusus untuk halaman ini, semua sudah di Base class

# ============================================================================
# HALAMAN PERHITUNGAN SESUAI TARGET
# ============================================================================
class TargetCountPage(BaseCountingPage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.target = 0
        self.final_count = 0

        # --- Tambahkan widget khusus untuk input target ---
        target_frame = ttk.Frame(self.left_panel)
        target_frame.pack(fill='x', pady=(10,20))

        ttk.Label(target_frame, text="Target Ikan:", font=("Helvetica", 12, "bold")).pack(fill = 'x')
        self.target_var = tk.StringVar(value="10")
        self.target_entry = ttk.Entry(target_frame, textvariable=self.target_var, width=10)
        self.target_entry.pack(fill='x', padx=5)

        # Sembunyikan tombol start utama, karena kita akan pakai tombol di sini
        self.start_button.pack_forget() 
        self.set_target_button = ttk.Button(target_frame, text="Set Target & Mulai", command=self.start_processing)
        self.set_target_button.pack(fill='x', padx=5)

    def set_target(self, target_value=None):
        # Fungsi ini dipanggil untuk mengatur ulang halaman saat ditampilkan
        try:
            # Jika tidak ada nilai baru, gunakan dari entry
            self.target = int(self.target_var.get())
        except ValueError:
            messagebox.showerror("Input Salah", "Target harus berupa angka.")
            return False
        
        # Reset hitungan saat target baru di-set
        detector = self.controller.get_detector()
        if detector:
            detector.fish_count_1 = 0
            detector.fish_count_2 = 0
            detector.fish_count_3 = 0
            detector.counted_ids_1.clear()
            detector.counted_ids_2.clear()
            detector.counted_ids_3.clear()
        
        # Reset label di GUI
        self.count_labels["count_1"].set("0")
        self.count_labels["count_2"].set("0")
        self.count_labels["count_3"].set("0")
        return True

    def start_processing(self):
        if not self.set_target(): # Set dan validasi target sebelum mulai
            return
        super().start_processing() # Panggil fungsi start dari Base class
        self.set_target_button.config(state=tk.DISABLED)
        self.target_entry.config(state=tk.DISABLED)
        self.target_sound_played = False

    def stop_processing(self):
        is_stopping_normally = not self.stop_event.is_set() #ambil data hitungan sebelum berhenti total (biar ga bug)

        super().stop_processing() # Panggil fungsi stop dari Base class
        # 3. Tampilkan hasil HANYA jika proses dihentikan secara normal oleh pengguna.
        if is_stopping_normally:
            try:
                # Ambil hitungan paling akurat (asumsi dari metode 3)
                count3 = int(self.count_labels["count_3"].get()) 
                excess = count3 - self.target
                
                if excess >= 0:
                    message = f"Penghitungan Selesai!\n\nTarget: {self.target}\nTerhitung: {count3}\n\nKelebihan: {excess} ekor"
                else:
                    message = f"Penghitungan Selesai!\n\nTarget: {self.target}\nTerhitung: {count3}\n\nKekurangan: {abs(excess)} ekor"

                messagebox.showinfo("Hasil Penghitungan", message)
            except ValueError:
                # Menangani jika label hitungan bukan angka saat diakses
                messagebox.showwarning("Hasil", "Tidak dapat menampilkan hasil akhir karena data hitungan tidak valid.")

        # 4. Atur ulang state tombol khusus untuk halaman ini.
        self.set_target_button.config(state=tk.NORMAL)
        self.target_entry.config(state=tk.NORMAL)


    
    def _update_gui(self, frame, counts):
        # First, let the base class do its job
        super()._update_gui(frame, counts)

        # Now, add our target-specific logic
        current_count = counts["count_3"] # Assuming method 3 is the primary count

        if not self.target_sound_played and current_count >= self.target:
            print("Target reached! Beeping.") # For debugging
            self.controller.buzzer.beep(duration=0.5) # A longer beep for success
            self.target_sound_played = True


if __name__ == "__main__":
    app = FishCounterApp()
    app.mainloop()
