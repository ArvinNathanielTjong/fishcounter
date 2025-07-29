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

# Pastikan file object_detector.py dan hardware_controller.py ada di dalam folder 'utils'
try:
    from utils.object_detector import ObjectDetector
    # ### PERUBAHAN ###
    # Impor BuzzerController dan MotorController dari hardware_controller
    from utils.hardware_controller import BuzzerController, MotorController
except ImportError as e:
    messagebox.showerror("Import Error", f"Tidak dapat mengimpor modul. Pastikan struktur folder sudah benar.\n\nError: {e}")
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
        
        # ### FITUR BARU: Inisialisasi Hardware Controllers ###
        self.buzzer = None
        self.motor = None
        
        # Inisialisasi Buzzer
        try:
            self.buzzer = BuzzerController(gpio_num=47) # Pin 7 di Orange Pi = GPIO47
        except Exception as e:
            messagebox.showwarning("Buzzer Warning", f"Could not initialize buzzer.\n\nError: {e}")

        # Inisialisasi Motor Controller (Serial)
        try:
            # Sesuaikan port serial dengan yang Anda gunakan
            SERIAL_PORT = '/dev/arduino' if platform.system() == "Linux" else 'COM8' 
            self.motor = MotorController(port=SERIAL_PORT)
            if not self.motor.ser:
                 messagebox.showwarning("Motor Warning", f"Could not connect to motor controller on {SERIAL_PORT}.")
                 self.motor = None # Nonaktifkan jika gagal konek
        except Exception as e:
            messagebox.showwarning("Motor Warning", f"Could not initialize motor controller.\n\nError: {e}")

        # ### FITUR BARU: Variabel untuk menampilkan status baterai di GUI ###
        self.battery_status_var = tk.StringVar(value="Baterai: --% ❔")
        self.is_system_charging = False # Status charging sistem, default tidak mengisi

        # Siapkan semua halaman yang ada
        for F in (MainMenu, FreeCountPage, TargetCountPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenu")

        # ### FITUR BARU: Mulai loop pembaruan status baterai jika motor terkoneksi ###
        if self.motor:
            self._update_battery_status()

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

    def _update_battery_status(self):
            """
            Meminta data baterai dan status sistem (AC & Charging), lalu memperbarui GUI.
            """
            if self.motor:
                # --- Inisialisasi nilai default ---
                percentage_val = "--"
                battery_icon = "❔"
                ac_plugged_in = False
                is_charging = False # <-- Tambahkan default untuk status charging

                # --- Langkah 1: Dapatkan Persentase Baterai ---
                self.motor.request_battery_voltage()
                time.sleep(0.1)
                bat_response = self.motor.read_response()
                
                if bat_response and bat_response.get('source') == 'BAT':
                    percentage = bat_response.get('percentage', '--')
                    if isinstance(percentage, (int, float)):
                        percentage_val = int(percentage)
                        if percentage_val > 75: battery_icon = "🟩"
                        elif percentage_val > 40: battery_icon = "🟨"
                        elif percentage_val > 10: battery_icon = "🟧"
                        else: battery_icon = "🟥"

                # --- Langkah 2: Dapatkan Status Adaptor AC & Charging ---
                self.motor.request_system_status()
                time.sleep(0.1)
                status_response = self.motor.read_response()

                if status_response and status_response.get('source') == 'STATUS':
                    ac_plugged_in = status_response.get('ac_ok', False)
                    # ### PERUBAHAN: Ambil status is_charging ###
                    is_charging = status_response.get('is_charging', False)

                 # ### Logika Otomatis untuk Charging ###
                # Kondisi untuk MULAI charging
                if ac_plugged_in and percentage_val != '--' and percentage_val < 95 and not self.is_system_charging: #saat dibawah 95% charging biar awet
                    print("[Auto-Charge] Memulai proses charging...")
                    self.motor.start_charging()
                    self.is_system_charging = True
                
                # Kondisi untuk BERHENTI charging
                elif (not ac_plugged_in or (percentage_val != '--' and percentage_val >= 100)) and self.is_system_charging:
                    print("[Auto-Charge] Menghentikan proses charging...")
                    self.motor.stop_charging()
                    self.is_system_charging = False

                # --- Langkah 3: Tentukan Ikon Final ---
                if ac_plugged_in:
                    final_icon = "⚡️" if is_charging else "🔌"
                else:
                    final_icon = battery_icon
                
                self.battery_status_var.set(f"Baterai: {percentage_val}% {final_icon}")

            # Jadwalkan fungsi ini untuk berjalan lagi
            self.after(5000, self._update_battery_status)


    def on_closing(self):
        """Menangani event penutupan aplikasi."""
        if self.buzzer:
            self.buzzer.cleanup()
        # ### PERUBAHAN ###
        # Tutup koneksi serial motor saat aplikasi ditutup
        if self.motor:
            self.motor.close()
        self.destroy()


# ============================================================================
# ON-SCREEN KEYBOARD
# ============================================================================
class OnScreenKeyboard(ttk.Frame):
    """
    Sebuah widget keyboard numerik on-screen yang sederhana.
    """
    def __init__(self, parent, target_entry_widget):
        super().__init__(parent)
        self.target_entry = target_entry_widget
        keys = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9'], ['Hapus', '0', '<-']]
        for y, row in enumerate(keys):
            self.columnconfigure(y, weight=1)
            for x, key in enumerate(row):
                button = ttk.Button(self, text=key, command=lambda k=key: self._on_press(k))
                button.grid(row=y, column=x, sticky="nsew", padx=2, pady=2, ipady=10)

    def _on_press(self, key):
        current_text = self.target_entry.get()
        if key.isdigit():
            self.target_entry.insert(tk.END, key)
        elif key == '<-':
            self.target_entry.delete(len(current_text) - 1, tk.END)
        elif key == 'Hapus':
            self.target_entry.delete(0, tk.END)


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
        style.configure("Left.TFrame", background='#259faf')  # Biru muda
        style.configure("Right.TFrame", background='#57b8bc') # Hijau muda

        # ### PERUBAHAN ###
        # Frame atas untuk menampung judul dan status baterai
        header_frame = ttk.Frame(self)
        header_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        ttk.Label(header_frame, text="Fish Counter Pro", font=("Helvetica", 16, "bold")).pack(side="left")
        battery_label = ttk.Label(header_frame, textvariable=self.controller.battery_status_var, font=("Helvetica", 12))
        battery_label.pack(side="right")

        # Frame utama untuk konten tombol menu
        content_frame = ttk.Frame(self)
        content_frame.pack(side="top", fill="both", expand=True)

        # 2. Konfigurasi grid agar membagi layar 50/50
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        # 3. BUAT KONTENAINER FRAME
        left_frame = ttk.Frame(content_frame, style="Left.TFrame")
        right_frame = ttk.Frame(content_frame, style="Right.TFrame")

        # 4. Tempatkan kontainer pada grid
        left_frame.grid(row=0, column=0, sticky="nsew")
        right_frame.grid(row=0, column=1, sticky="nsew")

        # 5. BUAT WIDGET YANG AKAN MASUK KE DALAM FRAME
        label_left = ttk.Label(left_frame, text="Mode Bebas", font=("Helvetica", 24, "bold"), background='#259faf')
        btn1 = ttk.Button(left_frame, text="Perhitungan\nBebas\n", style="Menu.TButton",
                          command=lambda: controller.show_frame("FreeCountPage"))

        label_right = ttk.Label(right_frame, text="Mode Target", font=("Helvetica", 24, "bold"), background='#57b8bc')
        btn2 = ttk.Button(right_frame, text="Perhitungan\n Sesuai \nJumlah", style="Menu.TButton",
                          command=lambda: controller.show_frame("TargetCountPage"))

        # 6. Atur layout widget di dalam framenya masing-masing
        label_left.pack(pady=(100, 20), padx=20)
        btn1.pack(pady=20, padx=50, fill="x")
        label_right.pack(pady=(100, 20), padx=20)
        btn2.pack(pady=20, padx=50, fill="x")


# ============================================================================
# KELAS DASAR UNTUK HALAMAN PENGHITUNGAN
# ============================================================================
class BaseCountingPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.video_thread = None
        self.stop_event = threading.Event()
        self.cap = None

        ## UNTUK PWM MOTOR ##
        self.current_speed_level = 3
        self.max_speed_level = 5

        # ### untuk kecerahan ###
        self.current_brightness_level = 3 # Kecerahan default
        self.max_brightness_level = 5     # Jumlah level kecerahan

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
        self._create_count_widgets(parent_frame=self.left_panel)

    # --- Kode BARU (Gunakan ini sebagai penggantinya) ---
    def _create_control_widgets(self):
        # Konfigurasi grid di dalam top_frame.
        # Kolom 13 akan menjadi "pegas" yang meregang.
        self.top_frame.columnconfigure(13, weight=1)

        # --- Penempatan Widget secara Eksplisit ---
        col = 0 # Mulai dari kolom pertama

        # Tombol Kembali
        self.back_button = ttk.Button(self.top_frame, text="< Kembali", command=lambda: self.controller.show_frame("MainMenu"))
        self.back_button.grid(row=0, column=col, padx=(5, 10)); col += 1

        # Kontrol Kamera
        ttk.Label(self.top_frame, text="Kamera:").grid(row=0, column=col, padx=(0, 5)); col += 1
        self.camera_options = self._find_cameras()
        self.camera_var = tk.StringVar(value=self.camera_options[0] if self.camera_options else "")
        self.camera_dropdown = ttk.Combobox(self.top_frame, textvariable=self.camera_var, values=self.camera_options, state="readonly", width=12)
        self.camera_dropdown.grid(row=0, column=col, padx=(0, 10)); col += 1

        # Tombol Start/Stop
        self.start_button = ttk.Button(self.top_frame, text="Start", command=self.start_processing)
        self.start_button.grid(row=0, column=col, padx=(0, 5)); col += 1
        self.stop_button = ttk.Button(self.top_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=col, padx=(0, 10)); col += 1

        # Kontrol Kecepatan
        ttk.Label(self.top_frame, text="Speed:").grid(row=0, column=col, padx=(0, 5)); col += 1
        down_button_speed = ttk.Button(self.top_frame, text="-", width=2, command=self._decrease_speed)
        down_button_speed.grid(row=0, column=col); col += 1
        self.speed_label_var = tk.StringVar(value=f"{self.current_speed_level}")
        speed_label = ttk.Label(self.top_frame, textvariable=self.speed_label_var, width=3, anchor="center")
        speed_label.grid(row=0, column=col, padx=2); col += 1
        up_button_speed = ttk.Button(self.top_frame, text="+", width=2, command=self._increase_speed)
        up_button_speed.grid(row=0, column=col, padx=(0, 10)); col += 1

        # Kontrol Kecerahan
        ttk.Label(self.top_frame, text="Light:").grid(row=0, column=col, padx=(0, 5)); col += 1
        down_button_bright = ttk.Button(self.top_frame, text="-", width=2, command=self._decrease_brightness)
        down_button_bright.grid(row=0, column=col); col += 1
        self.brightness_label_var = tk.StringVar(value=f"{self.current_brightness_level}")
        brightness_label = ttk.Label(self.top_frame, textvariable=self.brightness_label_var, width=3, anchor="center")
        brightness_label.grid(row=0, column=col, padx=2); col += 1
        up_button_bright = ttk.Button(self.top_frame, text="+", width=2, command=self._increase_brightness)
        up_button_bright.grid(row=0, column=col); col += 1

        # Kolom 13 adalah pegas (dilewati)
        col += 1


    def _create_count_widgets(self, parent_frame):
         # --- TAMBAHKAN BLOK INI DI PALING ATAS FUNGSI ---
        # Frame untuk menampung status baterai dengan rapi
        battery_frame = ttk.Frame(parent_frame)
        battery_frame.pack(fill='x', pady=(5, 10), padx=10)

        # Label dan Status
        ttk.Label(battery_frame, text="Status Baterai:", font=("Helvetica", 12, "bold")).pack(side="left")
        battery_label = ttk.Label(battery_frame, textvariable=self.controller.battery_status_var)
        battery_label.pack(side="left", padx=5)
        
        # Garis pemisah untuk kerapian
        ttk.Separator(parent_frame, orient='horizontal').pack(fill='x', padx=5, pady=(0, 10))
    
#----------------------------------------------------------------------------------------------------
        self.count_labels = {}
        count_names = ["Metode 1 (Garis Batas)", "Metode 2 (Zona)", "Metode 3 (Lintasan)"]
        for i, name in enumerate(count_names):
            ttk.Label(parent_frame, text=f"{name}:", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(10, 2), padx=10)
            count_var = tk.StringVar(value="0")
            label = ttk.Label(parent_frame, textvariable=count_var, font=("Helvetica", 16))
            label.pack(anchor="w", pady=(0, 10), padx=10)
            self.count_labels[f"count_{i+1}"] = count_var
    
    def start_processing(self):
        if "No camera" in self.camera_var.get() or not self.camera_var.get():
            messagebox.showerror("Error", "Kamera tidak dipilih atau tidak tersedia.")
            return
        
        detector = self.controller.get_detector()
        if detector:
            detector.reset()
        else:
            return

        # ### FITUR BARU: Kirim sinyal START MOTOR ke Arduino ###
        if self.controller.motor:
            self.controller.motor.set_motor_speed(self.current_speed_level)

        for i in range(1, 4):
            self.count_labels[f"count_{i}"].set("0")
            
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.back_button.config(state=tk.DISABLED)
        self.camera_dropdown.config(state=tk.DISABLED)
        
        self.stop_event.clear()
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_processing(self):
        # ### FITUR BARU: Kirim sinyal STOP MOTOR ke Arduino ###
        if self.stop_event.is_set():
            return # Jika sudah dihentikan, jangan lakukan apapun lagi
        
        if self.controller.motor:
            self.controller.motor.stop_motor()

        self.stop_event.set()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.back_button.config(state=tk.NORMAL)
        self.camera_dropdown.config(state="readonly")
        
        # Tampilkan frame hitam saat video berhenti
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        img_pil = Image.fromarray(black_frame)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk

    def _video_loop(self):
        detector = self.controller.get_detector()
        if detector is None:
            self.controller.after(0, self.stop_processing)
            return
        
        try:
            camera_path = self.camera_var.get()
            # Opsi untuk menggunakan file video untuk testing di Linux
            if platform.system() == "Linux" and camera_path == "../output.avi":
                 self.cap = cv2.VideoCapture(camera_path)
            # Opsi untuk kamera live
            else:
                 self.cap = cv2.VideoCapture(int(camera_path))

            if not self.cap.isOpened():
                raise IOError(f"Tidak dapat membuka kamera {camera_path}")
            
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Peringatan: Tidak dapat membaca frame. Mungkin akhir video.")
                    # Jika ini file video, loop kembali ke awal
                    if platform.system() == "Linux" and camera_path == "../output.avi":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break # Keluar loop jika kamera live tidak memberikan frame

                processed_frame, counts = detector.process_frame(frame)
                self.controller.after(0, self._update_gui, processed_frame, counts)
        
        except Exception as e:
            messagebox.showerror("Error Proses Video", f"Terjadi kesalahan:\n{e}")
        
        finally:
            if self.cap:
                self.cap.release()
            # Pastikan GUI kembali ke state "stopped"
            self.controller.after(0, self.stop_processing)

    def _update_gui(self, frame, counts):
        if self.stop_event.is_set():
            return
        
        # Logika untuk resize frame agar pas di label (letterboxing)
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w <= 1 or label_h <= 1:
            self.controller.after(20, lambda: self._update_gui(frame, counts))
            return
        
        frame_h, frame_w, _ = frame.shape
        scale = min(label_w / frame_w, label_h / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        canvas = np.zeros((label_h, label_w, 3), dtype=np.uint8)
        y_offset = (label_h - new_h) // 2
        x_offset = (label_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk
        
        # Update label hitungan
        for i in range(1, 4):
            self.count_labels[f"count_{i}"].set(str(counts[f"count_{i}"]))

    def _find_cameras(self):
        available_devices = []
        if platform.system() == "Linux":
            # Di Linux, cari /dev/video*
            video_paths = sorted(glob.glob("/dev/video*"))
            # Tambahkan opsi video file untuk testing
            available_devices.append("../output.avi") 
            for path in video_paths:
                available_devices.append(path.replace("/dev/video", ""))
        else: # Windows
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap is not None and cap.isOpened():
                    available_devices.append(str(i))
                    cap.release()
        return available_devices if available_devices else ["No camera found"]



    # untuk PWM motor
    def _increase_speed(self):
        if self.current_speed_level < self.max_speed_level:
            self.current_speed_level += 1
            self.speed_label_var.set(f"{self.current_speed_level}")
            if self.controller.motor and not self.stop_event.is_set():
                self.controller.motor.set_motor_speed(self.current_speed_level)

    def _decrease_speed(self):
        if self.current_speed_level > 1:
            self.current_speed_level -= 1
            self.speed_label_var.set(f"{self.current_speed_level}")
            if self.controller.motor and not self.stop_event.is_set():
                self.controller.motor.set_motor_speed(self.current_speed_level)



    # untuk kecerahan LED
    def _increase_brightness(self):
        if self.current_brightness_level < self.max_brightness_level:
            self.current_brightness_level += 1
            self.brightness_label_var.set(f"{self.current_brightness_level}")
            if self.controller.motor:
                self.controller.motor.set_led_brightness(self.current_brightness_level)

    def _decrease_brightness(self):
        if self.current_brightness_level > 1:
            self.current_brightness_level -= 1
            self.brightness_label_var.set(f"{self.current_brightness_level}")
            if self.controller.motor:
                self.controller.motor.set_led_brightness(self.current_brightness_level)

# ============================================================================
# HALAMAN PERHITUNGAN BEBAS
# ============================================================================
class FreeCountPage(BaseCountingPage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        # Tidak ada kustomisasi khusus, semua sudah di Base class


# ============================================================================
# HALAMAN PERHITUNGAN SESUAI TARGET
# ============================================================================
class TargetCountPage(BaseCountingPage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        self.target = 0
        self.target_sound_played = False

        # Hancurkan widget "Metode" yang dibuat oleh parent
        for widget in self.left_panel.winfo_children():
            widget.destroy()

        # Bangun infrastruktur scroll di dalam left_panel
        canvas = tk.Canvas(self.left_panel, borderwidth=0)
        scrollbar = ttk.Scrollbar(self.left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Panggil kembali _create_count_widgets, targetkan ke scrollable_frame
        self._create_count_widgets(parent_frame=scrollable_frame)

        # Tambahkan widget khusus untuk halaman ini
        ttk.Label(scrollable_frame, text="Target Ikan:", font=("Helvetica", 12, "bold")).pack(fill='x', pady=(20, 2), padx=10)
        self.target_var = tk.StringVar(value="10")
        self.target_entry = ttk.Entry(scrollable_frame, textvariable=self.target_var, width=10)
        self.target_entry.pack(fill='x', padx=10)

        # Ganti tombol Start default dengan tombol Set Target
        self.start_button.pack_forget() 
        self.set_target_button = ttk.Button(scrollable_frame, text="Set Target & Mulai", command=self.start_processing)
        self.set_target_button.pack(fill='x', padx=10, pady=(10, 20))

        self.keyboard = OnScreenKeyboard(scrollable_frame, self.target_entry)
        self.keyboard.pack(fill='x', padx=10, pady=(0, 10))

    def set_target(self):
        """Membaca dan memvalidasi nilai target dari entry box."""
        try:
            self.target = int(self.target_var.get())
            if self.target <= 0:
                messagebox.showerror("Input Salah", "Target harus lebih dari 0.")
                return False
        except ValueError:
            messagebox.showerror("Input Salah", "Target harus berupa angka.")
            return False
        
        detector = self.controller.get_detector()
        if detector:
            detector.reset()
            
        for i in range(1, 4):
            self.count_labels[f"count_{i}"].set("0")
        return True

    def start_processing(self):
        """Override fungsi start_processing untuk validasi target terlebih dahulu."""
        if not self.set_target():
            return
        
        super().start_processing() # Panggil fungsi start_processing dari parent class
        
        self.set_target_button.config(state=tk.DISABLED)
        self.target_entry.config(state=tk.DISABLED)
        self.target_sound_played = False

    def stop_processing(self):
        """Override fungsi stop_processing untuk menampilkan hasil akhir."""
        is_stopping_normally = not self.stop_event.is_set()
        
        super().stop_processing() # Panggil fungsi stop_processing dari parent class
        
        if is_stopping_normally:
            try:
                count3 = int(self.count_labels["count_3"].get()) 
                excess = count3 - self.target
                if excess >= 0:
                    message = f"Penghitungan Selesai!\n\nTarget: {self.target}\nTerhitung: {count3}\n\nKelebihan: {excess} ekor"
                else:
                    message = f"Penghitungan Selesai!\n\nTarget: {self.target}\nTerhitung: {count3}\n\nKekurangan: {abs(excess)} ekor"
                messagebox.showinfo("Hasil Penghitungan", message)
            except ValueError:
                messagebox.showwarning("Hasil", "Tidak dapat menampilkan hasil akhir karena data hitungan tidak valid.")
        
        self.set_target_button.config(state=tk.NORMAL)
        self.target_entry.config(state=tk.NORMAL)

    def _update_gui(self, frame, counts):
        """Override fungsi _update_gui untuk mengecek target dan membunyikan buzzer."""
        super()._update_gui(frame, counts) # Panggil fungsi update dari parent
        
        current_count = counts["count_3"] 
        if self.controller.buzzer and not self.target_sound_played and current_count >= self.target:
            print("Target tercapai! Membunyikan buzzer.") 
            self.controller.buzzer.beep(duration=1) 
            self.target_sound_played = True


if __name__ == "__main__":
    app = FishCounterApp()
    app.mainloop()
