import sys
import os

# --- KONFIGURASI UTAMA (Atur parameter Anda di sini) ---

path_ke_folder_yolov6 = './YOLOv6' 

# 2. Path ke model Anda (relatif dari skrip ini)
path_ke_model_pt = './YOLOv6/best_ckpt.pt'

# 3. Path ke file data.yaml (untuk nama kelas)
path_ke_yaml = './Rupiah-2/data.yaml' # Ganti dengan path data.yaml Anda

# 4. ID Webcam Anda (coba '0' atau '1')
id_webcam = '1'

# 5. Konfigurasi Deteksi
conf_threshold = 0.4
iou_threshold = 0.45
device_to_use = 'cpu'

# -----------------------------------------------------------


# --- BAGIAN LOGIKA (Tidak Perlu Diubah) ---

# Mengubah path relatif menjadi path absolut yang lengkap
# Ini penting agar script bisa dijalankan dari mana saja
CWD = os.getcwd()
path_ke_folder_yolov6 = os.path.join(CWD, path_ke_folder_yolov6)
path_ke_model_pt = os.path.join(CWD, path_ke_model_pt)
path_ke_yaml = os.path.join(CWD, path_ke_yaml)

# Menambahkan folder YOLOv6 ke path Python agar bisa di-import
if path_ke_folder_yolov6 not in sys.path:
    sys.path.append(path_ke_folder_yolov6)

# Mengubah direktori kerja ke folder YOLOv6
# Ini krusial agar semua path di dalam library YOLOv6 berfungsi
try:
    os.chdir(path_ke_folder_yolov6)
    print(f"Direktori kerja diubah ke: {os.getcwd()}")
except FileNotFoundError:
    print(f"Error: Folder YOLOv6 tidak ditemukan di '{path_ke_folder_yolov6}'")
    sys.exit()

# Mengimpor fungsi 'run' langsung dari skrip infer.py
from tools.infer import run

print("\nMemulai proses inferensi menggunakan library YOLOv6...")
print(f"Model: {path_ke_model_pt}")
print(f"Webcam ID: {id_webcam}")
print("Tekan 'q' pada jendela video untuk keluar.")

# Menjalankan fungsi inferensi dengan konfigurasi yang sudah diatur
run(
    weights=path_ke_model_pt,
    source=id_webcam,
    webcam=True,
    webcam_addr=id_webcam,
    yaml=path_ke_yaml,
    img_size=640,
    conf_thres=conf_threshold,
    iou_thres=iou_threshold,
    device=device_to_use,
    view_img=True,       # WAJIB True untuk menampilkan jendela video
    not_save_img=True    # Tidak menyimpan hasil sebagai file
)

print("\nAplikasi ditutup.")