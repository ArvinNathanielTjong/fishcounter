# 📦 HOW TO COMBINE MODELS

## ✅ General Steps

1. Make sure you have the **dataset** (that is not trained yet).  
   > (Personally, I use Roboflow to label)

2. Label it with the name `"fish"` (to keep it general).

3. Download the dataset (new dataset).

4. Change the `data.yaml` `train`, `val`, and `test` paths. Example:

    ```
    train: /content/drive/MyDrive/fishcounter/patin-dataset/images/train
    val: /content/drive/MyDrive/fishcounter/patin-dataset/images/valid
    test: /content/drive/MyDrive/fishcounter/patin-dataset/images/test
    ```

5. Put it in your Google Drive like this:

    ```
    My Drive/
    └── YOLOv6_Project/
        ├── A.pt
        └── my_new_dataset/
            ├── images/
            │   ├── train/
            │   └── val/
            └── labels/
                ├── train/
                └── val/
    ```

6. Open this Colab notebook:  
   [Google Colab Link](https://colab.research.google.com/drive/1DRv1PBJXkkRe2cn3OSftZ1sD2MUQTKk8?usp=sharing)  
   *(Currently not done)*

---

# 🖥️ Combining Models Locally (PC Setup)

## 🧪 1. System Requirements

- Check your NVIDIA version:

    ```bash
    nvidia-smi
    ```

- Install CUDA:  
  [Download CUDA 12.5](https://developer.nvidia.com/cuda-12-5-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

- Install GPU-compatible PyTorch:  
  *(Personally using PyTorch 2.5.1)*

---

## 📁 2. Folder Structure
First, you need to download the YOLOv6 source code. Navigate to where you want your project to live and run the following command. This is the airockchip fork, which is helpful for later conversion steps to RKNN.

```
git clone [https://github.com/airockchip/YOLOv6.git](https://github.com/airockchip/YOLOv6.git)
```

After cloning the repository, arrange your project so it matches the structure below. This involves creating your custom configs and patin-dataset folders and placing your FISHCOUNTER_TRAINING.ipynb notebook at the top level.

Ensure your training directory looks like this:

```
training/
├── FISHCOUNTER_TRAINING/
│   ├── configs/
│   │   └── yolov6n_config.py     # <- change pretrained path here
│   ├── patin-dataset/
│   │   ├── data.yaml             # <- change image paths here
│   │   └── images/, labels/, etc.
│   ├── scripts/
│   └── YOLOv6/
└── FISHCOUNTER_TRAINING.ipynb
```




- Run `FISHCOUNTER_TRAINING.ipynb`
- In the **local section**, check:
  - `data.yaml` path
  - config file path (`yolov6n_config.py`)

---

# 🍊 Orange Pi 5 Pro Setup

## 📦 Install Python & Libraries

```bash
sudo apt update
sudo apt install python3-tk python3-pil.imagetk python3-pip
/bin/python -m pip install opencv-python
git submodule update --init --recursive
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install request filterpy
```

Follow YOLOv6 NPU GitHub:  
🔗 https://github.com/Qengineering/YoloV6-NPU

---

## PT file -> ONNX
```
pip install torch onnx
```
then follow the instruction inside the training folder called ONNX_RKNNexport.ipynb

## 🧠 RKNN Setup (For NPU)

```bash
sudo apt-get update
sudo apt-get install cmake
pip3 install rknn-toolkit2
```

notes : in the link below you need to download the convert.py and follow step number 4

Convert ONNX to RKNN:  
🔗 https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov6

---

## 🔔 GPIO (Buzzer) Setup

```bash
sudo apt install python3-periphery
sudo nano /etc/udev/rules.d/99-gpio.rules
```

Paste this content:

```bash
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", ACTION=="add", PROGRAM="/bin/sh -c 'chown root:gpio /sys/class/gpio/export /sys/class/gpio/unexport ; chmod 220 /sys/class/gpio/export /sys/class/gpio/unexport'"
SUBSYSTEM=="gpio", KERNEL=="gpio*", ACTION=="add", PROGRAM="/bin/sh -c 'chown root:gpio /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value ; chmod 660 /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value'"
```

Then run:

```bash
sudo groupadd gpio        # Skip if already exists
sudo usermod -aG gpio orangepi
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

# 🔌 UART Setup

```bash
pip install pyserial
python3 launcher_app.py
```

---

# 🕒 Run on Boot Using Crontab

```bash
crontab -e
```

Add this line:

```bash
@reboot (sleep 15 && export DISPLAY=:0 && cd /home/orangepi/github/fishcounter/src/ && /usr/bin/python3 launcher_app.py) >> /home/orangepi/github/fishcounter/launcher.log 2>&1
```

✅ Make sure your paths are correct!

---

# 📝 Notes

- Ensure your `.rknn` model file is ready to use NPU.
- Orange Pi 5 Pro uses **Rockchip RK3588S** chip.

---

# 🔁 Udev Rules for Arduino Port (Optional)

Create a file:

```bash
sudo nano /etc/udev/rules.d/99-fishcounter.rules
```

Paste this content:

```bash
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0666", SYMLINK+="arduino"
```

Then reload the rules:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
