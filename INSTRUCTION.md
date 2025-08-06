# 📦 HOW TO COMBINE MODELS

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

follow this github : 
```
git clone https://github.com/ArvinNathanielTjong/fishcounter-training.git
```


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
