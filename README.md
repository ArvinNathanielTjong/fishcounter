## clone this REPO!

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
git clone --depth 1 https://github.com/ArvinNathanielTjong/fishcounter-training.git
```


# 🍊 Orange Pi 5 Pro Setup

## clone this repo
``` bash
git clone --depth 1 https://github.com/ArvinNathanielTjong/fishcounter.git
```


## 📦 Install Python & Libraries

```bash
sudo apt update
sudo apt install python3-tk python3-pil.imagetk python3-pip
/bin/python -m pip install opencv-python
git submodule update --init --recursive
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install requests filterpy
```



REFERENCE FOR THE THINGS BELOW :  
🔗 https://github.com/Qengineering/YoloV6-NPU

---
### Installing the dependencies.
Start with the usual 
```
sudo apt-get update 
sudo apt-get upgrade
sudo apt-get install cmake wget curl
```

#### RKNPU2
```
$ git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git
```
We only use a few files.
```
rknn-toolkit2-master
│      
└── rknpu2
    │      
    └── runtime
        │       
        └── Linux
            │      
            └── librknn_api
                ├── aarch64
                │   └── librknnrt.so
                └── include
                    ├── rknn_api.h
                    ├── rknn_custom_op.h
                    └── rknn_matmul_api.h

cd ~/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/aarch64
sudo cp ./librknnrt.so /usr/local/lib
cd ~/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/include
sudo cp ./rknn_* /usr/local/include
```
Save 2 GB of disk space by removing the toolkit. We do not need it anymore.
```
cd ~
sudo rm -rf ./rknn-toolkit2-master
```

---

### make sure to install the torch version to the one at the start if it uninstall and install the torch
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 #if neccesary
```


Try to follow the instructions below:

Clone the rknpu2 repository: git clone https://github.com/rockchip-linux/rknpu2.git

Copy the shared lib file to the lib dir: sudo cp rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/librknnrt.so

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
sudo apt install python3-serial
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


---

# 🔧 Tuning Detection & Tracking Sensitivity

Your fish counter's performance is controlled by six key parameters. Tweaking them can significantly change how accurately it detects and tracks fish. These settings are found in two different files.

## Part 1: Initial Detection (obj_thresh & nms_thresh)

These two parameters control the initial object detection performed by the AI model. They determine what the system considers a valid "fish" in a single frame, before tracking even begins.

➡️ Where to Edit: You can find these settings in the launcher_app.py file, inside the get_detector method.

``` bash
# In launcher_app.py

def get_detector(self):
    """Membuat instance detector jika belum ada."""
    if self.detector is None:
        try:
            # ...
            self.detector = ObjectDetector(model_path=model_path, 
            img_size=(640, 640), 
            obj_thresh=0.048,   # <-- EDIT HERE
            nms_thresh=0.048)   # <-- AND HERE
        # ...
```

### 1. obj_thresh (Object Threshold)
What it is: The minimum confidence score (from 0.0 to 1.0) the AI must have to consider a detection valid.

Analogy: Think of it as asking the AI, "How sure are you that this is a fish?" A value of 0.048 means the AI only needs to be 4.8% sure.

How to Tune:

Increase this value (e.g., to 0.3) if you are getting too many false positives (detecting things that aren't fish). This makes the AI more "picky."

Decrease this value if the AI is missing fish that are hard to see. This makes the AI less "picky."

---
### 2. nms_thresh (Non-Max Suppression Threshold)
What it is: The overlap threshold. It's used to clean up cases where the AI draws multiple bounding boxes on the same single fish.

Analogy: If the AI draws three boxes on one fish, NMS decides to keep only the "best" one and discard the others that overlap too much.

How to Tune:

Decrease this value (e.g., to 0.2) if single fish are being counted multiple times. This will more aggressively merge overlapping boxes.

Increase this value (e.g., to 0.6) if you have many fish very close together and the tracker is mistakenly merging them into one box.

---

## Part 2: Fine-Tuning the Tracking
These four parameters control the Sort tracker. They don't affect the initial detection; they affect how the system connects detections from one frame to the next to maintain a consistent ID for each fish.

➡️ Where to Edit: You can find these settings in the object_detector.py file, inside the __init__ method where mot_tracker is created.

``` bash
# In /utils/object_detector.py

class ObjectDetector:
    def __init__(self, ...):
        # ...
        self.mot_tracker = Sort(max_age=1,            # <-- EDIT HERE
        min_hits=1,           # <-- EDIT HERE
        diou_threshold=0.3,   # <-- EDIT HERE
        dij_threshold=0.9)    # <-- EDIT HERE
        # ...
```

### 3. max_age ⏳
What it is: The tracker's short-term memory. It's the max number of frames a track can exist without being matched to a detection before it's deleted.

Analogy: If a fish disappears behind an obstacle, this is how many frames the tracker will "remember" it and keep looking for it.

How to Tune: The current value of 1 is very low. Increase it (e.g., to 5 or 10) if fish that are momentarily lost are getting assigned new IDs when they reappear.

### 4. min_hits ✨
What it is: The confirmation rule. It's the number of consecutive frames a detection must appear before it's given a permanent track ID.

Analogy: This prevents a random, one-frame glitch from being counted as a fish. It needs to see the object a few times to be sure.

How to Tune: The current value of 1 means detections are trusted instantly. Increase it (e.g., to 2 or 3) if you are getting phantom tracks from noise or false detections.

### 5. dij_threshold 🎯
What it is: The strict matching rule for high-confidence detections. It requires a high similarity score (based on the distance between object centers) for a match.

Analogy: Matching a crystal-clear photo to a passport photo. It needs to be a near-perfect match.

How to Tune: A value of 0.9 is very strict. Decrease it slightly (e.g., to 0.85) if stable tracks are being lost because the fish moves too fast between frames.

### 6. diou_threshold 🖇️
What it is: The lenient matching rule for low-confidence detections. It uses a combined score of distance and box overlap.

Analogy: Matching a blurry security camera photo. You're more forgiving just to keep the track from being lost.

How to Tune: A value of 0.3 is fairly lenient. This is generally a good value, but you could lower it (e.g., to 0.2) to try and hold onto tracks even more tenaciously, at the risk of an incorrect match.
