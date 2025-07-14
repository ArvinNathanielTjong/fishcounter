import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import platform

YOLOV6_DIR = Path(__file__).parent / "YOLOv6"
sys.path.insert(0, str(YOLOV6_DIR))

from yolov6.utils.events import load_yaml
from yolov6.utils.checkpoint import load_checkpoint
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression


class YOLOv6Detector:
    def __init__(self, model_path, yaml_path, device="cpu", img_size=640):
        self.device = device
        self.img_size = img_size
        self.model = torch.load(model_path, map_location=device)['model'].float().eval()
        self.model.to(device)
        # Load class names from a yaml file if you have one, or set manually
        self.class_names = load_yaml(yaml_path)['names']

    def detect(self, frame, conf_thres=0.25, iou_thres=0.45):
        img = letterbox(frame, new_shape=(self.img_size, self.img_size), auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
        detections = []
        if pred is not None and len(pred):
            pred[:, :4] = pred[:, :4] / self.img_size  # scale boxes to 0-1
            h, w = frame.shape[:2]
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = [int(x * w if i % 2 == 0 else x * h) for i, x in enumerate(xyxy)]
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': float(conf),
                    'class_id': int(cls),
                    'class_name': self.class_names[int(cls)]
                })
        return detections

    def visualize(self, frame, detections):
        img = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['score']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        return img

if __name__ == "__main__":
    model_path = "../models/model1.pt"
    yaml_path = "../models/dataset.yaml"
    camera_path = "/dev/video0" if platform.system() == "Linux" else 0  
    detector = YOLOv6Detector(model_path, yaml_path, device="cpu")  # or "cuda:0" for GPU

    cap = cv2.VideoCapture(camera_path)
    if not cap.isOpened():
        print("Error: Could not open camera. Please check your camera index or connection.")
        exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue 

            detections = detector.detect(frame)
            vis_frame = detector.visualize(frame, detections)

            cv2.imshow("YOLOv6 Webcam Detection", vis_frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Detection stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

    cap.release()
    cv2.destroyAllWindows()