# main_detector.py

import cv2
import time
import numpy as np
import torch # Required for the dfl function in post-processing

# Assuming the previous code is in a file named fish_detector_components.py
from fish_detector_components import Sort, MetricsLogger, BBoxUtils, KalmanBoxTracker

# These imports would typically be handled by your environment setup
# For this example, we assume they are available.
from py_utils.coco_utils import COCO_test_helper
from py_utils.rknn_executor import RKNN_model_container

class ObjectDetector:
    """
    The main class to handle object detection, tracking, and counting in a video stream.
    It orchestrates the pre-processing, inference, post-processing, tracking, and visualization.
    """
    def __init__(self, model_path, img_size=(640, 640), obj_thresh=0.048, nms_thresh=0.048):
        # --- Configuration ---
        self.IMG_SIZE = img_size
        self.OBJ_THRESH = obj_thresh
        self.NMS_THRESH = nms_thresh
        np.random.seed(0)

        # --- Initialization of Components ---
        self.model = RKNN_model_container(model_path, 'rk3588')
        self.mot_tracker = Sort(max_age=5, min_hits=3)
        self.metrics_logger = MetricsLogger('detection_metrics.csv')
        self.co_helper = COCO_test_helper(enable_letter_box=True)

        # --- State Variables for Tracking and Counting ---
        self.frame_counter = 0
        self.previous_track_2 = np.empty((0,5))
        self.previous_track_3 = np.empty((0,5))
        
        # --- State Variables for Multiple Counting Methods ---
        self.fish_count_1 = 0
        self.counted_ids_1 = set()
        self.fish_count_2 = 0
        self.counted_ids_2 = set()
        self.fish_count_3 = 0
        self.counted_ids_3 = set()

        # --- Define Counting Boundaries ---
        self.line_x_pos = int(0.645 * self.IMG_SIZE[0])
        self.line_x_pos_right = int(0.78 * self.IMG_SIZE[0])
        self.line_x_pos_left = int(0.58 * self.IMG_SIZE[0])
        
        self.boundary1 = [(int(0.58*self.IMG_SIZE[0]), 0), (int(0.58*self.IMG_SIZE[0]), self.IMG_SIZE[1])]
        self.boundary2 = [(int(0.68*self.IMG_SIZE[0]), 0), (int(0.68*self.IMG_SIZE[0]), self.IMG_SIZE[1])]
        self.boundary3 = [(int(0.78*self.IMG_SIZE[0]), 0), (int(0.78*self.IMG_SIZE[0]), self.IMG_SIZE[1])]

    def process_frame(self, frame):
        """
        Accepts a single frame, performs detection and tracking, and returns the annotated frame.
        :param frame: A single video frame (NumPy array).
        :return: A tuple containing (annotated_frame, dictionary_of_counts).
        """
        self.frame_counter += 1
        
        # --- 1. Pre-processing ---
        start_time = time.time()
        img_rgb = self._preprocess(frame)
        preprocess_time = time.time() - start_time

        # --- 2. Model Inference ---
        start_time = time.time()
        model_outputs = self.model.run([img_rgb])
        detection_time = time.time() - start_time

        # --- 3. Post-processing ---
        start_time = time.time()
        boxes, _, scores = self._post_process(model_outputs)
        postprocess_time = time.time() - start_time
        
        # --- 4. Tracking ---
        start_time = time.time()
        detections = np.column_stack((boxes, scores)) if boxes.size > 0 else np.empty((0, 5))
        active_tracks = self.mot_tracker.update(detections)
        tracking_time = time.time() - start_time

        # --- 5. Counting and Visualization ---
        start_time = time.time()
        self._update_counts_and_visualize(frame, active_tracks)
        counting_time = time.time() - start_time

        # --- 6. Logging ---
        self.metrics_logger.log(
            frame_number=self.frame_counter,
            preprocess_time=f'{preprocess_time:.6f}',
            detection_time=f'{detection_time:.6f}',
            postprocess_time=f'{postprocess_time:.6f}',
            tracking_time=f'{tracking_time:.6f}',
            counting_time=f'{counting_time:.6f}',
            fish_count=self.fish_count_1
        )
        
        counts = {
            "count_1": self.fish_count_1,
            "count_2": self.fish_count_2,
            "count_3": self.fish_count_3
        }
        return frame, counts

    def _preprocess(self, frame):
        """Prepares a frame for the model."""
        img = self.co_helper.letter_box(im=frame.copy(), new_shape=self.IMG_SIZE, pad_color=(0,0,0))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _update_counts_and_visualize(self, frame, tracks):
        """Updates all fish counts and draws visualizations on the frame."""
        # Draw all boundary lines
        cv2.line(frame, (self.line_x_pos, 0), (self.line_x_pos, self.IMG_SIZE[1]), (0, 255, 0), 5)
        cv2.line(frame, (self.line_x_pos_left, 0), (self.line_x_pos_left, self.IMG_SIZE[1]), (0, 0, 255), 2)
        cv2.line(frame, (self.line_x_pos_right, 0), (self.line_x_pos_right, self.IMG_SIZE[1]), (0, 0, 255), 2)
        cv2.line(frame, self.boundary1[0], self.boundary1[1], (0, 255, 0), 2)
        cv2.line(frame, self.boundary2[0], self.boundary2[1], (0, 255, 0), 2)
        cv2.line(frame, self.boundary3[0], self.boundary3[1], (0, 255, 0), 2)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # --- Counting Method 1: Simple Right Boundary Cross ---
            if track_id not in self.counted_ids_1 and x2 > self.line_x_pos:
                self.fish_count_1 += 1
                self.counted_ids_1.add(track_id)

            # --- Counting Method 2: Enter Region ---
            if track_id not in self.counted_ids_2 and x2 > self.line_x_pos_left and x1 < self.line_x_pos_right:
                self.fish_count_2 += 1
                self.counted_ids_2.add(track_id)

            # --- Counting Method 3: Line Intersection Check ---
            xywh_track = BBoxUtils.convert_bbox_to_xywh(track)
            current_x, current_y = int(xywh_track[0][0]), int(xywh_track[0][1])
            cv2.circle(frame, (current_x, current_y), 5, (0, 255, 255), -1)
            
            # Find previous position of this track ID
            prev_pos = None
            prev_track_3_match = self.previous_track_3[self.previous_track_3[:, 4] == track_id]
            if len(prev_track_3_match) > 0:
                prev_pos = BBoxUtils.convert_bbox_to_xywh(prev_track_3_match[0])[0]
            else:
                prev_track_2_match = self.previous_track_2[self.previous_track_2[:, 4] == track_id]
                if len(prev_track_2_match) > 0:
                    prev_pos = BBoxUtils.convert_bbox_to_xywh(prev_track_2_match[0])[0]

            if prev_pos is not None and track_id not in self.counted_ids_3:
                prev_x, prev_y = int(prev_pos[0]), int(prev_pos[1])
                if self._check_cross(self.boundary1, (prev_x, prev_y), (current_x, current_y)) or \
                   self._check_cross(self.boundary2, (prev_x, prev_y), (current_x, current_y)) or \
                   self._check_cross(self.boundary3, (prev_x, prev_y), (current_x, current_y)):
                    self.fish_count_3 += 1
                    self.counted_ids_3.add(track_id)

        # Update historical track data for next frame
        self.previous_track_3 = self.previous_track_2
        self.previous_track_2 = tracks

        # Draw counters on frame
        cv2.putText(frame, f'Frame: {self.frame_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Fish Count 1: {self.fish_count_1}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Fish Count 2: {self.fish_count_2}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Fish Count 3: {self.fish_count_3}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def _check_cross(self, boundary, start_centroid, current_centroid):
        """Helper function to check if a line segment intersects a boundary line."""
        p1 = np.array(boundary[0])
        p2 = np.array(boundary[1])
        p3 = np.array(start_centroid)
        p4 = np.array(current_centroid)

        denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
        if denom == 0: return False # Parallel lines
        
        t_num = (p1[1] - p3[1]) * (p4[0] - p3[0]) - (p1[0] - p3[0]) * (p4[1] - p3[1])
        u_num = (p1[1] - p3[1]) * (p2[0] - p1[0]) - (p1[0] - p3[0]) * (p2[1] - p1[1])
        
        t = t_num / denom
        u = u_num / denom
        
        return 0 < t < 1 and 0 < u < 1

    # --- Post-processing methods specific to the YOLOv6 model ---
    def _post_process(self, input_data):
        """Converts raw model output into filtered bounding boxes, classes, and scores."""
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3
        pair_per_branch = len(input_data)//defualt_branch
        for i in range(defualt_branch):
            boxes.append(self._box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        boxes, classes, scores = self._filter_boxes(boxes, scores, classes_conf)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b, c_arr, s = boxes[inds], classes[inds], scores[inds]
            keep = self._nms_boxes(b, s)
            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c_arr[keep])
                nscores.append(s[keep])

        if not nboxes:
            return np.array([]), np.array([]), np.array([])

        return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score * box_confidences >= self.OBJ_THRESH)
        return boxes[_class_pos], classes[_class_pos], (class_max_score * box_confidences)[_class_pos]

    def _nms_boxes(self, boxes, scores):
        x, y = boxes[:, 0], boxes[:, 1]
        w, h = boxes[:, 2] - x, boxes[:, 3] - y
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1)
            h1 = np.maximum(0.0, yy2 - yy1)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            inds = np.where(ovr <= self.NMS_THRESH)[0]
            order = order[inds + 1]
        return np.array(keep)

    def _box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        grid = np.concatenate((col.reshape(1, 1, grid_h, grid_w), row.reshape(1, 1, grid_h, grid_w)), axis=1)
        stride = np.array([self.IMG_SIZE[1]//grid_h, self.IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

        if position.shape[1] == 68: # Using DFL
            position = self._dfl(position)
        
        box_xy = (position[:, :2, :, :] * 2 - 0.5 + grid) * stride
        box_wh = (position[:, 2:4, :, :] * 2)**2 * stride
        
        x1y1 = box_xy - box_wh / 2
        x2y2 = box_xy + box_wh / 2
        return np.concatenate((x1y1, x2y2), axis=1)

    @staticmethod
    def _dfl(position):
        x = torch.tensor(position)
        n,c,h,w = x.shape
        p_num = 4
        mc = c//p_num
        y = x.reshape(n,p_num,mc,h,w).softmax(2)
        acc_metrix = torch.arange(mc).float().reshape(1,1,mc,1,1)
        return (y*acc_metrix).sum(2).numpy()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    MODEL_PATH = "./yolov6.rknn"
    VIDEO_SOURCE = "./videotest/videobaru.mp4" # or use a device ID like "/dev/video11"

    try:
        detector = ObjectDetector(model_path=MODEL_PATH)
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {VIDEO_SOURCE}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        video_writer = cv2.VideoWriter("./output_counting.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, counts = detector.process_frame(frame)
            
            cv2.imshow("Fish Counting", processed_frame)
            video_writer.write(processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'cap' in locals(): cap.release()
        if 'video_writer' in locals(): video_writer.release()
        cv2.destroyAllWindows()
        print("Resources released.")
