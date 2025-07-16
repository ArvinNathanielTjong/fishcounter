import os
import sys
import numpy as np
import math

import subprocess

#NULIS CSV
import csv

import time
from filterpy.kalman import KalmanFilter

import cv2

#Cam Setup
# ===================================================================
# BAGIAN BARU: PENGATURAN KAMERA ADAPTIF (MODIFIKASI)
# ===================================================================
import platform
import glob
import argparse

def list_video_devices_linux():
    """Mendeteksi semua kamera yang tersedia di Linux."""
    available_devices = []
    video_paths = sorted(glob.glob("/dev/video*"))
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if cap is not None and cap.isOpened():
            available_devices.append(path)
            cap.release()
    return available_devices

# Cek apakah script dijalankan di Linux, jika tidak, langsung hentikan.
if platform.system() != "Linux":
    print("ERROR: Skrip ini dirancang khusus untuk berjalan di Linux dengan NPU Rockchip (seperti Orange Pi).")
    sys.exit()

# Pengaturan argumen terminal untuk memilih kamera
parser = argparse.ArgumentParser(description="RKNN Fish Counter untuk Orange Pi dengan kamera adaptif.")
parser.add_argument('--camera', type=str, help='Path spesifik ke kamera, contoh: /dev/video0')
args = parser.parse_args()

# Logika pemilihan kamera
if args.camera:
    device_id = args.camera
    print(f"INFO: Mencoba membuka kamera yang dipilih: {device_id}")
else:
    print("INFO: Mendeteksi kamera yang tersedia...")
    available_cams = list_video_devices_linux()
    if not available_cams:
        print("ERROR: Tidak ada kamera yang ditemukan di /dev/video*")
        sys.exit(1)
    device_id = available_cams[0]
    print(f"INFO: Kamera tidak dipilih, menggunakan kamera pertama yang ditemukan: {device_id}")

# Inisialisasi Video Capture (tetap menggunakan V4L2 untuk Linux)
cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
# ===================================================================

IMG_SIZE = (640, 640)
NMS_THRESH = 0.1 #0.048
OBJ_THRESH = 0.1 #0.048

# add path
#realpath = os.path.abspath(__file__)
#_sep = os.path.sep
#realpath = realpath.split(_sep)
#sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.coco_utils import COCO_test_helper

np.random.seed(0)

#NULIS CSV
class MetricsLogger:
    def __init__(self, output_path):
        self.output_path = output_path
        # Create/open the CSV file and write headers
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_number',
                'preprocess_time',
                'detection_time',
                'postprocess_time',
                'tracking_time',
                'counting_time',
                'fish_count'
            ])
    
    def log_metrics(self, frame_number, preprocess_time, detection_time, 
                   postprocess_time, tracking_time, counting_time, fish_count):
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_number,
                f'{preprocess_time:.6f}',
                f'{detection_time:.6f}',
                f'{postprocess_time:.6f}',
                f'{tracking_time:.6f}',
                f'{counting_time:.6f}',
                fish_count
            ])

# Association Matching
def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))
  
# BBox Related
def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  bb_test: dets
  bb_gt: trks
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  # print(f"bb_gt:{bb_gt}")
  # print(f"bb_test:{bb_test,np.shape(bb_test)}")

  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  # print(bb_test[..., 0],"---",bb_gt[..., 0])
  # print(f"xx1:{xx1}")
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  # print( "w",w,"h",h,"wh",wh)
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
  return(o)

def convert_bbox_to_xywh(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,w,h] 
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  ID = bbox[4]
  return np.array([x, y, w, h, ID]).reshape((1, 5))

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))

def convert_bbox_to_z2(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((1, 4))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
    
def L_diagonal(x1min,x2min,x1max,x2max,y1min,y2min,y1max,y2max):
  """
  L is diagonal distance between the smallest outer rectangle of two bounding boxes
  """
  xmin = min(x1min,x2min)
  xmax = max(x1max,x2max)
  ymin = min(y1min,y2min)
  ymax = max(y1max,y2max)

  L = (xmax-xmin)**2 + (ymax-ymin)**2
  L = math.sqrt(L)
  return L

def dij_distance(dets, trks):
  """
  Dij distance is euclidean distance between two center point of objects
  dets/trks is in [x1,y1,x2,y2,score]
  """
  # Check if dets or trks are empty and return appropriate empty array shapes
  if dets.shape[0] == 0:
    return np.empty((0, len(trks)),dtype=int)
  # if trks.shape[0] == 0:
  #   return np.empty((len(dets), 0),dtype=int)
  xdetmin = dets[...,0]
  xtrkmin = trks[...,0]
  ydetmax = dets[...,1]
  ytrkmax = trks[...,1]
  xdetmax = dets[...,2]
  xtrkmax = trks[...,2]
  ydetmin = dets[...,3]
  ytrkmin = trks[...,3]

  # print(f"xdetmin:{xdetmin}")
  # print(f"xtrkmax:{xtrkmax}")

  # convert bbox to centroid
  # score = dets[:,-1]
  dets = dets[:, :-1]
  dets_cp = []
  for det in dets:
    det = convert_bbox_to_z2(det)
    # print(det, np.shape(det))
    dets_cp.append(det)
  dets_cp = np.array(dets_cp)
  # print(dets_cp)
  # score = trks[:,-1]
  trks = trks[:, :-1]
  trks_cp = []
  for trk in trks:
    trk = convert_bbox_to_z2(trk)
    # print(trk, np.shape(trk))
    trks_cp.append(trk)
  trks_cp = np.array(trks_cp)


  x1 = dets_cp[...,0]
  y1 = dets_cp[...,1]
  x2 = trks_cp[...,0]
  y2 = trks_cp[...,1]

  # print(f"x1:{x1}")
  # print(f"x2:{x2}")
  # print(f"y1:{y1}")
  # print(f"y2:{y2}")

  dij_matrix = np.zeros([len(x1),len(x2)])
  for i in range(len(x1)):
    for j in range(len(x2)):
      L = L_diagonal(xdetmin[i],xtrkmin[j],xdetmax[i],xtrkmax[j],ydetmin[i],ytrkmin[j],ydetmax[i],ytrkmax[j])
      # print(f"L:{L}")
      dij_matrix[i][j] = 1 - (((x1[i,0]-x2[j,0])**2 + (y1[i,0]-y2[j,0])**2)/(L**2))

  return dij_matrix


def DIOU_2(iou_matrix, dij_matrix):
  """
  DIOU_2
  """
  diou_2 = (iou_matrix + dij_matrix)/2
  return diou_2

def divide_dets_byscore(dets):
  """
  divide the detections result to high-score and low-score
  """
  if dets.shape[0]==0:
    return np.empty((0, 5),dtype=int),np.empty((0, 5),dtype=int)

  # highscore_dets = np.array(np.empty((0,5)))
  dets = dets.tolist()
  # print(dets)
  highscore_dets = []
  lowscore_dets = []
  # print(highscore_dets,",",lowscore_dets)
  for det in dets:
    # print(det)
    if det[-1] >= 0.7:
      highscore_dets.append(det)  # Add to high-score list
    else:
      lowscore_dets.append(det)  # Add to low-score list
  if len(highscore_dets)==0:
    highscore_dets = np.empty((0,5))
  highscore_dets = np.array(highscore_dets)
  if len(lowscore_dets)==0:
    lowscore_dets = np.empty((0,5))
  lowscore_dets = np.array(lowscore_dets)
  # print(highscore_dets.shape,lowscore_dets.shape)
  return highscore_dets, lowscore_dets


# Kalman Filter
class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.kf.x[4] = 65
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)
    
# Associate Detections to Trackers
# 1st Stage
def associate_detections_firststage(highscore_detections,trackers,dij_threshold = 0.9):
  """
  Assigns detections to tracked object (both represented as bounding boxes) with DiJ Euclidean Distance Method
  For High Score Detections
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0): # what if there is no high-score detections? -> already handled in dij_distance
    return np.empty((0,2),dtype=int), np.arange(len(highscore_detections)), np.empty((0,5),dtype=int)

  dij_matrix = dij_distance(highscore_detections, trackers)

  if min(dij_matrix.shape) > 0:
    a = (dij_matrix > dij_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-dij_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(highscore_detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)


  #filter out matched with low dij value
  matches = []
  for m in matched_indices:
    if(dij_matrix[m[0], m[1]]<dij_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  # print("---stage 1---")
  # print("matches:",matches)
  # print("unmatched_trackers:",unmatched_trackers)
  # print("unmatched_detections:",unmatched_detections)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
  
# 2nd Stage
def associate_detections_secondstage(lowscore_dets,trks,unmatched_trackers_prev,diou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes) for low-score with DIOU_2 method
  lowscore_dets; trks; unmatched_trks: indices of trks that is not matched
  Returns 2 lists of matched and unmatched_trackers
  """
  # if(len(trks)==0): # what if there is no low-score detections?
  #   return np.empty((0,2),dtype=int), np.arange(len(lowscore_dets)), np.empty((0,5),dtype=int)

  if (len(trks)==0) or (len(unmatched_trackers_prev)==0):
    return np.empty((0,2),dtype=int), unmatched_trackers_prev

  # extract the value of unmatched_trackers_prev from trks to create unmatched_trks
  unmatched_trks = []
  for index in unmatched_trackers_prev:
    unmatched_trks.append(trks[index])
  unmatched_trks = np.array(unmatched_trks)
  # print("unmatched_trks:",unmatched_trks)

  # matrix calc
  iou_matrix = iou_batch(lowscore_dets,unmatched_trks)
  dij_matrix = dij_distance(lowscore_dets,unmatched_trks)
  diou_matrix = DIOU_2(iou_matrix, dij_matrix)

  if min(diou_matrix.shape) > 0:
    a = (diou_matrix > diou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-diou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(lowscore_dets):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(unmatched_trks):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low DIOU
  matches = []
  for m in matched_indices:
    if(diou_matrix[m[0], m[1]]<diou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  # print("---stage 2---")
  # print("matches:",matches)
  # print("unmatched_trackers:",unmatched_trackers)
  # print("unmatched_detections:",unmatched_detections)

  #replace with elements with true value
  # Replace the second element in each row of `matches` using the second element as the index for unmatched_trks
  for i in range(matches.shape[0]):
      # Use the second element in the row as the index for unmatched_trks
      index = matches[i, 1]
      matches[i, 1] = unmatched_trackers_prev[index]
  # Replace the element in unmatched_trks with the corresponding value from unmatched_trks_prev
  for i,value in enumerate(unmatched_trackers):
    unmatched_trackers[i] = unmatched_trackers_prev[value]

  # print("---stage 2---")
  # print("matches:",matches)
  # print("unmatched_trackers:",unmatched_trackers)
  # print("unmatched_detections:",unmatched_detections)

  return matches, np.array(unmatched_trackers)
  
########################### Third Stage #################################
def associate_detections_thirdstage(dets,trks,unmatched_trackers_prev,unmatched_detections_prev,diou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes) for low-score with DIOU_2 method with higher threshold

  Returns 3 lists of matched and unmatched_trackers
  """
  # if(len(trks)==0): # what if there is no low-score detections?
  #   return np.empty((0,2),dtype=int), unmatched_detections_prev, unmatched_trackers_prev

  if (len(dets)==0) or (len(trks)==0) or (len(unmatched_trackers_prev)==0) or (len(unmatched_detections_prev)==0):
    return np.empty((0,2),dtype=int), unmatched_detections_prev, unmatched_trackers_prev

  # extract the value of dets[unmatched_detections_prev] to create unmatched_trks
  unmatched_dets = []
  for index in unmatched_detections_prev:
    unmatched_dets.append(dets[index])
  unmatched_dets = np.array(unmatched_dets)
  # print("unmatched_dets:",unmatched_dets)

  # extract the value of unmatched_trackers_prev from trks to create unmatched_trks
  unmatched_trks = []
  for index in unmatched_trackers_prev:
    unmatched_trks.append(trks[index])
  unmatched_trks = np.array(unmatched_trks)
  # print("unmatched_trks:",unmatched_trks)

  # matrix calc
  iou_matrix = iou_batch(unmatched_dets,unmatched_trks)
  dij_matrix = dij_distance(unmatched_dets,unmatched_trks)
  diou_matrix = DIOU_2(iou_matrix, dij_matrix)

  if min(diou_matrix.shape) > 0:
    a = (diou_matrix > diou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-diou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(unmatched_dets):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(unmatched_trks):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low DIOU
  matches = []
  for m in matched_indices:
    if(diou_matrix[m[0], m[1]]<diou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  # print("---stage 3---")
  # print("matches:",matches)
  # print("unmatched_trackers:",unmatched_trackers)
  # print("unmatched_detections:",unmatched_detections)

  # replace with elements with true value for detections
  # Replace the second element in each row of `matches` using the second element as the index for unmatched_dets
  for i in range(matches.shape[0]):
      # Use the second element in the row as the index for unmatched_trks
      index = matches[i, 0]
      matches[i, 0] = unmatched_detections_prev[index]
  # Replace the element in unmatched_trks with the corresponding value from unmatched_trks_prev
  for i,value in enumerate(unmatched_detections):
    unmatched_detections[i] = unmatched_detections_prev[value]

  # replace with elements with true value for trackers
  # Replace the second element in each row of `matches` using the second element as the index for unmatched_trks
  for i in range(matches.shape[0]):
      # Use the second element in the row as the index for unmatched_trks
      index = matches[i, 1]
      matches[i, 1] = unmatched_trackers_prev[index]
  # Replace the element in unmatched_trks with the corresponding value from unmatched_trks_prev
  for i,value in enumerate(unmatched_trackers):
    unmatched_trackers[i] = unmatched_trackers_prev[value]

  # print("---stage 3---")
  # print("matches:",matches)
  # print("unmatched_trackers:",unmatched_trackers)
  # print("unmatched_detections:",unmatched_detections)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
  
 #################################  The SORT itself ############################################

class Sort(object):
  def __init__(self, max_age=1, min_hits=1, iou_threshold=0.3, dij_threshold=0.9):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.diou_threshold = iou_threshold
    self.dij_theshold = dij_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    # print(f"self.trackers:{self.trackers}")
    # print(f"len:{len(self.trackers)}")
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    # check the prediction
    # for trk in reversed(self.trackers):
    #   d = trk.get_state()[0]
    #   # print("d:",d)
    #   x_min, y_min, x_max, y_max = d
    #   x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    #   # Draw the bounding box
    #   # Define the color for the bounding box (e.g., green) and thickness
    #   color = (255, 0, 255)  # Green color in BGR format
    #   thickness = 2  # Thickness of the bounding box
    #   cv2.rectangle(im0, (x_min, y_min), (x_max, y_max), color, thickness)
    #print("---")

    #print(f"dets: {dets}")
    #print(f"trks: {trks}")
    #print("---")
    # divide high-score and low-score
    highscore_dets, lowscore_dets = divide_dets_byscore(dets)
    #print(f"highscore_dets:{highscore_dets}")
    #print(f"lowscore_dets:{lowscore_dets}")

    # matching: following the pipeline from the paper
    # first stage
    matched1, unmatched_dets1, unmatched_trks = associate_detections_firststage(highscore_dets,trks,self.dij_theshold)
    # second stage
    matched2, unmatched_trks = associate_detections_secondstage(lowscore_dets,trks,unmatched_trks,self.diou_threshold)
    # third stage
    matched3, unmatched_dets3, unmatched_trks = associate_detections_thirdstage(highscore_dets,trks,unmatched_trks,unmatched_dets1,self.diou_threshold)
    # print("---")
    # update matched trackers with assigned detections
    dets = np.concatenate((highscore_dets,lowscore_dets),axis=0)
    #print(f"dets:{dets}")
    matched2[:,0]+=len(highscore_dets)
    # print(f"matched1:{matched1}")
    # print(f"matched2:{matched2}")
    # print(f"matched3:{matched3}")
    matched = np.concatenate((matched1, matched3, matched2), axis=0)
    #print(f"matched:{matched}")
    unmatched_dets = unmatched_dets3

    # print("---")
    # print("matched:")
    # print(matched,matched.shape)
    # print("unmatched dets:")
    # print(unmatched_dets, unmatched_dets.shape)
    # print("unmatched trks:")
    # print(unmatched_trks, unmatched_trks.shape)

    # for trk in self.trackers:


    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])
    # print(f"self.trackers:{self.trackers}")
    # print(f"len:{len(self.trackers)}")
    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    # print(f"self.trackers:{self.trackers}")
    # print(f"len:{len(self.trackers)}")
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      # print(trk)
      d = trk.get_state()[0]
      # print("d:",d)
      if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
      i -= 1
      # remove dead tracklet
      if(trk.time_since_update > self.max_age):
        self.trackers.pop(i)
    # print(f"self.trackers:{self.trackers}")
    # print(f"len:{len(self.trackers)}")
    i = len(self.trackers)
    # print(f"ret:{ret}")
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))


##### COUNTING
def check_cross(boundary, start_centroid, current_centroid):
    x0_0, y0_0 = boundary[0]
    x1_0, y1_0 = boundary[1]
    x0_1, y0_1 = start_centroid
    x1_1, y1_1 = current_centroid
    
    dx0 = x1_0 - x0_0
    dy0 = y1_0 - y0_0
    dx1 = x1_1 - x0_1
    dy1 = y1_1 - y0_1
    
    denominator = dx1 * dy0 - dy1 * dx0
    if denominator == 0:
        return False
      
    t = ((x0_0 - x0_1) * dy1 - (y0_0 - y0_1) * dx1) / denominator
    u = ((x0_0 - x0_1) * dy0 - (y0_0 - y0_1) * dx0) / denominator
    if 0 <= t <= 1 and 0 <= u <= 1:
      print("yes")
      return x1_1 > x0_1  
    return False

#########

def setup_model(model_path):
    platform = 'rknn'
    from py_utils.rknn_executor import RKNN_model_container 
    model = RKNN_model_container(model_path, 'rk3588')
    return model, platform

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores
    
def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

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

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    if position.shape[1]==4:
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    else:
        position = dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
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

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return np.array([]), np.array([]), np.array([])

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

######################### MAIN CODE ############################

# Video Path
#vid_path = "./videotest/videobaru.mp4"
#vid_path = "/home/orangepi/fishcounter/rknn_model_zoo/examples/yolov5/model/lele.png"

#NULIS CSV
# metrics_logger = MetricsLogger('/home/orangepi/fishcounter/program_1/detection_metrics.csv')

 # '0' for the default camera
#cap = cv2.VideoCapture(vid_path)
assert cap.isOpened(), "Error accessing camera"

# Get video properties for saving
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 30)
#w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
w=640
h=640
# Calculate the x-coordinate for the vertical line (70% of the frame width)
line_x_pos = int(0.645 * 640)
line_x_pos_right = int(0.78 * 640)
line_x_pos_left = int(0.58 * 640)

# Setup video writer to save output
output_path = "./output_counting.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
video_writer = cv2.VideoWriter(output_path,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               30.0,
                               (640, 640))


############### INITIALIZE MODEL ###############
model_path = "./yolov6.rknn"
model, platform = setup_model(model_path)


############### INITIALIZE TRACKER ###############
mot_tracker = Sort()
track_bbs_ids = np.empty((0,5))
previous_track_2 = np.empty((0,5))
previous_track_3 = np.empty((0,5))

############### INITIALIZE PREVIOUS TRACKING DATA ###############
previous_track_bbs_ids = np.empty((0,5))  # This will hold tracking data from the previous frame



############### INITIALIZE FISH COUNTER ###############
ID_already_counted = []
fish_count=0

ID_already_counted_2 = []
fish_count_2=0

ID_already_counted_3 = []
fish_count_3=0
boundary1 = [(int(0.58*640),0),(int(0.58*640),640)]
boundary2 = [(int(0.68*640),0),(int(0.68*640),640)]
boundary3 = [(int(0.78*640),0),(int(0.78*640),640)]

co_helper = COCO_test_helper(enable_letter_box=True)

############### INITIALIZE FRAME COUNTER ###############
counter = 0

############### START ITERATING THROUGH VIDEO'S FRAMES ###############
while cap.isOpened():
  success, frame = cap.read()
  if not success:
      break  # Exit loop if no more frames

  # Add frame counter on top of the video
  counter+=1
  print(f"frame ke-{counter}")


  ############### PRE PROCESSING ###############
  preprocess_time_1 = time.time()
  pad_color = (0, 0, 0)
  img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  preprocess_time_2 = time.time()
  preprocess_time_total = preprocess_time_2 - preprocess_time_1
  print(f"Time taken to process the preprocessing: {preprocess_time_total:.6f} seconds")

  ############### DETECTION ###############
  det_time_1 = time.time()
  results = model.run([img])
  det_time_2 = time.time()
  det_time_total = det_time_2 - det_time_1
  print(f"Time taken to process the detection: {det_time_total:.6f} seconds")

  ############### POST PROCESSING ###############
  #boxes = np.empty((0,4))
  postprocess_time_1 = time.time()
  boxes, classes, scores = post_process(results)
  postprocess_time_2 = time.time()
  postprocess_time_total = postprocess_time_2 - postprocess_time_1
  print(f"Time taken to process the postprocessing: {postprocess_time_total:.6f} seconds")

  
  #print("ini boxes")
  #print(boxes.any())
  #print("ini scores")
  #print(scores)
  #print("ini classes")
  #print(classes)

  ############ DRAW DETECTIONS BOX ############
  if boxes.any() == False:
     continue
  else:
      for box in boxes:
	    #print("box:",box)
          x_min, y_min, x_max, y_max = box
          x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
          color = (0, 255, 0)  # Green color in BGR format
          thickness = 2  # Thickness of the bounding box
          cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

  # print("previous_track_bbs_ids:")
  # print(previous_track_bbs_ids)

  ############### TRACKER ###############
  track_time_1 = time.time()
  detections = np.empty((0,5))
  detections = np.column_stack((boxes, scores))
  #print("detections")
  #print(detections)
  # Update tracker with current frame's detections
  if np.any(detections): # if array detections is not empty
    print("a")
    track_bbs_ids = mot_tracker.update(np.array(detections))
  else: # if array detections is empty
    print("b")
    track_bbs_ids = mot_tracker.update(np.empty((0, 5)))
  # End timing
  track_time_2 = time.time()
  # Compute the time taken
  track_time_total = track_time_2 - track_time_1
  print(f"Time taken to process the tracking: {track_time_total:.6f} seconds")
  #print("tracks")
  #print(track_bbs_ids)

  ############### DRAW TRACKED FISH AND THE ID ###############
  ############### AND FISH COUNTER ###############
  count_time_1 = time.time()
  for track in track_bbs_ids:
    x_min, y_min, x_max, y_max, track_id = track.astype(int)
    color = (255, 0, 0)
    thickness = 2
    # Draw the bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
    # Display the track ID
    cv2.putText(frame, f'ID: {track_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    ############### FISH COUNTER ###############
    ID = int(track[4])
    right_x = int(track[2])
    left_x = int(track[0])
    #if (ID not in ID_already_counted) and (right_x>line_x_pos):
    if (ID not in ID_already_counted) and (right_x>line_x_pos):
      fish_count+=1
      ID_already_counted.append(ID)
        
    if (ID not in ID_already_counted_2) and (right_x>line_x_pos_left) and (left_x<line_x_pos_right):
      fish_count_2+=1
      ID_already_counted_2.append(ID)
    track = convert_bbox_to_xywh(track)
    current_x, current_y, _, _, current_id = track[0]
    cv2.circle(frame, (int(current_x), int(current_y)), 5, (0, 255, 255), -1)
    previous = previous_track_3[previous_track_3[:,4] == current_id]
    if len(previous) == 0:
       previous = previous_track_2[previous_track_2[:,4] == current_id]
    print(previous)
    if len(previous) > 0:
      previous = convert_bbox_to_xywh(previous[0])
      prev_x, prev_y, _, _, _ = previous[0]
      if (current_id not in ID_already_counted_3) and check_cross(boundary1, (prev_x, prev_y), (current_x,current_y)):
        fish_count_3+=1
        ID_already_counted_3.append(ID)
      if (current_id not in ID_already_counted_3) and check_cross(boundary2, (prev_x, prev_y), (current_x,current_y)):
        fish_count_3+=1
        ID_already_counted_3.append(ID)
      if (current_id not in ID_already_counted_3) and check_cross(boundary3, (prev_x, prev_y), (current_x,current_y)):
        fish_count_3+=1
        ID_already_counted_3.append(ID)
		
  ############### DISPLAY NUMBER OF FISH COUNTED ###############
  print(f"fish_count:{fish_count}")
  print(f"fish_count:{fish_count_2}")
  print(f"fish_count:{fish_count_3}")
  count_time_2 = time.time()
  count_time_total = count_time_2 - count_time_1
  print(f"Time taken to process the counting: {count_time_total:.6f} seconds")
  cv2.putText(frame, f'fish_count: {fish_count}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(frame, f'fish_count2: {fish_count_2}', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(frame, f'fish_count3: {fish_count_3}', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  previous_track_3 = previous_track_2
  previous_track_2 = track_bbs_ids
  print(previous_track_3)
  print(previous_track_2)

  # Draw the vertical line on the frame
  cv2.line(frame, (line_x_pos, 0), (line_x_pos, h), (0, 255, 0), 5)  # Green color (0, 255, 0) and 5 px thickness
  cv2.line(frame, (line_x_pos_left, 0), (line_x_pos_left, h), (0, 0, 255), 2)  # Green color (0, 255, 0) and 5 px thickness
  cv2.line(frame, (line_x_pos_right, 0), (line_x_pos_right, h), (0, 0, 255), 2)  # Green color (0, 255, 0) and 5 px thickness
  
  print(boundary1[0])
  
  cv2.line(frame, boundary1[0], boundary1[1], (0, 255, 0), 2)  # Green color (0, 255, 0) and 5 px thickness
  cv2.line(frame, boundary2[0], boundary2[1], (0, 255, 0), 2)  # Green color (0, 255, 0) and 5 px thickness
  cv2.line(frame, boundary3[0], boundary3[1], (0, 255, 0), 2)  # Green color (0, 255, 0) and 5 px thickness
  
  cv2.putText(frame, f'Frame: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  #NULIS CSV
  # metrics_logger.log_metrics(
  #    frame_number=counter,
  #    preprocess_time=preprocess_time_total,
  #    detection_time=det_time_total,
  #    postprocess_time=postprocess_time_total,
  #    tracking_time=track_time_total,
  #    counting_time=count_time_total,
  #    fish_count=fish_count)

  # Write frame to output video
  video_writer.write(frame)

  # Optionally display the frame in real-time
  cv2.imshow("Live Object Counting", frame)
  if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
      break

cap.release()
video_writer.release()
cv2.destroyAllWindows()