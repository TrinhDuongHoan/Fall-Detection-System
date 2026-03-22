import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

class PoseFeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.yolo_model = YOLO(cfg.MODEL.YOLO_MODEL_NAME)
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def choose_best_person_box(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            return None
            
        best_box, best_conf = None, -1.0
        
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if cls_id == self.cfg.MODEL.PERSON_CLASS_ID and conf >= self.cfg.MODEL.MIN_BBOX_CONF:
                if conf > best_conf:
                    best_box = box.xyxy[0].cpu().numpy().astype(int)
                    best_conf = conf
        return best_box
        
    def safe_crop(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x2 <= x1 or y2 <= y1:
            return None
        return np.ascontiguousarray(frame[y1:y2, x1:x2])

    def extract_from_roi(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return None
            
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(roi_rgb)
        
        if results.pose_landmarks is None:
            return None
            
        landmarks = results.pose_landmarks.landmark
        if len(landmarks) != self.cfg.MODEL.NUM_KEYPOINTS:
            return None
            
        feat = []
        valid_count = 0
        
        for lm in landmarks:
            x, y = float(lm.x), float(lm.y)
            vis = getattr(lm, "visibility", 1.0)
            
            if np.isfinite(x) and np.isfinite(y) and vis >= self.cfg.MODEL.VISIBILITY_THRESHOLD:
                feat.extend([x, y])
                valid_count += 1
            else:
                feat.extend([0.0, 0.0])
                
        if valid_count < self.cfg.MODEL.MIN_VALID_KEYPOINTS:
            return None
            
        feat = np.array(feat, dtype=np.float32)
        return feat if feat.shape[0] == self.cfg.MODEL.FEATURE_DIM else None

def frame_label_from_intervals(frame_idx, fall_start=None, fall_end=None, lying_start=None):
    if fall_start is None or fall_end is None:
        return 0
    if frame_idx < fall_start:
        return 0
    if fall_start <= frame_idx <= fall_end:
        return 1
    if lying_start is None:
        return 2 if frame_idx > fall_end else 0
    if frame_idx >= lying_start:
        return 2
    return 2

def build_sequences_from_valid_frames(features, labels, seq_len=30, stride=1):
    X, y = [], []
    if len(features) < seq_len:
        return np.empty((0, seq_len, len(features[0]) if features else 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
    for start in range(0, len(features) - seq_len + 1, stride):
        end = start + seq_len
        X.append(np.stack(features[start:end], axis=0))
        y.append(labels[end - 1])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
