import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from fall_detection.utils.config import get_config
from fall_detection.data.lei2_parser import build_le2i_annotations
from fall_detection.data.feature_extractor import (
    PoseFeatureExtractor, 
    frame_label_from_intervals, 
    build_sequences_from_valid_frames
)

def get_total_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def validate_row(row):
    fs, fe, ls, total = row["fall_start"], row["fall_end"], row["lying_start"], row["total_frames"]
    if pd.isna(fs) or pd.isna(fe): return "normal_only"
    if total <= 0: return "bad_video"
    if fs < 0 or fe < 0: return "bad_annotation"
    if fs >= total or fe >= total: return "out_of_range"
    if fe < fs: return "reversed"
    return "ok"

def process_video(video_info, extractor, cfg, verbose=False):
    video_path = video_info["video_path"]
    fall_start = video_info.get("fall_start")
    fall_end = video_info.get("fall_end")
    lying_start = video_info.get("lying_start")

    if pd.isna(fall_start): fall_start = None
    if pd.isna(fall_end): fall_end = None
    if pd.isna(lying_start): lying_start = None

    if not os.path.exists(video_path):
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_features, valid_labels = [], []
    pbar = tqdm(total=total_frames, desc=Path(video_path).name, disable=not verbose)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_label = frame_label_from_intervals(
            frame_idx=frame_idx,
            fall_start=fall_start,
            fall_end=fall_end,
            lying_start=lying_start
        )

        try:
            yolo_results = extractor.yolo_model.predict(
                source=frame, verbose=False, conf=cfg.MODEL.MIN_BBOX_CONF, device="cpu"
            )
        except Exception:
            frame_idx += 1
            pbar.update(1)
            continue

        if len(yolo_results) > 0:
            best_box = extractor.choose_best_person_box(yolo_results[0])
            if best_box is not None:
                roi = extractor.safe_crop(frame, best_box)
                feat = extractor.extract_from_roi(roi)
                if feat is not None:
                    valid_features.append(feat)
                    valid_labels.append(frame_label)

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    return build_sequences_from_valid_frames(
        valid_features, valid_labels, 
        seq_len=cfg.MODEL.SEQ_LEN, stride=cfg.MODEL.WINDOW_STRIDE
    )

if __name__ == "__main__":
    cfg = get_config("configs/default.yaml")
    DATASET_DIR = Path(cfg.DATASET_DIR)
    OUTPUT_DIR = Path(cfg.OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading LEI2 dataset from: {DATASET_DIR}")
    
    if not DATASET_DIR.exists():
        print("Data directory not found. Please ensure kaggle API downloaded the dataset or adjust config.")
        exit(0)
        
    df = build_le2i_annotations(DATASET_DIR)
    df["total_frames"] = df["video_path"].apply(get_total_frames)
    df["status"] = df.apply(validate_row, axis=1)

    usable_df = df[df["status"].isin(["ok", "normal_only"])]
    
    extractor = PoseFeatureExtractor(cfg)
    all_X, all_y = [], []
    
    for _, row in usable_df.iterrows():
        X_vid, y_vid = process_video(row, extractor, cfg, verbose=True)
        if X_vid is not None and len(X_vid) > 0:
            all_X.append(X_vid)
            all_y.append(y_vid)
            
    if len(all_X) > 0:
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        np.save(OUTPUT_DIR / "X_sequences.npy", X)
        np.save(OUTPUT_DIR / "y_sequences.npy", y)
        print(f"Dataset extracted: {X.shape}, {y.shape}")
    else:
        print("No valid sequences generated.")
