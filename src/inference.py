import cv2
import torch
import argparse
import numpy as np
from collections import deque
from pathlib import Path

from fall_detection.utils.config import get_config
from fall_detection.models.lstm import FallLSTM
from fall_detection.data.feature_extractor import PoseFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Inference for Fall Detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4/.avi) or '0' for webcam")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video result")
    args = parser.parse_args()

    cfg = get_config("configs/default.yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = Path(cfg.OUTPUT_DIR) / "best_fall_lstm.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")

    model = FallLSTM(
        seq_len=cfg.MODEL.SEQ_LEN,
        feature_dim=cfg.MODEL.FEATURE_DIM,
        num_classes=cfg.MODEL.NUM_CLASSES
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    extractor = PoseFeatureExtractor(cfg)

    class_map = {0: "Normal", 1: "Falling", 2: "Lying"}
    color_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)}

    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    if not cap.isOpened():
        print(f"Failed to open video: {args.video}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    sequence_buffer = deque(maxlen=cfg.MODEL.SEQ_LEN)
    current_label_id = 0
    current_prob = 0.0

    print("Running inference... Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yolo_results = extractor.yolo_model.predict(frame, verbose=False, conf=cfg.MODEL.MIN_BBOX_CONF)
        feat = None
        best_box = None

        if len(yolo_results) > 0:
            best_box = extractor.choose_best_person_box(yolo_results[0])
            if best_box is not None:
                roi = extractor.safe_crop(frame, best_box)
                feat = extractor.extract_from_roi(roi)
                x1, y1, x2, y2 = best_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[current_label_id], 2)

        if feat is not None:
            sequence_buffer.append(feat)
        else:
            sequence_buffer.append(np.zeros((cfg.MODEL.FEATURE_DIM,), dtype=np.float32))

        if len(sequence_buffer) == cfg.MODEL.SEQ_LEN:
            seq_np = np.stack(sequence_buffer, axis=0)
            seq_t = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(device) # (1, seq_len, dim)
            with torch.no_grad():
                logits = model(seq_t)
                probs = torch.softmax(logits, dim=1)[0]
                current_label_id = torch.argmax(probs).item()
                current_prob = probs[current_label_id].item()

        label_text = f"Status: {class_map.get(current_label_id, 'Unknown')} ({current_prob*100:.1f}%)"
        cv2.putText(frame, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_map.get(current_label_id, (255,255,255)), 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Inference complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
