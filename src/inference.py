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
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.mp4")
    args = parser.parse_args()

    cfg = get_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(cfg.OUTPUT_DIR) / "best_fall_lstm.pth"
    model = FallLSTM(cfg.MODEL.SEQ_LEN, cfg.MODEL.FEATURE_DIM, cfg.MODEL.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    extractor = PoseFeatureExtractor(cfg)
    class_map = {0: "Normal", 1: "Falling", 2: "Lying"}
    color_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)}

    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(5)) or 30
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    sequence_buffer = deque(maxlen=cfg.MODEL.SEQ_LEN)
    curr_id, curr_prob = 0, 0.0

    print("Đang chạy... Hãy kiểm tra cửa sổ hiển thị.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # TẠO BẢN SAO SẠCH - ĐÂY LÀ FRAME DUY NHẤT DÙNG ĐỂ HIỂN THỊ
        display_frame = frame.copy() 

        # GỌI PREDICT VỚI CÁC THAM SỐ KHÓA VẼ
        # show=False, save=False, visualize=False để ép YOLO không được chạm vào frame
        results = extractor.yolo_model.predict(
            frame, 
            verbose=False, 
            conf=cfg.MODEL.MIN_BBOX_CONF,
            show=False, 
            save=False
        )
        
        feat = None
        best_box = None
        if len(results) > 0:
            best_box = extractor.choose_best_person_box(results[0])
            if best_box is not None:
                # Trích xuất đặc trưng từ frame gốc (kệ nó nếu bị YOLO vẽ lên)
                roi = extractor.safe_crop(frame, best_box)
                feat = extractor.extract_from_roi(roi)

        # Xử lý Sequence cho LSTM
        if feat is not None:
            sequence_buffer.append(feat)
        else:
            sequence_buffer.append(np.zeros((cfg.MODEL.FEATURE_DIM,), dtype=np.float32))

        if len(sequence_buffer) == cfg.MODEL.SEQ_LEN:
            seq_t = torch.tensor(np.stack(sequence_buffer), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(seq_t), dim=1)[0]
                curr_id = torch.argmax(probs).item()
                curr_prob = probs[curr_id].item()

        # --- TỰ VẼ LẠI MỌI THỨ LÊN DISPLAY_FRAME (FRAME SẠCH) ---
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            color = color_map.get(curr_id, (0, 255, 0))
            
            # Vẽ Box cực mảnh
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)

            #