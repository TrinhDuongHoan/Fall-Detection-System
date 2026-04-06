import os
import sys
import cv2
import torch
import shutil
import numpy as np
from pathlib import Path
from collections import deque
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import imageio

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))

from fall_detection.utils.config import get_config
from fall_detection.data.feature_extractor import PoseFeatureExtractor
from fall_detection.models.lstm import FallLSTM
from fall_detection.models.hybrid import FallHybrid
from fall_detection.models.bi_lstm import FallBiLSTM

app = FastAPI(title="Fall Detection Guardian API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server is powering up on device: {device}")

cfg = None
extractor = None
model = None

# Initialize AI on app startup
@app.on_event("startup")
def load_ai_models():
    global cfg, extractor, model
    config_path = SRC_DIR / "configs" / "default.yaml"
    cfg = get_config(str(config_path))
    
    extractor = PoseFeatureExtractor(cfg)
    
    model_paths = [
        (SRC_DIR / "output" / "best_fall_hybrid.pth", FallHybrid),
        (SRC_DIR / "output" / "best_fall_bi_lstm.pth", FallBiLSTM),
    ]
    
    model_loaded = False
    for path, ModelClass in model_paths:
        if path.exists():
            print(f"Loading weights from {path.name}...")
            model = ModelClass(seq_len=cfg.MODEL.SEQ_LEN, feature_dim=cfg.MODEL.FEATURE_DIM, num_classes=cfg.MODEL.NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            model_loaded = True
            break
            
    if not model_loaded:
        print("WARNING: No trained weights found. Initializing an empty FallLSTM just so the UI doesn't crash.")
        model = FallLSTM(seq_len=cfg.MODEL.SEQ_LEN, feature_dim=cfg.MODEL.FEATURE_DIM, num_classes=cfg.MODEL.NUM_CLASSES).to(device)
        model.eval()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

def generate_video_stream(filename: str):
    file_path = UPLOAD_DIR / filename
    
    class_map = {0: "Normal", 1: "Falling", 2: "Lying"}
    color_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)}
    sequence_buffer = deque(maxlen=cfg.MODEL.SEQ_LEN)
    
    current_label_id = 0
    current_prob = 0.99
    
    try:
        reader = imageio.get_reader(str(file_path), 'ffmpeg')
    except Exception as e:
        print(f"Error opening video: {e}")
        return

    for frame_rgb in reader:
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_drawn = frame.copy()
        
        yolo_results = extractor.yolo_model.predict(frame, verbose=False, conf=cfg.MODEL.MIN_BBOX_CONF)
        feat = None
        
        if len(yolo_results) > 0:
            best_person = None
            best_conf = 0.0
            
            for box in yolo_results[0].boxes:
                if int(box.cls[0].item()) == 0:
                    conf = box.conf[0].item()
                    if conf > best_conf:
                        best_conf = conf
                        best_person = box
                        
            if best_person is not None:
                x_box, y_box, w_box, h_box = best_person.xyxy[0].cpu().numpy().astype(int)
                roi = extractor.safe_crop(frame, (x_box, y_box, w_box, h_box))
                feat = extractor.extract_from_roi(roi)
                
                if feat is not None:
                    sequence_buffer.append(feat)
                else:
                    sequence_buffer.append(np.zeros((cfg.MODEL.FEATURE_DIM,), dtype=np.float32))
                    
                if len(sequence_buffer) == cfg.MODEL.SEQ_LEN:
                    seq_np = np.stack(sequence_buffer, axis=0)
                    seq_t = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(seq_t)
                        probs = torch.softmax(logits, dim=1)[0]
                        current_label_id = torch.argmax(probs).item()
                        
                        if best_conf < 0.6: current_label_id = 0
                        
                # Draw UI
                status_name = class_map.get(current_label_id, 'Normal')
                status_color = color_map.get(current_label_id, (0, 255, 0))
                
                ui_text = f"Person {best_conf:.2f} | {status_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Calculate text size and ensure rectangle stays within bounds
                (text_w, text_h), _ = cv2.getTextSize(ui_text, font, 1.0, 3)
                y_bg_top = max(0, y_box - text_h - 15)
                
                cv2.rectangle(frame_drawn, (x_box, y_box), (w_box, h_box), status_color, 4)
                cv2.rectangle(frame_drawn, (x_box, y_bg_top), (x_box + text_w + 10, y_box), status_color, -1)
                cv2.putText(frame_drawn, ui_text, (x_box + 5, y_box - 8), font, 1.0, (255, 255, 255), 3)

        # Convert to JPEG bytes to stream over HTTP
        ret, buffer = cv2.imencode('.jpg', frame_drawn)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    try:
        reader.close()
    except:
        pass

@app.get("/video_feed/{filename}")
async def video_feed(filename: str):
    return StreamingResponse(generate_video_stream(filename), media_type="multipart/x-mixed-replace; boundary=frame")
