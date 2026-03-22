import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from fall_detection.utils.config import get_config
from fall_detection.models.transformer import FallTransformer
from fall_detection.data.dataset import FallSequenceDataset
from fall_detection.utils.metrics import print_evaluation_report

def evaluate_model():
    cfg = get_config("configs/default.yaml")
    output_dir = Path(cfg.OUTPUT_DIR)
    
    X_test_path = output_dir / "X_test.npy"
    y_test_path = output_dir / "y_test.npy"
    model_path = output_dir / "best_fall_transformer.pth"

    if not (X_test_path.exists() and y_test_path.exists()):
        print("Test split not found. Please run train.py first to generate the test split.")
        return

    if not model_path.exists():
        print(f"Trained model checkpoint not found at {model_path}")
        return

    print("Loading test dataset...")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    test_loader = DataLoader(FallSequenceDataset(X_test, y_test), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FallTransformer(
        seq_len=cfg.MODEL.SEQ_LEN,
        feature_dim=cfg.MODEL.FEATURE_DIM,
        num_classes=cfg.MODEL.NUM_CLASSES,
        d_model=cfg.MODEL.D_MODEL,
        num_heads=cfg.MODEL.NUM_HEADS,
        ff_dim=cfg.MODEL.FF_DIM,
        num_layers=cfg.MODEL.NUM_LAYERS,
        dropout=cfg.MODEL.DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    class_names = getattr(cfg, "CLASS_NAMES", {0: "Normal", 1: "Falling", 2: "Lying"})
    # Convert config recursive dict-like object to real dict if needed
    if not isinstance(class_names, dict):
        class_names = {k: v for k, v in class_names.__dict__.items() if not k.startswith('_')}

    print_evaluation_report(all_labels, all_preds, class_names)

if __name__ == "__main__":
    evaluate_model()
