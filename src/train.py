import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from fall_detection.utils.config import get_config
from fall_detection.models.lstm import FallLSTM
from fall_detection.data.dataset import FallSequenceDataset

def train_model():
    cfg = get_config("configs/default.yaml")
    output_dir = Path(cfg.OUTPUT_DIR)
    
    X_path = output_dir / "X_sequences.npy"
    y_path = output_dir / "y_sequences.npy"
    
    if not (X_path.exists() and y_path.exists()):
        print("Data sequences not found. Run prepare_data.py first.")
        return

    print("Loading extracted sequences...")
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=cfg.SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=cfg.SEED, stratify=y_temp
    )
    
    print("Train label distribution:", dict(Counter(y_train.tolist())))
    print("Val label distribution  :", dict(Counter(y_val.tolist())))

    train_loader = DataLoader(FallSequenceDataset(X_train, y_train), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FallSequenceDataset(X_val, y_val), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FallLSTM(
        seq_len=cfg.MODEL.SEQ_LEN,
        feature_dim=cfg.MODEL.FEATURE_DIM,
        num_classes=cfg.MODEL.NUM_CLASSES
    ).to(device)

    # Compute class weights to handle imbalanced dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # Chặn không để thuật toán phạt quá đà gây ra tâm lý đoán nhầm giả
    class_weights = np.clip(class_weights, 0.5, 3.5)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Applying bounded class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    epochs = cfg.TRAIN.EPOCHS
    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0

    best_model_path = output_dir / "best_fall_lstm.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]")
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Val]")
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        

        print(f"Epoch {epoch+1:02d}/{epochs} - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] val_loss improved to {val_loss:.4f}, model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
    # Save the split as well so evaluate.py can use the exact same test sequence
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    print("Training finished.")

if __name__ == "__main__":
    train_model()
