import torch
import numpy as np
from fall_detection.utils.metrics import print_evaluation_report
from tqdm.notebook import tqdm

def evaluate_model(model, test_loader, device, class_names=None):
    
    if class_names is None:
        class_names = {0: "Normal", 1: "Falling", 2: "Lying"}
        
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc="Evaluation"):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    print_evaluation_report(all_labels, all_preds, class_names)
