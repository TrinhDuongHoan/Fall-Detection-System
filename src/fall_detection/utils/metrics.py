from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score    
import numpy as np

def print_evaluation_report(y_true, y_pred, class_names_dict):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    names = [class_names_dict.get(i, str(i)) for i in range(len(class_names_dict))]
    
    print("=" * 50)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Overall Precision: {precision * 100:.2f}%")
    print(f"Overall Recall: {recall * 100:.2f}%")
    print(f"Overall F1 Score: {f1 * 100:.2f}%")
    print("=" * 50)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=names, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return acc, cm
