from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def print_evaluation_report(y_true, y_pred, class_names_dict):
    """
    Prints a formatted evaluation report including accuracy, confusion matrix, 
    and detailed classification report per class.
    
    Args:
        y_true (list or np.array): True labels
        y_pred (list or np.array): Predicted labels
        class_names_dict (dict): Dictionary mapping class indices to names (e.g., {0: 'Normal', 1: 'Falling', 2: 'Lying'})
    """
    acc = accuracy_score(y_true, y_pred)
    names = [class_names_dict.get(i, str(i)) for i in range(len(class_names_dict))]
    
    print("=" * 50)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print("=" * 50)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=names, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return acc, cm
