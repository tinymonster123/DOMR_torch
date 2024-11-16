import torch
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

def evaluate_model(classifier, X_test, y_test, device):
    classifier.eval()
    with torch.no_grad():
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.to(device)
        elif isinstance(X_test, pd.DataFrame) or isinstance(X_test, pd.Series):
            X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        elif isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        else:
            raise TypeError(f"Unsupported type for X_test: {type(X_test)}")
        
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.to(device)
        elif isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
            y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)
        elif isinstance(y_test, np.ndarray):
            y_test = torch.tensor(y_test, dtype=torch.long).to(device)
        else:
            raise TypeError(f"Unsupported type for y_test: {type(y_test)}")

        outputs = classifier(X_test)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        y_true = y_test.cpu().numpy()
        report = classification_report(y_true, preds, zero_division=0)
        print("Evaluation Report:\n", report)
