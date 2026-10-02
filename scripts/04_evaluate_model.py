import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def evaluate_model():
    # Load class map
    with open(os.path.join(PROCESSED_DATA_DIR, 'class_map.json'), 'r') as f:
        class_map = json.load(f)
    num_classes = len(class_map)
    # Reverse map for labels
    idx_to_class = {v: k for k, v in class_map.items()}

    # Data transformation for test (no augmentation)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DATA_DIR, 'test'), data_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    model_path = os.path.join(MODELS_DIR, 'species_classifier.pth')
    if not os.path.exists(model_path):
        print("Model file not found. Run 03_train_model.py first.")
        return
        
    # Since we are probably training and evaluating on CPU for this project, let's load with map_location
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    metrics = {
        "overall_accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1_score": float(f1)
    }

    metrics_file = os.path.join(RESULTS_DIR, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved metrics to {metrics_file}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save CSV
    class_names = [idx_to_class[i] for i in range(num_classes)]
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join(RESULTS_DIR, 'confusion_matrix.csv')
    cm_df.to_csv(cm_csv_path)
    
    # Save visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Saved confusion matrix to {RESULTS_DIR}")

if __name__ == '__main__':
    evaluate_model()
