import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
GRAD_CAM_DIR = os.path.join(BASE_DIR, 'results', 'grad_cam_outputs')

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[:, class_idx]
        score.backward()
        
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def run_grad_cam():
    os.makedirs(GRAD_CAM_DIR, exist_ok=True)
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'class_map.json'), 'r') as f:
        class_map = json.load(f)
    num_classes = len(class_map)
    idx_to_class = {v: k for k, v in class_map.items()}

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DATA_DIR, 'test'), data_transform)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    model_path = os.path.join(MODELS_DIR, 'species_classifier.pth')
    if not os.path.exists(model_path):
        print("Model file not found.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # The target layer in MobileNetV3 is typically the last convolutional layer in features
    # features[-1] is a Conv2dNormActivation which contains Conv2d
    target_layer = model.features[-1][0] 
    grad_cam = GradCAM(model, target_layer)
    
    # Pick a few images from test set
    num_samples = min(5, len(test_dataset))
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img_tensor, label = test_dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Original image path to load for visualization
        img_path, _ = test_dataset.samples[idx]
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (224, 224))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        cam = grad_cam(img_tensor, label)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original_img)
        ax[0].set_title(f'Original: {idx_to_class[label]}')
        ax[0].axis('off')
        
        ax[1].imshow(overlay)
        ax[1].set_title('Grad-CAM')
        ax[1].axis('off')
        
        out_path = os.path.join(GRAD_CAM_DIR, f'gradcam_{i}.png')
        plt.savefig(out_path)
        plt.close()
        
    print(f"Generated 5 Grad-CAM visualizations in {GRAD_CAM_DIR}")

if __name__ == '__main__':
    run_grad_cam()
