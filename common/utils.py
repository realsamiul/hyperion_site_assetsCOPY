"""
Utility functions for Hyperion Flood Detection
"""
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

# Backward compatibility
ensure = ensure_dir

def ts():
    """Timestamp for logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_json(data, filepath):
    """Save data as JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    return filepath

def load_json(filepath):
    """Load JSON data"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_img(array, filepath, cmap=None, dpi=100):
    """Save numpy array as image"""
    if array.dtype != np.uint8:
        # Normalize to 0-255 if needed
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
    
    if len(array.shape) == 2:
        # Grayscale
        if cmap:
            plt.imsave(filepath, array, cmap=cmap, dpi=dpi)
        else:
            Image.fromarray(array, mode='L').save(filepath)
    else:
        # Color
        if array.shape[2] == 3:
            Image.fromarray(array, mode='RGB').save(filepath)
        elif array.shape[2] == 4:
            Image.fromarray(array, mode='RGBA').save(filepath)
    
    return filepath

def normalize_sar(data):
    """Normalize SAR data to 0-1 range"""
    return np.clip((10*np.log10(data+1e-6)+25)/30, 0, 1)

def create_water_mask(ndwi, threshold=0.3):
    """Create water mask from NDWI"""
    return (ndwi > threshold).astype(np.uint8)

def calculate_metrics(pred, target):
    """Calculate evaluation metrics"""
    intersection = (pred * target).sum()
    union = ((pred + target) > 0).sum()
    
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    accuracy = (pred == target).mean()
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'accuracy': float(accuracy)
    }

def create_overlay(base_img, mask, color=[0, 0, 255], alpha=0.3):
    """Create overlay visualization"""
    if len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    
    overlay = base_img.copy()
    overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha
    
    return overlay.astype(np.uint8)

def create_comparison_grid(images, titles=None, figsize=(15, 10)):
    """Create a grid of comparison images"""
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
        ax.axis('off')
    
    plt.tight_layout()
    return fig

# For backward compatibility with existing scripts
def download_url(url, filepath):
    """Download file from URL"""
    import requests
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    return filepath

# Legacy functions for backward compatibility
def add_kv(d, path, value):
    """Insert value deep inside dict (path = 'results.quantitative.iou')"""
    keys = path.split("."); cur=d
    for k in keys[:-1]:
        if k not in cur: cur[k]={}
        cur=cur[k]
    cur[keys[-1]]=value

def deep_set(d, path, value):
    """Insert value deep inside dict (path = 'results.quantitative.iou')"""
    keys = path.split("."); cur=d
    for k in keys[:-1]:
        if k not in cur: cur[k]={}
        cur=cur[k]
    cur[keys[-1]]=value

def load_schema():
    return json.load(open("../_schema/template_schema.json"))

def timestamp():  
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
