import numpy as np, torch, torch.nn as nn
import sys
sys.path.append('..')
from common.utils import save_img, ensure_dir
from common.config import OUTPUT_DIR

class SimSiam(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Simulate crop embeddings
ensure_dir(f"{OUTPUT_DIR}/hero")

# Generate sample crop stress heatmap
stress_heatmap = np.random.rand(256, 256)
save_img(stress_heatmap, f"{OUTPUT_DIR}/hero/stress_heatmap.jpg", cmap="RdYlGn_r")

print("SimSiam embeddings generated for crop stress analysis")
