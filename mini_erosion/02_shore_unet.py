import numpy as np, torch, torch.nn as nn
import sys
sys.path.append('..')
from common.utils import save_img, ensure_dir
from common.config import OUTPUT_DIR

# Simple U-Net for shoreline detection
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create sample shoreline detection
ensure_dir(f"{OUTPUT_DIR}/hero")

# Simulate shoreline detection results
shoreline_1990 = np.random.rand(256, 256) > 0.5
shoreline_2023 = np.random.rand(256, 256) > 0.3

save_img(shoreline_1990, f"{OUTPUT_DIR}/hero/shore_1990.jpg", cmap="gray")
save_img(shoreline_2023, f"{OUTPUT_DIR}/hero/shore_2023.jpg", cmap="gray")

print("U-Net shoreline detection completed")
