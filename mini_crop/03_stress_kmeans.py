import numpy as np
from sklearn.cluster import KMeans
import sys
sys.path.append('..')
from common.utils import save_img, ensure_dir
from common.config import OUTPUT_DIR

# Simulate crop stress clustering
print("Running KMeans clustering for crop stress classification...")

# Generate sample data
n_samples = 1000
features = np.random.rand(n_samples, 4)  # NDVI, NDWI, temperature, moisture

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features)

# Create stress classification map
stress_map = np.random.rand(256, 256)
stress_map = (stress_map * 3).astype(int)  # 3 stress levels

ensure_dir(f"{OUTPUT_DIR}/charts")
save_img(stress_map, f"{OUTPUT_DIR}/charts/stress_classification.png", cmap="RdYlGn_r")

print("KMeans clustering completed - 3 stress levels identified")
