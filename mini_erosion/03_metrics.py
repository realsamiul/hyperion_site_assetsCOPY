import numpy as np, matplotlib.pyplot as plt
import sys
sys.path.append('..')
from common.utils import save_img, ensure_dir
from common.config import OUTPUT_DIR

# Simulate erosion metrics over time
years = np.arange(1990, 2024)
retreat_rate = np.random.normal(2.5, 0.5, len(years))  # meters per year

# Create retreat curve
plt.figure(figsize=(10, 6))
plt.plot(years, retreat_rate, 'b-', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Shoreline Retreat Rate (m/year)')
plt.title('Coastal Erosion Trends - Kutubdia Island')
plt.grid(True, alpha=0.3)
plt.tight_layout()

ensure_dir(f"{OUTPUT_DIR}/charts")
plt.savefig(f"{OUTPUT_DIR}/charts/retreat_curve.png", dpi=300, bbox_inches='tight')
plt.close()

print("Erosion metrics calculated and retreat curve generated")
