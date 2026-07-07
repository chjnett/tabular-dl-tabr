"""
California Housing Dataset Preparation Script
Prepares data in the exact format expected by tabular-dl-tabr:
  - X_num_{split}.npy (float32)
  - Y_{split}.npy     (float32, regression)
  - info.json
  - READY
"""
import io
import os
import sys
import json
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Fix Windows cp949 encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Output directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "california")
os.makedirs(DATA_DIR, exist_ok=True)

print("Fetching California Housing dataset via sklearn...")
cal = fetch_california_housing()
X, y = cal.data.astype(np.float32), cal.target.astype(np.float32)

# Train/Val/Test split: 60% / 20% / 20%
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

print(f"Split sizes - train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# Save numpy arrays
for split, X_s, y_s in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    np.save(os.path.join(DATA_DIR, f"X_num_{split}.npy"), X_s)
    np.save(os.path.join(DATA_DIR, f"Y_{split}.npy"), y_s)
    print(f"  Saved X_num_{split}.npy  {X_s.shape}  Y_{split}.npy  {y_s.shape}")

# Write info.json
info = {
    "task_type": "regression",
    "name": "California Housing",
    "id": "california",
    "n_num_features": X.shape[1],
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
}
with open(os.path.join(DATA_DIR, "info.json"), "w") as f:
    json.dump(info, f, indent=2)
print("  Saved info.json")

# Create READY sentinel file
open(os.path.join(DATA_DIR, "READY"), "w").close()
print("  Created READY")

print(f"\n✅ California dataset ready at: {DATA_DIR}")
