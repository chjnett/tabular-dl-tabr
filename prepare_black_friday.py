import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import openml

def main():
    # 1. Create output directory
    out_dir = Path("data/black-friday")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching Black Friday dataset from OpenML (ID: 41540)...")
    dataset = openml.datasets.get_dataset(41540, download_data=True)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    
    print(f"Dataset loaded! Shape: {X.shape}")
    
    # 2. Handle missing values
    # Product_Category_2 and Product_Category_3 have NaNs. We fill with a string placeholder.
    X = X.fillna("Unknown")
    
    # Convert target to float32 (Regression)
    y = y.astype(np.float32).values
    
    # 3. Split the data (60/20/20)
    print("Splitting dataset...")
    X_str = X.astype(str).values  # Convert all categorical to string for TabR CatPolicy
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_str, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42 # 0.25 x 0.8 = 0.2
    )
    
    print("Saving to X_cat_*.npy and Y_*.npy")
    np.save(out_dir / "X_cat_train.npy", X_train)
    np.save(out_dir / "X_cat_val.npy", X_val)
    np.save(out_dir / "X_cat_test.npy", X_test)
    
    np.save(out_dir / "Y_train.npy", y_train)
    np.save(out_dir / "Y_val.npy", y_val)
    np.save(out_dir / "Y_test.npy", y_test)
    
    # 4. Save info.json
    info = {
        "task_type": "regression",
        "name": "Black Friday",
        "id": "black-friday",
        "n_num_features": 0,
        "n_cat_features": X.shape[1],
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test)
    }
    
    with open(out_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
        
    print(f"Done! Data prepared in {out_dir}")

if __name__ == '__main__':
    main()
