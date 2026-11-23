"""
Preprocess and split the Female/Hybrid/Male maize seedling dataset.

Splits images into train (70%) / validation (15%) / test (15%) sets,
stratified so each split has a balanced mix of Female/Hybrid/Male.

Run this with your venv activated:
    python preprocess_fhm_dataset.py
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = "maize_fhm_dataset_large"
OUTPUT_DIR = "fhm_dataset_split_large"
RANDOM_SEED = 42

CLASSES = ["Female", "Hybrid", "Male"]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def main():
    metadata_path = os.path.join(DATASET_DIR, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find {metadata_path}.")

    df = pd.read_csv(metadata_path)
    print("Total images found in metadata.csv:", len(df))
    print(df["class_label"].value_counts(), "\n")

    train_df, temp_df = train_test_split(
        df,
        test_size=(SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]),
        stratify=df["class_label"],
        random_state=RANDOM_SEED,
    )
    val_ratio_within_temp = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_within_temp),
        stratify=temp_df["class_label"],
        random_state=RANDOM_SEED,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split_name in splits:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split_name, cls.lower()), exist_ok=True)

    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            src = os.path.join(DATASET_DIR, row["image_path"])
            cls_folder = row["class_label"].lower()
            dst = os.path.join(OUTPUT_DIR, split_name, cls_folder, os.path.basename(row["image_path"]))
            if os.path.exists(src):
                shutil.copyfile(src, dst)
            else:
                print(f"WARNING: missing file {src}, skipped.")

    print("Split complete. Summary:\n")
    for split_name, split_df in splits.items():
        print(f"{split_name.upper()} — total: {len(split_df)}")
        print(split_df["class_label"].value_counts().to_string())
        print()

    print(f"Preprocessed, split dataset is ready in: ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
