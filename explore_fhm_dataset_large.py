"""
Explore the Female/Hybrid/Male maize seedling dataset.

Run this with your venv activated:
    python explore_fhm_dataset.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

DATASET_DIR = "maize_fhm_dataset_large"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")


def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Could not find {METADATA_PATH}. Make sure this script is in the same "
            f"folder as '{DATASET_DIR}/'."
        )

    df = pd.read_csv(METADATA_PATH)
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total images: {len(df)}\n")
    print("Images per class:")
    print(df["class_label"].value_counts().to_string())
    print()

    print("=" * 50)
    print("SAMPLE IMAGE INFO")
    print("=" * 50)

    classes = ["Female", "Hybrid", "Male"]
    sample_images = {}

    for cls in classes:
        row = df[df["class_label"] == cls].iloc[0]
        img_path = os.path.join(DATASET_DIR, row["image_path"])
        if not os.path.exists(img_path):
            print(f"WARNING: sample image for {cls} not found at {img_path}")
            continue
        img = Image.open(img_path)
        sample_images[cls] = img
        print(f"{cls}: {row['image_path']}  |  size={img.size}  |  mode={img.mode}")

    print()

    fig, axes = plt.subplots(1, len(sample_images), figsize=(4 * len(sample_images), 4))
    if len(sample_images) == 1:
        axes = [axes]

    for ax, (cls, img) in zip(axes, sample_images.items()):
        ax.imshow(img)
        ax.set_title(cls, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.suptitle("One sample seedling image per class (Female / Hybrid / Male)", fontsize=12)
    plt.tight_layout()
    plt.savefig("fhm_dataset_large_preview.png", dpi=120)
    print("Saved a preview image as 'fhm_dataset_large_preview.png' — open it to view the samples.")
    plt.show()

    print("\nIf you saw three different-looking seedlings above, your dataset is loading correctly.")
    print("Next step: preprocessing and train/val/test split.")


if __name__ == "__main__":
    main()
