"""
STEP 5: Evaluate the trained Female/Hybrid/Male model on the held-out test set.

This script does NOT retrain anything — it loads your already-trained model
and measures how well it performs on test images it has never influenced
(not used in training, and not used for early-stopping decisions either).

What this generates:
  - Overall accuracy
  - Macro and weighted precision, recall, F1-score
  - Per-class accuracy
  - A confusion matrix (printed + saved as an image)
  - A CSV of every single test image's prediction (correct/incorrect, confidence)
  - A JSON summary of all metrics

Run this with your venv activated:
    python evaluate_fhm_model.py
"""

import os
import json
import csv
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no GUI window needed
import matplotlib.pyplot as plt

# ---- CONFIG ----
MODEL_PATH = "best_fhm_model_large.keras"
TEST_DIR = os.path.join("fhm_dataset_split_large", "test")
CLASS_NAMES_PATH = "fhm_class_names_large.txt"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
RESULTS_DIR = "results"


def main():
    print("=" * 60)
    print("  GENETIC PURITY MODEL — EVALUATION (Female / Hybrid / Male)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- STEP 1: Load trained model ----
    print(f"\n[1/5] Loading trained model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    print("  [OK] Model loaded successfully")
    print(f"  [OK] Input shape : {model.input_shape}")
    print(f"  [OK] Output shape: {model.output_shape}")
    print("  [OK] Total params: {:,}".format(model.count_params()))

    # ---- STEP 2: Load class names (order matters — must match training) ----
    print(f"\n[2/5] Loading class names from: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"  [OK] Classes (in training order): {class_names}")

    # ---- STEP 3: Load test dataset ----
    print(f"\n[3/5] Loading test dataset from: {TEST_DIR}")
    if not os.path.isdir(TEST_DIR):
        print(f"[ERROR] Test directory not found: {TEST_DIR}")
        return

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,  # IMPORTANT: keep order so filenames match predictions
    )
    test_class_names = test_ds.class_names
    print(f"  [OK] Test classes detected: {test_class_names}")
    print(f"  [OK] Total test images: {tf.data.experimental.cardinality(test_ds).numpy() * BATCH_SIZE} (approx)")

    if [c.lower() for c in class_names] != [c.lower() for c in test_class_names]:
        print("  [WARNING] Class order mismatch between training and test set!")
        print(f"    Training: {class_names}")
        print(f"    Test:     {test_class_names}")

    # Collect filenames in the same order the dataset will be iterated
    file_paths = test_ds.file_paths

    # Preprocess exactly the same way training data was preprocessed
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    test_ds_processed = test_ds.map(lambda x, y: (preprocess_input(x), y))

    # ---- STEP 4: Generate predictions ----
    print(f"\n[4/5] Running predictions on test images...")
    predictions = model.predict(test_ds_processed, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_confidences = np.max(predictions, axis=1)

    # Extract true labels in the same order
    true_classes = np.concatenate([np.argmax(y.numpy(), axis=1) for x, y in test_ds])

    # ---- STEP 5: Calculate and report metrics ----
    print(f"\n[5/5] Computing evaluation metrics...\n")
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"\n  Overall Accuracy: {accuracy * 100:.2f}%")

    precision_macro = precision_score(true_classes, predicted_classes, average="macro", zero_division=0)
    recall_macro = recall_score(true_classes, predicted_classes, average="macro", zero_division=0)
    f1_macro = f1_score(true_classes, predicted_classes, average="macro", zero_division=0)

    precision_weighted = precision_score(true_classes, predicted_classes, average="weighted", zero_division=0)
    recall_weighted = recall_score(true_classes, predicted_classes, average="weighted", zero_division=0)
    f1_weighted = f1_score(true_classes, predicted_classes, average="weighted", zero_division=0)

    print("\n  --- Macro Averaged Metrics ---")
    print(f"  Precision : {precision_macro * 100:.2f}%")
    print(f"  Recall    : {recall_macro * 100:.2f}%")
    print(f"  F1-Score  : {f1_macro * 100:.2f}%")

    print("\n  --- Weighted Averaged Metrics ---")
    print(f"  Precision : {precision_weighted * 100:.2f}%")
    print(f"  Recall    : {recall_weighted * 100:.2f}%")
    print(f"  F1-Score  : {f1_weighted * 100:.2f}%")

    print("\n  --- Per-Class Classification Report ---\n")
    report = classification_report(
        true_classes, predicted_classes, target_names=test_class_names, digits=4, zero_division=0
    )
    print(report)

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_classes, predicted_classes)
    print("  --- Confusion Matrix (Rows = True, Columns = Predicted) ---\n")
    header = "  " + " " * 10 + "".join(f"{name:>10}" for name in test_class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {test_class_names[i]:>10}" + "".join(f"{val:>10}" for val in row)
        print(row_str)

    # --- Per-Class Accuracy ---
    print("\n  --- Per-Class Accuracy ---\n")
    per_class_accuracy = {}
    for i, name in enumerate(test_class_names):
        class_mask = true_classes == i
        class_total = int(np.sum(class_mask))
        class_correct = int(np.sum(predicted_classes[class_mask] == i))
        class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
        per_class_accuracy[name] = {"correct": class_correct, "total": class_total, "accuracy": f"{class_acc:.1f}%"}
        print(f"  {name:>10}: {class_correct}/{class_total} correct = {class_acc:.1f}%")

    # --- Incorrect predictions detail ---
    print("\n  --- Incorrect Predictions ---\n")
    incorrect_details = []
    incorrect_count = 0
    for i in range(len(file_paths)):
        true_label = test_class_names[true_classes[i]]
        pred_label = test_class_names[predicted_classes[i]]
        if true_label != pred_label:
            incorrect_count += 1
            detail = {
                "file": os.path.basename(file_paths[i]),
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": f"{predicted_confidences[i]:.4f}",
            }
            incorrect_details.append(detail)
            print(f"  [{incorrect_count}] {detail['file']} — True: {true_label} | Predicted: {pred_label} | Confidence: {predicted_confidences[i]*100:.1f}%")

    total = len(file_paths)
    print(f"\n  Total incorrect: {incorrect_count}/{total} ({incorrect_count/total*100:.1f}%)")
    print(f"  Total correct:   {total - incorrect_count}/{total} ({(total - incorrect_count)/total*100:.1f}%)")

    # ---- SAVE RESULTS ----
    eval_results = {
        "overall_accuracy": f"{accuracy * 100:.2f}%",
        "total_test_images": total,
        "correct_predictions": total - incorrect_count,
        "incorrect_predictions": incorrect_count,
        "macro_metrics": {
            "precision": f"{precision_macro * 100:.2f}%",
            "recall": f"{recall_macro * 100:.2f}%",
            "f1_score": f"{f1_macro * 100:.2f}%",
        },
        "weighted_metrics": {
            "precision": f"{precision_weighted * 100:.2f}%",
            "recall": f"{recall_weighted * 100:.2f}%",
            "f1_score": f"{f1_weighted * 100:.2f}%",
        },
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm.tolist(),
        "class_names": test_class_names,
        "incorrect_predictions_detail": incorrect_details,
    }

    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"\n[INFO] Evaluation results saved to: {eval_path}")

    # Save per-image predictions CSV
    csv_path = os.path.join(RESULTS_DIR, "predictions_detail.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "predicted_label", "correct", "confidence"] +
                         [f"prob_{name}" for name in test_class_names])
        for i in range(len(file_paths)):
            true_label = test_class_names[true_classes[i]]
            pred_label = test_class_names[predicted_classes[i]]
            is_correct = true_label == pred_label
            row = [os.path.basename(file_paths[i]), true_label, pred_label, is_correct, f"{predicted_confidences[i]:.4f}"]
            row += [f"{predictions[i][j]:.4f}" for j in range(len(test_class_names))]
            writer.writerow(row)
    print(f"[INFO] Per-image predictions saved to: {csv_path}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix — Female / Hybrid / Male", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(test_class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(test_class_names, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(test_class_names, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     fontsize=13, fontweight="bold",
                     color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to: {cm_path}")

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
