"""
PHASE 2: Fine-tune the already-trained Female/Hybrid/Male model.

WHAT THIS DOES DIFFERENTLY FROM PHASE 1 (train_fhm_model_large.py):
Phase 1 trained only a small new head on top of a FULLY FROZEN MobileNetV2 —
the pretrained layers were never touched. That's safe and fast, but caps how
well the model can adapt to your specific images.

Phase 2 unfreezes the LAST 40 layers of MobileNetV2 (the more specialized,
higher-level feature layers — the early layers that detect basic edges/colors
stay frozen) and continues training with a MUCH lower learning rate (100x
smaller than Phase 1). This lets the model fine-tune its pretrained features
specifically for maize seedling images, without a high learning rate
destroying the useful pretrained knowledge in a few noisy gradient steps.

This mirrors the two-phase approach used in the reference project, which is
a standard, well-established transfer learning technique.

BEFORE RUNNING: make sure best_fhm_model_large.keras (from Phase 1) already
exists in this folder — this script loads and continues from it.

Run this with your venv activated:
    python finetune_fhm_model.py
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json

DATA_DIR = "fhm_dataset_split_large"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
PHASE1_MODEL_PATH = "best_fhm_model_large.keras"
PHASE2_MODEL_PATH = "best_fhm_model_finetuned.keras"
FINE_TUNE_EPOCHS = 20
UNFREEZE_LAST_N_LAYERS = 40
FINE_TUNE_LEARNING_RATE = 1e-5   # 100x smaller than Phase 1's 1e-3


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"), image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True, seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"), image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
    )
    class_names = train_ds.class_names
    print("Detected classes (in this order):", class_names)

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
    ])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def plot_combined_history(phase1_history, phase2_history):
    """Plot Phase 1 and Phase 2 together as one continuous timeline, like the
    reference project's merged training curve."""
    acc = phase1_history["accuracy"] + phase2_history.history["accuracy"]
    val_acc = phase1_history["val_accuracy"] + phase2_history.history["val_accuracy"]
    loss = phase1_history["loss"] + phase2_history.history["loss"]
    val_loss = phase1_history["val_loss"] + phase2_history.history["val_loss"]
    phase1_len = len(phase1_history["accuracy"])

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.axvline(x=phase1_len - 0.5, color="gray", linestyle="--", label="Phase 2 starts")
    plt.legend(loc="lower right")
    plt.title("Accuracy — Phase 1 (frozen) + Phase 2 (fine-tuned)")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.axvline(x=phase1_len - 0.5, color="gray", linestyle="--", label="Phase 2 starts")
    plt.legend(loc="upper right")
    plt.title("Loss — Phase 1 (frozen) + Phase 2 (fine-tuned)")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig("fhm_finetuning_history.png", dpi=120)
    print("Saved combined training chart as 'fhm_finetuning_history.png'")
    plt.show()


def main():
    if not os.path.exists(PHASE1_MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find '{PHASE1_MODEL_PATH}'. Run train_fhm_model_large.py "
            f"first (Phase 1) before running this fine-tuning script."
        )

    phase1_history = None
    if os.path.exists("fhm_phase1_history.json"):
        with open("fhm_phase1_history.json") as f:
            phase1_history = json.load(f)

    print("Loading datasets...")
    train_ds, val_ds, class_names = build_datasets()

    print(f"\nLoading Phase 1 model from: {PHASE1_MODEL_PATH}")
    model = tf.keras.models.load_model(PHASE1_MODEL_PATH)

    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise RuntimeError("Could not find the MobileNetV2 base layer inside the loaded model.")

    print(f"Base model has {len(base_model.layers)} layers.")

    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfroze the last {UNFREEZE_LAST_N_LAYERS} layers "
          f"({trainable_count} of {len(base_model.layers)} base layers are now trainable).")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\nStarting Phase 2 fine-tuning (lr={FINE_TUNE_LEARNING_RATE})...\n")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            PHASE2_MODEL_PATH, monitor="val_accuracy", save_best_only=True
        ),
    ]

    phase2_history = model.fit(
        train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks
    )

    best_val_acc = max(phase2_history.history["val_accuracy"])
    print(f"\nFine-tuning complete. Best model saved to '{PHASE2_MODEL_PATH}'.")
    print(f"Best fine-tuned validation accuracy: {best_val_acc:.4f}")

    if phase1_history:
        plot_combined_history(phase1_history, phase2_history)
    else:
        print("\n(No saved Phase 1 history found — plotting Phase 2 only.)")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(phase2_history.history["accuracy"], label="Training Accuracy")
        plt.plot(phase2_history.history["val_accuracy"], label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Phase 2 Fine-Tuning Accuracy")
        plt.subplot(1, 2, 2)
        plt.plot(phase2_history.history["loss"], label="Training Loss")
        plt.plot(phase2_history.history["val_loss"], label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Phase 2 Fine-Tuning Loss")
        plt.tight_layout()
        plt.savefig("fhm_finetuning_history.png", dpi=120)
        plt.show()


if __name__ == "__main__":
    main()
