"""
Train the Female/Hybrid/Male maize classifier using MobileNetV2 transfer learning.

Run this with your venv activated:
    python train_fhm_model.py
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

DATA_DIR = "fhm_dataset_split_large"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 40
NUM_CLASSES = 3
MODEL_SAVE_PATH = "best_fhm_model_large.keras"


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


def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def plot_history(history):
    acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
    loss, val_loss = history.history["loss"], history.history["val_loss"]
    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training vs Validation Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig("fhm_training_history_large.png", dpi=120)
    print("Saved chart as 'fhm_training_history_large.png'")
    plt.show()


def main():
    print("Loading datasets...")
    train_ds, val_ds, class_names = build_datasets()

    print("\nBuilding MobileNetV2 transfer learning model...")
    model, base_model = build_model()
    print(f"Base model frozen: {not base_model.trainable}")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True
        ),
    ]

    print("\nStarting training...\n")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
    )

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nTraining complete. Best model saved to '{MODEL_SAVE_PATH}'.")
    print(f"Best validation accuracy achieved: {best_val_acc:.4f}")

    with open("fhm_class_names_large.txt", "w") as f:
        f.write("\n".join(class_names))

    plot_history(history)


if __name__ == "__main__":
    main()
