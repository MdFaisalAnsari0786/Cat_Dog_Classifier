"""Train a cats-vs-dogs classifier using MobileNetV2 transfer learning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
IMG_SIZE = (160, 160)


def load_datasets(batch_size: int):
    common = dict(
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=batch_size,
    )
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", shuffle=True, seed=42, **common
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "val", shuffle=False, **common
    )
    test_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "test", shuffle=False, **common
    )

    class_names = train_ds.class_names
    if class_names != ["cats", "dogs"]:
        raise ValueError(
            f"Expected class folders ['cats', 'dogs']; received {class_names}. "
            "Run download_data.py to create the expected directory structure."
        )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_model() -> tuple[keras.Model, keras.Model]:
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_preprocessing")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="dog_probability")(x)
    model = keras.Model(inputs, outputs, name="cat_dog_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def save_training_plot(history: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("accuracy", [])) + 1)
    if not list(epochs):
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="Training accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Cat vs Dog Classifier - Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_accuracy.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Cat vs Dog image classifier.")
    parser.add_argument("--epochs", type=int, default=5, help="Initial training epochs (default: 5).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze the last MobileNetV2 layers for fine-tuning.")
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    args = parser.parse_args()

    train_path = DATA_DIR / "train"
    if not train_path.exists() or not any(train_path.rglob("*.jpg")):
        raise SystemExit("Training images not found. First run: python download_data.py")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds, class_names = load_datasets(args.batch_size)
    model, base_model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "cat_dog_model.keras", monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.CSVLogger(OUTPUT_DIR / "training_log.csv"),
    ]

    print("\nStage 1: training the new classification head...")
    history_all = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks
    )
    combined_history = {key: list(value) for key, value in history_all.history.items()}

    if args.fine_tune:
        print("\nStage 2: fine-tuning the last 30 layers of MobileNetV2...")
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        # Keep Batch Normalization layers frozen for stable small-dataset fine-tuning.
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        fine_callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                MODEL_DIR / "cat_dog_model.keras", monitor="val_loss", save_best_only=True
            ),
            keras.callbacks.CSVLogger(OUTPUT_DIR / "fine_tuning_log.csv"),
        ]
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=fine_callbacks,
        )
        for key, values in history_ft.history.items():
            combined_history.setdefault(key, []).extend(values)

    # Load the best checkpoint before the final test evaluation.
    best_model = keras.models.load_model(MODEL_DIR / "cat_dog_model.keras")
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
    best_model.save(MODEL_DIR / "cat_dog_model.keras")

    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy), "class_names": class_names},
            file,
            indent=2,
        )
    save_training_plot(combined_history)

    print("\nTraining complete!")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Saved model: {MODEL_DIR / 'cat_dog_model.keras'}")
    print("Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
