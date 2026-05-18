import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


DEFAULT_DATASET_DIR = "data/RAF_DB/archive-3/DATASET"
DEFAULT_MODEL_PATH = "models/emotion_custom_cnn.keras"

LABEL_MAP = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happy",
    "5": "Sad",
    "6": "Angry",
    "7": "Neutral",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a custom CNN on RAF-DB without touching the existing files."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def build_datasets(train_dir, test_dir, image_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = list(train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, test_ds, class_names


def build_custom_cnn(image_size, num_classes):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="augmentation",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),
            augmentation,
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="emotion_custom_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def friendly_class_names(class_names):
    return [LABEL_MAP.get(class_name, class_name) for class_name in class_names]


def evaluate_model(model, test_ds, class_names):
    predictions = model.predict(test_ds, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = np.concatenate(
        [labels.numpy() for _, labels in test_ds],
        axis=0,
    )

    labels = list(range(len(class_names)))
    readable_names = friendly_class_names(class_names)

    print("\nClassification Report:")
    print(
        classification_report(
            true_classes,
            predicted_classes,
            labels=labels,
            target_names=readable_names,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes, labels=labels))


def main():
    args = parse_args()

    train_dir = args.train_dir or os.path.join(args.dataset_dir, "train")
    test_dir = args.test_dir or os.path.join(args.dataset_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    model_dir = os.path.dirname(args.model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("Preparing datasets...")
    train_ds, test_ds, class_names = build_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    print("Keras class indices:")
    print({name: index for index, name in enumerate(class_names)})
    print("Emotion class names:")
    print(friendly_class_names(class_names))

    model = build_custom_cnn(
        image_size=args.image_size,
        num_classes=len(class_names),
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_path,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("\nTraining custom CNN...")
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\nLoading best saved custom CNN...")
    best_model = tf.keras.models.load_model(args.model_path)
    evaluate_model(best_model, test_ds, class_names)

    print(f"\nBest custom CNN saved to: {args.model_path}")


if __name__ == "__main__":
    main()
