import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = "data/RAF_DB/archive-3/DATASET"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_SAVE_PATH = "models/emotion_mobilenetv2.keras"

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

# RAF-DB label mapping
LABEL_MAP = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happy",
    "5": "Sad",
    "6": "Angry",
    "7": "Neutral"
}


# -----------------------------
# Data generators
# -----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_generator.num_classes

print("Class indices from Keras:")
print(train_generator.class_indices)

class_names_numeric = list(test_generator.class_indices.keys())
class_names = [LABEL_MAP[label] for label in class_names_numeric]

print("Emotion class names:")
print(class_names)


# -----------------------------
# MobileNetV2 base model
# -----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# First stage: freeze base model
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )
]


# -----------------------------
# Stage 1: Train classifier head
# -----------------------------
print("\nStage 1: Training classification head...")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


# -----------------------------
# Stage 2: Fine-tuning
# -----------------------------
print("\nStage 2: Fine-tuning last layers of MobileNetV2...")

base_model.trainable = True

# Freeze earlier layers, fine-tune only last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=8,
    callbacks=callbacks
)


# -----------------------------
# Evaluation
# -----------------------------
print("\nLoading best saved model...")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

test_generator.reset()

pred_probs = model.predict(test_generator)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = test_generator.classes

print("\nClassification Report:")
print(classification_report(
    true_classes,
    pred_classes,
    target_names=class_names
))

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, pred_classes))

print(f"\nBest model saved to: {MODEL_SAVE_PATH}")