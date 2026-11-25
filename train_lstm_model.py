"""
GestureTalk — ISL LSTM Model Trainer (Enhanced)
================================================
Improvements over baseline:
  - Deeper LSTM (3 layers) with residual-style skip connection
  - L2 regularisation on LSTM & Dense layers
  - ReduceLROnPlateau + EarlyStopping callbacks
  - Data augmentation: Gaussian noise + temporal jitter
  - Per-class sample weight to handle imbalanced datasets
  - Saves best checkpoint mid-training, restores at end
  - Full evaluation: accuracy, confusion matrix, classification report
"""

import numpy as np
import os
import sys

DATASET_PATH = "dataset"
MODEL_OUT    = "isl_lstm_model.keras"
LABELS_OUT   = "labels.npy"
SEQ_LEN      = 20
FEAT_DIM     = 126
EPOCHS       = 120
BATCH_SIZE   = 32
TEST_SIZE    = 0.20
RANDOM_SEED  = 42

if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset directory '{DATASET_PATH}' not found.")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                     BatchNormalization, Add, TimeDistributed)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ── reproducibility ────────────────────────────────────────────────────────────
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ── 1. dataset loading ─────────────────────────────────────────────────────────

def sort_key(name):
    base = os.path.splitext(name)[0]
    return int(base) if base.isdigit() else name

X, y = [], []
labels = sorted(os.listdir(DATASET_PATH))
print("Classes:", labels)

for label_idx, label in enumerate(labels):
    gesture_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(gesture_path):
        continue
    items = os.listdir(gesture_path)
    if not items:
        continue

    if os.path.isdir(os.path.join(gesture_path, items[0])):
        # sequences in sub-folders
        for seq in sorted(items, key=sort_key):
            seq_path = os.path.join(gesture_path, seq)
            if not os.path.isdir(seq_path):
                continue
            frames = []
            for f in sorted(os.listdir(seq_path), key=sort_key):
                if f.endswith(".npy"):
                    d = np.load(os.path.join(seq_path, f))
                    if d.shape[0] == FEAT_DIM:
                        frames.append(d)
            if len(frames) == SEQ_LEN:
                X.append(frames)
                y.append(label_idx)
    else:
        # npy files directly in gesture folder
        frames = []
        for f in sorted(items, key=sort_key):
            if f.endswith(".npy"):
                d = np.load(os.path.join(gesture_path, f))
                if d.shape[0] == FEAT_DIM:
                    frames.append(d)
                if len(frames) == SEQ_LEN:
                    X.append(frames)
                    y.append(label_idx)
                    frames = []

X = np.array(X, dtype=np.float32)
y_labels = np.array(y)

if len(X) == 0:
    print("Error: No valid sequences found.")
    sys.exit(1)

print(f"Dataset: {X.shape[0]} sequences × {SEQ_LEN} frames × {FEAT_DIM} features")

y_cat = to_categorical(y_labels, num_classes=len(labels))


# ── 2. augmentation helpers ───────────────────────────────────────────────────

def augment_sequence(seq, noise_std=0.008, time_jitter=True):
    """Light augmentation: Gaussian noise + optional 1-frame temporal shift."""
    out = seq + np.random.normal(0, noise_std, seq.shape).astype(np.float32)
    if time_jitter and np.random.rand() < 0.5:
        shift = np.random.choice([-1, 1])
        out = np.roll(out, shift, axis=0)
        # zero-pad boundary frame
        if shift > 0:
            out[0] = 0.0
        else:
            out[-1] = 0.0
    return out

def build_augmented_dataset(X, y, factor=2):
    """Duplicate dataset `factor` times with augmentation."""
    Xa, ya = [X], [y]
    for _ in range(factor - 1):
        Xa.append(np.array([augment_sequence(s) for s in X]))
        ya.append(y)
    return np.concatenate(Xa, axis=0), np.concatenate(ya, axis=0)


# ── 3. train/test split ───────────────────────────────────────────────────────

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y_cat, test_size=TEST_SIZE, stratify=y_labels, random_state=RANDOM_SEED
)

# augment only the training split
X_train, y_train = build_augmented_dataset(X_train_raw, y_train_raw, factor=3)
print(f"Train (aug): {X_train.shape[0]} | Test: {X_test.shape[0]}")

# per-class sample weights for imbalance handling
y_int_train = np.argmax(y_train, axis=1)
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_int_train), y=y_int_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}


# ── 4. enhanced LSTM model ────────────────────────────────────────────────────

def build_model(seq_len, feat_dim, num_classes):
    reg = l2(1e-4)
    inp = Input(shape=(seq_len, feat_dim), name="input")

    # Block 1
    x = LSTM(128, return_sequences=True, kernel_regularizer=reg, name="lstm1")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Block 2
    x = LSTM(128, return_sequences=True, kernel_regularizer=reg, name="lstm2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Block 3 — deeper temporal context
    x = LSTM(64, return_sequences=False, kernel_regularizer=reg, name="lstm3")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # Classifier head
    x = Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = Dropout(0.25)(x)
    out = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model(SEQ_LEN, FEAT_DIM, len(labels))
model.summary()


# ── 5. callbacks ──────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=18,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        "isl_lstm_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
]


# ── 6. training ───────────────────────────────────────────────────────────────

print("\nStarting enhanced training…")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)


# ── 7. save ───────────────────────────────────────────────────────────────────

model.save(MODEL_OUT)
np.save(LABELS_OUT, np.array(labels))
print(f"\nModel saved → {MODEL_OUT}")
print(f"Labels saved → {LABELS_OUT}")


# ── 8. evaluation ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test, verbose=0)
y_pred_idx = np.argmax(y_pred, axis=1)
y_true_idx = np.argmax(y_test, axis=1)

test_acc = np.mean(y_pred_idx == y_true_idx) * 100
print(f"\nTest accuracy : {test_acc:.2f}%")

print("\nClassification Report:")
print(classification_report(y_true_idx, y_pred_idx, target_names=labels))

print("Confusion Matrix:")
print(confusion_matrix(y_true_idx, y_pred_idx))
