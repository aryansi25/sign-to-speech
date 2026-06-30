import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# DATASET LOADING
# =========================

DATASET_PATH = "dataset"

X = []
y = []

labels = sorted(os.listdir(DATASET_PATH))
print("Classes:", labels)

for label_idx, label in enumerate(labels):

    gesture_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(gesture_path):
        continue

    items = os.listdir(gesture_path)

    if len(items) == 0:
        continue

    # CASE 1: sequences stored in folders
    if os.path.isdir(os.path.join(gesture_path, items[0])):

        for sequence in items:

            sequence_path = os.path.join(gesture_path, sequence)

            if not os.path.isdir(sequence_path):
                continue

            frames = []

            for frame_file in sorted(os.listdir(sequence_path)):

                frame_path = os.path.join(sequence_path, frame_file)

                if frame_file.endswith(".npy"):

                    data = np.load(frame_path)

                    # ensure correct landmark size
                    if data.shape[0] == 126:
                        frames.append(data)

            if len(frames) == 20:
                X.append(frames)
                y.append(label_idx)

    # CASE 2: npy files directly inside gesture folder
    else:

        frames = []

        for file in sorted(items):

            frame_path = os.path.join(gesture_path, file)

            if file.endswith(".npy"):

                data = np.load(frame_path)

                if data.shape[0] == 126:
                    frames.append(data)

                if len(frames) == 20:
                    X.append(frames)
                    y.append(label_idx)
                    frames = []

# =========================
# DATA PREPARATION
# =========================

X = np.array(X)
y = np.array(y)

# convert labels to categorical
y = to_categorical(y)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# LSTM MODEL
# =========================

model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(20,126)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(128))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))

# IMPORTANT FIX: output size from dataset
model.add(Dense(y.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAINING
# =========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=32,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL
# =========================

model.save("isl_lstm_model.keras")

# save labels
np.save("labels.npy", np.array(labels))

print("Model saved as isl_lstm_model.keras")
print("Labels saved as labels.npy")

# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)

y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true_labels, y_pred_labels))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true_labels, y_pred_labels))