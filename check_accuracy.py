import os
import sys
import numpy as np
import tensorflow as tf

DATASET_PATH = "dataset"

if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset directory '{DATASET_PATH}' was not found.")
    print("Please place dataset directory in the project root to evaluate accuracy.")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical

def sort_key(name):
    base = os.path.splitext(name)[0]
    return int(base) if base.isdigit() else name

sequences = []
labels = []
label_map = {}

# Load label names
actions = sorted(os.listdir(DATASET_PATH))
for idx, action in enumerate(actions):
    label_map[action] = idx

# Load dataset
for action in actions:
    action_path = os.path.join(DATASET_PATH, action)
    if not os.path.isdir(action_path):
        continue

    for sequence in sorted(os.listdir(action_path), key=sort_key):
        seq_path = os.path.join(action_path, sequence)
        if not os.path.isdir(seq_path):
            continue

        frames = []
        for frame in sorted(os.listdir(seq_path), key=sort_key):
            frame_path = os.path.join(seq_path, frame)
            if frame.endswith(".npy"):
                data = np.load(frame_path)
                if data.shape[0] == 126:
                    frames.append(data)

        if len(frames) == 20:
            sequences.append(frames)
            labels.append(label_map[action])

if len(sequences) == 0:
    print("Error: No 20-frame sequences found in dataset.")
    sys.exit(1)

X = np.array(sequences, dtype=np.float32)
y = to_categorical(labels)

# Train/Test Split (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# Load trained model
model = tf.keras.models.load_model("isl_lstm_model.keras")

# Predict
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=actions))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
