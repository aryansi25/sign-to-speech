import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "dataset"

sequences = []
labels = []
label_map = {}

actions = sorted(os.listdir(DATASET_PATH))

for idx, action in enumerate(actions):
    label_map[action] = idx

# Load dataset
for action in actions:
    action_path = os.path.join(DATASET_PATH, action)

    for sequence in os.listdir(action_path):

        seq_path = os.path.join(action_path, sequence)
        frames = []

        for frame in sorted(os.listdir(seq_path)):

            data = np.load(os.path.join(seq_path, frame))

            if data.shape[0] == 126:
                frames.append(data)

        if len(frames) == 20:
            sequences.append(frames)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# Load model
model = tf.keras.models.load_model("isl_lstm_model.keras")

# Predict
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Accuracy
accuracy = np.mean(y_pred == y_true)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Classification report
report = classification_report(y_true, y_pred, target_names=actions, output_dict=True)

precision = []
recall = []
f1 = []

for label in actions:
    precision.append(report[label]['precision'])
    recall.append(report[label]['recall'])
    f1.append(report[label]['f1-score'])

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=actions,
            yticklabels=actions)

plt.title("GestureTalk Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -------------------------------
# Precision Recall F1 chart
# -------------------------------

x = np.arange(len(actions))

plt.figure(figsize=(12,6))

plt.bar(x-0.2, precision, width=0.2, label="Precision")
plt.bar(x, recall, width=0.2, label="Recall")
plt.bar(x+0.2, f1, width=0.2, label="F1 Score")

plt.xticks(x, actions, rotation=90)

plt.title("Per-Class Performance")
plt.legend()

plt.tight_layout()
plt.savefig("per_class_metrics.png")
plt.show()

# -------------------------------
# Overall Metrics
# -------------------------------

macro_precision = report["macro avg"]["precision"]
macro_recall = report["macro avg"]["recall"]
macro_f1 = report["macro avg"]["f1-score"]

metrics = [accuracy, macro_precision, macro_recall, macro_f1]
names = ["Accuracy", "Precision", "Recall", "F1"]

plt.figure(figsize=(6,6))
plt.bar(names, metrics)

plt.ylim(0,1)

plt.title("GestureTalk Model Performance")

plt.savefig("overall_metrics.png")
plt.show()