import cv2
import os
import sys
import time
import numpy as np
import mediapipe as mp

# =========================
# SETTINGS
# =========================

DATA_PATH = "dataset"
SEQUENCE_LENGTH = 20
NEW_SEQUENCES = 30

# =========================
# INPUT LABEL
# =========================

label = input("Enter gesture label: ").strip().upper()
if not label:
    print("Label cannot be empty.")
    sys.exit(1)

label_path = os.path.join(DATA_PATH, label)
os.makedirs(label_path, exist_ok=True)

existing_sequences = len([d for d in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, d))])
start_seq = existing_sequences
end_seq = existing_sequences + NEW_SEQUENCES

print(f"Collecting sequences {start_seq} to {end_seq-1} for gesture '{label}'")
print("Press 'q' at any time to finish and save.")

# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# LANDMARK EXTRACTION
# =========================

def extract_landmarks(results):
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None

    num_hands = len(results.multi_hand_landmarks)
    left = [0.0] * 63
    right = [0.0] * 63

    if num_hands == 1:
        # Single-hand: map to Right slot, Left slot stays zero
        hand_pts = []
        for lm in results.multi_hand_landmarks[0].landmark:
            hand_pts.extend([lm.x, lm.y, lm.z])
        right = hand_pts
        points = np.array(left + right, dtype=np.float32).reshape(42, 3)

        # Normalize relative to wrist (point 0) — matches app.py inference exactly
        base = points[0]
        points = points - base
        scale = np.linalg.norm(points[12][:2])
        if scale == 0 or np.isnan(scale):
            scale = 1.0
        points = points / scale
        return np.nan_to_num(points.reshape(-1), nan=0.0)
    else:
        hand_entries = []
        for idx, hand_lms in enumerate(results.multi_hand_landmarks[:2]):
            coords = []
            for lm in hand_lms.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            wrist_x = hand_lms.landmark[0].x
            hand_entries.append((wrist_x, coords))

        hand_entries.sort(key=lambda item: item[0])
        left = hand_entries[0][1]
        right = hand_entries[1][1]

        points = np.array(left + right, dtype=np.float32).reshape(42, 3)
        base = points[0]
        points = points - base

        scale = np.linalg.norm(points[12][:2])
        if scale == 0 or np.isnan(scale):
            scale = 1.0

        points = points / scale
        return np.nan_to_num(points.reshape(-1), nan=0.0)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

should_exit = False

for seq in range(start_seq, end_seq):
    if should_exit:
        break

    seq_dir = os.path.join(label_path, str(seq))
    os.makedirs(seq_dir, exist_ok=True)

    # 1.5-second readiness countdown before each sequence
    start_time = time.time()
    while time.time() - start_time < 1.5:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.flip(frame, 1)
        remaining = round(1.5 - (time.time() - start_time), 1)
        cv2.putText(frame, f"Get Ready for Seq {seq} in {remaining}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            should_exit = True
            break

    if should_exit:
        break

    print(f"Recording sequence {seq}...")

    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        landmarks = extract_landmarks(results)

        np.save(os.path.join(seq_dir, f"{frame_num}.npy"), landmarks)

        cv2.putText(frame,
                    f"{label} : Seq {seq} Frame {frame_num+1}/{SEQUENCE_LENGTH}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2)

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            should_exit = True
            break

cap.release()
cv2.destroyAllWindows()

print("\nData collection finished.")
