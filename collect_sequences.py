import cv2
import os
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

label = input("Enter gesture label: ").upper()

label_path = os.path.join(DATA_PATH, label)

os.makedirs(label_path, exist_ok=True)

existing_sequences = len(os.listdir(label_path))

start_seq = existing_sequences
end_seq = existing_sequences + NEW_SEQUENCES

print(f"Collecting sequences {start_seq} to {end_seq-1}")

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

    left = [0]*63
    right = [0]*63

    if results.multi_hand_landmarks and results.multi_handedness:

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            hand_label = results.multi_handedness[idx].classification[0].label

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            if hand_label == "Left":
                left = coords
            else:
                right = coords

    points = np.array(left + right).reshape(42,3)

    base = points[0]
    points = points - base

    scale = np.linalg.norm(points[12][:2])
    if scale == 0:
        scale = 1

    points = points / scale

    return points.reshape(-1)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

for seq in range(start_seq, end_seq):

    os.makedirs(os.path.join(label_path, str(seq)))

    print(f"\nRecording sequence {seq}")

    for frame_num in range(SEQUENCE_LENGTH):

        ret, frame = cap.read()

        frame = cv2.flip(frame,1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        landmarks = extract_landmarks(results)

        np.save(
            os.path.join(label_path, str(seq), str(frame_num)),
            landmarks
        )

        cv2.putText(frame,
            f"{label} : Seq {seq} Frame {frame_num}",
            (10,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2)

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\nData collection complete!")







"""import cv2
import numpy as np
import mediapipe as mp
import os

# ===============================
# SETTINGS
# ===============================
DATA_PATH = "dataset"
SEQUENCE_LENGTH = 20
NUM_SEQUENCES = 50  # change to 60 if problem occurs 
# ===============================

label = input("Enter gesture label (EXACT name): ").strip().upper()

save_path = os.path.join(DATA_PATH, label)
os.makedirs(save_path, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    left = [0] * 63
    right = [0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if hand_label == "Left":
                left = landmarks
            else:
                right = landmarks

    return np.array(left + right)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    print("\nINSTRUCTIONS:")
    print("1. Perform the sign naturally")
    print("2. Hold briefly")
    print("3. Press 's' to record a sequence")
    print("4. Press 'q' to quit\n")

    sequence_count = 0

    while sequence_count < NUM_SEQUENCES:

        key = cv2.waitKey(1)

        if key & 0xFF == ord('s'):
            sequence = []

            print(f"\nRecording sequence {sequence_count+1}/{NUM_SEQUENCES}")

            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                landmarks = extract_landmarks(results)
                sequence.append(landmarks)

                cv2.putText(frame, f"Recording {frame_num+1}/{SEQUENCE_LENGTH}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

                cv2.imshow("Collecting Sequences", frame)
                cv2.waitKey(30)

            np.save(os.path.join(save_path, f"{sequence_count}.npy"), np.array(sequence))
            sequence_count += 1
            print("Saved.")

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Label: {label}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

        cv2.putText(frame, f"Collected: {sequence_count}/{NUM_SEQUENCES}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)

        cv2.imshow("Collecting Sequences", frame)

        if key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()"""