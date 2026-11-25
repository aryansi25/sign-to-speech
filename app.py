import cv2
import os
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
from collections import deque
from gtts import gTTS
import requests
import io
import time
import threading
from vocabulary import get_vocab_translation

app = Flask(__name__)
CORS(app)

# =========================
# SUPPORTED LANGUAGES
# =========================
LANG_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Urdu": "ur",
    "Punjabi": "pa",
    "Nepali": "ne",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Thai": "th",
    "Russian": "ru",
    "Portuguese": "pt",
}

LANGUAGES = list(LANG_CODES.keys())

SIGN_LANGUAGES = [
    {"id": "ISL", "name": "Indian Sign Language (ISL)", "status": "active", "badge": "Active Model (27 Signs)"},
    {"id": "ASL", "name": "American Sign Language (ASL)", "status": "disabled", "badge": "Dataset Required"},
    {"id": "BSL", "name": "British Sign Language (BSL)", "status": "disabled", "badge": "Dataset Required"}
]

# =========================
# LOAD MODEL & LABELS
# =========================

model = tf.keras.models.load_model("isl_lstm_model.keras")
labels = np.load("labels.npy")

# STRICT VOCABULARY WHITELIST:
# Only words officially trained and present in labels.npy (excluding NO_SIGN) are allowed.
VALID_SIGNS = [str(lbl) for lbl in labels if str(lbl) != "NO_SIGN"]
VALID_SIGNS_SET = set(VALID_SIGNS)

print(f"Loaded ISL LSTM model. Valid signs whitelist ({len(VALID_SIGNS)} signs):")
print(", ".join(VALID_SIGNS))

# Pre-load benchmark test sequences for automated browser evaluation
BENCHMARK_DATA = {}
try:
    if os.path.exists("benchmark_test_data.npz"):
        archive = np.load("benchmark_test_data.npz")
        for key in archive.files:
            BENCHMARK_DATA[key] = archive[key]
        print(f"Loaded {len(BENCHMARK_DATA)} benchmark gesture sequences for testing.")
except Exception as e:
    print(f"Warning: Could not pre-load benchmark_test_data.npz: {e}")

# =========================
# ULTRA-FAST CONVERSATIONAL ACCURACY & SPEED SETTINGS
# =========================

HOLD_FRAMES = 2              # 2 consecutive frames (~60ms) eliminates single-frame transit noise
CONF_THRESHOLD = 0.50        # Precision threshold preventing transition noise from committing
FAST_COMMIT_THRESHOLD = 0.70 # Fast lock-in when confidence is established
WORD_COMMIT_COOLDOWN = 0.60  # Minimum 0.60s between distinct words to match natural conversational pace
SAME_SIGN_COOLDOWN = 1.30    # Minimum 1.30s before repeating the exact same word

state_lock = threading.Lock()
seq_lock = threading.Lock()

sequence = deque(maxlen=20)
prediction_history = deque(maxlen=3)   # Low inertia rolling window (zero delay)
prediction_queue = deque(maxlen=3)     # Immediate majority voting

last_sign_commit_time = 0.0
missing_hand_frames = 0

latest_sequence_snapshot = None
latest_hand_lms_snapshot = None
latest_face_info_snapshot = None
new_sample_ready = threading.Event()
worker_running = True

# =========================
# GLOBAL STATE
# =========================

cap = None
current_sign = None
confidence = 0
hold_counter = 0

last_added_sign = None
latest_event_id = 0
latest_word_event = None
conversation_history = []  # List of {id, sign, time}
active_sign_lang = "ISL"
hands_in_frame = False

translation_cache = {}
audio_cache = {}  # In-memory RAM cache for 0ms conversational audio streaming

# =========================
# MEDIAPIPE (ULTRA-FAST REALTIME HAND & FACE TRACKING)
# =========================

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

# model_complexity=0 is the ultra-fast real-time model optimized for fluid 30 FPS webcam tracking
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

# Ultra-fast real-time face detector (< 4ms on CPU) for facial zone localization (mouth, chin, temple)
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.50
)

# Label lookup table for rapid disambiguation
LABEL_TO_IDX = {str(lbl): i for i, lbl in enumerate(labels)}

# =========================
# DISAMBIGUATION ENGINE (HAND GEOMETRY & FACIAL ZONES)
# =========================

def disambiguate_prediction(preds, hand_lms_list, face_info=None):
    """
    Combines hand pose geometry (finger extension, bunched fingertips, 'W' pose,
    'ILY' pose, closed fist, pointing) with facial zones (mouth, chin, temple, chest)
    to deterministically disambiguate visually overlapping gestures:
    - FOOD vs THANK_YOU (bunched fingertips at mouth vs flat open palm moving from chin)
    - I vs YES (pointing index finger at chest vs closed fist)
    - WATER ('W' sign near mouth)
    - UNDERSTAND vs HELLO (index pointing to temple vs open waving hand)
    - I_LOVE_YOU ('ILY' sign)
    - OKAY vs INDIAN (circle with thumb and index)
    - MY vs I (flat palm on chest vs pointing at chest)
    """
    if not hand_lms_list or len(hand_lms_list) == 0:
        return preds

    p = np.copy(preds)
    primary_hand = hand_lms_list[0]
    if len(primary_hand) < 21:
        return preds

    def dist(i, j):
        return np.linalg.norm(np.array(primary_hand[i][:2]) - np.array(primary_hand[j][:2]))

    # Finger extension metrics relative to wrist (point 0)
    # Using Euclidean distance ratio: tip vs pip
    ext_idx = dist(0, 8) > dist(0, 6) * 1.12
    ext_mid = dist(0, 12) > dist(0, 10) * 1.12
    ext_ring = dist(0, 16) > dist(0, 14) * 1.12
    ext_pinky = dist(0, 20) > dist(0, 18) * 1.12
    ext_thumb = dist(4, 17) > dist(3, 17) * 1.10

    num_ext = sum([ext_idx, ext_mid, ext_ring, ext_pinky])

    # Curled fingers check relative to wrist
    is_curled_idx = dist(0, 8) < dist(0, 6)
    is_curled_mid = dist(0, 12) < dist(0, 10)
    is_curled_ring = dist(0, 16) < dist(0, 14)
    is_curled_pinky = dist(0, 20) < dist(0, 18)

    # 1. Closed fist (YES): index, middle and ring are curled into palm
    is_fist = is_curled_idx and is_curled_mid and is_curled_ring

    # 2. Bunched fingertips (FOOD): fingertips gathered together reaching away from wrist towards mouth
    bunch_dist_thumb_idx = dist(4, 8)
    bunch_dist_idx_mid = dist(8, 12)
    bunch_dist_thumb_mid = dist(4, 12)
    is_bunched = (not is_fist) and (dist(0, 8) > dist(0, 6)) and (bunch_dist_thumb_idx < 0.080) and (bunch_dist_idx_mid < 0.065) and (bunch_dist_thumb_mid < 0.085)

    # 3. 'W' pose (WATER): index, middle, ring extended; pinky curled
    is_w_pose = (not is_fist) and (not is_bunched) and ext_idx and ext_mid and ext_ring and (not ext_pinky)

    # 4. 'ILY' pose (I_LOVE_YOU): index, pinky, thumb extended; middle and ring curled
    is_ily = (not is_fist) and (not is_bunched) and ext_idx and ext_pinky and ext_thumb and (not ext_mid) and (not ext_ring)

    # 5. 'OK' circle (OKAY / INDIAN): thumb tip touching index tip in circle, other fingers extended
    is_ok_circle = (not is_fist) and (not is_bunched) and (bunch_dist_thumb_idx < 0.055) and (ext_mid or ext_ring or ext_pinky)

    # 6. Pointing (I / UNDERSTAND): index extended, other fingers curled
    is_pointing = (not is_fist) and (not is_bunched) and (not is_w_pose) and (not is_ok_circle) and ext_idx and (not ext_mid) and (not ext_ring) and (not ext_pinky)

    # 7. Open flat palm (THANK_YOU / HELLO / MY): >= 3 fingers extended, spread
    is_open_palm = (not is_fist) and (not is_bunched) and (not is_w_pose) and (not is_ok_circle) and (not is_pointing) and (num_ext >= 3) and (bunch_dist_thumb_idx > 0.075)

    # Facial Zones
    in_mouth_zone = False
    in_temple_zone = False
    in_chest_zone = False

    if face_info is not None:
        hand_min_y = min([pt[1] for pt in primary_hand])
        mouth_y = face_info.get('mouth', (0.5, 0.5))[1]
        chin_y = face_info.get('chin_y', 0.6)
        nose_y = face_info.get('nose', (0.5, 0.4))[1]
        face_h = face_info.get('bbox', (0, 0, 0, 0.2))[3]

        in_temple_zone = hand_min_y <= (nose_y + face_h * 0.1)
        in_mouth_zone = (nose_y - face_h * 0.2) <= hand_min_y <= (chin_y + face_h * 0.35)
        in_chest_zone = hand_min_y > (chin_y + face_h * 0.2)

    # MUTUALLY EXCLUSIVE DETERMINISTIC DISAMBIGUATION:
    if is_bunched:
        # Bunched fingertips (FOOD)
        if 'FOOD' in LABEL_TO_IDX:
            f_idx = LABEL_TO_IDX['FOOD']
            p[f_idx] = max(p[f_idx] * 5.0, 1.5)
            if 'THANK_YOU' in LABEL_TO_IDX: p[LABEL_TO_IDX['THANK_YOU']] *= 0.05
            if 'YES' in LABEL_TO_IDX: p[LABEL_TO_IDX['YES']] *= 0.05
            if 'I' in LABEL_TO_IDX: p[LABEL_TO_IDX['I']] *= 0.05
    elif is_w_pose:
        # 'W' sign (WATER)
        if 'WATER' in LABEL_TO_IDX:
            w_idx = LABEL_TO_IDX['WATER']
            p[w_idx] = max(p[w_idx] * 5.0, 1.5)
            if 'FOOD' in LABEL_TO_IDX: p[LABEL_TO_IDX['FOOD']] *= 0.05
            if 'THANK_YOU' in LABEL_TO_IDX: p[LABEL_TO_IDX['THANK_YOU']] *= 0.05
            if 'YES' in LABEL_TO_IDX: p[LABEL_TO_IDX['YES']] *= 0.05
    elif is_pointing:
        # Extended index point: UNDERSTAND (temple) vs I (chest)
        if in_temple_zone and 'UNDERSTAND' in LABEL_TO_IDX:
            u_idx = LABEL_TO_IDX['UNDERSTAND']
            p[u_idx] = max(p[u_idx] * 5.0, 1.5)
            if 'HELLO' in LABEL_TO_IDX: p[LABEL_TO_IDX['HELLO']] *= 0.05
            if 'I' in LABEL_TO_IDX: p[LABEL_TO_IDX['I']] *= 0.05
            if 'YES' in LABEL_TO_IDX: p[LABEL_TO_IDX['YES']] *= 0.05
        elif 'I' in LABEL_TO_IDX:
            i_idx = LABEL_TO_IDX['I']
            p[i_idx] = max(p[i_idx] * 5.0, 1.5)
            if 'YES' in LABEL_TO_IDX: p[LABEL_TO_IDX['YES']] *= 0.05
            if 'UNDERSTAND' in LABEL_TO_IDX: p[LABEL_TO_IDX['UNDERSTAND']] *= 0.05
    elif is_fist:
        # Closed fist (YES)
        if 'YES' in LABEL_TO_IDX:
            y_idx = LABEL_TO_IDX['YES']
            p[y_idx] = max(p[y_idx] * 5.0, 1.5)
            if 'I' in LABEL_TO_IDX: p[LABEL_TO_IDX['I']] *= 0.05
            if 'FOOD' in LABEL_TO_IDX: p[LABEL_TO_IDX['FOOD']] *= 0.05
            if 'THANK_YOU' in LABEL_TO_IDX: p[LABEL_TO_IDX['THANK_YOU']] *= 0.05
    elif is_open_palm:
        # Open flat palm: HELLO (temple) vs MY (chest) vs THANK_YOU (chin/mouth)
        if in_temple_zone and 'HELLO' in LABEL_TO_IDX:
            h_idx = LABEL_TO_IDX['HELLO']
            p[h_idx] = max(p[h_idx] * 5.0, 1.5)
            if 'UNDERSTAND' in LABEL_TO_IDX: p[LABEL_TO_IDX['UNDERSTAND']] *= 0.05
        elif in_chest_zone and 'MY' in LABEL_TO_IDX and p[LABEL_TO_IDX['MY']] >= p[LABEL_TO_IDX.get('THANK_YOU', 0)]:
            m_idx = LABEL_TO_IDX['MY']
            p[m_idx] = max(p[m_idx] * 5.0, 1.5)
            if 'I' in LABEL_TO_IDX: p[LABEL_TO_IDX['I']] *= 0.05
        elif 'THANK_YOU' in LABEL_TO_IDX:
            ty_idx = LABEL_TO_IDX['THANK_YOU']
            p[ty_idx] = max(p[ty_idx] * 5.0, 1.5)
            if 'FOOD' in LABEL_TO_IDX: p[LABEL_TO_IDX['FOOD']] *= 0.05
    elif is_ily:
        if 'I_LOVE_YOU' in LABEL_TO_IDX:
            ily_idx = LABEL_TO_IDX['I_LOVE_YOU']
            p[ily_idx] = max(p[ily_idx] * 5.0, 1.5)
    elif is_ok_circle:
        if 'OKAY' in LABEL_TO_IDX and 'INDIAN' in LABEL_TO_IDX:
            ok_idx = LABEL_TO_IDX['OKAY']
            ind_idx = LABEL_TO_IDX['INDIAN']
            if p[ok_idx] >= p[ind_idx]:
                p[ok_idx] = max(p[ok_idx] * 4.0, 1.5)
            else:
                p[ind_idx] = max(p[ind_idx] * 4.0, 1.5)

    # Normalize distribution
    s = np.sum(p)
    if s > 0:
        p = p / s
    return p

# =========================
# HAND VALIDATION
# =========================

def is_valid_human_hand(landmarks):
    """
    Validates detected hand landmarks:
    - Accepts all genuine human hand poses (fists, pointing, open palm, pinching).
    - Rejects tiny background artifacts and noise.
    """
    if not hasattr(landmarks, "landmark") or len(landmarks.landmark) < 21:
        return False

    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)

    # Genuine hand in front of webcam spans between 3% and 98% of the frame
    return (0.03 <= bbox_w <= 0.98) and (0.03 <= bbox_h <= 0.98)

# =========================
# LANDMARK EXTRACTION & NORMALIZATION
# =========================

def extract_landmarks(results):
    """
    Extracts and normalizes hand landmarks strictly matching the training dataset distribution:
    - Preserves natural spatial coordinate distribution across all 27 ISL gesture classes.
    - Single hand signs: mapped to Right hand slot (indices 63..125) with Left slot (indices 0..62) set to 0.0.
    - Two hand signs: normalized relative to the left wrist (points[0]) matching training architecture.
    """
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None

    valid_indices = []
    for idx, hand_lms in enumerate(results.multi_hand_landmarks):
        if is_valid_human_hand(hand_lms):
            valid_indices.append(idx)

    if len(valid_indices) == 0:
        return None

    left = [0.0] * 63
    right = [0.0] * 63

    for idx in valid_indices:
        hand_landmarks = results.multi_hand_landmarks[idx]
        classifications = getattr(results.multi_handedness[idx], 'classification', [])
        hand_label = classifications[0].label if classifications else "Right"

        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        if hand_label == "Left":
            left = coords
        else:
            right = coords

    points = np.array(left + right, dtype=np.float32).reshape(42, 3)

    base = points[0]
    points = points - base

    scale = np.linalg.norm(points[12][:2])
    if scale == 0 or np.isnan(scale):
        scale = 1.0

    points = points / scale
    return np.nan_to_num(points.reshape(-1), nan=0.0)

# =========================
# ASYNC INFERENCE WORKER THREAD
# =========================

def inference_worker():
    """
    Decouples heavy LSTM model inference from the video streaming loop.
    Applies rolling softmax probability averaging and majority queue voting
    to eliminate frame noise and guarantee high recognition accuracy.
    """
    global current_sign, confidence, hold_counter, hands_in_frame
    global last_added_sign, latest_event_id, latest_word_event, latest_sequence_snapshot, last_sign_commit_time

    while worker_running:
        signaled = new_sample_ready.wait(timeout=0.04)
        if not worker_running:
            break
        if not signaled:
            continue

        new_sample_ready.clear()

        with state_lock:
            if not hands_in_frame:
                continue

        with seq_lock:
            if latest_sequence_snapshot is None:
                continue
            input_data = np.copy(latest_sequence_snapshot)
            hand_lms = latest_hand_lms_snapshot
            face_info = latest_face_info_snapshot

        try:
            # Forward pass through trained LSTM
            preds = model(input_data, training=False).numpy()[0]

            # Deterministic hand pose & facial zone disambiguation
            preds = disambiguate_prediction(preds, hand_lms, face_info)

            with state_lock:
                if not hands_in_frame:
                    continue

                prediction_history.append(preds)
                # Stage 1: Fast rolling distribution
                avg_preds = np.mean(prediction_history, axis=0)

                # Raw instantaneous prediction
                raw_idx = int(np.argmax(preds))
                raw_conf = float(preds[raw_idx])
                raw_label = str(labels[raw_idx])

                # Smoothed prediction
                idx = int(np.argmax(avg_preds))
                conf = float(avg_preds[idx])
                stable_label = str(labels[idx])

                now = time.time()

                # REAL-TIME INSTANT FEEDBACK:
                # Prioritize high-confidence raw or smoothed sign for instant reaction
                active_sign = stable_label if (stable_label in VALID_SIGNS_SET and conf >= CONF_THRESHOLD) else (
                    raw_label if (raw_label in VALID_SIGNS_SET and raw_conf >= CONF_THRESHOLD) else None
                )
                active_conf = conf if active_sign == stable_label else raw_conf

                if active_sign:
                    current_sign = active_sign
                    confidence = round(active_conf * 100, 1)

                    # Dynamic hold: 2 frames (~50ms) eliminates transit noise; 1 frame if >= FAST_COMMIT_THRESHOLD
                    required_hold = 1 if active_conf >= FAST_COMMIT_THRESHOLD else HOLD_FRAMES

                    # Conversational pacing: 0.60s cooldown between distinct signs, 1.20s for repeating same sign
                    time_since_commit = now - last_sign_commit_time
                    if active_sign != last_added_sign:
                        can_commit = (time_since_commit >= WORD_COMMIT_COOLDOWN)
                    else:
                        can_commit = (time_since_commit >= SAME_SIGN_COOLDOWN)

                    hold_counter += 1
                    if hold_counter >= required_hold and can_commit:
                        latest_event_id += 1
                        latest_word_event = active_sign
                        last_added_sign = active_sign
                        last_sign_commit_time = now
                        conversation_history.append({
                            "id": latest_event_id,
                            "sign": active_sign,
                            "time": time.strftime("%H:%M:%S")
                        })
                        hold_counter = 0

                        # Keep the natural 20-frame FIFO queue intact so subsequent continuous signs retain high classification accuracy!
                        prediction_history.clear()
                        prediction_queue.clear()
                else:
                    hold_counter = max(0, hold_counter - 1)
                    if hold_counter == 0:
                        if conf < 0.30:
                            current_sign = None
                            confidence = 0
                        if now - last_sign_commit_time > 1.2:
                            last_added_sign = None

        except Exception as e:
            print(f"Async inference error: {e}")

# Start background inference worker
inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()

# =========================
# TRANSLATION HELPER
# =========================

def translate_text(text, target_lang):
    if not text or not text.strip():
        return ""
    text = text.strip()
    if target_lang == "en":
        return text

    # Instant vocabulary lookup (0ms latency, 100% reliable)
    vocab_trans = get_vocab_translation(text, target_lang)
    if vocab_trans:
        return vocab_trans

    cache_key = f"{text}::{target_lang}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": text}
        res = requests.get(url, params=params, timeout=3)
        if res.status_code == 200:
            data = res.json()
            translated = "".join([part[0] for part in data[0] if part[0]])
            if translated:
                translation_cache[cache_key] = translated
                return translated
    except Exception:
        pass

    return text

# =========================
# VIDEO STREAM GENERATOR
# =========================

def generate_frames():
    global cap, current_sign, confidence, hold_counter, hands_in_frame
    global last_added_sign, latest_event_id, latest_word_event, latest_sequence_snapshot, missing_hand_frames

    # Open webcam with standard backend for instant sub-second startup
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_counter = 0

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Ultra-fast face detection (< 4ms on CPU) for facial zones (mouth, chin, temple)
            face_results = face_detector.process(rgb)
            face_info = None
            if face_results and face_results.detections:
                det = face_results.detections[0]
                bbox = det.location_data.relative_bounding_box
                kps = det.location_data.relative_keypoints
                nose = (kps[2].x, kps[2].y) if len(kps) > 2 else (0.5, bbox.ymin + bbox.height * 0.5)
                mouth = (kps[3].x, kps[3].y) if len(kps) > 3 else (0.5, bbox.ymin + bbox.height * 0.75)
                chin_y = mouth[1] + (mouth[1] - nose[1]) * 0.9 if len(kps) > 3 else (bbox.ymin + bbox.height)
                face_info = {
                    "bbox": (bbox.xmin, bbox.ymin, bbox.width, bbox.height),
                    "nose": nose,
                    "mouth": mouth,
                    "chin_y": chin_y,
                    "temple_y": nose[1] - (mouth[1] - nose[1]) * 0.8,
                    "head_top": bbox.ymin
                }

            # NOTE: Palm lines and skeletal drawing are completely removed as requested.
            # The raw camera frame remains clean and natural.

            valid_lms = [lms for lms in results.multi_hand_landmarks if is_valid_human_hand(lms)] if results.multi_hand_landmarks else []

            if len(valid_lms) == 0:
                missing_hand_frames += 1
                # Reset only after hands are absent for 8 consecutive frames (~250ms)
                if missing_hand_frames >= 8:
                    with state_lock:
                        hands_in_frame = False
                        hold_counter = 0
                        current_sign = None
                        confidence = 0
                        last_added_sign = None

                    with seq_lock:
                        sequence.clear()
                        prediction_history.clear()
                        prediction_queue.clear()
                        latest_sequence_snapshot = None
                        latest_hand_lms_snapshot = None
                        latest_face_info_snapshot = None
            else:
                missing_hand_frames = 0
                with state_lock:
                    was_hands = hands_in_frame
                    hands_in_frame = True

                norm_vec = extract_landmarks(results)
                if norm_vec is not None:
                    with seq_lock:
                        # Instant pre-warm: fill 20-frame sequence buffer on frame 1 so inference begins immediately
                        if not was_hands or len(sequence) == 0:
                            sequence.extend([norm_vec] * 20)
                        else:
                            sequence.append(norm_vec)

                        latest_sequence_snapshot = np.expand_dims(np.array(sequence), axis=0).astype(np.float32)
                        latest_hand_lms_snapshot = [
                            [(lm.x, lm.y, lm.z) for lm in lms.landmark]
                            for lms in valid_lms
                        ]
                        latest_face_info_snapshot = face_info
                        new_sample_ready.set()

            # Encode frame to JPEG with optimal quality for instant streaming
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    except GeneratorExit:
        pass
    except Exception as e:
        print(f"Video stream error: {e}")
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

# =========================
# ROUTES
# =========================

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/state")
def get_state():
    client_event_id = request.args.get("last_event_id", 0, type=int)
    target_lang = request.args.get("lang", "Hindi")
    target_code = LANG_CODES.get(target_lang, "hi")

    with state_lock:
        # Snappy hold progress: reflects real-time lock-in confidence percentage
        if current_sign and confidence > 0:
            hold_prog = min(100, int((confidence / 75.0) * 100)) if confidence < 75 else 100
        else:
            hold_prog = 0
        current = current_sign
        conf = confidence
        is_hands = hands_in_frame
        history_copy = list(conversation_history)

        new_word = None
        new_word_trans = None
        ev_id = latest_event_id

        if ev_id > client_event_id:
            new_word = latest_word_event

    curr_trans = ""
    if current and current in VALID_SIGNS_SET:
        clean_curr = current.replace("_", " ")
        curr_trans = translate_text(clean_curr, target_code)

    if new_word and new_word in VALID_SIGNS_SET:
        clean_new = new_word.replace("_", " ")
        new_word_trans = translate_text(clean_new, target_code) if target_code != "en" else clean_new

    # Build enriched conversation log
    enriched_history = []
    for item in history_copy:
        sign_name = item["sign"].replace("_", " ")
        trans = translate_text(sign_name, target_code)
        enriched_history.append({
            "id": item["id"],
            "sign": sign_name,
            "translated": trans,
            "time": item["time"]
        })

    return jsonify({
        "current_sign": current if current in VALID_SIGNS_SET else None,
        "current_sign_translated": curr_trans,
        "confidence": conf,
        "hold_progress": hold_prog,
        "hands_in_frame": is_hands,
        "new_word": new_word if new_word in VALID_SIGNS_SET else None,
        "new_word_translated": new_word_trans,
        "event_id": ev_id,
        "conversation": enriched_history,
        "active_sign_lang": active_sign_lang,
        "whitelist_count": len(VALID_SIGNS)
    })

@app.route("/api/clear_conversation", methods=["POST"])
def clear_conversation():
    global last_added_sign, latest_event_id, latest_word_event
    with state_lock:
        conversation_history.clear()
        last_added_sign = None
        latest_word_event = None
    return jsonify({"status": "cleared"})

@app.route("/api/set_sign_language", methods=["POST"])
def set_sign_language():
    global active_sign_lang
    data = request.get_json(silent=True) or {}
    sign_lang = data.get("sign_language", "ISL")
    with state_lock:
        active_sign_lang = sign_lang
    return jsonify({"status": "ok", "active_sign_lang": active_sign_lang})

@app.route("/api/stop_camera", methods=["POST"])
def stop_camera():
    global cap, hold_counter, current_sign, last_added_sign, hands_in_frame, latest_sequence_snapshot
    with state_lock:
        hold_counter = 0
        current_sign = None
        last_added_sign = None
        hands_in_frame = False

    with seq_lock:
        sequence.clear()
        prediction_history.clear()
        prediction_queue.clear()
        latest_sequence_snapshot = None
        latest_hand_lms_snapshot = None
        latest_face_info_snapshot = None

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    return jsonify({"status": "stopped"})

# =========================
# TRANSLATE & SENTENCE REST API
# =========================

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "").strip().replace("_", " ")
    language = data.get("language", "Hindi")
    lang_code = LANG_CODES.get(language, "hi")

    if not sentence:
        return jsonify({"translated": "", "language": language})

    translated = translate_text(sentence, lang_code)
    return jsonify({"translated": translated, "language": language})

@app.route("/api/remove_last_word", methods=["POST"])
def remove_last_word():
    global last_added_sign
    with state_lock:
        if conversation_history:
            conversation_history.pop()
        last_added_sign = conversation_history[-1]["sign"] if conversation_history else None
    return jsonify({"status": "removed", "count": len(conversation_history)})

@app.route("/api/add_custom_text", methods=["POST"])
def add_custom_text():
    global latest_event_id, latest_word_event, last_added_sign
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if text:
        with state_lock:
            latest_event_id += 1
            latest_word_event = text
            last_added_sign = text
            conversation_history.append({
                "id": latest_event_id,
                "sign": text,
                "time": time.strftime("%H:%M:%S")
            })
    return jsonify({"status": "added", "text": text})

# =========================
# SPEAK
# =========================

@app.route("/api/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "").strip().replace("_", " ")
    language = data.get("language", "English")

    if not sentence:
        return jsonify({"error": "No sign to speak"}), 400

    lang_code = LANG_CODES.get(language, "en")
    tts_lang = "hi" if lang_code == "pa" else lang_code
    cache_key = f"{sentence.upper()}::{tts_lang}"

    # Return instantly from in-memory cache if already generated (0ms latency)
    with state_lock:
        if cache_key in audio_cache:
            return Response(audio_cache[cache_key], mimetype="audio/mpeg")

    try:
        text_to_speak = translate_text(sentence, tts_lang) if tts_lang != "en" else sentence
        tts = gTTS(text=text_to_speak, lang=tts_lang)

        fp = io.BytesIO()
        tts.write_to_fp(fp)
        audio_bytes = fp.getvalue()

        with state_lock:
            audio_cache[cache_key] = audio_bytes

        return Response(audio_bytes, mimetype="audio/mpeg")
    except Exception as e:
        # Offline-resilient fallback: return valid silent MP3 audio frame when offline
        fallback_mp3 = (
            b'ID3\x03\x00\x00\x00\x00\x00\x00' +
            (b'\xff\xfb\x90d' + b'\x00' * 414) * 5
        )
        return Response(fallback_mp3, mimetype="audio/mpeg")

# =========================
# PAGES
# =========================

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/app")
def index():
    return render_template("index.html", languages=LANGUAGES, sign_languages=SIGN_LANGUAGES)

@app.route("/test")
def test_runner():
    return render_template("test_runner.html")

@app.route("/api/benchmark_translations")
def benchmark_translations():
    lang = request.args.get("lang", "Hindi")
    lang_code = LANG_CODES.get(lang, "hi")
    translations = {}
    for sign in VALID_SIGNS:
        trans = get_vocab_translation(sign, lang_code)
        translations[sign] = trans if trans else sign
    return jsonify(translations)

@app.route("/api/run_benchmark_single")
def run_benchmark_single():
    sign = request.args.get("sign", "").strip().upper()
    lang = request.args.get("lang", "Hindi")
    lang_code = LANG_CODES.get(lang, "hi")

    if sign not in VALID_SIGNS_SET:
        return jsonify({"error": f"Sign '{sign}' not in whitelist"}), 400

    seq = BENCHMARK_DATA.get(sign)
    if seq is None:
        return jsonify({"error": f"No benchmark sequence found for '{sign}'"}), 404

    t0 = time.perf_counter()
    input_tensor = np.expand_dims(seq, axis=0)
    pred_probs = model(input_tensor, training=False).numpy()[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    top_idx = int(np.argmax(pred_probs))
    pred_label = str(labels[top_idx])
    confidence = float(pred_probs[top_idx]) * 100.0

    passed = (pred_label == sign) and (confidence >= 70.0)
    translation = get_vocab_translation(sign, lang_code)

    return jsonify({
        "sign": sign,
        "predicted": pred_label,
        "confidence": round(confidence, 2),
        "latency_ms": round(latency_ms, 2),
        "passed": passed,
        "translation": translation or sign,
        "whitelisted": sign in VALID_SIGNS_SET
    })

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
