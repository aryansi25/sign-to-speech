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

# Graph-compiled inference (~1.9ms/pass vs ~187ms eager) + startup warmup
@tf.function(reduce_retracing=True)
def _fast_infer(x):
    return model(x, training=False)

_ = _fast_infer(tf.zeros((1, 20, 126), dtype=tf.float32))

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
prediction_history = deque(maxlen=5)   # Exponential-weighted 5-frame rolling window

last_sign_commit_time = 0.0
missing_hand_frames = 0

latest_sequence_snapshot   = None
latest_hand_lms_snapshot   = None
latest_face_info_snapshot  = None
latest_hand_velocity_snapshot = 0.0   # wrist speed (screen-units/frame)
wrist_history = deque(maxlen=6)        # recent wrist positions for velocity estimate
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
# MEDIAPIPE (HIGH-ACCURACY REALTIME HAND & FACE TRACKING)
# =========================

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

# model_complexity=1 gives much sharper finger-curl & bunched-hand landmark accuracy
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45
)

# model_selection=1 (full-range 0-5m) detects faces reliably at any webcam distance
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.40
)

# CLAHE contrast normalizer for MediaPipe input (improves palm/face detection in dim/backlit rooms)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Label lookup table for rapid disambiguation
LABEL_TO_IDX = {str(lbl): i for i, lbl in enumerate(labels)}

# =========================
# DISAMBIGUATION ENGINE (HAND GEOMETRY + VELOCITY + FACIAL ZONES)
# =========================

def joint_angle(a, b, c):
    """
    Returns the interior angle (degrees) at joint b in the chain a->b->c.
    Uses raw (x, y) screen coordinates. 180 = fully extended, 0 = fully folded.
    Scale-invariant: works at any hand distance from the camera.
    """
    v1 = np.array(a[:2]) - np.array(b[:2])
    v2 = np.array(c[:2]) - np.array(b[:2])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-7 or n2 < 1e-7:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def disambiguate_prediction(preds, hand_lms_list, face_info=None, velocity=0.0):
    """
    Precision disambiguation pipeline combining:
      1. Joint-angle-based finger state detection (scale-invariant, camera-distance-invariant)
      2. Wrist velocity — static vs dynamic hand motion separates overlapping signs
      3. Facial zone estimation (temple / mouth / chest) using face detector + absolute fallback
      4. Two-hand presence detection for WHAT / WHERE / NICE
    """
    if not hand_lms_list or len(hand_lms_list) == 0:
        return preds

    p = np.copy(preds)
    primary = hand_lms_list[0]
    if len(primary) < 21:
        return preds

    # Angle-based finger state (MCP -> PIP -> TIP)
    def ja(a, b, c):
        return joint_angle(primary[a], primary[b], primary[c])

    ang_idx   = ja(5, 6, 8)
    ang_mid   = ja(9, 10, 12)
    ang_ring  = ja(13, 14, 16)
    ang_pinky = ja(17, 18, 20)
    ang_thumb = ja(1, 2, 4)

    EXT  = 148   # >= 148 deg -> extended
    CURL = 115   # <= 115 deg -> curled

    ext_idx   = ang_idx   >= EXT
    ext_mid   = ang_mid   >= EXT
    ext_ring  = ang_ring  >= EXT
    ext_pinky = ang_pinky >= EXT
    ext_thumb = ang_thumb >= 128

    curl_idx   = ang_idx   <= CURL
    curl_mid   = ang_mid   <= CURL
    curl_ring  = ang_ring  <= CURL
    curl_pinky = ang_pinky <= CURL

    num_ext  = sum([ext_idx, ext_mid, ext_ring, ext_pinky])
    num_curl = sum([curl_idx, curl_mid, curl_ring, curl_pinky])

    def dist(i, j):
        return np.linalg.norm(np.array(primary[i][:2]) - np.array(primary[j][:2]))

    d_thumb_idx = dist(4, 8)
    d_idx_mid   = dist(8, 12)
    d_thumb_mid = dist(4, 12)

    is_moving = velocity > 0.013
    is_static = velocity < 0.007
    two_hands = len(hand_lms_list) >= 2

    # 1. CLOSED FIST (YES)
    is_fist = curl_idx and curl_mid and curl_ring and curl_pinky

    # 2. BUNCHED FINGERTIPS (FOOD)
    tips_clustered = (d_thumb_idx < 0.09 and d_idx_mid < 0.08 and d_thumb_mid < 0.10)
    is_bunched = (not is_fist) and tips_clustered and (num_ext >= 1 or num_curl <= 2)

    # 3. 'W' POSE (WATER)
    is_w_pose = (not is_fist) and (not is_bunched) and ext_idx and ext_mid and ext_ring and curl_pinky

    # 4. 'ILY' POSE (I_LOVE_YOU)
    is_ily = (
        (not is_fist) and (not is_bunched)
        and ext_idx and ext_pinky and ext_thumb
        and curl_mid and curl_ring
    )

    # 5. OK CIRCLE (OKAY / INDIAN)
    is_ok_circle = (
        (not is_fist) and (not is_bunched)
        and d_thumb_idx < 0.055
        and (ext_mid or ext_ring or ext_pinky)
    )

    # 6. POINTING (I / UNDERSTAND / WHERE)
    is_pointing = (
        (not is_fist) and (not is_bunched) and (not is_w_pose)
        and (not is_ok_circle) and (not is_ily)
        and ext_idx and curl_mid and curl_ring and curl_pinky
    )

    # 7. OPEN FLAT PALM (HELLO / MY / THANK_YOU / WANT / WELCOME)
    is_open_palm = (
        (not is_fist) and (not is_bunched) and (not is_w_pose)
        and (not is_ok_circle) and (not is_pointing) and (not is_ily)
        and num_ext >= 3 and d_thumb_idx > 0.065
    )

    # Facial / body zone estimation (using fingertip top hand_min_y for clean zone separation)
    hand_min_y = min(pt[1] for pt in primary)

    if face_info is not None:
        chin_y = face_info.get('chin_y', 0.65)
        nose_y = face_info.get('nose', (0.5, 0.40))[1]
        face_h = face_info.get('bbox', (0, 0, 0, 0.22))[3]
        in_temple_zone = hand_min_y <= (nose_y + face_h * 0.10)
        in_chest_zone  = hand_min_y >  (chin_y + face_h * 0.05)
    else:
        in_temple_zone = hand_min_y < 0.26
        in_chest_zone  = hand_min_y > 0.46

    top_idx = int(np.argmax(p))
    top_label = str(labels[top_idx])
    top_conf = float(p[top_idx])

    def should_disambiguate(group_keys):
        """
        Only disambiguate when the LSTM's top prediction is inside this confusion group,
        OR when the LSTM is uncertain (top_conf < 0.55) and a group member is plausible (>= 0.08).
        Prevents hijacking confident predictions for unrelated signs (PLEASE, SORRY, HELP, YOU, etc.).
        """
        if top_label in group_keys:
            return True
        if top_conf < 0.55:
            return any(p[LABEL_TO_IDX[k]] >= 0.08 for k in group_keys if k in LABEL_TO_IDX)
        return False

    def boost(key, factor=7.0, suppress=None):
        if key not in LABEL_TO_IDX:
            return
        p[LABEL_TO_IDX[key]] = max(p[LABEL_TO_IDX[key]] * factor, 2.0)
        for k in (suppress or []):
            if k in LABEL_TO_IDX:
                p[LABEL_TO_IDX[k]] *= 0.02

    # Helper for any hand's finger states
    def _hand_stats(h):
        def h_ja(a, b, c):
            return joint_angle(h[a], h[b], h[c])
        def h_dist(i, j):
            return np.linalg.norm(np.array(h[i][:2]) - np.array(h[j][:2]))
        e_idx   = h_ja(5, 6, 8)   >= EXT
        e_mid   = h_ja(9, 10, 12) >= EXT
        e_ring  = h_ja(13, 14, 16) >= EXT
        e_pinky = h_ja(17, 18, 20) >= EXT
        e_thumb = h_ja(1, 2, 4)   >= 128
        c_idx   = h_ja(5, 6, 8)   <= CURL
        c_mid   = h_ja(9, 10, 12) <= CURL
        c_ring  = h_ja(13, 14, 16) <= CURL
        c_pinky = h_ja(17, 18, 20) <= CURL
        n_ext   = sum([e_idx, e_mid, e_ring, e_pinky])
        n_curl  = sum([c_idx, c_mid, c_ring, c_pinky])
        return {
            "e_idx": e_idx, "e_mid": e_mid, "e_ring": e_ring, "e_pinky": e_pinky, "e_thumb": e_thumb,
            "c_idx": c_idx, "c_mid": c_mid, "c_ring": c_ring, "c_pinky": c_pinky,
            "n_ext": n_ext, "n_curl": n_curl,
            "is_fist": c_idx and c_mid and c_ring and c_pinky,
            "is_point": e_idx and c_mid and c_ring and c_pinky,
            "is_two": e_idx and e_mid and c_ring and c_pinky,
            "is_open": n_ext >= 3 and h_dist(4, 8) > 0.065,
        }

    # Additional single-hand poses (BATHROOM, NO, YOU)
    # BATHROOM: closed fist with thumb sticking clearly outward (away from index PIP)
    is_bathroom_thumb = is_fist and ext_thumb and dist(4, 6) > 0.09 and dist(4, 17) > dist(3, 17) * 1.15
    # NO: index + middle extended together, ring + pinky curled
    is_two_fingers = (not is_fist) and (not is_bunched) and ext_idx and ext_mid and curl_ring and curl_pinky
    # YOU: pointing straight forward at camera (foreshortened 2D index & tip z closer than MCP z)
    is_pointing_forward = is_pointing and (
        (primary[8][2] < primary[5][2] - 0.045 and dist(5, 8) < 0.095) or top_label == 'YOU'
    )

    # ── TWO-HAND SIGNS (ALL 2-HAND ISL POSTER SIGNS) ─────────────────────────
    two_hand_matched = False
    if two_hands:
        sec = hand_lms_list[1]
        if len(sec) >= 21:
            # Sort left vs right in screen space
            h_left, h_right = (primary, sec) if primary[0][0] <= sec[0][0] else (sec, primary)
            sL = _hand_stats(h_left)
            sR = _hand_stats(h_right)

            def cross_dist(a_idx, b_idx):
                return np.linalg.norm(np.array(h_left[a_idx][:2]) - np.array(h_right[b_idx][:2]))

            wrist_dist = cross_dist(0, 0)
            palm_dist  = cross_dist(9, 9)
            tip_dist   = cross_dist(8, 8)

            # 1. TIME: one hand pointing at the wrist of the other hand
            l_to_r_wrist = cross_dist(8, 0) < 0.12 and sL["is_point"]
            r_to_l_wrist = np.linalg.norm(np.array(h_right[8][:2]) - np.array(h_left[0][:2])) < 0.12 and sR["is_point"]
            if l_to_r_wrist or r_to_l_wrist:
                boost('TIME', factor=8.0, suppress=['I', 'YOU', 'UNDERSTAND', 'THIS_PLACE', 'WHERE'])
                two_hand_matched = True

            # 2. THIS_PLACE: both index fingers pointing downward (tip_y > mcp_y + 0.03)
            elif (sL["is_point"] and sR["is_point"]
                  and h_left[8][1] > h_left[5][1] + 0.03
                  and h_right[8][1] > h_right[5][1] + 0.03):
                boost('THIS_PLACE', factor=8.0, suppress=['TO_MEET', 'WHERE', 'I', 'UNDERSTAND'])
                two_hand_matched = True

            # 3. TO_MEET: both index fingers pointing upward near each other
            elif (sL["is_point"] and sR["is_point"]
                  and h_left[8][1] < h_left[5][1] - 0.03
                  and h_right[8][1] < h_right[5][1] - 0.03
                  and tip_dist < 0.24):
                boost('TO_MEET', factor=8.0, suppress=['THIS_PLACE', 'WHERE', 'I', 'UNDERSTAND'])
                two_hand_matched = True

            # 4. WHERE: both index fingers pointing outward/wide
            elif sL["is_point"] and sR["is_point"]:
                boost('WHERE', factor=7.0, suppress=['UNDERSTAND', 'I', 'WHAT', 'THIS_PLACE'])
                two_hand_matched = True

            # 5. NAME_IS: both hands have index+middle extended (H-shape) close together
            elif sL["is_two"] and sR["is_two"] and tip_dist < 0.22:
                boost('NAME_IS', factor=8.0, suppress=['NO', 'NICE', 'PLEASE', 'HELP'])
                two_hand_matched = True

            # 6. HELP: one closed fist resting on top of the other open palm
            elif ((sL["is_fist"] and sR["is_open"]) or (sR["is_fist"] and sL["is_open"])) and palm_dist < 0.24:
                boost('HELP', factor=8.0, suppress=['YES', 'SORRY', 'MY', 'THANK_YOU', 'PLEASE'])
                two_hand_matched = True

            # 7. PLEASE: both open palms pressed together (namaste/prayer) in front of chest
            elif sL["is_open"] and sR["is_open"] and palm_dist < 0.14:
                boost('PLEASE', factor=8.0, suppress=['MY', 'THANK_YOU', 'NICE', 'WELCOME', 'WHAT'])
                two_hand_matched = True

            # 8. DONT: both open palms crossed in X (wrists crossed or very close, fingertips crossed)
            elif sL["is_open"] and sR["is_open"] and (h_left[8][0] > h_right[8][0] - 0.04) and wrist_dist < 0.25:
                boost('DONT', factor=8.0, suppress=['WELCOME', 'WHAT', 'NICE', 'HELLO'])
                two_hand_matched = True

            # 9. WELCOME / WHAT / NICE / WANT (both open palms)
            elif sL["is_open"] and sR["is_open"]:
                if wrist_dist > 0.30 and is_static:
                    boost('WELCOME', factor=7.0, suppress=['HELLO', 'MY', 'THANK_YOU'])
                    two_hand_matched = True
                elif is_moving and should_disambiguate({'WHAT', 'WELCOME', 'WANT', 'WHERE', 'HELLO'}):
                    boost('WHAT', factor=7.0, suppress=['HELLO', 'WELCOME', 'WANT', 'WHERE'])
                    two_hand_matched = True
                elif is_static and palm_dist < 0.24 and should_disambiguate({'NICE', 'TO_MEET', 'WELCOME', 'PLEASE'}):
                    boost('NICE', factor=6.0, suppress=['HELLO', 'MY', 'THANK_YOU'])
                    two_hand_matched = True

    # ── SINGLE-HAND DISAMBIGUATION (when no 2-hand pose already matched) ──────
    if not two_hand_matched:
        if is_bunched and should_disambiguate({'FOOD', 'THANK_YOU', 'WATER', 'YES', 'I'}):
            boost('FOOD', factor=9.0, suppress=['THANK_YOU', 'YES', 'I', 'MY', 'HELLO', 'WANT', 'WATER'])
        elif is_w_pose and should_disambiguate({'WATER', 'FOOD', 'THANK_YOU', 'YES', 'HELLO'}):
            boost('WATER', factor=9.0, suppress=['FOOD', 'THANK_YOU', 'YES', 'HELLO', 'WANT'])
        elif is_ily:
            boost('I_LOVE_YOU', factor=8.0, suppress=['I', 'YES', 'HELLO', 'WHAT', 'WANT'])
        elif is_bathroom_thumb and should_disambiguate({'BATHROOM'}):
            boost('BATHROOM', factor=8.0, suppress=['YES', 'SORRY', 'I', 'MY'])
        elif is_two_fingers and should_disambiguate({'NO', 'NAME_IS', 'WATER', 'I', 'YES'}):
            boost('NO', factor=7.0, suppress=['YES', 'I', 'WATER'])
        elif is_pointing and should_disambiguate({'I', 'YOU', 'UNDERSTAND', 'YES', 'HELLO', 'MY'}):
            if in_temple_zone:
                boost('UNDERSTAND', factor=9.0, suppress=['HELLO', 'I', 'YOU', 'YES', 'MY', 'WHERE'])
            elif is_pointing_forward:
                boost('YOU', factor=8.0, suppress=['I', 'YES', 'UNDERSTAND', 'MY'])
            else:
                boost('I', factor=9.0, suppress=['YES', 'YOU', 'UNDERSTAND', 'MY', 'FOOD', 'WANT', 'WHERE'])
        elif is_fist and should_disambiguate({'YES', 'SORRY', 'I', 'FOOD', 'THANK_YOU', 'MY'}):
            if in_chest_zone and top_label == 'SORRY':
                boost('SORRY', factor=8.0, suppress=['YES', 'I', 'MY'])
            else:
                boost('YES', factor=9.0, suppress=['I', 'FOOD', 'THANK_YOU', 'MY', 'WANT', 'WHAT'])
        elif is_open_palm and should_disambiguate({'HELLO', 'UNDERSTAND', 'MY', 'WANT', 'THANK_YOU', 'FOOD', 'PLEASE'}):
            if in_temple_zone:
                boost('HELLO', factor=9.0, suppress=['UNDERSTAND', 'MY', 'THANK_YOU', 'WANT', 'WHAT'])
            elif in_chest_zone:
                if top_label == 'PLEASE':
                    boost('PLEASE', factor=7.0, suppress=['MY', 'THANK_YOU', 'HELLO'])
                elif is_moving:
                    boost('WANT', factor=8.0, suppress=['MY', 'THANK_YOU', 'I', 'HELLO', 'FOOD'])
                else:
                    boost('MY', factor=9.0, suppress=['I', 'THANK_YOU', 'FOOD', 'HELLO', 'WANT'])
            else:
                boost('THANK_YOU', factor=8.0, suppress=['FOOD', 'MY', 'HELLO', 'WANT', 'YES'])
        elif is_ok_circle and should_disambiguate({'OKAY', 'INDIAN'}):
            ok_i  = LABEL_TO_IDX.get('OKAY', -1)
            ind_i = LABEL_TO_IDX.get('INDIAN', -1)
            if ok_i >= 0 and ind_i >= 0:
                winner = 'INDIAN' if (in_chest_zone or p[ind_i] > p[ok_i]) else 'OKAY'
                boost(winner, factor=6.0)

    # Normalize distribution
    s = np.sum(p)
    if s > 0:
        p = p / s
    return p

# =========================
# HAND VALIDATION
# =========================

def is_valid_human_hand(landmarks, handedness_score=1.0, face_info=None):
    """
    Validates detected hand landmarks and rejects background noise / distant bystanders:
    - Rejects low-confidence MediaPipe handedness (< 0.55)
    - Rejects tiny or full-frame artifacts
    - Verifies anatomical palm-span ratio (wrist-to-middle-MCP vs bounding box)
    - Rejects distant background hands much smaller than the primary signer's face
    """
    if handedness_score < 0.55:
        return False
    if not hasattr(landmarks, "landmark") or len(landmarks.landmark) < 21:
        return False

    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)

    if not ((0.03 <= bbox_w <= 0.98) and (0.03 <= bbox_h <= 0.98)):
        return False

    # Anatomical palm-span check: wrist (0) to middle MCP (9) relative to hand size
    palm_span = np.hypot(
        landmarks.landmark[9].x - landmarks.landmark[0].x,
        landmarks.landmark[9].y - landmarks.landmark[0].y,
    )
    max_dim = max(bbox_w, bbox_h)
    if max_dim > 0 and not (0.10 <= (palm_span / max_dim) <= 0.95):
        return False

    # Background person suppression: ignore tiny hands far behind primary signer
    if face_info is not None:
        face_h = face_info.get("bbox", (0, 0, 0, 0.0))[3]
        if face_h > 0.12 and bbox_h < face_h * 0.25:
            return False

    return True

# =========================
# LANDMARK EXTRACTION & NORMALIZATION
# =========================

def extract_landmarks(results, face_info=None):
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
        cls_list = getattr(results.multi_handedness[idx], 'classification', [])
        h_score = getattr(cls_list[0], 'score', 1.0) if cls_list else 1.0
        if is_valid_human_hand(hand_lms, handedness_score=h_score, face_info=face_info):
            valid_indices.append(idx)

    if len(valid_indices) == 0:
        return None

    left = [0.0] * 63
    right = [0.0] * 63

    if len(valid_indices) == 1:
        # Always map single-hand gestures to the Right slot (63..125) regardless of mirrored handedness
        hand_landmarks = results.multi_hand_landmarks[valid_indices[0]]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        right = coords
    else:
        # Two-hand gestures: assign by wrist x-position (leftmost -> Left slot, rightmost -> Right slot)
        hand_entries = []
        for idx in valid_indices[:2]:
            hand_landmarks = results.multi_hand_landmarks[idx]
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            hand_entries.append((hand_landmarks.landmark[0].x, coords))
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
            velocity = latest_hand_velocity_snapshot

        try:
            # Graph-compiled forward pass through trained LSTM (~1.9ms)
            preds = _fast_infer(tf.convert_to_tensor(input_data, dtype=tf.float32)).numpy()[0]

            # Deterministic hand pose, velocity & facial zone disambiguation
            preds = disambiguate_prediction(preds, hand_lms, face_info, velocity=velocity)

            with state_lock:
                if not hands_in_frame:
                    continue

                prediction_history.append(preds)
                # Exponential-weighted rolling average (recent frames weighted higher)
                hist_list = list(prediction_history)
                weights = np.array([0.5 ** (len(hist_list) - 1 - i) for i in range(len(hist_list))], dtype=np.float32)
                weights /= np.sum(weights)
                avg_preds = np.average(hist_list, axis=0, weights=weights)

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
    cached_face_info = None
    face_cache_age = 0

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break

            # Mirror view for natural interaction (display frame stays untouched)
            frame = cv2.flip(frame, 1)

            # Enhance luminance with CLAHE for MediaPipe detection only
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = _clahe.apply(ycrcb[:, :, 0])
            enhanced_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Full-range face detection with 15-frame occlusion cache (keeps zones stable when hand covers mouth)
            face_results = face_detector.process(rgb)
            face_info = None
            if face_results and face_results.detections:
                # Pick primary signer's face (largest bounding box) to ignore background faces
                det = max(
                    face_results.detections,
                    key=lambda d: (
                        d.location_data.relative_bounding_box.width
                        * d.location_data.relative_bounding_box.height
                    ),
                )
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
                cached_face_info = face_info
                face_cache_age = 0
            elif cached_face_info is not None and face_cache_age < 15:
                face_info = cached_face_info
                face_cache_age += 1

            # NOTE: Palm lines and skeletal drawing are completely removed as requested.
            # The raw camera frame remains clean and natural.

            valid_lms = []
            if results.multi_hand_landmarks:
                handedness_list = results.multi_handedness or []
                for idx, lms in enumerate(results.multi_hand_landmarks):
                    cls_list = getattr(handedness_list[idx], 'classification', []) if idx < len(handedness_list) else []
                    h_score = getattr(cls_list[0], 'score', 1.0) if cls_list else 1.0
                    if is_valid_human_hand(lms, handedness_score=h_score, face_info=face_info):
                        valid_lms.append(lms)

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
                        wrist_history.clear()
                        latest_sequence_snapshot = None
                        latest_hand_lms_snapshot = None
                        latest_face_info_snapshot = None
                        latest_hand_velocity_snapshot = 0.0
            else:
                missing_hand_frames = 0
                with state_lock:
                    was_hands = hands_in_frame
                    hands_in_frame = True

                norm_vec = extract_landmarks(results, face_info=face_info)
                if norm_vec is not None:
                    with seq_lock:
                        # Instant pre-warm: fill 20-frame sequence buffer on frame 1 so inference begins immediately
                        if not was_hands or len(sequence) == 0:
                            sequence.extend([norm_vec] * 20)
                            wrist_history.clear()
                        else:
                            sequence.append(norm_vec)

                        latest_sequence_snapshot = np.expand_dims(np.array(sequence), axis=0).astype(np.float32)
                        latest_hand_lms_snapshot = [
                            [(lm.x, lm.y, lm.z) for lm in lms.landmark]
                            for lms in valid_lms
                        ]
                        # Track wrist velocity (screen-units/frame)
                        w_pt = latest_hand_lms_snapshot[0][0][:2]
                        wrist_history.append(w_pt)
                        if len(wrist_history) >= 2:
                            diffs = [
                                np.linalg.norm(np.array(wrist_history[i]) - np.array(wrist_history[i - 1]))
                                for i in range(1, len(wrist_history))
                            ]
                            latest_hand_velocity_snapshot = float(np.mean(diffs))
                        else:
                            latest_hand_velocity_snapshot = 0.0

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
    global cap, hold_counter, current_sign, last_added_sign, hands_in_frame
    global latest_sequence_snapshot, latest_hand_lms_snapshot, latest_face_info_snapshot, latest_hand_velocity_snapshot
    with state_lock:
        hold_counter = 0
        current_sign = None
        last_added_sign = None
        hands_in_frame = False

    with seq_lock:
        sequence.clear()
        prediction_history.clear()
        wrist_history.clear()
        latest_sequence_snapshot = None
        latest_hand_lms_snapshot = None
        latest_face_info_snapshot = None
        latest_hand_velocity_snapshot = 0.0

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
        return Response(fallback_mp3, mimetype="audio/mpeg", headers={"X-Audio-Fallback": "silent"})

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
    pred_probs = _fast_infer(tf.convert_to_tensor(input_tensor, dtype=tf.float32)).numpy()[0]
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
