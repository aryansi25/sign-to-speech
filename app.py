import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
from collections import deque
from gtts import gTTS
from deep_translator import GoogleTranslator
import tempfile

app = Flask(__name__)
CORS(app)

# =========================
# SUPPORTED LANGUAGES
# =========================
# Display name -> language code.
# These codes work for BOTH Google Translate (text) and gTTS (speech),
# so every language in the dropdown can be translated AND spoken.

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

# The list of names shown in the dropdown on the page
LANGUAGES = list(LANG_CODES.keys())

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("isl_lstm_model.keras")
labels = np.load("labels.npy")

# =========================
# SETTINGS
# =========================

HOLD_FRAMES = 3
CONF_THRESHOLD = 0.40

sequence = deque(maxlen=20)
prediction_history = deque(maxlen=4)
prediction_queue = deque(maxlen=4)

# =========================
# GLOBAL STATE
# =========================

cap = None
sentence_words = []
current_sign = None
confidence = 0
pending_word = None
hold_counter = 0

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
# VIDEO STREAM
# =========================

def generate_frames():

    global cap,current_sign,confidence,pending_word,hold_counter

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

    frame_counter = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame,1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        landmarks = extract_landmarks(results)

        frame_counter += 1

        if frame_counter % 2 == 0:

            sequence.append(landmarks)

            if len(sequence) == 20:

                input_data = np.expand_dims(sequence, axis=0)

                prediction = model.predict(input_data, verbose=0)[0]

                prediction_history.append(prediction)

                avg_pred = np.mean(prediction_history, axis=0)

                idx = np.argmax(avg_pred)
                conf = avg_pred[idx]
                label = labels[idx]

                prediction_queue.append(label)

                if len(prediction_queue) == prediction_queue.maxlen:
                    label = max(set(prediction_queue), key=prediction_queue.count)

                if conf > CONF_THRESHOLD and label != "NO_SIGN":

                    current_sign = label
                    confidence = round(conf*100,2)

                    hold_counter += 1

                    if hold_counter >= HOLD_FRAMES:
                        pending_word = label

                else:

                    hold_counter = max(0,hold_counter-1)

                    if hold_counter == 0:
                        current_sign = None
                        confidence = 0

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# =========================
# ROUTES
# =========================

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/state")
def get_state():

    return jsonify({
        "current_sign": current_sign,
        "confidence": confidence,
        "pending_word": pending_word,
        "sentence_words": sentence_words,
        "sentence": " ".join(sentence_words)
    })

@app.route("/api/confirm_word", methods=["POST"])
def confirm_word():

    global pending_word

    if pending_word:
        sentence_words.append(pending_word)

    pending_word = None

    return jsonify({"status":"ok"})

@app.route("/api/clear_sentence", methods=["POST"])
def clear_sentence():

    sentence_words.clear()

    return jsonify({"status":"cleared"})

@app.route("/api/add_custom_text", methods=["POST"])
def add_custom_text():

    # Adds a typed word or name (from the "Type a name or word" box) to the sentence
    text = request.json.get("text", "").strip()

    if text:
        sentence_words.append(text)

    return jsonify({"status":"ok"})

@app.route("/api/remove_last_word", methods=["POST"])
def remove_last_word():

    # Powers the "Undo" button: removes the most recently added word
    if sentence_words:
        sentence_words.pop()

    return jsonify({"status":"removed"})

@app.route("/api/skip_word", methods=["POST"])
def skip_word():

    # Powers the "Skip" button: discards the word currently waiting to be confirmed
    global pending_word

    pending_word = None

    return jsonify({"status":"skipped"})

@app.route("/api/stop_camera", methods=["POST"])
def stop_camera():

    # Powers the "Stop" button: releases the webcam so the camera light turns off
    global cap

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    return jsonify({"status":"stopped"})

# =========================
# TRANSLATE
# =========================

@app.route("/api/translate", methods=["POST"])
def translate():

    sentence = request.json.get("sentence")
    language = request.json.get("language")

    if sentence:
        sentence = sentence.replace("_"," ")

    # Convert the dropdown name (e.g. "Hindi") into its code (e.g. "hi").
    # If something unexpected comes in, fall back to using it as-is.
    target_code = LANG_CODES.get(language, language)

    translated = GoogleTranslator(source="auto", target=target_code).translate(sentence)

    return jsonify({"translated":translated})

# =========================
# SPEAK
# =========================

@app.route("/api/speak", methods=["POST"])
def speak():

    sentence = request.json.get("sentence")
    language = request.json.get("language")

    sentence = sentence.replace("_"," ")

    # The speech tool (gTTS) needs the short code (e.g. "en"), not "English".
    lang_code = LANG_CODES.get(language, language)

    try:
        # First translate the sentence into the chosen language, THEN speak that.
        # (Without this, gTTS would read the English words using the language's
        #  voice, which sounds like English with an accent instead of real speech.)
        text_to_speak = GoogleTranslator(source="auto", target=lang_code).translate(sentence)

        tts = gTTS(text=text_to_speak, lang=lang_code)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

        tts.save(temp_file.name)

        return send_file(temp_file.name, mimetype="audio/mpeg")

    except Exception:
        # If speech can't be produced (e.g. no internet), tell the page kindly
        return jsonify({"error": f"Speech is not available for {language} right now."}), 400

# =========================

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/app")
def index():
    return render_template("index.html", languages=LANGUAGES)

# =========================

if __name__ == "__main__":
    app.run(debug=True)