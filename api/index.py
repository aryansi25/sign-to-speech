import csv
import io
import os
import sys
import time
import requests
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from gtts import gTTS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vocabulary import get_vocab_translation

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
CORS(app)

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
    {"id": "BSL", "name": "British Sign Language (BSL)", "status": "disabled", "badge": "Dataset Required"},
]


def _load_valid_signs():
    csv_path = os.path.join(BASE_DIR, "labels.csv")
    signs = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = (row.get("label") or row.get("Label") or "").strip()
                if not lbl and len(row) > 0:
                    lbl = list(row.values())[-1].strip()
                if lbl and lbl != "NO_SIGN":
                    signs.append(lbl)
    return signs


VALID_SIGNS = _load_valid_signs()
VALID_SIGNS_SET = set(VALID_SIGNS)
audio_cache = {}
translation_cache = {}


def translate_text(text, target_lang_code):
    if not text or target_lang_code == "en":
        return text

    cache_key = f"{text.strip().upper()}::{target_lang_code}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    vocab_hit = get_vocab_translation(text, target_lang_code)
    if vocab_hit:
        translation_cache[cache_key] = vocab_hit
        return vocab_hit

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "en", "tl": target_lang_code, "dt": "t", "q": text}
        r = requests.get(url, params=params, timeout=3)
        if r.status_code == 200:
            data = r.json()
            result = "".join([part[0] for part in data[0] if part[0]])
            if result:
                translation_cache[cache_key] = result
                return result
    except Exception:
        pass

    words = text.strip().split()
    if len(words) > 1:
        translated_words = []
        all_found = True
        for w in words:
            wh = get_vocab_translation(w, target_lang_code)
            if wh:
                translated_words.append(wh)
            else:
                all_found = False
                translated_words.append(w)
        if all_found or any(tw != orig for tw, orig in zip(translated_words, words)):
            joined = " ".join(translated_words)
            translation_cache[cache_key] = joined
            return joined

    return text


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/app")
def index():
    return render_template(
        "index.html",
        languages=LANGUAGES,
        sign_languages=SIGN_LANGUAGES,
        vercel_mode=True,
    )


@app.route("/test")
def test_runner():
    return render_template("test_runner.html")


@app.route("/api/state")
def get_state():
    return jsonify({
        "current_sign": None,
        "current_sign_translated": None,
        "confidence": 0,
        "hold_progress": 0,
        "hands_in_frame": False,
        "event_id": 0,
        "new_word": None,
        "new_word_translated": None,
        "whitelist_count": len(VALID_SIGNS),
        "valid_signs": VALID_SIGNS,
        "vercel_mode": True,
    })


@app.route("/api/set_sign_language", methods=["POST"])
def set_sign_language():
    data = request.get_json(silent=True) or {}
    return jsonify({"status": "ok", "sign_language": data.get("sign_language", "ISL")})


@app.route("/api/stop_camera", methods=["POST"])
def stop_camera():
    return jsonify({"status": "stopped"})


@app.route("/api/clear_conversation", methods=["POST"])
@app.route("/api/clear_sentence", methods=["POST"])
def clear_conversation():
    return jsonify({"status": "cleared"})


@app.route("/api/add_custom", methods=["POST"])
def add_custom():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    return jsonify({"status": "added", "text": text})


@app.route("/api/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "").strip()
    language = data.get("language", "English")
    if not sentence:
        return jsonify({"translated": ""})
    lang_code = LANG_CODES.get(language, "en")
    translated = translate_text(sentence, lang_code)
    return jsonify({"translated": translated})


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
    if cache_key in audio_cache:
        return Response(audio_cache[cache_key], mimetype="audio/mpeg")

    try:
        text_to_speak = translate_text(sentence, tts_lang) if tts_lang != "en" else sentence
        tts = gTTS(text=text_to_speak, lang=tts_lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        audio_bytes = fp.getvalue()
        audio_cache[cache_key] = audio_bytes
        return Response(audio_bytes, mimetype="audio/mpeg")
    except Exception:
        fallback_mp3 = (
            b"ID3\x03\x00\x00\x00\x00\x00\x00"
            + (b"\xff\xfb\x90d" + b"\x00" * 414) * 5
        )
        return Response(fallback_mp3, mimetype="audio/mpeg", headers={"X-Audio-Fallback": "silent"})


@app.route("/api/benchmark_translations")
def benchmark_translations():
    lang = request.args.get("lang", "Hindi")
    lang_code = LANG_CODES.get(lang, "hi")
    return jsonify({s: (get_vocab_translation(s, lang_code) or s) for s in VALID_SIGNS})


@app.route("/api/feedback_correct", methods=["POST"])
def api_feedback_correct():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "").strip().upper().replace(" ", "_")
    if not word or word not in VALID_SIGNS_SET:
        return jsonify({"error": "No valid detected word to confirm"}), 400
    return jsonify({
        "status": "calibrated_browser",
        "word": word,
        "total_samples": int(data.get("local_count", 1)),
    })


@app.route("/api/feedback_stats")
def api_feedback_stats():
    return jsonify({"total_samples": 0, "per_class": {}, "vercel_mode": True})
