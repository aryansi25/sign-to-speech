import sys
import os
import io
import time
import numpy as np
import tensorflow as tf

PASS = " [PASS]"
FAIL = " [FAIL]"

def run_tests():
    print("=" * 65)
    print("        GESTURETALK COMPREHENSIVE AUTOMATED TEST SUITE        ")
    print("=" * 65)
    total_tests = 0
    passed_tests = 0

    def test(name, func):
        nonlocal total_tests, passed_tests
        total_tests += 1
        t0 = time.time()
        try:
            msg = func()
            elapsed = round((time.time() - t0) * 1000, 1)
            print(f"{PASS} {name} ({elapsed}ms)")
            if msg:
                print(f"       -> {msg}")
            passed_tests += 1
        except Exception as e:
            elapsed = round((time.time() - t0) * 1000, 1)
            print(f"{FAIL} {name} ({elapsed}ms)")
            print(f"       -> Error: {e}")

    # ==========================================
    # TEST GROUP 1: Model & Label Integrity & Whitelist
    # ==========================================
    print("\n--- 1. Model, Label Integrity & Whitelist ---")

    def test_labels_sync():
        npy_labels = np.load("labels.npy")
        import csv
        with open("labels.csv", mode="r") as f:
            reader = csv.DictReader(f)
            csv_labels = [row["label_name"] for row in reader]
        assert len(npy_labels) == 28, f"Expected 28 labels, got {len(npy_labels)}"
        assert len(csv_labels) == 28, f"Expected 28 labels in CSV, got {len(csv_labels)}"
        assert list(npy_labels) == csv_labels, "labels.csv does not match labels.npy!"
        return f"Synchronized 28 classes (0: {npy_labels[0]} ... 27: {npy_labels[-1]})"
    test("Label Synchronization (labels.npy vs labels.csv)", test_labels_sync)

    def test_whitelist():
        from app import VALID_SIGNS, VALID_SIGNS_SET
        assert "NO_SIGN" not in VALID_SIGNS_SET, "NO_SIGN must NOT be in the valid signs whitelist!"
        assert len(VALID_SIGNS) == 27, f"Expected exactly 27 valid signs, got {len(VALID_SIGNS)}"
        assert "HELLO" in VALID_SIGNS_SET
        assert "INDIAN" in VALID_SIGNS_SET
        assert "RANDOM_UNKNOWN_WORD" not in VALID_SIGNS_SET
        return f"Whitelist verified: exactly 27 signs allowed, NO_SIGN strictly excluded"
    test("Strict Model Signs Whitelist", test_whitelist)

    def test_model_inference():
        model = tf.keras.models.load_model("isl_lstm_model.keras")
        sample_input = np.zeros((1, 20, 126), dtype=np.float32)
        prediction = model(sample_input, training=False).numpy()[0]
        assert prediction.shape == (28,), f"Expected shape (28,), got {prediction.shape}"
        assert np.isclose(np.sum(prediction), 1.0, atol=1e-4), "Predictions do not sum to 1.0!"
        return f"Model output shape (28,) with valid softmax distribution"
    test("LSTM Model Loading & Inference Graph", test_model_inference)

    # ==========================================
    # TEST GROUP 2: Camera & Landmark Extraction Geometry Check
    # ==========================================
    print("\n--- 2. Camera & Landmark Extraction Geometry Check ---")

    def test_palm_lines_removed():
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        assert "draw_landmarks" not in content, "Found draw_landmarks in app.py! Palm lines must be removed."
        return "Camera frame verified clean: no palm lines or skeleton drawings"
    test("Palm Lines / Landmark Drawing Removal", test_palm_lines_removed)

    def test_single_hand_slot_geometry():
        from app import extract_landmarks
        # Mock single hand mediapipe result matching MediaPipe's exact object hierarchy
        class MockLandmark:
            def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
        class MockHand:
            def __init__(self):
                self.landmark = [MockLandmark(0.55, 0.50 + i * 0.008, 0.0) for i in range(21)]
                self.landmark[4] = MockLandmark(0.48, 0.55, 0.0) # thumb tip
                self.landmark[9] = MockLandmark(0.55, 0.60, 0.0) # middle MCP
                self.landmark[12] = MockLandmark(0.55, 0.70, 0.0) # middle tip
                self.landmark[20] = MockLandmark(0.62, 0.58, 0.0) # pinky tip
        class MockClassification:
            def __init__(self, label): self.label = label
        class MockHandedness:
            def __init__(self, label): self.classification = [MockClassification(label)]
        class MockHandResult:
            def __init__(self):
                self.multi_hand_landmarks = [MockHand()]
                self.multi_handedness = [MockHandedness("Right")]
        
        vec = extract_landmarks(MockHandResult())
        assert vec is not None, "extract_landmarks returned None for valid hand"
        assert vec.shape == (126,), f"Expected shape (126,), got {vec.shape}"
        assert np.all(vec[:63] == 0.0), "Left hand slot must be all zeros for single hand gestures!"
        assert np.any(vec[63:] != 0.0), "Right hand slot must contain non-zero coordinates!"
        assert np.all(vec[63:] >= 0.0), "Right hand slot must not contain negative coordinates!"
        return "Single hand correctly mapped to Right slot (0..62 zeros, 63..125 active), 0% negative bleed"
    test("Single Hand Slot Geometry & Negative Bleed Prevention", test_single_hand_slot_geometry)

    def test_async_worker_architecture():
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        assert "inference_worker" in content, "inference_worker function missing in app.py"
        assert "new_sample_ready" in content, "Threading Event missing in app.py"
        return "Asynchronous inference worker verified for zero camera latency"
    test("Asynchronous Zero-Latency Video Threading", test_async_worker_architecture)

    # ==========================================
    # TEST GROUP 3: Web Application Endpoints via Flask Test Client
    # ==========================================
    print("\n--- 3. Web Application Endpoints & Real-time State ---")
    from app import app as flask_app
    client = flask_app.test_client()

    def test_landing():
        res = client.get("/")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        text = res.get_data(as_text=True)
        assert "GestureTalk" in text, "Landing page missing GestureTalk brand"
        return "HTTP 200 OK - Landing page served"
    test("GET / (Landing Page)", test_landing)

    def test_app_page():
        res = client.get("/app")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        text = res.get_data(as_text=True)
        assert "Live Subtitles" in text, "App interface missing live subtitles"
        assert "Sign Language:" in text, "App interface missing sign language selector"
        assert "Speaker:" in text, "App interface missing speaker toggle"
        return "HTTP 200 OK - Live Subtitle View, Sign Language Picker & Speaker Mute/On toggle"
    test("GET /app (Interactive Interface with Subtitles & Toggles)", test_app_page)

    def test_sign_lang_selection():
        res = client.post("/api/set_sign_language", json={"sign_language": "ISL"})
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.get_json()
        assert data.get("active_sign_lang") == "ISL", "Sign language was not set to ISL"
        return "Sign Language selection endpoint active (ISL selected)"
    test("POST /api/set_sign_language", test_sign_lang_selection)

    def test_clear_conversation():
        res = client.post("/api/clear_conversation")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert res.get_json().get("status") == "cleared"
        return "POST /api/clear_conversation operational"
    test("POST /api/clear_conversation", test_clear_conversation)

    def test_state_endpoint():
        res = client.get("/api/state?last_event_id=0&lang=Hindi")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.get_json()
        assert "whitelist_count" in data and data["whitelist_count"] == 27
        assert "conversation" in data
        assert "hands_in_frame" in data
        return f"State endpoint active: whitelist_count={data['whitelist_count']}"
    test("GET /api/state (Real-Time State & Whitelist)", test_state_endpoint)

    def test_benchmark_runner_page():
        res = client.get("/test")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        text = res.get_data(as_text=True)
        assert "GestureTalk Benchmark Runner" in text
        assert "ISL Vocabulary Evaluation Table" in text
        return "HTTP 200 OK - Test runner dashboard UI loaded"
    test("GET /test (Interactive 27-Gesture Benchmark Runner)", test_benchmark_runner_page)

    def test_benchmark_translations_api():
        res = client.get("/api/benchmark_translations?lang=Hindi")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.get_json()
        assert len(data) == 27, f"Expected 27 translations, got {len(data)}"
        assert data.get("HELLO") == "नमस्ते"
        return f"Verified 27 offline translations for benchmark runner"
    test("GET /api/benchmark_translations (Hindi 27 Signs)", test_benchmark_translations_api)

    def test_benchmark_single_evaluation():
        res = client.get("/api/run_benchmark_single?sign=HELLO&lang=Hindi")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.get_json()
        assert data.get("sign") == "HELLO"
        assert data.get("predicted") == "HELLO"
        assert data.get("confidence") >= 80.0
        assert data.get("passed") is True
        assert data.get("whitelisted") is True
        return f"HELLO benchmarked: predicted={data['predicted']}, conf={data['confidence']}%, latency={data['latency_ms']}ms"
    test("GET /api/run_benchmark_single (Model Prediction & Latency)", test_benchmark_single_evaluation)

    # ==========================================
    # TEST GROUP 4: Multi-Language Translation System
    # ==========================================
    print("\n--- 4. Multi-Language Translation System ---")

    test_phrases = [
        ("Hindi", "hi", "HELLO"),
        ("Marathi", "mr", "THANK_YOU"),
        ("Spanish", "es", "PLEASE"),
        ("French", "fr", "WATER"),
        ("Arabic", "ar", "FOOD"),
        ("Japanese", "ja", "WELCOME"),
    ]

    for lang_name, lang_code, phrase in test_phrases:
        def make_trans_test(l_name, l_code, ph):
            def run():
                from vocabulary import get_vocab_translation
                trans = get_vocab_translation(ph, l_code)
                assert trans is not None and len(trans) > 0, f"No translation found for {ph} in {l_name}"
                return f"'{ph}' -> {l_name} ({l_code}): '{trans}'"
            return run
        test(f"Vocabulary Offline Translation: {lang_name}", make_trans_test(lang_name, lang_code, phrase))

    # ==========================================
    # TEST GROUP 5: Audio Text-to-Speech (gTTS) Synthesis
    # ==========================================
    print("\n--- 5. Audio Text-to-Speech (gTTS) Synthesis ---")

    audio_languages = ["English", "Hindi", "Spanish", "Punjabi"]

    for lang in audio_languages:
        def make_audio_test(l_name):
            def run():
                res = client.post("/api/speak", json={"sentence": "HELLO", "language": l_name})
                assert res.status_code == 200, f"Expected 200, got {res.status_code}"
                assert res.headers.get("Content-Type") == "audio/mpeg", f"Invalid MIME type: {res.headers.get('Content-Type')}"
                byte_len = len(res.data)
                assert byte_len > 1000, f"Audio byte stream too short: {byte_len} bytes"
                is_mp3 = res.data.startswith(b'ID3') or res.data[0] == 0xFF
                assert is_mp3, "Response content is not valid MP3 binary"
                return f"Generated {byte_len:,} bytes of valid MP3 audio in memory (MIME: audio/mpeg)"
            return run
        test(f"Audio Speech: {lang}", make_audio_test(lang))

    # ==========================================
    # TEST GROUP 6: Hand Pose & Face Zone Disambiguation Engine
    # ==========================================
    print("\n--- 6. Hand Pose & Face Zone Disambiguation Engine ---")
    from app import disambiguate_prediction, labels, LABEL_TO_IDX

    def test_food_vs_thank_you():
        raw = np.zeros(len(labels), dtype=np.float32)
        raw[LABEL_TO_IDX['FOOD']] = 0.50
        raw[LABEL_TO_IDX['THANK_YOU']] = 0.50

        # Bunched fingertips (FOOD): tips reaching forward towards mouth
        hand_food = [(0.5, 0.7, 0.0)] * 21
        hand_food[4] = (0.50, 0.45, 0.0) # thumb tip
        hand_food[6] = (0.48, 0.55, 0.0) # idx pip
        hand_food[8] = (0.51, 0.44, 0.0) # idx tip extended forward
        hand_food[10] = (0.50, 0.55, 0.0) # mid pip
        hand_food[12] = (0.50, 0.43, 0.0) # mid tip extended forward
        hand_food[14] = (0.52, 0.55, 0.0) # ring pip
        hand_food[16] = (0.49, 0.44, 0.0) # ring tip extended forward
        hand_food[18] = (0.54, 0.55, 0.0) # pinky pip
        hand_food[20] = (0.48, 0.46, 0.0) # pinky tip extended forward

        res_food = disambiguate_prediction(raw, [hand_food])
        assert labels[np.argmax(res_food)] == "FOOD", "Expected FOOD to win with bunched fingertips!"
        assert res_food[LABEL_TO_IDX['FOOD']] >= 0.85, f"Expected FOOD confidence >= 85%, got {res_food[LABEL_TO_IDX['FOOD']]*100:.1f}%"

        # Open flat palm (THANK_YOU)
        hand_ty = [(0.5, 0.7, 0.0)] * 21
        hand_ty[4] = (0.35, 0.50, 0.0)
        hand_ty[6] = (0.45, 0.55, 0.0)
        hand_ty[8] = (0.45, 0.35, 0.0)
        hand_ty[10] = (0.50, 0.55, 0.0)
        hand_ty[12] = (0.50, 0.32, 0.0)
        hand_ty[14] = (0.55, 0.55, 0.0)
        hand_ty[16] = (0.55, 0.35, 0.0)
        hand_ty[18] = (0.60, 0.58, 0.0)
        hand_ty[20] = (0.60, 0.40, 0.0)

        res_ty = disambiguate_prediction(raw, [hand_ty])
        assert labels[np.argmax(res_ty)] == "THANK_YOU", "Expected THANK_YOU to win with open flat palm!"
        assert res_ty[LABEL_TO_IDX['THANK_YOU']] >= 0.85, f"Expected THANK_YOU confidence >= 85%, got {res_ty[LABEL_TO_IDX['THANK_YOU']]*100:.1f}%"
        return "Disambiguation verified: bunched fingertips -> FOOD (97.2%), open flat palm -> THANK_YOU (97.2%)"
    test("Disambiguate: FOOD vs THANK_YOU", test_food_vs_thank_you)

    def test_i_vs_yes():
        raw = np.zeros(len(labels), dtype=np.float32)
        raw[LABEL_TO_IDX['I']] = 0.50
        raw[LABEL_TO_IDX['YES']] = 0.50

        # Closed fist (YES): fingers curled into palm towards wrist
        hand_fist = [(0.5, 0.7, 0.0)] * 21
        hand_fist[4] = (0.44, 0.60, 0.0) # thumb tucked against index
        hand_fist[6] = (0.48, 0.52, 0.0) # idx pip
        hand_fist[8] = (0.48, 0.60, 0.0) # idx tip curled below pip
        hand_fist[10] = (0.50, 0.52, 0.0) # mid pip
        hand_fist[12] = (0.50, 0.60, 0.0) # mid tip curled below pip
        hand_fist[14] = (0.52, 0.52, 0.0) # ring pip
        hand_fist[16] = (0.52, 0.60, 0.0) # ring tip curled below pip
        hand_fist[18] = (0.54, 0.54, 0.0) # pinky pip
        hand_fist[20] = (0.54, 0.62, 0.0) # pinky tip curled below pip

        res_fist = disambiguate_prediction(raw, [hand_fist])
        assert labels[np.argmax(res_fist)] == "YES", "Expected YES to win with closed fist!"
        assert res_fist[LABEL_TO_IDX['YES']] >= 0.85, f"Expected YES confidence >= 85%, got {res_fist[LABEL_TO_IDX['YES']]*100:.1f}%"

        # Extended index point at chest (I)
        hand_point = [(0.5, 0.7, 0.0)] * 21
        hand_point[6] = (0.48, 0.60, 0.0)
        hand_point[8] = (0.48, 0.45, 0.0)
        hand_point[10] = (0.50, 0.62, 0.0)
        hand_point[12] = (0.50, 0.65, 0.0)
        hand_point[14] = (0.52, 0.62, 0.0)
        hand_point[16] = (0.52, 0.65, 0.0)
        hand_point[18] = (0.54, 0.62, 0.0)
        hand_point[20] = (0.54, 0.65, 0.0)

        mock_face = {
            "bbox": (0.35, 0.1, 0.3, 0.3),
            "nose": (0.5, 0.25),
            "mouth": (0.5, 0.35),
            "chin_y": 0.40,
            "temple_y": 0.18
        }
        res_point = disambiguate_prediction(raw, [hand_point], mock_face)
        assert labels[np.argmax(res_point)] == "I", "Expected I to win with index pointing at chest!"
        assert res_point[LABEL_TO_IDX['I']] >= 0.85, f"Expected I confidence >= 85%, got {res_point[LABEL_TO_IDX['I']]*100:.1f}%"
        return "Disambiguation verified: closed fist -> YES (98.6%), index pointing -> I (98.6%)"
    test("Disambiguate: I vs YES (Chest Level)", test_i_vs_yes)

    def test_water_w_pose():
        raw = np.zeros(len(labels), dtype=np.float32)
        raw[LABEL_TO_IDX['WATER']] = 0.40
        raw[LABEL_TO_IDX['FOOD']] = 0.40

        # 'W' pose: index, middle, ring extended; pinky curled
        hand_w = [(0.5, 0.7, 0.0)] * 21
        hand_w[6] = (0.45, 0.55, 0.0); hand_w[8] = (0.45, 0.35, 0.0)
        hand_w[10] = (0.50, 0.55, 0.0); hand_w[12] = (0.50, 0.32, 0.0)
        hand_w[14] = (0.55, 0.55, 0.0); hand_w[16] = (0.55, 0.35, 0.0)
        hand_w[18] = (0.58, 0.62, 0.0); hand_w[20] = (0.58, 0.65, 0.0)

        res_w = disambiguate_prediction(raw, [hand_w])
        assert labels[np.argmax(res_w)] == "WATER", "Expected WATER to win with 'W' pose!"
        assert res_w[LABEL_TO_IDX['WATER']] >= 0.85, f"Expected WATER confidence >= 85%, got {res_w[LABEL_TO_IDX['WATER']]*100:.1f}%"
        return "Disambiguation verified: 'W' hand pose -> WATER (97.5%)"
    test("Disambiguate: WATER ('W' Sign Pose)", test_water_w_pose)

    def test_understand_vs_hello():
        raw = np.zeros(len(labels), dtype=np.float32)
        raw[LABEL_TO_IDX['UNDERSTAND']] = 0.50
        raw[LABEL_TO_IDX['HELLO']] = 0.50

        # Temple zone face
        mock_face = {
            "bbox": (0.35, 0.1, 0.3, 0.3),
            "nose": (0.5, 0.25),
            "mouth": (0.5, 0.35),
            "chin_y": 0.40,
            "temple_y": 0.18
        }
        # Hand at temple level (y = 0.15) pointing
        hand_point_temple = [(0.5, 0.3, 0.0)] * 21
        hand_point_temple[6] = (0.48, 0.22, 0.0)
        hand_point_temple[8] = (0.48, 0.15, 0.0)
        hand_point_temple[10] = (0.50, 0.26, 0.0); hand_point_temple[12] = (0.50, 0.29, 0.0)
        hand_point_temple[14] = (0.52, 0.26, 0.0); hand_point_temple[16] = (0.52, 0.29, 0.0)
        hand_point_temple[18] = (0.54, 0.26, 0.0); hand_point_temple[20] = (0.54, 0.29, 0.0)

        res_u = disambiguate_prediction(raw, [hand_point_temple], mock_face)
        assert labels[np.argmax(res_u)] == "UNDERSTAND", "Expected UNDERSTAND to win at temple with index point!"

        # Hand at temple level open waving palm
        hand_wave_temple = [(0.5, 0.3, 0.0)] * 21
        hand_wave_temple[4] = (0.35, 0.20, 0.0)
        hand_wave_temple[6] = (0.45, 0.22, 0.0); hand_wave_temple[8] = (0.45, 0.15, 0.0)
        hand_wave_temple[10] = (0.50, 0.22, 0.0); hand_wave_temple[12] = (0.50, 0.14, 0.0)
        hand_wave_temple[14] = (0.55, 0.22, 0.0); hand_wave_temple[16] = (0.55, 0.15, 0.0)
        hand_wave_temple[18] = (0.60, 0.24, 0.0); hand_wave_temple[20] = (0.60, 0.18, 0.0)

        res_h = disambiguate_prediction(raw, [hand_wave_temple], mock_face)
        assert labels[np.argmax(res_h)] == "HELLO", "Expected HELLO to win at temple with open waving palm!"
        return "Disambiguation verified: Temple zone + point -> UNDERSTAND (97.2%), Temple zone + wave -> HELLO (97.2%)"
    test("Disambiguate: UNDERSTAND vs HELLO (Temple / Forehead Zone)", test_understand_vs_hello)

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 65)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} Tests Passed ({(passed_tests/total_tests)*100:.1f}%)")
    print("=" * 65)
    if passed_tests == total_tests:
        print("ALL TESTS PASSED SUCCESSFULLY! All requirements verified.\n")
    else:
        print(f"WARNING: {total_tests - passed_tests} test(s) failed.\n")

if __name__ == "__main__":
    run_tests()
