from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os
import threading
import h5py

app = Flask(__name__)

# ==================== CONFIGURATION ====================
ALPHABET_MODEL_PATH = 'models/alphabet_model.h5'
DIGIT_MODEL_PATH    = 'models/digit_model.h5'

MAX_SEQ_LEN  = 20
MIN_FRAMES   = 10
NUM_FEATURES = 62   # 31 landmarks × 2

# ==================== MEDIAPIPE SETUP ====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Lip landmark indices (MediaPipe Face Mesh)
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
             324, 318, 402, 317, 14, 87, 178, 88, 95]
LIP_INDICES = sorted(list(set(OUTER_LIP + INNER_LIP)))

# Lip open/close detection indices
LIP_TOP_INNER    = 13
LIP_BOTTOM_INNER = 14
LIP_LEFT_CORNER  = 78
LIP_RIGHT_CORNER = 308
LIP_OPEN_THRESHOLD = 0.04   # MAR threshold; tune between 0.02–0.06

DIGIT_CLASSES = [str(i) for i in range(10)]
ALPHA_CLASSES = [chr(ord('A') + i) for i in range(26)]

# ==================== GLOBAL VARIABLES ====================
camera      = None
camera_lock = threading.Lock()

landmark_buffer = deque(maxlen=MAX_SEQ_LEN)

current_mode        = "alphabet"   # "alphabet" | "digit"
current_prediction  = "-"
current_confidence  = 0.0
is_detecting        = False
is_lip_open         = False

# ==================== LOAD MODELS ====================
alphabet_model = None
digit_model    = None

models_status = {
    "alphabet": False,
    "digit":    False,
}


def _build_model(num_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, BatchNormalization, MaxPooling1D, Dropout,
        Bidirectional, LSTM, Dense
    )
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same',
               input_shape=(MAX_SEQ_LEN, NUM_FEATURES)),
        BatchNormalization(momentum=0.99, epsilon=0.001),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    return model


def _get_all_arrays(path):
    arrays = []
    with h5py.File(path, 'r') as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim > 0:
                arrays.append((name, np.array(obj)))
        f.visititems(_visit)
    return arrays


def _load_weights_positional(model, path):
    all_arrays    = _get_all_arrays(path)
    needed_shapes = [tuple(w.shape) for w in model.weights]

    matched = []
    arr_idx = 0
    for shape in needed_shapes:
        found = False
        while arr_idx < len(all_arrays):
            name, arr = all_arrays[arr_idx]
            arr_idx += 1
            if arr.shape == shape:
                matched.append(arr)
                found = True
                break
        if not found:
            print(f"   ❌ Could not find array with shape {shape}")
            return False

    model.set_weights(matched)
    w0 = matched[0]
    if w0.std() < 1e-6:
        print("   ⚠️  First weight tensor near-zero — weights may not have loaded correctly!")
        return False

    print(f"   ✅ Loaded {len(matched)} weight tensors. "
          f"First weight: mean={w0.mean():.4f}, std={w0.std():.4f}")
    return True


def _load_single_model(path, num_classes):
    print(f"\n[..] Loading: {path}")
    if not os.path.exists(path):
        print(f"   File not found.")
        return None

    model = _build_model(num_classes)

    try:
        model.load_weights(path)
        w0 = model.weights[0].numpy()
        if w0.std() > 1e-6:
            print(f"   ✅ load_weights() succeeded. std={w0.std():.4f}")
            return model
        else:
            print("   ⚠️  Weights look zero — trying positional load.")
    except Exception as e:
        print(f"   load_weights() failed: {e} — trying positional load.")

    model2 = _build_model(num_classes)
    if _load_weights_positional(model2, path):
        return model2

    print(f"   ❌ All loading strategies failed for {path}")
    return None


print("\nLoading Deep Learning models...")
alphabet_model = _load_single_model(ALPHABET_MODEL_PATH, 26)
models_status["alphabet"] = alphabet_model is not None
if alphabet_model: print("✓ Alphabet model loaded")

digit_model = _load_single_model(DIGIT_MODEL_PATH, 10)
models_status["digit"] = digit_model is not None
if digit_model: print("✓ Digit model loaded")

# ==================== LIP OPEN/CLOSE DETECTION ====================

def check_lip_open(landmarks, img_w, img_h):
    """
    Determine whether lips are open using the mouth aspect ratio (MAR).
    Returns (bool, float): is_open, mar_value
    """
    from scipy.spatial.distance import euclidean

    top    = landmarks.landmark[LIP_TOP_INNER]
    bottom = landmarks.landmark[LIP_BOTTOM_INNER]
    left   = landmarks.landmark[LIP_LEFT_CORNER]
    right  = landmarks.landmark[LIP_RIGHT_CORNER]

    top_pt    = np.array([top.x    * img_w, top.y    * img_h])
    bottom_pt = np.array([bottom.x * img_w, bottom.y * img_h])
    left_pt   = np.array([left.x   * img_w, left.y   * img_h])
    right_pt  = np.array([right.x  * img_w, right.y  * img_h])

    opening = euclidean(top_pt, bottom_pt)
    width   = euclidean(left_pt, right_pt)

    if width < 1e-6:
        return False, 0.0

    mar = opening / width
    return mar >= LIP_OPEN_THRESHOLD, mar


# ==================== FEATURE EXTRACTION ====================

def normalize_landmarks(landmarks):
    """Normalize lip landmark coordinates to [-1, 1]."""
    lms      = np.array(landmarks, dtype=np.float32)
    centroid = np.mean(lms, axis=0)
    centered = lms - centroid
    max_val  = np.max(np.abs(centered))
    return (centered / max_val if max_val > 0 else centered).flatten()


# ==================== PREDICTION ====================

def predict_from_buffer(buf, model, classes):
    """Pad buffer to MAX_SEQ_LEN and run model inference."""
    if model is None:
        return None, None
    n = len(buf)
    if n < MIN_FRAMES:
        return None, None

    seq    = list(buf)
    padded = np.zeros((MAX_SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    padded[:n] = seq[:MAX_SEQ_LEN]

    preds = model.predict(padded[np.newaxis, ...], verbose=0)[0]
    idx   = int(np.argmax(preds))
    conf  = float(preds[idx])

    print(f"   predict: class={classes[idx]} conf={conf:.3f}  "
          f"top3={sorted(enumerate(preds), key=lambda x: -x[1])[:3]}")
    return classes[idx], conf


# ==================== VIDEO PROCESSING ====================

def generate_frames():
    """Generate annotated MJPEG frames — matches the working RF app's structure."""
    global camera, landmark_buffer
    global current_prediction, current_confidence
    global current_mode, is_detecting, is_lip_open

    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0

    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
            success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(rgb_frame)

        # Mode indicator (top-left)
        mode_color = (0, 255, 0) if is_detecting else (128, 128, 128)
        cv2.putText(frame, f"Mode: {current_mode.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Lip open/closed check
            lip_open, mar = check_lip_open(landmarks, w, h)
            is_lip_open   = lip_open

            lip_state_color = (0, 255, 0) if lip_open else (0, 0, 255)
            cv2.putText(frame, f"Lip: {'OPEN' if lip_open else 'CLOSED'}  MAR:{mar:.3f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lip_state_color, 2)

            # Clear buffer & reset prediction when lips close
            if not lip_open:
                landmark_buffer.clear()
                current_prediction = "-"
                current_confidence = 0.0

            # Draw GREEN dots + polyline on lip landmarks
            lip_points = []
            for i in LIP_INDICES:
                lm = landmarks.landmark[i]
                px, py = int(lm.x * w), int(lm.y * h)
                lip_points.append([px, py])
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

            lip_pts_array = np.array(lip_points, dtype=np.int32)
            cv2.polylines(frame, [lip_pts_array], True, (0, 255, 0), 1)

            # Collect feature vectors and predict when lips are open
            if lip_open:
                # Build raw (x, y) list for feature extraction
                feature_points = []
                for i in LIP_INDICES:
                    lm = landmarks.landmark[i]
                    feature_points.append([lm.x * w, lm.y * h])

                fv = normalize_landmarks(feature_points)
                landmark_buffer.append(fv)

                if is_detecting and frame_count % 15 == 0 and len(landmark_buffer) >= MIN_FRAMES:
                    model  = alphabet_model if current_mode == "alphabet" else digit_model
                    classes = ALPHA_CLASSES  if current_mode == "alphabet" else DIGIT_CLASSES
                    pred, conf = predict_from_buffer(landmark_buffer, model, classes)
                    if pred is not None:
                        current_prediction = pred
                        current_confidence = conf

        else:
            # No face detected
            is_lip_open = False
            cv2.putText(frame, "No face detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Prediction overlay (top-right box)
        if is_detecting and current_prediction != "-":
            text      = current_prediction
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
            cv2.rectangle(frame,
                          (w - text_size[0] - 30, 10),
                          (w - 10, text_size[1] + 30),
                          (0, 255, 0), -1)
            cv2.putText(frame, text,
                        (w - text_size[0] - 20, text_size[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4)

            conf_text = f"Conf: {current_confidence:.1%}"
            cv2.putText(frame, conf_text,
                        (w - 150, text_size[1] + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Buffer fill & status (bottom)
        cv2.putText(frame, f"Buffer: {len(landmark_buffer)}/{MAX_SEQ_LEN}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        status_color = (0, 255, 0) if is_detecting else (128, 128, 128)
        cv2.putText(frame, "DETECTING..." if is_detecting else "PAUSED",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        frame_count += 1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with camera_lock:
        camera_active = camera is not None and camera.isOpened()

    return jsonify({
        'camera_active':         camera_active,
        'alphabet_model_loaded': models_status["alphabet"],
        'digit_model_loaded':    models_status["digit"],
        'current_mode':          current_mode,
        'is_detecting':          is_detecting,
        'current_prediction':    current_prediction,
        'current_confidence':    float(current_confidence),
        'is_lip_open':           is_lip_open,
    })


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode, current_prediction, current_confidence

    data = request.json
    mode = data.get('mode', 'alphabet')

    if mode in ['alphabet', 'digit']:
        current_mode       = mode
        current_prediction = "-"
        current_confidence = 0.0
        landmark_buffer.clear()
        print(f"Mode switched to: {mode}")
        return jsonify({'success': True, 'mode': current_mode})

    return jsonify({'success': False, 'error': 'Invalid mode'})


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global is_detecting, current_prediction, current_confidence

    is_detecting = not is_detecting

    if not is_detecting:
        current_prediction = "-"
        current_confidence = 0.0
        landmark_buffer.clear()

    print(f"Detection {'started' if is_detecting else 'stopped'}")
    return jsonify({'success': True, 'is_detecting': is_detecting})


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎥 DUAL MODE LIP READING — DEEP LEARNING (Conv1D + BiLSTM)".center(70))
    print("=" * 70)

    print("\n📋 MODEL STATUS:")
    print(f"   [{'✓' if models_status['alphabet'] else '✗'}] Alphabet Model  ({ALPHABET_MODEL_PATH})")
    print(f"   [{'✓' if models_status['digit'] else '✗'}] Digit Model     ({DIGIT_MODEL_PATH})")

    if not any(models_status.values()):
        print("\n⚠️  WARNING: No models loaded! "
              "Place alphabet_model.h5 and digit_model.h5 inside a 'models/' folder.")

    print("\n📡 SERVER INFO:")
    print("   URL: http://localhost:5000")
    print("\n⏹️  TO STOP: Press Ctrl+C")
    print("=" * 70 + "\n")

    try:
        app.run(debug=False, host='0.0.0.0', port=5000,
                threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        with camera_lock:
            if camera is not None:
                camera.release()
        print("✓ Camera released")
        print("✓ Server stopped\n")