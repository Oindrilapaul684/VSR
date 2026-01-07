from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pickle
import os
import threading
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean

app = Flask(__name__)

# ==================== CONFIGURATION ====================
ALPHABET_MODEL_PATH = 'rf_model.pkl'
DIGIT_MODEL_PATH = 'random_forest_model.pkl'

# ==================== GLOBAL VARIABLES ====================
camera = None
camera_lock = threading.Lock()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Lip landmark indices (MediaPipe Face Mesh)
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
lip_indices = list(set(OUTER_LIP + INNER_LIP))

# Buffer for temporal analysis
landmark_buffer = deque(maxlen=30)

# Current state
current_mode = "alphabet"  # "alphabet" or "digit"
current_prediction = "-"
current_confidence = 0.0
is_detecting = False

# ==================== LOAD MODELS ====================
alphabet_model = None
digit_model = None
models_status = {"alphabet": False, "digit": False}

# Load Alphabet Model
try:
    print("Loading alphabet model...")
    with open(ALPHABET_MODEL_PATH, 'rb') as f:
        alphabet_model = pickle.load(f)
    models_status["alphabet"] = True
    print(f"✓ Alphabet model loaded successfully")
    print(f"  - Type: {type(alphabet_model)}")
    print(f"  - Has predict: {hasattr(alphabet_model, 'predict')}")
    if hasattr(alphabet_model, 'n_features_in_'):
        print(f"  - Expected features: {alphabet_model.n_features_in_}")
except Exception as e:
    print(f"✗ Could not load alphabet model: {e}")

# Load Digit Model
try:
    print("Loading digit model...")
    with open(DIGIT_MODEL_PATH, 'rb') as f:
        digit_model = pickle.load(f)
    models_status["digit"] = True
    print(f"✓ Digit model loaded successfully")
    print(f"  - Type: {type(digit_model)}")
    print(f"  - Has predict: {hasattr(digit_model, 'predict')}")
    if hasattr(digit_model, 'n_features_in_'):
        print(f"  - Expected features: {digit_model.n_features_in_}")
except Exception as e:
    print(f"✗ Could not load digit model: {e}")

# ==================== SIMPLE FEATURE EXTRACTION ====================

def extract_simple_lip_features(landmarks):
    """
    Extract simple, robust features from lip landmarks
    Returns a flattened array of coordinates and basic geometric features
    """
    features = []
    points = landmarks.reshape(-1, 2)

    # 1. Raw coordinates
    features.extend(landmarks.flatten())

    # 2. Centroid
    centroid = np.mean(points, axis=0)
    features.extend(centroid)

    # 3. Distances from centroid
    distances = [euclidean(p, centroid) for p in points]
    features.extend([np.mean(distances), np.std(distances),
                    np.max(distances), np.min(distances)])

    # 4. Bounding box features
    x_coords, y_coords = points[:, 0], points[:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    aspect_ratio = width / (height + 1e-6)
    features.extend([width, height, aspect_ratio])

    # 5. Statistical moments
    features.extend([np.mean(x_coords), np.std(x_coords), skew(x_coords), kurtosis(x_coords)])
    features.extend([np.mean(y_coords), np.std(y_coords), skew(y_coords), kurtosis(y_coords)])

    # 6. Pairwise distances (key points)
    key_indices = [0, len(points)//4, len(points)//2, 3*len(points)//4, -1]
    for i in range(len(key_indices)-1):
        dist = euclidean(points[key_indices[i]], points[key_indices[i+1]])
        features.append(dist)

    # 7. Mouth opening
    if len(points) > 10:
        mouth_opening = euclidean(points[3], points[9]) if len(points) > 9 else 0
        features.append(mouth_opening)

    return np.array(features)


def extract_temporal_features(landmark_sequence):
    """
    Extract temporal features from a sequence of landmarks
    """
    if len(landmark_sequence) == 0:
        return None
    
    # Extract features from each frame
    frame_features = [extract_simple_lip_features(lm) for lm in landmark_sequence]
    frame_features_array = np.array(frame_features)
    
    # Aggregate over time
    features = []
    
    # Mean over time
    features.extend(np.mean(frame_features_array, axis=0))
    
    # Std over time
    features.extend(np.std(frame_features_array, axis=0))
    
    # Max over time
    features.extend(np.max(frame_features_array, axis=0))
    
    # Min over time
    features.extend(np.min(frame_features_array, axis=0))
    
    # Velocity (if we have multiple frames)
    if len(landmark_sequence) > 1:
        velocities = np.diff(frame_features_array, axis=0)
        features.extend(np.mean(velocities, axis=0))
        features.extend(np.std(velocities, axis=0))
    else:
        # Pad with zeros if only one frame
        features.extend(np.zeros(len(frame_features_array[0])))
        features.extend(np.zeros(len(frame_features_array[0])))
    
    return np.array(features)

def predict_from_landmarks(landmark_sequence, mode):
    """Predict based on current mode (alphabet or digit)"""
    global current_prediction, current_confidence
    
    if len(landmark_sequence) < 5:
        return None, 0.0
    
    # Select model based on mode
    model = alphabet_model if mode == "alphabet" else digit_model
    
    if model is None:
        return None, 0.0
    
    try:
        # Extract temporal features
        features = extract_temporal_features(landmark_sequence)
        
        if features is None:
            return None, 0.0
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Debug: print feature shape
        print(f"Feature shape: {features.shape}, Model expects: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown'}")
        
        # Handle feature dimension mismatch
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            current_features = features.shape[1]
            
            if current_features < expected_features:
                # Pad with zeros
                padding = np.zeros((1, expected_features - current_features))
                features = np.hstack([features, padding])
                print(f"Padded features from {current_features} to {expected_features}")
            elif current_features > expected_features:
                # Truncate
                features = features[:, :expected_features]
                print(f"Truncated features from {current_features} to {expected_features}")
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Get confidence if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            class_idx = np.where(model.classes_ == prediction)[0]
            if len(class_idx) > 0:
                confidence = probabilities[class_idx[0]]
            else:
                confidence = 0.0
        else:
            confidence = 1.0  # If no predict_proba, assume 100% confidence
        
        # Convert to character
        if mode == "alphabet":
            # Handle both 0-25 range and other ranges
            if 0 <= prediction <= 25:
                predicted_char = chr(int(prediction) + ord('A'))
            else:
                predicted_char = chr(int(prediction % 26) + ord('A'))
        else:
            # For digits, ensure 0-9 range
            predicted_char = str(int(prediction) % 10)
        
        current_prediction = predicted_char
        current_confidence = float(confidence)
        
        print(f"Predicted: {predicted_char} (raw: {prediction}) with confidence: {confidence:.2%}")
        
        return predicted_char, float(confidence)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

# ==================== VIDEO PROCESSING ====================

def generate_frames():
    """Generate video frames with lip landmarks"""
    global camera, landmark_buffer, current_prediction, current_confidence
    global current_mode, is_detecting
    
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
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        # Draw mode indicator
        mode_text = f"Mode: {current_mode.upper()}"
        mode_color = (0, 255, 0) if is_detecting else (128, 128, 128)
        cv2.putText(frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Extract and draw lip landmarks with GREEN DOTS
            lip_points = []
            for i in lip_indices:
                x = int(landmarks.landmark[i].x * w)
                y = int(landmarks.landmark[i].y * h)
                lip_points.append([x, y])
                
                # Draw GREEN dots on lip landmarks (ALWAYS visible)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw connections between lip points
            lip_points_array = np.array(lip_points, dtype=np.int32)
            cv2.polylines(frame, [lip_points_array], True, (0, 255, 0), 1)
            
            # Normalize landmarks
            if len(lip_points_array) > 0:
                x_min, y_min = np.min(lip_points_array, axis=0)
                x_max, y_max = np.max(lip_points_array, axis=0)
                
                x_range = x_max - x_min if x_max > x_min else 1
                y_range = y_max - y_min if y_max > y_min else 1
                
                normalized_points = (lip_points_array - np.array([x_min, y_min])) / \
                                   np.array([x_range, y_range])
                
                # Add to buffer
                landmark_buffer.append(normalized_points)
                
                # Predict if detecting is enabled
                if is_detecting and frame_count % 15 == 0 and len(landmark_buffer) >= 10:
                    active_model = alphabet_model if current_mode == "alphabet" else digit_model
                    if active_model is not None:
                        predicted_char, confidence = predict_from_landmarks(
                            list(landmark_buffer), current_mode
                        )
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display prediction if detecting
        if is_detecting and current_prediction != "-":
            # Prediction text with background
            text = f"{current_prediction}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
            
            # Draw background rectangle
            cv2.rectangle(frame, (w - text_size[0] - 30, 10), 
                         (w - 10, text_size[1] + 30), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (w - text_size[0] - 20, text_size[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4)
            
            # Confidence bar
            conf_text = f"{current_confidence:.1%}"
            cv2.putText(frame, conf_text, (w - 120, text_size[1] + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            bar_width = int(current_confidence * 200)
            cv2.rectangle(frame, (w - 220, text_size[1] + 70), 
                         (w - 220 + bar_width, text_size[1] + 85), (0, 255, 0), -1)
            cv2.rectangle(frame, (w - 220, text_size[1] + 70), 
                         (w - 20, text_size[1] + 85), (0, 255, 0), 2)
        
        # Status text
        status_text = "DETECTING..." if is_detecting else "PAUSED"
        status_color = (0, 255, 0) if is_detecting else (128, 128, 128)
        cv2.putText(frame, status_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Landmark buffer info
        buffer_text = f"Buffer: {len(landmark_buffer)}/30"
        cv2.putText(frame, buffer_text, (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        frame_count += 1
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Get system status"""
    with camera_lock:
        camera_active = camera is not None and camera.isOpened()
    
    return jsonify({
        'camera_active': camera_active,
        'alphabet_model_loaded': models_status["alphabet"],
        'digit_model_loaded': models_status["digit"],
        'current_mode': current_mode,
        'is_detecting': is_detecting,
        'current_prediction': current_prediction,
        'current_confidence': float(current_confidence)
    })

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Set detection mode (alphabet or digit)"""
    global current_mode, current_prediction, landmark_buffer
    
    data = request.json
    mode = data.get('mode', 'alphabet')
    
    if mode in ['alphabet', 'digit']:
        current_mode = mode
        current_prediction = "-"
        landmark_buffer.clear()
        print(f"Mode switched to: {mode}")
        return jsonify({'success': True, 'mode': current_mode})
    
    return jsonify({'success': False, 'error': 'Invalid mode'})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle detection on/off"""
    global is_detecting, current_prediction, landmark_buffer
    
    is_detecting = not is_detecting
    
    if not is_detecting:
        current_prediction = "-"
        landmark_buffer.clear()
    
    print(f"Detection {'started' if is_detecting else 'stopped'}")
    
    return jsonify({'success': True, 'is_detecting': is_detecting})

# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎥 DUAL MODE LIP READING APPLICATION".center(70))
    print("=" * 70)
    
    print("\n📋 MODEL STATUS:")
    print(f"   [{'✓' if models_status['alphabet'] else '✗'}] Alphabet Model (rf_model.pkl)")
    print(f"   [{'✓' if models_status['digit'] else '✗'}] Digit Model (random_forest_model.pkl)")
    
    if not any(models_status.values()):
        print("\n⚠️  WARNING: No models loaded!")
        print("   Please ensure at least one model file exists:")
        print("   - rf_model.pkl (for alphabets)")
        print("   - random_forest_model.pkl (for digits)")
    
    print("\n📡 SERVER INFO:")
    print("   URL: http://localhost:5000")
    
    print("\n🎯 FEATURES:")
    print("   ✓ Continuous camera access")
    print("   ✓ Real-time lip landmark tracking (green dots)")
    print("   ✓ Switch between Alphabet and Digit modes")
    print("   ✓ Start/Stop detection control")
    
    print("\n⏹️  TO STOP: Press Ctrl+C")
    print("=" * 70 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        with camera_lock:
            if camera is not None:
                camera.release()
        print("✓ Camera released")
        print("✓ Server stopped\n")