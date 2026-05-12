import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque


# -----------------------------
# Settings
# -----------------------------

MODEL_PATH = "models/emotion_mobilenetv2.keras"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.35
SMOOTHING_WINDOW = 8



LABEL_MAP = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"
}

prediction_history = deque(maxlen=SMOOTHING_WINDOW)



def apply_emotion_filter(frame, emotion):
    """
    Applies a visual filter to the full webcam frame based on the predicted emotion.
    """

    if emotion == "Happy":
        return apply_happy_filter(frame)

    elif emotion == "Sad":
        return apply_sad_filter(frame)

    elif emotion == "Angry":
        return apply_angry_filter(frame)

    elif emotion == "Surprise":
        return apply_surprise_filter(frame)

    elif emotion == "Fear":
        return apply_fear_filter(frame)

    elif emotion == "Disgust":
        return apply_disgust_filter(frame)

    else:
        # Neutral, Uncertain, No face
        return frame


def apply_happy_filter(frame):
    # Increase brightness and saturation using HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    s = cv2.add(s, 40)
    v = cv2.add(v, 35)

    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_frame = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return enhanced_frame


def apply_sad_filter(frame):
    # Convert to grayscale but keep 3 channels for display consistency
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Add slight blur for softer sad effect
    blurred = cv2.GaussianBlur(gray_bgr, (7, 7), 0)

    return blurred


def apply_angry_filter(frame):
    # Apply red overlay
    red_overlay = np.zeros_like(frame)
    red_overlay[:, :, 2] = 180  # Red channel in BGR

    filtered = cv2.addWeighted(frame, 0.65, red_overlay, 0.35, 0)

    # Increase contrast slightly
    filtered = cv2.convertScaleAbs(filtered, alpha=1.2, beta=0)

    return filtered


def apply_surprise_filter(frame):
    # Canny edge overlay effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    filtered = cv2.addWeighted(frame, 0.75, edges_bgr, 0.45, 0)

    return filtered


def apply_fear_filter(frame):
    # Dark vignette effect
    rows, cols = frame.shape[:2]

    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)

    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()

    vignette = np.copy(frame)

    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask

    # Darken overall image
    vignette = cv2.convertScaleAbs(vignette, alpha=0.85, beta=-20)

    return vignette


def apply_disgust_filter(frame):
    # Greenish and desaturated effect
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Reduce saturation and slightly shift hue
    s = cv2.multiply(s, 0.6)
    h = cv2.add(h, 15)

    disgust_hsv = cv2.merge([h, s, v])
    filtered = cv2.cvtColor(disgust_hsv, cv2.COLOR_HSV2BGR)

    # Add subtle green overlay
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 80

    filtered = cv2.addWeighted(filtered, 0.8, green_overlay, 0.2, 0)

    return filtered


# -----------------------------
# Load emotion model
# -----------------------------
print("Loading emotion model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


# -----------------------------
# MediaPipe Face Detection
# -----------------------------
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
) as face_detection:

    prev_time = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Webcam görüntüsü alınamadı.")
            break
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        h, w, _ = frame.shape

        predicted_label = "No face"
        predicted_confidence = 0.0

        if results.detections:
            # Şimdilik ilk yüzü alıyoruz
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            # Margin ekle
            margin = 30

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + box_w + margin)
            y2 = min(h, y + box_h + margin)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size != 0:
                # Model input hazırlığı
                face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

                # BGR → RGB
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                # Batch dimension ekle
                face_array = np.expand_dims(face_rgb, axis=0)

                # MobileNetV2 preprocessing
                face_array = preprocess_input(face_array.astype(np.float32))

                # Prediction
                pred_probs = model.predict(face_array, verbose=0)[0]


                # Add prediction probabilities to history
                prediction_history.append(pred_probs)

                # Average probabilities over recent frames
                avg_probs = np.mean(prediction_history, axis=0)

                smoothed_class = int(np.argmax(avg_probs))
                smoothed_confidence = float(np.max(avg_probs))
                raw_label = LABEL_MAP[smoothed_class]


                top_indices = np.argsort(avg_probs)[::-1][:3]

                top_texts = []
                for idx in top_indices:
                    emotion = LABEL_MAP[int(idx)]
                    prob = avg_probs[idx]
                    top_texts.append(f"{emotion}: {prob:.2f}")

                if frame_count % 10 == 0:
                    print(" | ".join(top_texts))

                # Confidence threshold after smoothing
                if smoothed_confidence >= CONFIDENCE_THRESHOLD:
                    predicted_label = raw_label
                else:
                    predicted_label = "Uncertain"

                predicted_confidence = smoothed_confidence

                # Bounding box çiz
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Yüz crop penceresi
                cv2.imshow("Face Crop 224x224", face_resized)

                # Label yaz
                label_text = f"{predicted_label}: {predicted_confidence:.2f}"

                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        # FPS hesaplama
        # Apply emotion-based filter to the full frame
        display_frame = apply_emotion_filter(frame.copy(), predicted_label)

        # FPS hesaplama
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(
            display_frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # Show current emotion label on top-left
        cv2.putText(
            display_frame,
            f"Emotion: {predicted_label} ({predicted_confidence:.2f})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Real-Time Emotion Recognition", display_frame)

        # Çıkmak için pencereye tıkla ve q bas
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()