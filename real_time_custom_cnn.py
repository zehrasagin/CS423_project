import argparse
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


DEFAULT_MODEL_PATH = "models/emotion_custom_cnn.keras"
SMOOTHING_WINDOW = 8
FACE_MARGIN = 30
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

LEFT_EYE_INDICES = [33, 133, 159, 145]
RIGHT_EYE_INDICES = [362, 263, 386, 374]

LABEL_MAP = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run webcam emotion recognition with the custom CNN model."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def apply_happy_filter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 40)
    v = cv2.add(v, 35)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def apply_sad_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.GaussianBlur(gray_bgr, (7, 7), 0)


def apply_angry_filter(frame):
    red_overlay = np.zeros_like(frame)
    red_overlay[:, :, 2] = 180
    filtered = cv2.addWeighted(frame, 0.65, red_overlay, 0.35, 0)
    return cv2.convertScaleAbs(filtered, alpha=1.2, beta=0)


def apply_surprise_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 0.75, edges_bgr, 0.45, 0)


def apply_fear_filter(frame):
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)
    mask = (kernel_y * kernel_x.T) / np.max(kernel_y * kernel_x.T)
    vignette = np.copy(frame)

    for channel in range(3):
        vignette[:, :, channel] = vignette[:, :, channel] * mask

    return cv2.convertScaleAbs(vignette, alpha=0.85, beta=-20)


def apply_disgust_filter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 0.6)
    h = cv2.add(h, 15)
    filtered = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 80
    return cv2.addWeighted(filtered, 0.8, green_overlay, 0.2, 0)


def apply_emotion_filter(frame, emotion):
    if emotion == "Happy":
        return apply_happy_filter(frame)
    if emotion == "Sad":
        return apply_sad_filter(frame)
    if emotion == "Angry":
        return apply_angry_filter(frame)
    if emotion == "Surprise":
        return apply_surprise_filter(frame)
    if emotion == "Fear":
        return apply_fear_filter(frame)
    if emotion == "Disgust":
        return apply_disgust_filter(frame)
    return frame


def prepare_input(face_crop, image_size):
    resized = cv2.resize(face_crop, (image_size, image_size))
    rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(rgb_face.astype(np.float32), axis=0)
    # The trained custom CNN already contains a Rescaling(1.0 / 255.0) layer.
    # Passing 0-1 inputs here would normalize twice and distort predictions.
    return batch


def enhance_face(face_crop):
    ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    y_channel = clahe.apply(y_channel)

    enhanced = cv2.merge([y_channel, cr_channel, cb_channel])
    return cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)


def align_face(face_crop, face_mesh):
    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    mesh_results = face_mesh.process(rgb_face)

    if not mesh_results.multi_face_landmarks:
        return face_crop

    landmarks = mesh_results.multi_face_landmarks[0].landmark
    height, width = face_crop.shape[:2]

    def eye_center(indices):
        points = np.array(
            [
                (landmarks[index].x * width, landmarks[index].y * height)
                for index in indices
            ],
            dtype=np.float32,
        )
        return points.mean(axis=0)

    left_eye = eye_center(LEFT_EYE_INDICES)
    right_eye = eye_center(RIGHT_EYE_INDICES)

    angle = np.degrees(
        np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    )
    eyes_center = tuple(((left_eye + right_eye) / 2.0).tolist())

    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, -angle, 1.0)
    return cv2.warpAffine(
        face_crop,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def predict_with_tta(model, face_crop, image_size):
    variants = [
        face_crop,
        cv2.flip(face_crop, 1),
    ]

    probabilities = []
    for variant in variants:
        face_input = prepare_input(variant, image_size)
        probabilities.append(model.predict(face_input, verbose=0)[0])

    return np.mean(probabilities, axis=0)


def build_square_crop(frame, x, y, box_w, box_h, margin):
    height, width = frame.shape[:2]
    side = max(box_w, box_h) + 2 * margin

    center_x = x + box_w // 2
    center_y = y + box_h // 2

    x1 = max(0, center_x - side // 2)
    y1 = max(0, center_y - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)

    # Keep crop square when hitting borders.
    actual_w = x2 - x1
    actual_h = y2 - y1
    side = min(actual_w, actual_h)
    x2 = x1 + side
    y2 = y1 + side

    return x1, y1, x2, y2


def main():
    args = parse_args()

    print("Loading custom CNN model...")
    model = tf.keras.models.load_model(args.model_path)
    image_size = int(model.input_shape[1]) if model.input_shape[1] else 128
    print(f"Model loaded successfully. Input size: {image_size}x{image_size}")

    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        raise RuntimeError("Webcam acilamadi.")

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6,
    ) as face_detection, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        prev_time = 0.0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Webcam goruntusu alinamadi.")
                break
            frame_count += 1

            predicted_label = "No face"
            predicted_confidence = 0.0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            height, width, _ = frame.shape

            if results.detections:
                detection = max(results.detections, key=lambda item: item.score[0])
                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                box_w = int(bbox.width * width)
                box_h = int(bbox.height * height)

                x1, y1, x2, y2 = build_square_crop(
                    frame,
                    x,
                    y,
                    box_w,
                    box_h,
                    FACE_MARGIN,
                )

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size != 0:
                    aligned_face = align_face(face_crop, face_mesh)
                    enhanced_face = enhance_face(aligned_face)
                    probabilities = predict_with_tta(
                        model,
                        enhanced_face,
                        image_size,
                    )

                    prediction_history.append(probabilities)
                    avg_probabilities = np.mean(prediction_history, axis=0)

                    class_index = int(np.argmax(avg_probabilities))
                    predicted_confidence = float(np.max(avg_probabilities))
                    sorted_indices = np.argsort(avg_probabilities)[::-1]
                    predicted_label = LABEL_MAP.get(class_index, str(class_index))

                    if args.debug and frame_count % 10 == 0:
                        top_three = []
                        for idx in sorted_indices[:3]:
                            label = LABEL_MAP.get(int(idx), str(idx))
                            top_three.append(f"{label}: {avg_probabilities[idx]:.2f}")
                        print(" | ".join(top_three))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{predicted_label}: {predicted_confidence:.2f}",
                        (x1, max(20, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
            else:
                prediction_history.clear()

            display_frame = apply_emotion_filter(frame.copy(), predicted_label)

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time else 0.0
            prev_time = current_time

            cv2.putText(
                display_frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"Emotion: {predicted_label} ({predicted_confidence:.2f})",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Custom CNN Real-Time Emotion Recognition", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
