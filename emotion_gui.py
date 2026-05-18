import os
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


MODEL_OPTIONS = {
    "MobileNetV2": {
        "path": "models/emotion_mobilenetv2.keras",
        "preprocess": "mobilenetv2",
    },
    "Custom CNN": {
        "path": "models/emotion_custom_cnn.keras",
        "preprocess": "rescale",
    },
}

LABEL_MAP = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral",
}

CONFIDENCE_THRESHOLD = 0.35
SMOOTHING_WINDOW = 8
FACE_MARGIN = 30


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


class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition GUI")
        self.root.geometry("1180x760")
        self.root.configure(bg="#f4efe8")

        self.model = None
        self.model_config = None
        self.loaded_model_name = None
        self.model_input_size = 224
        self.cap = None
        self.face_detector = None
        self.running = False
        self.last_frame_time = 0.0
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.video_photo = None

        self.selected_model_name = tk.StringVar(value=list(MODEL_OPTIONS.keys())[0])
        self.status_var = tk.StringVar(value="Select a model and start the camera.")
        self.emotion_var = tk.StringVar(value="Emotion: -")
        self.confidence_var = tk.StringVar(value="Confidence: -")
        self.fps_var = tk.StringVar(value="FPS: -")
        self.filter_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Card.TFrame", background="#fffaf3")
        style.configure("Panel.TFrame", background="#f4efe8")
        style.configure("Title.TLabel", background="#f4efe8", foreground="#184d47", font=("Helvetica", 18, "bold"))
        style.configure("Body.TLabel", background="#fffaf3", foreground="#3c3c3c", font=("Helvetica", 11))
        style.configure("Accent.TButton", font=("Helvetica", 11, "bold"))

        container = ttk.Frame(self.root, style="Panel.TFrame", padding=18)
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container, style="Card.TFrame", padding=12)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(container, style="Card.TFrame", padding=16, width=300)
        right.pack(side="right", fill="y")

        ttk.Label(left, text="Live Camera Feed", style="Title.TLabel").pack(anchor="w", pady=(0, 10))

        self.video_label = tk.Label(
            left,
            text="Select a model and start the camera.",
            bg="#d8e2dc",
            fg="#1f2933",
            font=("Helvetica", 14),
            width=72,
            height=32,
        )
        self.video_label.pack(fill="both", expand=True)

        ttk.Label(right, text="Model Selection", style="Title.TLabel").pack(anchor="w")
        model_combo = ttk.Combobox(
            right,
            textvariable=self.selected_model_name,
            values=list(MODEL_OPTIONS.keys()),
            state="readonly",
        )
        model_combo.pack(fill="x", pady=(10, 14))

        ttk.Button(
            right,
            text="Start Camera",
            command=self.start_camera,
            style="Accent.TButton",
        ).pack(fill="x", pady=4)

        ttk.Button(
            right,
            text="Stop Camera",
            command=self.stop_camera,
            style="Accent.TButton",
        ).pack(fill="x", pady=4)

        ttk.Checkbutton(
            right,
            text="Apply emotion filter",
            variable=self.filter_var,
        ).pack(anchor="w", pady=(16, 14))

        info_box = ttk.Frame(right, style="Card.TFrame", padding=12)
        info_box.pack(fill="x", pady=(4, 12))

        ttk.Label(info_box, textvariable=self.status_var, style="Body.TLabel", wraplength=240).pack(anchor="w", pady=4)
        ttk.Label(info_box, textvariable=self.emotion_var, style="Body.TLabel").pack(anchor="w", pady=4)
        ttk.Label(info_box, textvariable=self.confidence_var, style="Body.TLabel").pack(anchor="w", pady=4)
        ttk.Label(info_box, textvariable=self.fps_var, style="Body.TLabel").pack(anchor="w", pady=4)

        notes = (
            "Notes:\n"
            "- The selected model loads automatically when you start the camera.\n"
            "- MobileNetV2 uses the existing trained model.\n"
            "- For Custom CNN, run train_custom_cnn.py first.\n"
            "- You can close the window to exit."
        )
        ttk.Label(right, text=notes, style="Body.TLabel", wraplength=250, justify="left").pack(anchor="w")

    def _ensure_selected_model_loaded(self):
        selected_model_name = self.selected_model_name.get()
        if self.model is not None and self.loaded_model_name == selected_model_name:
            return True

        config = MODEL_OPTIONS[selected_model_name]
        model_path = config["path"]

        if not os.path.exists(model_path):
            messagebox.showerror(
                "Model not found",
                f"File not found:\n{model_path}\n\nCreate the relevant model first or select an existing one.",
            )
            self.status_var.set("Selected model not found.")
            return False

        self.status_var.set(f"Loading {selected_model_name}...")
        self.root.update_idletasks()

        try:
            model = tf.keras.models.load_model(model_path)
            input_shape = model.input_shape

            self.model = model
            self.model_config = config
            self.loaded_model_name = selected_model_name
            self.model_input_size = int(input_shape[1]) if input_shape and len(input_shape) > 2 else 224
            self.prediction_history.clear()
            self.status_var.set(f"Model ready: {selected_model_name}")
            return True
        except Exception as exc:
            self.status_var.set("Model could not be loaded.")
            messagebox.showerror("Loading error", str(exc))
            return False

    def start_camera(self):
        if not self._ensure_selected_model_loaded():
            return

        if self.running:
            self.status_var.set(f"Camera is active. Model: {self.loaded_model_name}")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Camera error", "Webcam could not be opened.")
            self.status_var.set("Webcam could not be opened.")
            return

        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        )
        self.running = True
        self.last_frame_time = 0.0
        self.prediction_history.clear()
        self.status_var.set(f"Camera is active. Model: {self.loaded_model_name}")
        self._update_frame()

    def stop_camera(self):
        self.running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.face_detector is not None:
            self.face_detector.close()
            self.face_detector = None

        self.video_label.configure(image="", text="Camera stopped.", bg="#d8e2dc")
        self.video_label.image = None
        self.emotion_var.set("Emotion: -")
        self.confidence_var.set("Confidence: -")
        self.fps_var.set("FPS: -")

    def _prepare_input(self, face_crop):
        resized = cv2.resize(face_crop, (self.model_input_size, self.model_input_size))
        rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb_face.astype(np.float32), axis=0)

        if self.model_config["preprocess"] == "mobilenetv2":
            return preprocess_input(batch)

        # The custom CNN already contains a Rescaling layer in the model.
        # Dividing here would normalize twice and distort predictions.
        return batch

    def _build_square_crop(self, frame, x, y, box_w, box_h):
        height, width = frame.shape[:2]
        side = max(box_w, box_h) + 2 * FACE_MARGIN

        center_x = x + box_w // 2
        center_y = y + box_h // 2

        x1 = max(0, center_x - side // 2)
        y1 = max(0, center_y - side // 2)
        x2 = min(width, x1 + side)
        y2 = min(height, y1 + side)

        actual_w = x2 - x1
        actual_h = y2 - y1
        side = min(actual_w, actual_h)
        x2 = x1 + side
        y2 = y1 + side

        return x1, y1, x2, y2

    def _predict_emotion(self, face_crop):
        input_batch = self._prepare_input(face_crop)
        probabilities = self.model.predict(input_batch, verbose=0)[0]
        self.prediction_history.append(probabilities)

        averaged = np.mean(self.prediction_history, axis=0)
        class_index = int(np.argmax(averaged))
        confidence = float(np.max(averaged))

        if confidence < CONFIDENCE_THRESHOLD:
            return "Uncertain", confidence

        return LABEL_MAP.get(class_index, str(class_index)), confidence

    def _update_frame(self):
        if not self.running or self.cap is None or self.face_detector is None:
            return

        success, frame = self.cap.read()
        if not success:
            self.status_var.set("Could not read webcam frame.")
            self.root.after(100, self._update_frame)
            return

        predicted_label = "No face"
        predicted_confidence = 0.0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        height, width, _ = frame.shape

        if results.detections:
            detection = max(results.detections, key=lambda item: item.score[0])
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            box_w = int(bbox.width * width)
            box_h = int(bbox.height * height)

            x1, y1, x2, y2 = self._build_square_crop(frame, x, y, box_w, box_h)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size != 0:
                predicted_label, predicted_confidence = self._predict_emotion(face_crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{predicted_label}: {predicted_confidence:.2f}",
                    (x1, max(20, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                self.prediction_history.clear()
        else:
            self.prediction_history.clear()

        display_frame = frame.copy()
        if self.filter_var.get():
            display_frame = apply_emotion_filter(display_frame, predicted_label)

        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0.0
        self.last_frame_time = current_time

        cv2.putText(
            display_frame,
            f"FPS: {fps:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Emotion: {predicted_label}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        self.emotion_var.set(f"Emotion: {predicted_label}")
        self.confidence_var.set(f"Confidence: {predicted_confidence:.2f}")
        self.fps_var.set(f"FPS: {fps:.2f}")

        rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_display)
        self.video_photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.video_photo, text="")
        self.video_label.image = self.video_photo

        self.root.after(15, self._update_frame)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = EmotionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
