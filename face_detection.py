import cv2
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
) as face_detection:

    prev_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Webcam görüntüsü alınamadı.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        h, w, _ = frame.shape

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)

                # Biraz margin ekleyelim ki yüz çok sıkı kesilmesin
                margin = 30

                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + box_w + margin)
                y2 = min(h, y + box_h + margin)

                # Yüz crop
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size != 0:
                    face_resized = cv2.resize(face_crop, (224, 224))

                    # Crop edilmiş yüzü ayrı pencerede göster
                    cv2.imshow("Face Crop 224x224", face_resized)

                # Ana frame üzerinde kutu çiz
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                confidence = detection.score[0]
                cv2.putText(
                    frame,
                    f"Face: {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # FPS hesaplama
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow("Face Detection", frame)

        # Çıkmak için Face Detection penceresine tıklayıp q bas
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()