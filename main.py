from ultralytics import YOLO
import cv2
import os
import face_recognition
from datetime import datetime
from utils.alert_email import send_email_alert

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load known face encodings
known_encodings = []
known_names = []

for file in os.listdir("known_faces"):
    if file.endswith(".jpg") or file.endswith(".png"):
        image = face_recognition.load_image_file(f"known_faces/{file}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(file.split('.')[0])

# Setup
if not os.path.exists("intruders"):
    os.makedirs("intruders")

cap = cv2.VideoCapture(0)
print("[INFO] Camera started. Press 'q' to quit.")
sent_recent_alert = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.data

    for detection in detections:
        class_id = int(detection[5])
        if class_id == 0:  # person
            x1, y1, x2, y2 = map(int, detection[:4])
            person_roi = frame[y1:y2, x1:x2]

            face_locations = face_recognition.face_locations(person_roi)
            face_encodings = face_recognition.face_encodings(person_roi, face_locations)

            is_intruder = True
            for encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                if True in matches:
                    matched_idx = matches.index(True)
                    name = known_names[matched_idx]
                    print(f"[INFO] Known person detected: {name}")
                    is_intruder = False
                    break

            if is_intruder:
                print("[INFO] Unknown person detected! Sending alert.")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Intruder!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = f"intruders/intruder_{timestamp}.jpg"
                cv2.imwrite(img_path, frame)

                if not sent_recent_alert:
                    send_email_alert(img_path)
                    sent_recent_alert = True
            else:
                sent_recent_alert = False

    cv2.imshow("AI Home Security", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
