import cv2
import numpy as np
from twilio.rest import Client
import firebase_admin
from firebase_admin import credentials, firestore
import time
import datetime 

# Twilio Client
account_sid = 'ACdf1086f0539977324e2dc952a855f358'
auth_token = '0d4a4e6bab5cd687ee5cd779c7068fc9'
twilio_client = Client(account_sid, auth_token)
twilio_number = '+12792064935'
alert_recipient_number = '+919783735904'

# Firebase
cred = credentials.Certificate(r'c:\Users\theda\Documents\phantompulse-firebase-adminsdk-fbsvc-875e865453.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load YOLO
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Video Capture (0 for webcam or provide video file path)
cap = cv2.VideoCapture(0)

# Function to send SMS alert
def send_sms_alert(message):
    twilio_client.messages.create(
        body=message,
        from_=twilio_number,
        to=alert_recipient_number
    )

# Function to log complaint to Firebase
def log_complaint(location, timestamp):
    doc_ref = db.collection('complaints').document()
    doc_ref.set({
        'location': location,
        'timestamp': timestamp,
        'status': 'Pending'
    })

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':  # Adjust class as needed
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Send SMS alert
            message = f'Alert: {label} detected with {confidence:.2f} confidence.'
            send_sms_alert(message)

            # Log complaint to Firebase
            location = 'Location details or coordinates'
            timestamp = datetime.datetime.now().isoformat()
            log_complaint(location, timestamp)

    cv2.imshow('Garbage Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
