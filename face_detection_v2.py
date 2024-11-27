import cv2
import torch
from pathlib import Path

# Load YOLOv9 model
model_path = './weights/yolov9-c.pt'  # Ensure the path points to the correct weights file
model = torch.hub.load('ultralytics/yolov9', 'custom', path=model_path)

# Video source (can be a video file or a webcam index)
video_path = './Stephen Curry Clips For Edits 4K - Reapther (1080p, h264, youtube).mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Update frame with results
    for *xyxy, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # Adjust the class index based on your model's specific training
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv9 Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
