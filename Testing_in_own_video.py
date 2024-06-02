import cv2
from pathlib import Path
from ultralytics import YOLO

# Path to YOLOv8 model weights
yolo_weights_path = r'C:\Users\pucso\Desktop\Vehicle_detecting\runs\detect\train2\weights\best.pt'

# Path to the video
video_path = r'C:\Users\pucso\Desktop\Vehicle_detecting\P1010559.MOV'

# Import YOLOv8 library
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(yolo_weights_path)

# Open video file
cap = cv2.VideoCapture(video_path)

while True:
    # Read video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLOv8 model on the frame
    results = model(frame)

    # Display results on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if len(box.xyxy[0]) != 0:
                x1, y1, x2, y2 = tuple(map(int, box.xyxy[0]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                labels = "Car"
                conf = "{:.2f}".format(box.conf[0])
                cv2.putText(frame, f'{labels} {conf}', [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display
    cv2.imshow('YOLOv8 Detection', frame)

    # Exit if the user presses the 'esc' key
    if cv2.waitKey(1) == 27:
        break


# Release resources
cap.release()
cv2.destroyAllWindows()