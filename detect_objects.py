import cv2
from ultralytics import YOLO

# --- MODEL INITIALIZATION ---
# Load the YOLOv8 model. 'yolov8n.pt' is the smallest and fastest version.
model = YOLO('yolo11s.pt')

# --- VIDEO CAPTURE INITIALIZATION ---
# Initialize a connection to the default webcam (ID 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- MAIN LOOP ---
while True:
    # Read one frame from the webcam feed
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # --- OBJECT DETECTION ---
    # Run the YOLOv8 model on the frame
    results = model(frame)

    # --- VISUALIZATION ---
    # The 'results[0].plot()' method returns a frame with bounding boxes and labels drawn on it.
    annotated_frame = results[0].plot()

    # Display the annotated frame in a window
    cv2.imshow("Object Detection", annotated_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()