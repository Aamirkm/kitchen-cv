import cv2
from flask import Flask, render_template, Response
import threading
from ultralytics import YOLO
import numpy as np

# =============================================================================
# Main Application
# =============================================================================

app = Flask(__name__)

# --- Load Your Custom Trained Model ---
model = YOLO('runs/detect/train/weights/best.pt')

# --- Video Capture ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Global Variables for Tracking and Counting ---
track_history = {}
thaal_out_count = 0
thaal_in_count = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- Define TWO separate lines for counting ---
LINE_OUT_POSITION = FRAME_WIDTH // 3
LINE_IN_POSITION = (FRAME_WIDTH // 3) * 2

LINE_OUT_START = (LINE_OUT_POSITION, 0)
LINE_OUT_END = (LINE_OUT_POSITION, FRAME_HEIGHT)
LINE_IN_START = (LINE_IN_POSITION, 0)
LINE_IN_END = (LINE_IN_POSITION, FRAME_HEIGHT)

lock = threading.Lock()
global_frame = None

def camera_and_cv_thread():
    """Main thread to read frames, run detection, tracking, and counting."""
    global global_frame, lock, thaal_out_count, thaal_in_count

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Use the model's built-in tracker
        results = model.track(frame, persist=True, verbose=False)

        # Draw the two virtual lines
        cv2.line(frame, LINE_OUT_START, LINE_OUT_END, (0, 255, 0), 2) # Green line for OUT
        cv2.line(frame, LINE_IN_START, LINE_IN_END, (0, 0, 255), 2)   # Red line for IN

        # Check if there are any detections and tracking IDs
        if results[0].boxes.id is not None:
            # Get boxes and track IDs
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                
                center_x = (x1 + x2) // 2
                
                # Initialize tracking history for new objects
                if track_id not in track_history:
                    track_history[track_id] = {
                        "positions": [],
                        "has_been_counted": False # NEW: Single flag for counting
                    }
                
                # Append current center position
                track_history[track_id]["positions"].append(center_x)

                # Check for line crossing only if we have a history
                if len(track_history[track_id]["positions"]) > 1:
                    prev_x = track_history[track_id]["positions"][-2]

                    # --- YOUR NEW, STRICTER COUNTING LOGIC ---
                    # An object can only be counted ONCE per appearance.
                    if not track_history[track_id]["has_been_counted"]:
                        # 1. OUTGOING: Crosses the LEFT green line moving RIGHT-TO-LEFT
                        if prev_x > LINE_OUT_POSITION and center_x <= LINE_OUT_POSITION:
                            thaal_out_count += 1
                            track_history[track_id]["has_been_counted"] = True

                        # 2. INCOMING: Crosses the RIGHT red line moving LEFT-TO-RIGHT
                        elif prev_x < LINE_IN_POSITION and center_x >= LINE_IN_POSITION:
                            thaal_in_count += 1
                            track_history[track_id]["has_been_counted"] = True

                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"ID:{track_id} thaal"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw the counts on the screen
        cv2.putText(frame, f"Thaal Out: {thaal_out_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Thaal In: {thaal_in_count}", (FRAME_WIDTH - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        with lock:
            global_frame = frame.copy()

def generate_frames():
    global global_frame, lock
    while True:
        with lock:
            if global_frame is None:
                continue
            
            stream_frame = cv2.resize(global_frame, (960, 540))
            (flag, encoded_image) = cv2.imencode('.jpg', stream_frame)
            if not flag:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    thread = threading.Thread(target=camera_and_cv_thread)
    thread.daemon = True
    thread.start()
    app.run(host='0.0.0.0', port=5001, debug=False)

