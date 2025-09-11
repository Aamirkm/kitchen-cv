import cv2
import os
from flask import Flask, render_template, Response, redirect, url_for
from datetime import datetime
import threading

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Camera and Folder Setup ---
# Set the camera to its highest resolution (e.g., 1080p)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

output_folder = 'capture_data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Threading for Camera Frames ---
lock = threading.Lock()
global_frame = None


def camera_reader_thread():
    """Reads high-resolution frames from the camera continuously."""
    global global_frame, lock
    while True:
        success, frame = cap.read()
        if not success:
            break
        with lock:
            global_frame = frame.copy()


def generate_frames():
    """Generator function to yield resized frames for the MJPEG stream."""
    global global_frame, lock
    while True:
        with lock:
            if global_frame is None:
                continue

            # --- THIS IS THE KEY CHANGE ---
            # Create a smaller copy of the frame for streaming
            stream_frame = cv2.resize(global_frame, (640, 360))  # Using 16:9 aspect ratio

            (flag, encoded_image) = cv2.imencode('.jpg', stream_frame)
            if not flag:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')


# --- Web Routes ---
@app.route('/')
def index():
    """Serves the main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    """Saves the full-resolution frame to disk."""
    global global_frame, lock

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    filename = os.path.join(output_folder, f'thaal_{timestamp}.jpg')

    with lock:
        if global_frame is not None:
            # This now saves the full-resolution global_frame
            cv2.imwrite(filename, global_frame)
            print(f"Image saved: {filename}")
        else:
            print("Error: No frame available to capture.")

    return redirect(url_for('index'))


# --- Main Execution ---
if __name__ == '__main__':
    thread = threading.Thread(target=camera_reader_thread)
    thread.daemon = True
    thread.start()

    app.run(host='0.0.0.0', port=5001)