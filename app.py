from flask import Flask, render_template, Response, jsonify, make_response
import threading
import io
import csv
import cv2

# Import the other layers of our application
from vision_processor import VisionProcessor
import database

# --- Main Application Setup ---
app = Flask(__name__)
# Create a single instance of our vision processor
vision_processor = VisionProcessor()

def generate_frames():
    """Generator function to yield frames for the MJPEG stream."""
    while True:
        # Get the latest frame from the vision processor
        frame = vision_processor.get_frame()
        if frame is None:
            continue
        
        # Resize frame for a smoother streaming experience
        stream_frame = cv2.resize(frame, (960, 540))
        (flag, encoded_image) = cv2.imencode('.jpg', stream_frame)
        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

# --- Flask Web Routes ---
@app.route('/')
def index():
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_service', methods=['POST'])
def start_service():
    if vision_processor.start_service():
        return jsonify(success=True, status="Service started.")
    return jsonify(success=False, status="Service already active.")

@app.route('/stop_service', methods=['POST'])
def stop_service():
    if vision_processor.stop_service():
        return jsonify(success=True, status="Service stopped.")
    return jsonify(success=False, status="Service not active.")

@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    vision_processor.reset_counts()
    return jsonify(success=True, status="Counts reset.")

@app.route('/status')
def status():
    return jsonify(vision_processor.get_status())

@app.route('/export_csv')
def export_csv():
    """Exports the last completed session's data as a CSV file."""
    result = database.get_last_session_events()

    if not result:
        return "No completed sessions found to export.", 404
    
    last_session_id, session_events = result

    # Generate CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'EventType'])
    writer.writerows(session_events)
    
    output.seek(0)
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=thaal_log_{last_session_id}.csv"
    response.headers["Content-type"] = "text/csv"
    
    return response

if __name__ == '__main__':
    # Initialize the database
    database.init_db()
    
    # Start the background thread for the vision processor
    cv_thread = threading.Thread(target=vision_processor.run)
    cv_thread.daemon = True
    cv_thread.start()
    
    # Start the Flask web server
    app.run(host='0.0.0.0', port=5001, debug=False)
