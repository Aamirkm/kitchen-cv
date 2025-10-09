import cv2
from flask import Flask, render_template, Response, request, jsonify
import threading
import time
from datetime import datetime, date

# Import from our other project files
from vision_processor import VisionProcessor
import database

# =============================================================================
# Main Application Setup
# =============================================================================

app = Flask(__name__)
database.init_db()

# --- Create and start the Vision Processor in a background thread ---
vision_processor = VisionProcessor(model_path='runs/detect/train/weights/best.pt')
vision_thread = threading.Thread(target=vision_processor.run, daemon=True)
vision_thread.start()

# --- Video Streaming ---
def generate_frames():
    """
    Generator function to yield annotated frames for the MJPEG stream.
    Saves resources by only running when a client is connected.
    """
    print("Client connected to video stream.")
    try:
        while True:
            frame = vision_processor.get_annotated_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            stream_frame = cv2.resize(frame, (960, 540))
            (flag, encoded_image) = cv2.imencode('.jpg', stream_frame)
            if not flag:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
            
            time.sleep(0.05)
    finally:
        print("Client disconnected from video stream.")


# =============================================================================
# Web Routes (Presentation Layer)
# =============================================================================

@app.route('/')
def index():
    """Serves the main control panel page."""
    return render_template('main.html')

@app.route('/dashboard')
def dashboard():
    """Serves the data dashboard page."""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Provides the video stream endpoint."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API Routes for Control ---

@app.route('/start_service', methods=['POST'])
def start_service():
    """Starts a new service session."""
    data = request.get_json()
    expected_thaals = data.get('expected_thaals')
    vision_processor.start_service(expected_thaals)
    return jsonify(status="Service started", service_active=True)

@app.route('/stop_service', methods=['POST'])
def stop_service():
    """Stops the current service session."""
    vision_processor.stop_service()
    return jsonify(status="Service stopped", service_active=False)

@app.route('/status', methods=['GET'])
def status():
    """Returns the current status of the application."""
    status_data = vision_processor.get_status()
    return jsonify(status_data)

# --- API Routes for Data ---

@app.route('/api/dashboard_data')
def dashboard_data():
    """Provides data for the historical dashboard."""
    date_str = request.args.get('date')
    if not date_str:
        return jsonify(message="No date provided"), 400

    try:
        query_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify(message="Invalid date format"), 400

    sessions = database.get_sessions_for_date(query_date)

    if not sessions:
        return jsonify(message="No event data found for the selected date.")

    total_expected = 0
    total_out = 0
    total_in = 0
    total_duration_minutes = 0
    session_ids = []

    for session in sessions:
        session_id, start_time_str, end_time_str, expected, final_out, final_in = session
        session_ids.append(session_id)
        
        if expected and isinstance(expected, int):
            total_expected += expected
        if final_out:
            total_out += final_out
        if final_in:
            total_in += final_in
        
        if start_time_str and end_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.fromisoformat(end_time_str)
            total_duration_minutes += (end_time - start_time).total_seconds() / 60

    # Get data for charts
    timeline_labels, cumulative_out, cumulative_in = database.get_timeseries_data(session_ids)
    throughput_per_minute = database.get_throughput_per_minute(session_ids)

    return jsonify({
        'total_thaals_out': total_out,
        'total_thaals_in': total_in,
        'total_expected': total_expected if total_expected > 0 else "N/A",
        'total_duration_minutes': round(total_duration_minutes),
        'timeline_labels': timeline_labels,
        'cumulative_out_data': cumulative_out,
        'cumulative_in_data': cumulative_in,
        'throughput_per_minute': throughput_per_minute
    })


@app.route('/export_by_date')
def export_by_date():
    """Exports all event logs for a given date as a CSV file."""
    date_str = request.args.get('date')
    if not date_str:
        return "Error: No date provided.", 400
    
    query_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    events = database.get_events_for_date(query_date)
    
    csv_data = "id,timestamp,event_type,session_id\n"
    for event in events:
        csv_data += ",".join(map(str, event)) + "\n"
        
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition":
                 f"attachment; filename=thaal_events_{date_str}.csv"}
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)

