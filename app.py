from flask import Flask, render_template, Response, jsonify, make_response, request
import threading
import io
import csv
import cv2
from datetime import datetime

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

# --- DASHBOARD ROUTES ---

@app.route('/dashboard')
def dashboard():
    """Renders the main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/dashboard_data')
def dashboard_data():
    """API endpoint to fetch processed data for the dashboard."""
    date_str = request.args.get('date')
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    try:
        selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    session_ids = database.get_sessions_for_date(selected_date)
    if not session_ids:
        return jsonify({"message": "No data available for this date."})

    all_events = []
    for session_id in session_ids:
        _, events = database.get_session_events(session_id)
        if events:
            all_events.extend(events)

    if not all_events:
        return jsonify({"message": "No event data found for the sessions on this date."})

    # Process events to calculate metrics
    total_out = sum(1 for _, event_type in all_events if event_type == 'THAAL_OUT')
    total_in = sum(1 for _, event_type in all_events if event_type == 'THAAL_IN')
    
    start_time = datetime.fromisoformat(all_events[0][0])
    end_time = datetime.fromisoformat(all_events[-1][0])
    duration_minutes = round((end_time - start_time).total_seconds() / 60)

    # Calculate hourly throughput for the bar chart
    hourly_throughput = {}
    for timestamp, event_type in all_events:
        if event_type == 'THAAL_OUT':
            hour = datetime.fromisoformat(timestamp).strftime('%H:00')
            hourly_throughput[hour] = hourly_throughput.get(hour, 0) + 1
    sorted_throughput = dict(sorted(hourly_throughput.items()))

    # --- NEW: Calculate cumulative data for the line chart ---
    timeline_labels, cumulative_out, cumulative_in = database.get_timeseries_data(session_ids)

    return jsonify({
        "date": date_str,
        "total_thaals_out": total_out,
        "total_thaals_in": total_in,
        "total_duration_minutes": duration_minutes,
        "hourly_throughput": sorted_throughput,
        "timeline_labels": timeline_labels,
        "cumulative_out_data": cumulative_out,
        "cumulative_in_data": cumulative_in
    })

@app.route('/export_by_date')
def export_by_date():
    """Exports all event data for a specific date as a CSV file."""
    date_str = request.args.get('date')
    if not date_str:
        return "Please provide a date.", 400
    
    try:
        selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD.", 400

    all_events = database.get_events_for_date(selected_date)
    
    if not all_events:
        return "No data for the selected date.", 404

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Timestamp', 'EventType', 'SessionID'])
    writer.writerows(all_events)
    output.seek(0)
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=thaal_log_{date_str}.csv"
    response.headers["Content-type"] = "text/csv"
    
    return response

if __name__ == '__main__':
    database.init_db()
    cv_thread = threading.Thread(target=vision_processor.run)
    cv_thread.daemon = True
    cv_thread.start()
    app.run(host='0.0.0.0', port=5001, debug=False)

