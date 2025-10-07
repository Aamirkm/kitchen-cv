from flask import Flask, render_template, Response, jsonify, make_response, request
import threading
import io
import csv
import cv2
from datetime import datetime

from vision_processor import VisionProcessor
import database

app = Flask(__name__)
vision_processor = VisionProcessor()

def generate_frames():
    """Generator function to yield frames for the MJPEG stream."""
    while True:
        frame = vision_processor.get_frame()
        if frame is None:
            continue
        
        stream_frame = cv2.resize(frame, (960, 540))
        (flag, encoded_image) = cv2.imencode('.jpg', stream_frame)
        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

# --- Main Control Routes ---
@app.route('/')
def index():
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_service', methods=['POST'])
def start_service():
    """Starts a new service session, optionally with an expected count."""
    data = request.get_json()
    expected_thaals = data.get('expected_thaals') if data else None
    
    if expected_thaals:
        try:
            expected_thaals = int(expected_thaals)
        except (ValueError, TypeError):
            expected_thaals = None

    if vision_processor.start_service(expected_thaals):
        return jsonify(success=True, status="Service started.")
    return jsonify(success=False, status="Service already active.")

@app.route('/stop_service', methods=['POST'])
def stop_service():
    if vision_processor.stop_service():
        return jsonify(success=True, status="Service stopped.")
    return jsonify(success=False, status="Service not active.")

@app.route('/status')
def status():
    return jsonify(vision_processor.get_status())

# --- DASHBOARD ROUTES ---
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/dashboard_data')
def dashboard_data():
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    try:
        selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    sessions = database.get_sessions_for_date(selected_date)
    if not sessions:
        return jsonify({"message": "No data available for this date."})

    total_out = 0
    total_in = 0
    total_expected = 0
    total_duration = 0
    session_ids = []

    for session in sessions:
        session_id, start_time, end_time, expected, final_out, final_in = session
        session_ids.append(session_id)
        
        if final_out is not None: total_out += final_out
        if final_in is not None: total_in += final_in
        
        # --- THIS IS THE FIX ---
        # Check if 'expected' is a valid number before adding it.
        if expected is not None and isinstance(expected, int):
            total_expected += expected
        
        if start_time and end_time:
            duration = datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)
            total_duration += duration.total_seconds()

    all_events = []
    # Fetch event logs only if needed for charts
    if session_ids:
        all_events = database.get_events_for_date(selected_date)


    hourly_throughput = {}
    if all_events:
        for _, timestamp, event_type, _ in all_events:
            if event_type == 'THAAL_OUT':
                hour = datetime.fromisoformat(timestamp).strftime('%H:00')
                hourly_throughput[hour] = hourly_throughput.get(hour, 0) + 1
    
    sorted_throughput = dict(sorted(hourly_throughput.items()))
    timeline_labels, cumulative_out, cumulative_in = database.get_timeseries_data(session_ids)

    return jsonify({
        "date": date_str,
        "total_thaals_out": total_out,
        "total_thaals_in": total_in,
        "total_duration_minutes": round(total_duration / 60),
        "total_expected_thaals": total_expected if total_expected > 0 else "N/A",
        "hourly_throughput": sorted_throughput,
        "timeline_labels": timeline_labels,
        "cumulative_out_data": cumulative_out,
        "cumulative_in_data": cumulative_in
    })

@app.route('/export_by_date')
def export_by_date():
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

