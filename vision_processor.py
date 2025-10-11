import cv2
from ultralytics import YOLO
import threading
import time
from datetime import datetime
import database

class VisionProcessor:
    """
    Handles all computer vision, object tracking, and counting logic
    in a self-contained, threaded class.
    """
    def __init__(self, model_path):
        # --- Store arguments ---
        self.model_path = model_path

        # --- CV and State Variables ---
        self.model = YOLO(self.model_path)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.track_history = {}
        self.thaal_out_count = 0
        self.thaal_in_count = 0
        self.is_service_active = False
        self.current_session_id = None
        self.annotated_frame = None
        self.start_time = None
        self.expected_thaals = None
        
        # --- Constants ---
        self.FRAME_WIDTH = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.FRAME_HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.LINE_OUT_POSITION = self.FRAME_WIDTH // 3
        self.LINE_IN_POSITION = 750 #(self.FRAME_WIDTH // 3) * 2

        # --- Threading ---
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.auto_stop_timer = None

    def _reset_counts(self):
        """Internal method to reset counts and tracking history."""
        self.thaal_out_count = 0
        self.thaal_in_count = 0
        self.track_history.clear()
        self.start_time = None
        self.expected_thaals = None

    def start_service(self, expected_thaals):
        """Starts a new service session."""
        with self.state_lock:
            if not self.is_service_active:
                self._reset_counts()
                self.is_service_active = True
                self.start_time = datetime.now()
                self.expected_thaals = expected_thaals if expected_thaals and expected_thaals.strip() != "" else "N/A"

                self.current_session_id = self.start_time.strftime("session-%Y-%m-%d-%H%M%S")
                
                db_expected = int(expected_thaals) if self.expected_thaals != "N/A" else None
                database.create_session(self.current_session_id, self.start_time, db_expected)
                print(f"Service started with Session ID: {self.current_session_id}")

                if self.auto_stop_timer:
                    self.auto_stop_timer.cancel()
                self.auto_stop_timer = threading.Timer(3 * 60 * 60, self.stop_service) 
                self.auto_stop_timer.start()

    def stop_service(self):
        """Stops the current service session and saves final counts."""
        with self.state_lock:
            if self.is_service_active:
                self.is_service_active = False
                stop_time = datetime.now()
                database.end_session(self.current_session_id, stop_time)
                database.update_session_summary(self.current_session_id, self.thaal_out_count, self.thaal_in_count)
                print(f"Service stopped for Session ID: {self.current_session_id}")
                # Don't reset counts immediately, so the UI can show final numbers
                self.current_session_id = None
                
                if self.auto_stop_timer:
                    self.auto_stop_timer.cancel()
                    self.auto_stop_timer = None
    
    def get_status(self):
        """Returns the current status for the web interface."""
        with self.state_lock:
            duration_minutes = 0
            if self.is_service_active and self.start_time:
                duration = datetime.now() - self.start_time
                duration_minutes = int(duration.total_seconds() // 60)

            return {
                "service_active": self.is_service_active,
                "thaal_out": self.thaal_out_count,
                "thaal_in": self.thaal_in_count,
                "expected_thaals": self.expected_thaals,
                "service_duration_minutes": duration_minutes
            }

    def get_annotated_frame(self):
        """Returns the latest annotated frame for streaming."""
        with self.frame_lock:
            if self.annotated_frame is not None:
                return self.annotated_frame.copy()
        return None

    def run(self):
        """The main loop for the computer vision thread."""
        while True:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.1)
                continue
            
            # --- CV processing happens here ---
            processed_frame = frame.copy()
            if self.is_service_active:
                results = self.model.track(processed_frame, persist=True, verbose=False)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(boxes, track_ids):
                        # ... (tracking and counting logic remains the same) ...
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) // 2

                        if track_id not in self.track_history:
                            self.track_history[track_id] = { "positions": [], "has_been_counted": False }
                        
                        self.track_history[track_id]["positions"].append(center_x)

                        if len(self.track_history[track_id]["positions"]) > 1:
                            prev_x = self.track_history[track_id]["positions"][-2]

                            if not self.track_history[track_id]["has_been_counted"]:
                                if prev_x > self.LINE_OUT_POSITION and center_x <= self.LINE_OUT_POSITION:
                                    with self.state_lock:
                                        self.thaal_out_count += 1
                                        database.log_event("THAAL_OUT", self.current_session_id)
                                    self.track_history[track_id]["has_been_counted"] = True

                                elif prev_x < self.LINE_IN_POSITION and center_x >= self.LINE_IN_POSITION:
                                    with self.state_lock:
                                        self.thaal_in_count += 1
                                        database.log_event("THAAL_IN", self.current_session_id)
                                    self.track_history[track_id]["has_been_counted"] = True
                
                processed_frame = results[0].plot()
            
            # --- Drawing on the frame ---
            # Draw lines regardless of service state
            cv2.line(processed_frame, (500, 0), (250, self.FRAME_HEIGHT), (0, 255, 0), 2)
            cv2.line(processed_frame, (self.LINE_IN_POSITION, 0), (self.LINE_IN_POSITION, self.FRAME_HEIGHT), (0, 0, 255), 2)
            
            # THE COUNT TEXT IS NO LONGER DRAWN HERE

            with self.frame_lock:
                self.annotated_frame = processed_frame

            time.sleep(0.01)

        self.cap.release()

