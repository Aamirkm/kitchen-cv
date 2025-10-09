import cv2
from ultralytics import YOLO
import threading
import time
from datetime import datetime
import database  # <-- Reverted to direct import

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
        
        # --- Constants ---
        self.FRAME_WIDTH = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.FRAME_HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.LINE_OUT_POSITION = self.FRAME_WIDTH // 3
        self.LINE_IN_POSITION = (self.FRAME_WIDTH // 3) * 2

        # --- Threading ---
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.auto_stop_timer = None

    def _reset_counts(self):
        """Internal method to reset counts and tracking history."""
        self.thaal_out_count = 0
        self.thaal_in_count = 0
        self.track_history.clear()

    def start_service(self, expected_thaals):
        """Starts a new service session."""
        with self.state_lock:
            if not self.is_service_active:
                self._reset_counts()
                self.is_service_active = True
                # --- CHANGE: Use a human-readable, timestamp-based session ID ---
                start_time = datetime.now()
                self.current_session_id = start_time.strftime("session-%Y-%m-%d-%H%M%S")
                
                # We need to pass the start_time object to the database function
                database.create_session(self.current_session_id, start_time, expected_thaals)
                print(f"Service started with Session ID: {self.current_session_id}")

                # Start the 3-hour auto-stop timer
                if self.auto_stop_timer:
                    self.auto_stop_timer.cancel()
                # Set timer for 3 hours (10800 seconds)
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
                self._reset_counts()
                self.current_session_id = None
                
                # Cancel the auto-stop timer if service is stopped manually
                if self.auto_stop_timer:
                    self.auto_stop_timer.cancel()
                    self.auto_stop_timer = None
    
    def get_status(self):
        """Returns the current status for the web interface."""
        with self.state_lock:
            return {
                "service_active": self.is_service_active,
                "thaal_out": self.thaal_out_count,
                "thaal_in": self.thaal_in_count
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

            # Only process if the service is active
            if self.is_service_active:
                results = self.model.track(frame, persist=True, verbose=False)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) // 2

                        if track_id not in self.track_history:
                            self.track_history[track_id] = { "positions": [], "has_been_counted": False }
                        
                        self.track_history[track_id]["positions"].append(center_x)

                        if len(self.track_history[track_id]["positions"]) > 1:
                            prev_x = self.track_history[track_id]["positions"][-2]

                            if not self.track_history[track_id]["has_been_counted"]:
                                # OUTGOING: Crosses LEFT line from RIGHT-TO-LEFT
                                if prev_x > self.LINE_OUT_POSITION and center_x <= self.LINE_OUT_POSITION:
                                    with self.state_lock:
                                        self.thaal_out_count += 1
                                        database.log_event("THAAL_OUT", self.current_session_id)
                                    self.track_history[track_id]["has_been_counted"] = True

                                # INCOMING: Crosses RIGHT line from LEFT-TO-RIGHT
                                elif prev_x < self.LINE_IN_POSITION and center_x >= self.LINE_IN_POSITION:
                                    with self.state_lock:
                                        self.thaal_in_count += 1
                                        database.log_event("THAAL_IN", self.current_session_id)
                                    self.track_history[track_id]["has_been_counted"] = True
                
                # Get the annotated frame from the results
                annotated_frame = results[0].plot()
            else:
                # If service is not active, just use the raw frame
                annotated_frame = frame
            
            # Draw lines and counts regardless of service state
            cv2.line(annotated_frame, (self.LINE_OUT_POSITION, 0), (self.LINE_OUT_POSITION, self.FRAME_HEIGHT), (0, 255, 0), 2)
            cv2.line(annotated_frame, (self.LINE_IN_POSITION, 0), (self.LINE_IN_POSITION, self.FRAME_HEIGHT), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Thaals Out: {self.thaal_out_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(annotated_frame, f"Thaals In: {self.thaal_in_count}", (self.FRAME_WIDTH - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Update the global annotated frame
            with self.frame_lock:
                self.annotated_frame = annotated_frame

            # Small delay to prevent this thread from hogging 100% CPU
            time.sleep(0.01)

        self.cap.release()

