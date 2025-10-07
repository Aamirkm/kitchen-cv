import cv2
import threading
from ultralytics import YOLO
import time
import database # Import our new data layer

class VisionProcessor:
    def __init__(self):
        # --- CV Model and Camera Setup ---
        self.model = YOLO('runs/detect/train/weights/best.pt')
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # --- State Variables ---
        self.lock = threading.Lock() # Protects shared data
        self.is_service_active = False
        self.thaal_out_count = 0
        self.thaal_in_count = 0
        self.current_session_id = None
        self.track_history = {}
        self.latest_frame = None

        # --- Constants ---
        self.FRAME_WIDTH = 1280
        self.FRAME_HEIGHT = 720
        self.LINE_OUT_POSITION = self.FRAME_WIDTH // 3
        self.LINE_IN_POSITION = 750 # (self.FRAME_WIDTH // 3) * 2
        self.LINE_OUT_START = (500, 0)
        self.LINE_OUT_END = (250, self.FRAME_HEIGHT)
        self.LINE_IN_START = (self.LINE_IN_POSITION, 0)
        self.LINE_IN_END = (self.LINE_IN_POSITION, self.FRAME_HEIGHT)

    def _process_frame(self):
        """Internal method to run the CV logic on a single frame."""
        success, frame = self.cap.read()
        if not success:
            time.sleep(0.1)
            return

        annotated_frame = frame.copy()

        if self.is_service_active:
            results = self.model.track(annotated_frame, persist=True, verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) // 2
                    
                    if track_id not in self.track_history:
                        self.track_history[track_id] = {"positions": [], "has_been_counted": False}
                    
                    self.track_history[track_id]["positions"].append(center_x)

                    if len(self.track_history[track_id]["positions"]) > 1:
                        prev_x = self.track_history[track_id]["positions"][-2]

                        if not self.track_history[track_id]["has_been_counted"]:
                            if prev_x > self.LINE_OUT_POSITION and center_x <= self.LINE_OUT_POSITION:
                                self.thaal_out_count += 1
                                database.log_event("THAAL_OUT", self.current_session_id)
                                self.track_history[track_id]["has_been_counted"] = True

                            elif prev_x < self.LINE_IN_POSITION and center_x >= self.LINE_IN_POSITION:
                                self.thaal_in_count += 1
                                database.log_event("THAAL_IN", self.current_session_id)
                                self.track_history[track_id]["has_been_counted"] = True
                    
                    label = f"ID:{track_id} thaal"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.line(annotated_frame, self.LINE_OUT_START, self.LINE_OUT_END, (0, 255, 0), 2)
        cv2.line(annotated_frame, self.LINE_IN_START, self.LINE_IN_END, (0, 0, 255), 2)
        
        status_text = "Service Active" if self.is_service_active else "Service Stopped"
        status_color = (0, 255, 0) if self.is_service_active else (0, 0, 255)
        cv2.putText(annotated_frame, status_text, (20, self.FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.putText(annotated_frame, f"Thaal Out: {self.thaal_out_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(annotated_frame, f"Thaal In: {self.thaal_in_count}", (self.FRAME_WIDTH - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        with self.lock:
            self.latest_frame = annotated_frame.copy()

    def run(self):
        """The main loop for the CV thread."""
        while True:
            self._process_frame()

    def get_frame(self):
        """Returns the latest annotated frame for streaming."""
        with self.lock:
            return self.latest_frame
            
    def start_service(self):
        with self.lock:
            if not self.is_service_active:
                self.is_service_active = True
                self.thaal_out_count = 0
                self.thaal_in_count = 0
                self.track_history.clear()
                self.current_session_id = f"session_{int(time.time())}"
                database.log_event("SERVICE_START", self.current_session_id)
                print(f"Service started with session ID: {self.current_session_id}")
                return True
        return False

    def stop_service(self):
        with self.lock:
            if self.is_service_active:
                self.is_service_active = False
                database.log_event("SERVICE_STOP", self.current_session_id)
                print(f"Service stopped for session ID: {self.current_session_id}")
                self.current_session_id = None
                return True
        return False

    def reset_counts(self):
        with self.lock:
            self.thaal_out_count = 0
            self.thaal_in_count = 0
            self.track_history.clear()
            print("Counts have been reset.")
            return True

    def get_status(self):
        with self.lock:
            return {
                "service_active": self.is_service_active,
                "thaal_out": self.thaal_out_count,
                "thaal_in": self.thaal_in_count
            }
