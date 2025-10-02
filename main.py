import cv2
from flask import Flask, render_template, Response
import threading
from ultralytics import YOLO
import numpy as np

# =============================================================================
# SORT (Simple Online and Realtime Tracking) Algorithm
# This code is based on the original implementation by Alex Bewley.
# =============================================================================

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h) if h != 0 else 0
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        w = np.sqrt(x[2] * x[3]) if (x[2] * x[3]) >= 0 else 0
        h = x[2] / w if w != 0 else 0
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) 
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold = 0.3):
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
        iou_matrix = iou_batch(detections, trackers)
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))
        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]]<iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# =============================================================================
# Main Application
# =============================================================================

app = Flask(__name__)
model = YOLO('runs/detect/train/weights/best.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
track_history = {}
thaal_out_count = 0
thaal_in_count = 0
# Placeholder for tray counts
# tray_out_count = 0
# tray_in_count = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
LINE_POSITION = FRAME_WIDTH // 2 # Center line

LINE_START = (400, 0)
LINE_END = (400, FRAME_HEIGHT)

lock = threading.Lock()
global_frame = None

def camera_and_cv_thread():
    global global_frame, lock, thaal_out_count, thaal_in_count

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True)
        
        detections_list = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf > 0.5:
                    detections_list.append([int(x1), int(y1), int(x2), int(y2), conf])

        if len(detections_list) > 0:
            detections_np = np.array(detections_list)
            tracked_objects = tracker.update(detections_np)
        else:
            tracked_objects = tracker.update(np.empty((0, 5)))


        cv2.line(frame, LINE_START, LINE_END, (0, 255, 0), 2)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)
            
            center_x = (x1 + x2) // 2

            if obj_id not in track_history:
                track_history[obj_id] = []
            track_history[obj_id].append(center_x)

            if len(track_history[obj_id]) > 1:
                prev_x = track_history[obj_id][-2]
                
                if prev_x < LINE_POSITION and center_x >= LINE_POSITION:
                    thaal_out_count += 1
                
                elif prev_x > LINE_POSITION and center_x <= LINE_POSITION:
                    thaal_in_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, f"Thaal Out: {thaal_out_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, f"Thaal In: {thaal_in_count}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        with lock:
            global_frame = frame.copy()

def generate_frames():
    global global_frame, lock
    while True:
        with lock:
            if global_frame is None:
                continue
            
            stream_frame = cv2.resize(global_frame, (960, 540)) # Resize for streaming
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

