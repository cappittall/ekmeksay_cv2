import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

import numpy as np
from filterpy.kalman import KalmanFilter

class SortTracker:
    count = 0

    def __init__(self, bbox, score, label_id):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R *= 10
        self.kf.P *= 1000
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = bbox.reshape((-1, 1))
        self.time_since_update = 0
        self.id = SortTracker.count
        SortTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = score
        self.label_id = label_id

    def update(self, bbox, score, label_id):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox.reshape((-1, 1)))
        self.score = score
        self.label_id = label_id

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1][:4]

    def get_state(self):
        return self.kf.x[:4].flatten()
    
def iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    X1, Y1, X2, Y2 = bbox2

    w = max(0, min(x2, X2) - max(x1, X1))
    h = max(0, min(y2, Y2) - max(y1, Y1))

    intersection = w * h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (X2 - X1) * (Y2 - Y1)
    union = area1 + area2 - intersection

    return intersection / union

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(matched_indices).T
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
                
    def update(self, dets, scores, label_ids):
        self.frame_count += 1
        to_del = []
        ret = []
 
        trks = []
        for tracker in self.trackers:
            pos = tracker.predict()
            if np.any(np.isnan(pos)):
                to_del.append(tracker)
            else:
                trks.append(pos)
        trks = np.array(trks)

        for t in reversed(to_del):
            self.trackers.remove(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]], scores[m[0]], label_ids[m[0]])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = SortTracker(dets[i], scores[i], label_ids[i])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                entry = np.zeros((1, 7))
                entry[0, :4] = d
                entry[0, 4] = trk.id
                entry[0, 5] = trk.score
                entry[0, 6] = trk.label_id
                ret.append(entry)
            i -= 1
            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        return np.vstack(ret) if ret else np.array([])


