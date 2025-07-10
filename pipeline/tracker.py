# Note: A typical tracker design implements a dedicated filter class for keeping the individual state of each track
# The filter class represents the current state of the track (predicted position, size, velocity) as well as additional information (track age, class, missing updates, etc..)
# The filter class is also responsible for assigning a unique ID to each newly formed track
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    computes the Intersection over Union (IoU) for all pairings of 
    two sets of bounding boxes in center format [cx, cy, w, h].
    The output is a matrix of shape (len(a), len(b)) where each element (i, j)
    represents the IoU between the i-th box in `a` and the j-th box in `b`.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    # Center‑ → Corner‑Format
    a_xy1 = a[:, :2] - a[:, 2:] / 2
    a_xy2 = a[:, :2] + a[:, 2:] / 2
    b_xy1 = b[:, :2] - b[:, 2:] / 2
    b_xy2 = b[:, :2] + b[:, 2:] / 2

    inter_tl = np.maximum(a_xy1[:, None, :], b_xy1[None, :, :])
    inter_br = np.minimum(a_xy2[:, None, :], b_xy2[None, :, :])
    inter_wh = np.clip(inter_br - inter_tl, 0, None)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    union = area_a[:, None] + area_b[None, :] - inter_area

    return inter_area / np.clip(union, 1e-6, None)

class Filter:
    '''
    Represents the state and update logic for a single track (object).
    Each track is represented by a filter that keeps track of its position, velocity, age, number of missing frames
    and class ID.
    '''
    
    _next_id = 0
    ALPHA = 0.6
    BETA = 0.15

    def __init__(self, det, cls):
        '''
        Initialize a new filter instance for a detected object.
        
        Parameters: 
        bbox: np.ndarray
            Bounding box of the detected object in the format [X, Y, W, H] where (X, Y) is the top-left corner 
            and (W, H) are the width and height.
        cls: int
            Class ID of the detected object.    
        '''
        self.state = det.astype(float) 
        self.velocity = np.zeros(2, dtype=float)
        self.cls = int(cls)

        self.age = 1
        self.missing = 0

        self.id = Filter._next_id
        Filter._next_id += 1

    def update(self, z):
        '''
        update the track's state with a new measurement and update its velocity. 
        innovation = z-x ; x = x + alpha * innovation; v = beta * v + (1 - beta) * innovation
        where z is the new measurement and x is the current state of the track.
        '''
        innovation = z[:2] - self.state[:2] 
        self.state[:2] += Filter.ALPHA * innovation 
        self.velocity = (1- Filter.BETA) * self.velocity + Filter.BETA * innovation 

        self.state[2:] = z[2:]
        self.missing = 0
        
    def predict(self, flow: np.ndarray):
        '''
        predict the next state of the track based on its current state, velocity and the given optical flow.
        x = x + v
        '''
        self.state[:2] += self.velocity - flow
        self.age += 1
        self.missing += 1
 
    def to_bbox(self):
        return self.state
    

class Tracker:
    def __init__(self, iou_thr: float = 0.3, max_missing: dict[int, int] | None = None, vmax_px: dict[int, int] | None = None):
        super().__init__()
        self.name = "Tracker" 
        self.iou_thr = iou_thr
        self.max_missing = max_missing or {0: 1, 1: 5, 2: 5, 3: 5}
        self.vmax_px = vmax_px or {0: 120, 1: 50, 2: 50, 3: 50}
        self.tracks = [] 
        
    def start(self, data):
        self.tracks = []

    def stop(self, data):
        pass

    def step(self, data):
        '''
        main tracking step: runs prediction, association, update and prepares the output.
        '''
        detections, classes, flow, img_h, img_w = self._parse_input(data)

        # a) Prädiktion
        for tr in self.tracks:
            tr.predict(flow)

        # b) Assoziation
        cost = 1 - iou_matrix(np.vstack([tr.to_bbox() for tr in self.tracks]) if self.tracks else np.empty((0, 4)), 
                              detections)

        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))

        if cost.size:
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1 - self.iou_thr) and self.tracks[r].cls == classes[c]:
                    matches.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_detections.discard(c)

        # c) Update
        for ti, di in matches:
            self.tracks[ti].update(detections[di])
        
        # d) Create new tracks for unmatched detections
        for di in unmatched_detections:
            if self.tracks:
                best_iou = iou_matrix(detections[di : di + 1], np.vstack([tr.to_bbox() for tr in self.tracks])).max()
                if best_iou > 0.45:
                    continue
            self.tracks.append(Filter(detections[di], classes[di]))

        # e) Remove tracks that are not valid anymore

        alive = []
        for tr in self.tracks:
            vmax = self.vmax_px.get(tr.cls, 50)
            tr.velocity = np.clip(tr.velocity, -vmax, vmax)

            cx, cy = tr.state[:2]
            if not self._is_on_field(tr, img_w, img_h):
                continue

            if tr.missing > self.max_missing.get(tr.cls, 5):
                continue

            alive.append(tr)

        self.tracks = alive



        return self._parse_output()  
    
    def _parse_input(self, data):
        """
        Parses the input data to extract necessary information for tracking.
        """
        detections = data.get("detections", np.array([]))
        classes = data.get("classes", np.array([]))
        flow = data.get("opticalFlow", np.zeros((2,)))
        img_h, img_w= data.get("image", np.zeros((1080, 1920, 3))).shape[:2]
        
        return detections, classes, flow, img_h, img_w

    def _is_on_field(self, track, img_w=1920, img_h=1080):
        cx, cy, w, h = track.to_bbox()
        return (0 <= cx <= img_w) and (0 <= cy <= img_h) and (w >= 2) and (h >= 2)
    
    def _parse_output(self):
        """
        returns the current state of the tracks.
        """
        if not self.tracks:
            return {
                "tracks": np.zeros((0, 4)),
                "trackVelocities": np.zeros((0, 2)),
                "trackAge": [],
                "trackClasses": [],
                "trackIds": []
            }
        

        return {
        "tracks": np.vstack([t.to_bbox() for t in self.tracks]),
        "trackVelocities": np.vstack([t.velocity for t in self.tracks]),
        "trackAge": [t.age for t in self.tracks],
        "trackClasses": [t.cls for t in self.tracks],
        "trackIds": [t.id for t in self.tracks]
        }