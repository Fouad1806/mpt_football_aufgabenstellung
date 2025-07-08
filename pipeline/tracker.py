# Note: A typical tracker design implements a dedicated filter class for keeping the individual state of each track
# The filter class represents the current state of the track (predicted position, size, velocity) as well as additional information (track age, class, missing updates, etc..)
# The filter class is also responsible for assigning a unique ID to each newly formed track
import numpy as np
from scipy.optimize import linear_sum_assignment


#global_id_counter = 0

class Filter:
    '''
    Represents the state and update logic for a single track (object).
    Each track is represented by a filter that keeps track of its position, velocity, age, number of missing frames
    and class ID.
    '''
    _next_id = 0
    vel_alpha = 0.8  # Gewicht für glättendes Velocity-Update

    def __init__(self, z, cls):
        '''
        Initialize a new filter instance for a detected object.
        
        Parameters: 
        bbox: np.ndarray
            Bounding box of the detected object in the format [X, Y, W, H] where (X, Y) is the top-left corner 
            and (W, H) are the width and height.
        cls: int
            Class ID of the detected object.    
        '''
        self.state = z.copy()  # [X, Y, W, H]
        self.age = 1 
        self.velocity = np.zeros(2, dtype=float)
        self.missing = 0 
        self.class_id = int(cls)

        self.id = Filter._next_id
        Filter._next_id += 1
        
    # TODO: Implement remaining funtionality for an individual track

    def update(self, z):
        '''
        update the track's state with a new measurement and update its velocity.
        '''
        old_center = self.state[:2].copy()
        self.state = Filter.vel_alpha * self.state + (1 - Filter.vel_alpha) * z  # Glättung der Position
        # Berechne gemessene Geschwindigkeit
        meas_vel = z[:2] - old_center
        # Glättung der Velocity
        self.velocity = self.vel_alpha * self.velocity + (1 - self.vel_alpha) * meas_vel
        
        # Update der Position
        #self.state[:2] += self.velocity
        self.missing = 0

    def predict(self, optical_flow: np.ndarray):
        '''
        predict the next state of the track based on its current state, velocity and the given optical flow.
        '''
        self.state[:2] += self.velocity # + optical_flow  # Update position based on 
        self.age += 1
        self.missing += 1
    
    def is_valid(self, max_missing=5):
        return self.missing <= max_missing
 
    def to_bbox(self):
        '''
        returns the bounding box of the current track.
        '''
        return self.state.copy() 
    

class Tracker:
    def __init__(self, max_missing=3, distance_threshold=80.0):
        super().__init__()
        self.name = "Tracker" # Do not change the name of the module as otherwise recording replay would break!
        self.max_missing = max_missing
        self.distance_threshold = distance_threshold
        self.tracks = [] 
        
    def start(self, data):
        # TODO: Implement start up procedure of the module
        self.tracks = []
        Filter._next_id = 0 
        # pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

    def step(self, data):
        detections, classes, optical_flow = self._parse_input(data)

        for tr in self.tracks:
            tr.predict(optical_flow)

        assignments, assigned_tracks, assigned_detections = self._assign(detections)

        self._update_tracks(assignments, assigned_tracks, assigned_detections, detections, classes)

        return self._parse_output()  
    
    def _parse_input(self, data):
        """
        Parses the input data to extract necessary information for tracking.
        """
        detections = data.get("detections", np.array([]))
        classes = data.get("classes", np.array([]))
        optical_flow = data.get("opticalFlow", np.zeros((2,)))
        
        return detections, classes, optical_flow

    def _assign(self, detections):
        '''
        assigns detections to existing tracks using the Hungarian algorithm and a distance threshold.
        '''
        M, N = len(self.tracks), len(detections)
        assignments, assigned_tracks, assigned_detections = [], set(), set()
        if M > 0 and N > 0:
            preds = np.array([tr.to_bbox()[:2] for tr in self.tracks])
            detxy = detections[:, :2]
            cost = np.linalg.norm(preds[:, None, :] - detxy[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            assignments = [(r, c) for r, c in zip(row_ind, col_ind) if cost[r, c] <= self.distance_threshold]
            assigned_tracks = {r for r, _ in assignments}
            assigned_detections = {c for _, c in assignments}
        return assignments, assigned_tracks, assigned_detections
    
   
    def _update_tracks(self, assignments, assigned_tr, assigned_d, detections, classes):
        """
        Updates all tracks based on the new detections and optical flow.
        """
        new_tracks = []

        # matched
        for ti, di in assignments:
            self.tracks[ti].update(detections[di])
            new_tracks.append(self.tracks[ti])
        # unmatched tracks
        for indx, track in enumerate(self.tracks):
            if indx not in assigned_tr:
                if track.is_valid(self.max_missing):
                    new_tracks.append(track)
        # new detections
        for dindx, det in enumerate(detections):
            if dindx not in assigned_d:
                new_tracks.append(Filter(det, classes[dindx]))

        self.tracks = new_tracks

    
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
        "tracks": np.array([t.to_bbox() for t in self.tracks]) if self.tracks else np.zeros((0, 4)),
        "trackVelocities": np.array([t.velocity for t in self.tracks]) if self.tracks else np.zeros((0, 2)),
        "trackAge": [t.age for t in self.tracks] if self.tracks else [],
        "trackClasses": [t.class_id for t in self.tracks] if self.tracks else [],
        "trackIds": [t.id for t in self.tracks] if self.tracks else []
        }