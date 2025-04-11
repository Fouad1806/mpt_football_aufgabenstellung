# Note: A typical tracker design implements a dedicated filter class for keeping the individual state of each track
# The filter class represents the current state of the track (predicted position, size, velocity) as well as additional information (track age, class, missing updates, etc..)
# The filter class is also responsible for assigning a unique ID to each newly formed track
import numpy as np

global_id_counter = 0

class Filter:
    def __init__(self, z, cls):
        # TODO: Implement filter initializstion
        global global_id_counter
        self.id = global_id_counter
        global_id_counter += 1

        self.bbox = z  # [X, Y, W, H]
        self.cls = cls  
        self.age = 1 
        self.velocity = np.array([0.0, 0.0]) 
        self.missing = 0 
        pass
        
    # TODO: Implement remaining funtionality for an individual track

    def update(self, z):
        # Berechne Velocity
        old_center = np.array(self.bbox[0:2])
        new_center = np.array(z[0:2])
        self.velocity = new_center - old_center
        self.bbox = z.copy()
        self.age += 1
        self.missing = 0

    def predict(self):
        dt = 1 
        self.bbox[0] += self.velocity[0] * dt
        self.bbox[1] += self.velocity[1] * dt
        self.age += 1
        self.missing += 1

    
class Tracker:
    def __init__(self):
        self.name = "Tracker" # Do not change the name of the module as otherwise recording replay would break!
        self.tracks = []
        
    def start(self, data):
        # TODO: Implement start up procedure of the module
        self.tracks = []
        pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

    def step(self, data):
        detections = data.get("detections", np.array([]))
        classes = data.get("classes", np.array([]))
        distance_threshold = 50.0 
        existing_tracks = self.tracks.copy()  # Kopie für matching
        updated_tracks = []

        for i in range(len(detections)):
            z = detections[i]
            cls = int(classes[i])
            center_det = z[0:2]
            best_track = None
            best_distance = float('inf')

            # Innerer Loop
            for track in existing_tracks:
                center_track = track.bbox[0:2]
                dist = np.linalg.norm(np.array(center_det) - np.array(center_track))
                if dist < best_distance:
                    best_distance = dist
                    best_track = track

        if best_track is not None and best_distance < distance_threshold:
                best_track.update(z)
                updated_tracks.append(best_track)
                existing_tracks.remove(best_track)
        else:
                new_track = Filter(z, cls)
                updated_tracks.append(new_track)

        for track in existing_tracks:
            track.predict() 
            updated_tracks.append(track)

        self.tracks = updated_tracks

        N = len(self.tracks)
        return {
            "tracks": np.zeros((N, 4)) if N == 0 else np.array([t.bbox for t in self.tracks]),
            "trackVelocities": np.zeros((N, 2)) if N == 0 else np.array([t.velocity for t in self.tracks]),
            "trackAge": [] if N == 0 else [t.age for t in self.tracks],
            "trackClasses": [] if N == 0 else [t.cls for t in self.tracks],
            "trackIds": [] if N == 0 else [t.id for t in self.tracks],

        }
