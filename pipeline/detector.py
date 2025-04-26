from ultralytics import YOLO
import numpy as np
import cv2

class Detector:
    def __init__(self):
        self.name = "Detector" # Do not change the name of the module as otherwise recording replay would break!
        self.model = None
        self.model_name = "yolov8n-football.pt"

    def start(self, data):
        # TODO: Implement start up procedure of the module
        if not self.model:
            self.model = YOLO(self.model_name)
        print(f"[{self.name}] Model '{self.model_name}' loaded") # For Debugging

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

    def step(self, data):
        # TODO: Implement processing of a single frame
        # The task of the detector is to detect the ball, the goal keepers, the players and the referees if visible.
        # A bounding box needs to be defined for each detected object including the objects center position (X,Y) and its width and height (W, H) 
        # You can return an arbitrary number of objects 
        
        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with detections and classes
        #
        # Detections must be a Nx4 NumPy Tensor, one 4-dimensional vector per detection
        # The detection vector itself is encoded as (X, Y, W, H), so X and Y coordinate first, then width and height of each detection box.
        # X and Y coordinates are the center point of the object, so the bounding box is drawn from (X - W/2, Y - H/2) to (X + W/2, Y + H/2)
        #
        # Classes must be Nx1 NumPy Tensor, one scalar entryx per detection
        # For each corresponding detection, the following mapping must be used
        #   0: Ball
        #   1: GoalKeeper
        #   2: Player
        #   3: Referee

        # Loading Frame and converting it into RGB
        image = data["image"]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Running YOLO on the Frame
        results = self.model(image_rgb)
        result = results[0]
        names = result.names
        
        # print(result.boxes) # Boxes contains the necessary information for detections and classes

        detections = result.boxes.xywh.numpy()
        classes = result.boxes.cls.numpy().reshape(-1, 1)


        return {
            "detections": detections,
            "classes": classes
        }
