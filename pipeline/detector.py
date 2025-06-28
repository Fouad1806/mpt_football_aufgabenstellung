from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Detector:
    def __init__(self):
        self.name = "Detector"
        self.model = None
        self.model_name = "yolov8n-football.pt"
        self.iou_per_frame = []

    def start(self, data):
        # TODO: Implement start up procedure of the module
        if not self.model:
            self.model = YOLO(self.model_name)
        print(f"[{self.name}] Model '{self.model_name}' loaded") # For Debugging

        # TODO:FRAGEN OB ES KLAR GEHT AUCH DIE EIGENEN AUFNAHMEN IN PICKLE FILES ZU SPEICHERN?

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        import matplotlib.pyplot as plt

        if self.iou_per_frame:
            plt.figure()
            plt.plot(self.iou_per_frame, marker='o')
            plt.xlabel("Frame")
            plt.ylabel("Durchschnittlicher IoU")
            plt.title("IoU-Verlauf pro Frame Nano YOLO | Video 20")
            plt.ylim(0.2, 1)  # <<< Y-Achse manuell festlegen
            plt.grid(True)
            plt.savefig("plots/iou_verlauf.png")
            plt.show()
        else:
            print("Keine IoU-Werte gesammelt.")


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

        # print("data:", data)

        image = data["image"]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Running YOLO on the Frame
        results = self.model(image_rgb)
        result = results[0]
        boxes = result.boxes
        detections = boxes.xywh.numpy()
        classes = result.boxes.cls.numpy().reshape(-1, 1)

        def extract_xyxy(x):
            return [x[0] - x[2]/2, x[1] - x[3]/2, x[0] + x[2]/2, x[1] + x[3]/2]

        if "detections" in data:
            ground_truth = data.get("detections") # contains x_center, y_center, width, height
            area_b = np.apply_along_axis(extract_xyxy, axis=1, arr=ground_truth)
            # print(result.boxes) # Boxes contains the necessary information for detections and classes

            area_a = boxes.xyxy # Contains x_left, y_top, x_right, y_bottom

            confidences = result.boxes.conf.numpy()

            ious = []

            for pred in area_a.numpy():
                best_iou = 0
                for gt in area_b:
                    iou = self.compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                ious.append(best_iou)

            if ious:
                mean_iou = np.mean(ious)
            else:
                mean_iou = 0

            self.iou_per_frame.append(mean_iou)

            print(f"[Frame {data['counter']}] Mean IoU: {mean_iou:.3f}")

            #self.plot_pr_curve_with_iou(area_a, area_b, confidences, data["counter"])

        else:
            ground_truth = None
        
        
        # self.plot_pr_curve(confidences)

        return {
            "detections": detections,
            "classes": classes
        }

        # TODO: Calculate IoU
        
  
    def compute_iou(self,boxA, boxB):
        """
        Calculates the IoU of two Bounding Boxes. Both of them have to be in the format (x1, y1, x2, y2)

        This function returns an IoU Value between 0.0 and 1.0
        """

        xA = max(boxA[0], boxB[0]) # top left corner (welche ist weiter rechts)
        yA = max(boxA[1], boxB[1]) # top W"kante" (Koordinatensystem wächst nach unten)
        xB = min(boxA[2], boxB[2]) # top right corner (welche ist weiter links)
        yB = min(boxA[3], boxB[3]) # bottom "kante"  

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight


        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        unionArea = boxAArea + boxBArea

        iou = interArea / unionArea

        return iou