import numpy as np
import cv2 as cv

class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier"  # Do not change the name!
        self.team_colors_initialized = False
        self.team_a_color = None
        self.team_b_color = None

    def start(self, data):
        self.team_colors_initialized = False
        self.team_a_color = None
        self.team_b_color = None

    def stop(self, data):
        pass

    def step(self, data):
        image = data["image"]
        tracks = data["tracks"]
        classes = data["trackClasses"]

        team_a_color = (255, 0, 0)  # Default red color for Team A
        team_b_color = (0, 0, 255)  # Default blue color for Team B
        team_classes = [0] * len(tracks)

        player_colors = []
        player_indices = []

        # Step 1: Extract shirt region color for each player
        for i, (x, y, w, h) in enumerate(tracks):
            if classes[i] not in [1, 2, 3]:
                continue

            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(image.shape[1] - 1, int(x + w / 2))
            y2 = min(image.shape[0] - 1, int(y + h / 2))

            shirt_box = image[y1 : y1 + (y2 - y1) // 2, x1:x2]
            if shirt_box.size > 0:
                avg_color = shirt_box.mean(axis=(0, 1))  # BGR
                player_colors.append(avg_color)
                player_indices.append(i)

        # Step 2: Only classify if we have at least 2 players
        if len(player_colors) < 2:
            return {
                "teamAColor": team_a_color,
                "teamBColor": team_b_color,
                "teamClasses": team_classes
            }

        # Step 3: Cluster the colors into 2 teams
        samples = np.float32(player_colors)
        _, labels, centers = cv.kmeans(samples, 2, None,
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            10, cv.KMEANS_PP_CENTERS)

        # Step 4: Set team colors once (after K-means)
        if not self.team_colors_initialized:
            self.team_a_color = tuple(map(int, centers[0]))
            self.team_b_color = tuple(map(int, centers[1]))
            self.team_colors_initialized = True
            print(f"Team colors initialized: A={self.team_a_color}, B={self.team_b_color}") #Debugging output
        
        team_a_color = self.team_a_color
        team_b_color = self.team_b_color

        for idx, label in enumerate(labels.flatten()):
            track_index = player_indices[idx]
            if label == 0:
                team_classes[track_index] = 1  # Team A
            else:  # label == 1
                team_classes[track_index] = -1  # Team B 

        return {
            "teamAColor": team_a_color,
            "teamBColor": team_b_color,
            "teamClasses": team_classes
        }