import numpy as np
import cv2 as cv

class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier"  # Do not change the name!

    def start(self, data):
        pass

    def stop(self, data):
        pass

    def step(self, data):
        image = data["image"]
        tracks = data["tracks"]
        classes = data["trackClasses"]

        team_a_color = (0, 0, 255)  # Red
        team_b_color = (255, 0, 0)  # Blue
        team_classes = [0] * len(tracks)

        player_colors = []
        player_indices = []

        # Step 1: Extract shirt region color for each player
        for i, (x, y, w, h) in enumerate(tracks):
            if classes[i] not in [2]:
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

        # Step 4: Determine which cluster is closer to red
        red = np.array([0, 0, 255], dtype=np.float32)
        dists_to_red = [np.linalg.norm(center - red) for center in centers]

        red_cluster = int(np.argmin(dists_to_red))
        blue_cluster = 1 - red_cluster

        # Step 5: Assign teamClasses accordingly
        for idx, label in enumerate(labels.flatten()):
            track_index = player_indices[idx]
            if label == red_cluster:
                team_classes[track_index] = 1  # Team A
            else:
                team_classes[track_index] = 2  # Team B

        return {
            "teamAColor": team_a_color,
            "teamBColor": team_b_color,
            "teamClasses": team_classes
        }