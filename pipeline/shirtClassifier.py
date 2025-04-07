import numpy as np
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

        player_colors = []
        player_indices = []

        for i, (x, y, w, h) in enumerate(tracks):
            if classes[i] not in [1, 2]: # Only consider players and goalkeepers
                continue

            # Calculate box and comply with image limits
            x1 = int(max(0, x - w / 2))
            y1 = int(max(0, y - h / 2))
            x2 = int(min(image.shape[1], x + w / 2))
            y2 = int(min(image.shape[0], y + h / 2))

            # Only use the jersey (upper) area of the boxes
            trickot_box = image[y1:y1 + (y2 - y1) // 2, x1:x2]
            if trickot_box.size > 0:
                # Extract average jersey colour
                avg_color = trickot_box.mean(axis=(0, 1))
                player_colors.append(avg_color)
                player_indices.append(i)
                # DEBUG: Round and print the average color
                avg_color = avg_color.round().astype(int)
                print(f"Player {i} color: {avg_color}")
        return {
            "teamAColor": (0, 0, 255),
            "teamBColor": (255, 0, 0),
            "teamClasses": [0] * len(tracks)
    }