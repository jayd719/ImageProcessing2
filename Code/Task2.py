import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN  # For clustering keypoints

# Set working directory
os.chdir('./Code')

# Constants
LOWES = 0.6
THRESHOLD = 30
OUTPUT_FOLDER = "../Detected Objects"
MIN_BOX_SIZE = 50  # Minimum width/height for bounding boxes
CLUSTER_EPS = 30
BOX_BUFFER = 10  # Add a small buffer to bounding box size

# Object names mapping
object_names = [
    "speed stick", "calculator", "instant rice", "toothpaste", "dryer sheets",
    "peanut butter", "gum", "old spice", "gift card", "seasoning"
]

# Scene-specific box limits
scene_object_limits = {  # Maximum boxes allowed for each scene
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10
}

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Function for clustering keypoints
def cluster_keypoints(points):
    if len(points) < 2:
        return points  # Not enough points for clustering
    clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=2).fit(points)
    labels = clustering.labels_
    clustered_points = [points[i] for i in range(len(points)) if labels[i] != -1]
    return clustered_points if clustered_points else points  # Fallback to all points if clustering fails

# Match function with refined bounding box logic
def match_with_bounding_box(OKeypoints, ODescriptors, SKeypoints, SDescriptors):
    distance = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = distance.knnMatch(ODescriptors, SDescriptors, k=2)
    best = []
    for firstMatch, secondMatch in matches:
        if firstMatch.distance < (LOWES * secondMatch.distance):
            best.append(firstMatch)
    if len(best) >= THRESHOLD:
        points = np.array([SKeypoints[m.trainIdx].pt for m in best])
        clustered_points = cluster_keypoints(points)
        if len(clustered_points) > 0:
            x_min, y_min = np.min(clustered_points, axis=0).astype(int)
            x_max, y_max = np.max(clustered_points, axis=0).astype(int)

            # Add a buffer to bounding box size
            x_min = max(0, x_min - BOX_BUFFER)
            y_min = max(0, y_min - BOX_BUFFER)
            x_max = x_max + BOX_BUFFER
            y_max = y_max + BOX_BUFFER

            # Ensure bounding box meets minimum size
            if (x_max - x_min) < MIN_BOX_SIZE or (y_max - y_min) < MIN_BOX_SIZE:
                x_min, y_min = np.min(points, axis=0).astype(int)
                x_max, y_max = np.max(points, axis=0).astype(int)

            return True, (x_min, y_min, x_max, y_max)
    return False, None

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Define paths for objects and scenes
OArray = ['../Objects/O1.jpg', '../Objects/O2.jpg', '../Objects/O3.jpg', 
          '../Objects/O4.jpg', '../Objects/O5.jpg', '../Objects/O6.jpg',
          '../Objects/O7.jpg', '../Objects/O8.jpg', '../Objects/O9.jpg',
          '../Objects/O10.jpg']

SArray = ['../Scenes/S1_front.jpg', '../Scenes/S1_right.jpg', '../Scenes/S1_left.jpg',
          '../Scenes/S2_front.jpg', '../Scenes/S2_right.jpg', '../Scenes/S2_left.jpg',
          '../Scenes/S3_front.jpg', '../Scenes/S3_right.jpg', '../Scenes/S3_left.jpg',
          '../Scenes/S4_front.jpg', '../Scenes/S4_right.jpg', '../Scenes/S4_left.jpg',
          '../Scenes/S5_front.jpg', '../Scenes/S5_right.jpg', '../Scenes/S5_left.jpg',
          '../Scenes/S6_front.jpg', '../Scenes/S6_right.jpg', '../Scenes/S6_left.jpg',
          '../Scenes/S7_front.jpg', '../Scenes/S7_right.jpg', '../Scenes/S7_left.jpg',
          '../Scenes/S8_front.jpg', '../Scenes/S8_right.jpg', '../Scenes/S8_left.jpg',
          '../Scenes/S9_front.jpg', '../Scenes/S9_right.jpg', '../Scenes/S9_left.jpg',
          '../Scenes/S10_front.jpg', '../Scenes/S10_right.jpg', '../Scenes/S10_left.jpg']


if __name__ =="__main__":

    # Precompute descriptors for objects
    OArrayKey = []
    OArrayDes = []
    for O in OArray:
        Oimg = cv2.imread(O, cv2.IMREAD_GRAYSCALE)
        OKeypoints, ODescriptors = sift.detectAndCompute(Oimg, None)
        OArrayKey.append(OKeypoints)
        OArrayDes.append(ODescriptors)

    # Process each scene
    for scene_idx, S in enumerate(SArray, start=1):
        # Determine maximum allowed boxes for the current scene
        max_boxes = scene_object_limits.get((scene_idx - 1) // 3 + 1, 0)  # Map images to scenes
        box_count = 0

        # Read scene image
        Simg = cv2.imread(S)
        SGray = cv2.cvtColor(Simg, cv2.COLOR_BGR2GRAY)
        SKeypoints, SDescriptors = sift.detectAndCompute(SGray, None)

        for OIndex, O in enumerate(OArray):
            if box_count >= max_boxes:
                break  # Stop if maximum boxes are reached

            # Match objects to the scene
            detected, bbox = match_with_bounding_box(OArrayKey[OIndex], OArrayDes[OIndex], SKeypoints, SDescriptors)
            if detected:
                x_min, y_min, x_max, y_max = bbox

                # Draw bounding box
                cv2.rectangle(Simg, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Annotate with object name
                object_name = object_names[OIndex]
                cv2.putText(Simg, object_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                box_count += 1  # Increment box count

        # Save annotated image
        scene_name = os.path.basename(S).split('.')[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{scene_name}_bb.jpg")
        cv2.imwrite(output_path, Simg)

    print("Task 2 completed. Annotated images saved in the 'Detected Objects' folder.")