"""
-------------------------------------------------------
CP467 Project: Task 3
Enhanced Image Stitching and Object Detection
-------------------------------------------------------
Author:  
__updated__ = "2024-11-26"
-------------------------------------------------------
"""

import cv2 as cv
import numpy as np
import os
from random import randint
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor

# Constants
DEFAULT_DIRECTORY = "./Panorama"
DEFAULT_OBJECTS_DIRECTORY = "./Objects"

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

object_names = [
    "speed stick",
    "seasoning",
    "calculator",
    "instant rice",
    "toothpaste",
    "dryer sheets",
    "peanut butter",
    "gum",
    "old spice",
    "gift card",
]


def prompt_for_directory(prompt, default):
    directory = input(f"{prompt} (default: {default}): ")
    return directory if directory else default


def calculate_bounding_box(cluster_points, object_name, image_shape):
    padding_ratio = 0.4
    x_min, y_min = np.min(cluster_points, axis=0).astype(int)
    x_max, y_max = np.max(cluster_points, axis=0).astype(int)

    box_width = x_max - x_min
    box_height = y_max - y_min

    padding_x = int(box_width * padding_ratio)
    padding_y = int(box_height * padding_ratio)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(image_shape[1], x_max + padding_x)
    y_max = min(image_shape[0], y_max + padding_y)

    return x_min, y_min, x_max, y_max


def detect_and_describe(image):
    sift = cv.SIFT_create()
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints, descriptors


def match_keypoints(descriptors1, descriptors2, ratio=0.50):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches


def compute_homography(kp1, kp2, matches):
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")

    matches = sorted(matches, key=lambda x: x.distance)[:10]
    src_pts = np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(
        -1, 1, 2
    )

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return H, mask


def draw_matches(index, img1, img2, kp1, kp2, good):
    output_path = os.path.join("./Keypoints", f"S{index}-S{index+1}.jpg")
    img3 = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv.imwrite(output_path, img3)


def stitch_images(img1, img2, homography):
    corners = np.array(
        [[0, 0], [0, img2.shape[0]], [img2.shape[1], img2.shape[0]], [img2.shape[1], 0]]
    )
    transformed_corners = cv.perspectiveTransform(np.float32([corners]), homography)
    min_x = int(min(transformed_corners[0][:, 0].min(), 0))
    max_x = int(max(transformed_corners[0][:, 0].max(), img1.shape[1]))
    min_y = int(min(transformed_corners[0][:, 1].min(), 0))
    max_y = int(max(transformed_corners[0][:, 1].max(), img1.shape[0]))

    width = max_x - min_x
    height = max_y - min_y

    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    stitched_image = cv.warpPerspective(
        img2, translation_matrix @ homography, (width, height)
    )
    stitched_image[-min_y : img1.shape[0] - min_y, -min_x : img1.shape[1] - min_x] = (
        img1
    )

    return stitched_image


def process_images(panorama_input):
    stitched_image = panorama_input[0]
    for i in range(1, len(panorama_input)):
        img1 = stitched_image
        img2 = panorama_input[i]
        kp1, des1 = detect_and_describe(img1)
        kp2, des2 = detect_and_describe(img2)
        matches = match_keypoints(des1, des2)

        if len(matches) < 10:
            print(f"Not enough matches for image S{i}.jpg")
            continue

        draw_matches(i, img1, img2, kp1, kp2, matches)
        homography, _ = compute_homography(kp1, kp2, matches)
        stitched_image = stitch_images(img2, img1, homography)

    return stitched_image


def process_objects(objects_directory, stitched_image):
    pano_kp, pano_des = detect_and_describe(stitched_image)
    output_path = os.path.join("./Panorama", "Panorama_bb.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i, objectImageFile in enumerate(sorted(os.listdir(objects_directory))):
        objectImage = cv.imread(os.path.join(objects_directory, objectImageFile))
        obj_kp, obj_des = detect_and_describe(objectImage)

        matches = match_keypoints(obj_des, pano_des)
        if len(matches) < 10:
            continue

        homography, _ = compute_homography(obj_kp, pano_kp, matches)
        draw_matches(i, objectImage, stitched_image, obj_kp, pano_kp, matches)

        matches_coords = np.array([pano_kp[match.trainIdx].pt for match in matches])

        clustering = DBSCAN(eps=30, min_samples=3).fit(matches_coords)
        labels = clustering.labels_

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        largest_cluster_id = max(
            unique_labels, key=lambda label: list(labels).count(label)
        )

        cluster_points = matches_coords[labels == largest_cluster_id]
        x, y, x_max, y_max = calculate_bounding_box(
            cluster_points, object_names[i], stitched_image.shape
        )

        color = (randint(1, 255), randint(1, 255), randint(1, 255))
        stitched_image = cv.rectangle(
            stitched_image, (x, y), (x_max, y_max), color, 5, cv.LINE_AA
        )
        cv.putText(
            stitched_image,
            object_names[i],
            (x + 20, y + 30),
            cv.FONT_HERSHEY_TRIPLEX,
            1.25,
            (0, 0, 0),
            3,
        )

    cv.imwrite(output_path, stitched_image)
    print("Panorama_bb created successfully:", output_path)


if __name__ == "__main__":
    DIRECTORY = prompt_for_directory("Enter panorama directory", DEFAULT_DIRECTORY)
    OBJECTS_DIRECTORY = prompt_for_directory(
        "Enter objects directory", DEFAULT_OBJECTS_DIRECTORY
    )

    panorama_input = []
    for input_image in sorted(os.listdir(DIRECTORY)):
        if (
            input_image.endswith((".png", ".jpg", ".jpeg"))
            and "Panorama" not in input_image
        ):
            file_path = os.path.join(DIRECTORY, input_image)
            img = cv.imread(file_path)
            panorama_input.append(img)

    if len(panorama_input) < 2:
        print("Not enough images to create a panorama.")
        exit()

    stitched_image = process_images(panorama_input)

    output_path = os.path.join(DIRECTORY, "Panorama.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv.imwrite(output_path, stitched_image)
    print("Panorama created successfully:", output_path)

    process_objects(OBJECTS_DIRECTORY, stitched_image)
    print("Processing complete.")
