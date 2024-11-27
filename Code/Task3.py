"""
-------------------------------------------------------
CP467 Project: Task 3
Image Stitching
-------------------------------------------------------
Author:  Jashandeep Singh
__updated__ = "2024-11-20"
-------------------------------------------------------
"""

import cv2 as cv
import numpy as np
import os
from random import randint
from sklearn.cluster import DBSCAN

# Constants
DIRECTORY = "./Panorama"
OBJECTS_DIRECTORY = "./Objects"

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


def calculate_bounding_box(cluster_points, object_name, image_shape):
    """
    -------------------------------------------------------
    Calculates a dynamic bounding box around clustered points with padding.
    Use: x_min, y_min, x_max, y_max = calculate_bounding_box(cluster_points, object_name, image_shape, padding_ratio)
    -------------------------------------------------------
    Parameters:
        cluster_points - the points in the cluster (numpy.ndarray, shape: N x 2)
        object_name - the name of the object (str, optional for future extensions)
        image_shape - the shape of the image as (height, width) (tuple)
        padding_ratio - the ratio of padding around the bounding box (float, default: 0.4)
    Returns:
        x_min - the minimum x-coordinate of the bounding box (int)
        y_min - the minimum y-coordinate of the bounding box (int)
        x_max - the maximum x-coordinate of the bounding box (int)
        y_max - the maximum y-coordinate of the bounding box (int)
    -------------------------------------------------------
    """
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
    """
    -------------------------------------------------------
    Detects keypoints and computes descriptors using SIFT.
    Converts the input image to grayscale before processing.
    Use: keypoints, descriptors = detect_and_describe(image)
    -------------------------------------------------------
    Parameters:
        image - the input image (numpy.ndarray)
    Returns:
        keypoints - the detected keypoints (list of cv.KeyPoint)
        descriptors - the computed descriptors (numpy.ndarray)
    -------------------------------------------------------
    """
    sift = cv.SIFT_create()
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints, descriptors


def match_keypoints(descriptors1, descriptors2, ratio=0.50):
    """
    -------------------------------------------------------
    Matches descriptors between two images using FLANN-based matcher.
    Use: good_matches = match_keypoints(descriptors1, descriptors2, ratio)
    -------------------------------------------------------
    Parameters:
        descriptors1 - descriptors from the first image (numpy.ndarray)
        descriptors2 - descriptors from the second image (numpy.ndarray)
        ratio - ratio for Lowe's test (float)
    Returns:
        good_matches - list of good matches (list of cv.DMatch)
    -------------------------------------------------------
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def compute_homography(kp1, kp2, matches):
    """
    -------------------------------------------------------
    Computes the homography matrix using RANSAC from matched keypoints.
    Use: H, mask = compute_homography(kp1, kp2, matches)
    -------------------------------------------------------
    Parameters:
        kp1 - keypoints from the first image (list of cv.KeyPoint)
        kp2 - keypoints from the second image (list of cv.KeyPoint)
        matches - list of matched keypoints (list of cv.DMatch)
    Returns:
        H - the homography matrix (numpy.ndarray)
        mask - mask of inlier matches (numpy.ndarray)
    -------------------------------------------------------
    """
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")

    num_matches = 10
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]
    src_pts = np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(
        -1, 1, 2
    )

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return H, mask


def draw_matches(i, img1, img2, kp1, kp2, good):
    """
    -------------------------------------------------------
    Draws matches between two images using their keypoints and match results.
    Use: draw_matches(i, img1, img2, kp1, kp2, good)
    -------------------------------------------------------
    Parameters:
        i - index of the current match (int)
        img1 - the first image (numpy.ndarray)
        img2 - the second image (numpy.ndarray)
        kp1 - keypoints of the first image (list of cv.KeyPoint)
        kp2 - keypoints of the second image (list of cv.KeyPoint)
        good - list of good matches
    Returns:
        None
    -------------------------------------------------------"""
    output_path = os.path.join("./Keypoints", f"S{i}-S{i+1}.jpg")
    img3 = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv.imwrite(output_path, img3)


def stitch_images(img1, img2, homography):
    """
    -------------------------------------------------------
    Warps the second image onto the first image using the homography matrix.
    Computes the output dimensions and offsets to create a seamless panorama.
    Use: stitched_img = warp_and_stitch(image1, image2, H)
    -------------------------------------------------------
    Parameters:
        image1 - the first input image (numpy.ndarray)
        image2 - the second input image (numpy.ndarray)
        homography - the homography matrix (numpy.ndarray)
    Returns:
        stitched_img - the stitched panorama image (numpy.ndarray)
    -------------------------------------------------------
    """
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


if __name__ == "__main__":
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
        # cv.imshow("Intermediate Result", stitched_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    output_path = os.path.join(DIRECTORY, "Panorama.jpg")
    cv.imwrite(output_path, stitched_image)
    print("-" * 20)
    print("Panorama created successfully:", output_path)
    print("-" * 20)
    # cv.imshow("Panorama", stitched_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    pano_kp, pano_des = detect_and_describe(stitched_image)
    for i, objectImageFile in enumerate(sorted(os.listdir(OBJECTS_DIRECTORY))):
        objectImage = cv.imread(os.path.join(OBJECTS_DIRECTORY, objectImageFile))
        obj_kp, obj_des = detect_and_describe(objectImage)

        matches = match_keypoints(obj_des, pano_des)

        homography, _ = compute_homography(obj_kp, pano_kp, matches)

        draw_matches(i, objectImage, stitched_image, obj_kp, pano_kp, matches)

        h, w = stitched_image.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, homography).reshape(-1, 2)

        matches_coords = []
        for match in matches:
            # Get the keypoints from both images
            obj_pt = obj_kp[match.queryIdx].pt
            pano_pt = pano_kp[match.trainIdx].pt
            matches_coords.append(pano_pt)

        matches_coords = np.array(matches_coords)

        clustering = DBSCAN(eps=30, min_samples=3).fit(matches_coords)
        labels = clustering.labels_

        unique_labels = set(labels)
        largest_cluster_id = max(
            unique_labels, key=lambda label: list(labels).count(label)
        )

        cluster_points = matches_coords[labels == largest_cluster_id]

        # Calculate bounding rectangle
        x, y, w, h = calculate_bounding_box(
            cluster_points, object_names[i], stitched_image.shape
        )

        # Draw bounding box and label on the stitched image
        color = (randint(1, 255), randint(1, 255), randint(1, 255))

        stitched_image = cv.rectangle(
            stitched_image, (x, y), (w, h), color, 5, cv.LINE_AA
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
        # Match objects to the scene
    output_path = os.path.join(DIRECTORY, "Panorama_bb.jpg")
    cv.imwrite(output_path, stitched_image)
    print("Panorama_bb created successfully:", output_path)
    print("-" * 20)
