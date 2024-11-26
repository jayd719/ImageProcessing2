import cv2 as cv
import numpy as np
import os


# Constants
DIRECTORY = "./Panorama_Testing/TestSet2/"


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


def match_keypoints(descriptors1, descriptors2, ratio=0.80):
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

    num_matches = 20
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]
    src_pts = np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

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
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(output_path, img3)


import cv2 as cv
import numpy as np


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

    stitched_image = cv.warpPerspective(img2, translation_matrix @ homography, (width, height))
    stitched_image[-min_y : img1.shape[0] - min_y, -min_x : img1.shape[1] - min_x] = (img1)

    return stitched_image


if __name__ == "__main__":
    panorama_input = []
    for input_image in sorted(os.listdir(DIRECTORY)):
        if (input_image.endswith((".png", ".jpg", ".jpeg"))and "Panorama" not in input_image):
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
        # warped_image = warp_and_stitch(img2, img1, homography)
        # warped_image = cv.warpPerspective(img1, homography, (img2.shape[1], img2.shape[0]),flags=cv.INTER_LINEAR)
        #

        # Apply panorama correction
        width = img1.shape[1] + img2.shape[1]
        height = img2.shape[0] + img1.shape[0]

        stitched_image = stitch_images(img2, img1, homography)
        cv.imshow("Intermediate Result", stitched_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    output_path = os.path.join(DIRECTORY, "Panorama.jpg")
    cv.imwrite(output_path, stitched_image)
    print("Panorama created successfully:", output_path)
    cv.imshow("Panorama", stitched_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
