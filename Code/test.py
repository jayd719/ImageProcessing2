import cv2 as cv
import numpy as np
import os

# CONSTANTS
DIRECTORY = "./Panorama"
MIN_MATCH_COUNT = 20  # Minimum number of matches required for stitching


# Load images from the specified directory
panorama_input = []
for input_image in sorted(os.listdir(DIRECTORY)):
    if "Panorama" not in input_image:  # Skip files with "Panorama" in their name
        file_path = os.path.join(DIRECTORY, input_image)
        img = cv.imread(file_path)
        if img is not None:
            panorama_input.append(img)
        else:
            print(f"Failed to load {file_path}")

if len(panorama_input) < 2:
    print("Not enough images to stitch. Exiting.")
    exit()

# Convert the first two images to grayscale
img1 = cv.cvtColor(panorama_input[0], cv.COLOR_BGR2GRAY)
img1 = cv.resize(img1, (2000, 1000))
img2 = cv.cvtColor(panorama_input[1], cv.COLOR_BGR2GRAY)
img2 = cv.resize(img2, (2000, 1000))

# Initialize SIFT
sift = cv.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match features using BFMatcher
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# Visualize matches (optional)
draw_params = dict(
    matchColor=(0, 255, 0),  # Matches in green
    singlePointColor=None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

img_matches = cv.drawMatches(
    panorama_input[0], kp1, panorama_input[1], kp2, good, None, **draw_params
)
cv.imshow("Feature Matches", img_matches)


# Homography and stitching
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    if m is None:
        print("Homography calculation failed. Exiting.")
        exit()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, m)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    cv.imshow("original_image_overlapping.jpg", img2)
    # Warp perspective
    dst = cv.warpPerspective(
        panorama_input[0],
        m,
        (
            panorama_input[1].shape[1] + panorama_input[0].shape[1],
            panorama_input[1].shape[0],
        ),
    )

    # Place the second image into the panorama
    dst[0 : panorama_input[1].shape[0], 0 : panorama_input[1].shape[1]] = (
        panorama_input[1]
    )

    # Trim black borders
    trimmed = dst

    # Save and display the stitched result
    cv.imwrite("stitched_output.jpg", trimmed)
    cv.imshow("Stitched Image", trimmed)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
