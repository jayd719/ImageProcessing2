import cv2 as cv
import numpy as np
import os

# Constants
DIRECTORY = "./Panorama"


def remove_borders(stitched_img):
    stitched_img = cv.copyMakeBorder(
        stitched_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, (0, 0, 0)
    )
    gray = cv.cvtColor(stitched_img, cv.COLOR_BGR2GRAY)
    _, thresh_img = cv.threshold(gray, 10, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)
    cropped_img = stitched_img[y : y + h, x : x + w]
    cv.imwrite("stitchedOutputProcessed.png", cropped_img)
    return cropped_img


def detect_and_describe(image):
    """
    Detect keypoints and compute descriptors using ORB.
    """
    sift = cv.SIFT_create()
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints, descriptors


def match_keypoints(descriptors1, descriptors2, ratio=0.75):
    """
    Match descriptors between two images using BFMatcher and apply Lowe's ratio test.
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def compute_homography(kp1, kp2, matches):
    """
    Compute homography using RANSAC.
    """
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")

    # Extract matching keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return H, mask


def warp_and_stitch(image1, image2, H):
    """
    Warp image2 onto image1 using the homography matrix and stitch the images.
    """
    # Calculate dimensions of the resulting panorama
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Warp image2
    corners = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv.perspectiveTransform(corners, H)
    all_corners = np.vstack((np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2), transformed_corners))
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Offset for translating the panorama
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Warp image2
    stitched_img = cv.warpPerspective(image2, H_translation @ H, (x_max - x_min, y_max - y_min))
    stitched_img[translation_dist[1]:height1 + translation_dist[1], translation_dist[0]:width1 + translation_dist[0]] = image1
    return stitched_img


if __name__ == "__main__":
    # Load input images
    panorama_input = []
    for input_image in sorted(os.listdir(DIRECTORY)):
        if (
            input_image.endswith((".png", ".jpg", ".jpeg"))
            and "Panorama" not in input_image
        ):
            file_path = os.path.join(DIRECTORY, input_image)
            img = cv.imread(file_path)
            img = cv.resize(img, (1000, 1000))
            panorama_input.append(img)

    if len(panorama_input) < 2:
        print("Not enough images to create a panorama.")
        exit()

    # Stitch images one by one
    stitched_image = panorama_input[0]
    for i in range(1, len(panorama_input)):
        img1 = stitched_image
        img2 = panorama_input[i]

        # Step 1: Detect and describe
        kp1, des1 = detect_and_describe(img1)
        kp2, des2 = detect_and_describe(img2)

        # Step 2: Match keypoints
        matches = match_keypoints(des1, des2)

        # Step 3: Compute homography
        H, _ = compute_homography(kp1, kp2, matches)

        # Step 4: Warp and stitch
        stitched_image = warp_and_stitch(img2, img1, H)
        stitched_image = remove_borders(stitched_image)
        # stitched_image = cv.resize(stitched_image, (2000, 600))
        cv.imshow("s", stitched_image)
        cv.waitKey(0)

    # Save and display the final panorama
    output_path = os.path.join(DIRECTORY, "Panorama.jpg")
    cv.imwrite(output_path, stitched_image)
    print("Panorama created successfully:", output_path)
    cv.imshow("Panorama", stitched_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
