import cv2 as cv
import numpy as np
import os

# Constants
DIRECTORY = "./Panorama"
MIN_MATCH_COUNT = 10


def warp_and_stitch(image1, image2, H):
    """
    Warp image2 onto image1 using the homography matrix and stitch the images.
    """
    # Calculate dimensions of the resulting panorama
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Warp image2
    corners = np.float32(
        [[0, 0], [0, height2], [width2, height2], [width2, 0]]
    ).reshape(-1, 1, 2)
    transformed_corners = cv.perspectiveTransform(corners, H)
    all_corners = np.vstack(
        (
            np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(
                -1, 1, 2
            ),
            transformed_corners,
        )
    )

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Offset for translating the panorama
    translation_dist = [-x_min, -y_min]
    H_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
    )

    # Warp image2
    stitched_img = cv.warpPerspective(
        image2, H_translation @ H, (x_max - x_min, y_max - y_min)
    )
    stitched_img[
        translation_dist[1] : height1 + translation_dist[1],
        translation_dist[0] : width1 + translation_dist[0],
    ] = image1
    return stitched_img


if __name__ == "__main__":
    # Load input images
    panorama_input = []
    for input_image in sorted(os.listdir(DIRECTORY)):
        if input_image.endswith((".png", ".jpg", ".jpeg")):
            if "Panorama" not in input_image:
                file_path = os.path.join(DIRECTORY, input_image)
                img = cv.imread(file_path)
                panorama_input.append(img)

    if len(panorama_input) < 2:
        print("Not enough images to create a panorama.")
        exit()

    sift = cv.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    # Stitch images one by one
    stitched_image = panorama_input[0]
    for i in range(1, len(panorama_input)):
        img1 = stitched_image
        img2 = panorama_input[i]

        # Step 1: Detect and describe
        img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        # Step 2: Match keypoints
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Step 3: Compute homography
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )

            H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # Step 4: Warp and stitch
        stitched_image = warp_and_stitch(img1, img2, H)

    # Save and display the final panorama
    output_path = os.path.join(DIRECTORY, "Panorama.jpg")
    cv.imwrite(output_path, stitched_image)
    print("Panorama created successfully:", output_path)
    cv.imshow("Panorama", stitched_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
