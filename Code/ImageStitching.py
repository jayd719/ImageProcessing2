"""
-------------------------------------------------------
CP467 Project: ImageStitching.py
-------------------------------------------------------
Author:  Jashandeep Singh
__updated__ = "2024-11-20"
-------------------------------------------------------
"""


# IMPORTS
import cv2 as cv
import numpy as np
import os

# CONSTANTS
DIRECTORY = "./Panorama"

"""
1.  Detect keypoints in all of the images --> SIFT, corner dector
2.  Match the descriptors between the two images
3.  Use RANSAC algorithm to estimate a homography matrix using the matched
    descriptors
4.  Apply warp transformation using the estimated homography matrix
"""


# Entry Point
if __name__=="__main__":

    # get all images for this task
    panorama_input = []
    for input_image in os.listdir(DIRECTORY):
        file_path = os.path.join(DIRECTORY, input_image)
        img = cv.imread(file_path)
        panorama_input.append(img)

    print(f"Number of Input Images: {len(panorama_input)}")


