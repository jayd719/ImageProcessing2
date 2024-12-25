"""-------------------------------------------------------
CP467 Project: convertToJPG.py
-------------------------------------------------------
Author:  JD
ID:      91786
Project: CP 467
Version:  1.0.8
__updated__ = Wed Dec 25 2024
-------------------------------------------------------
"""


# Imports
import cv2 as cv
import os


def convert_to_jpg(directory: str):
    """-------------------------------------------------------
    Converts all images in a directory to PNG format.
    Non-image files are ignored.

    Use: convert_to_png(directory)
    -------------------------------------------------------
    Parameters:
        directory - path to the directory containing the images (str)
    Returns:
        None
    -------------------------------------------------------
    """
    for root, _, files in os.walk(directory, topdown=True):
        for file in files:
            file_path = os.path.join(root, file)
            name, ext = os.path.splitext(file)
            img = cv.imread(file_path)

            if img is not None:
                output_path = os.path.join(root, f"{name}.jpg")
                print(f"converted {file}")
                cv.imwrite(output_path, img)
            else:
                print(f"Could not read file: {file_path}")

            if not file.endswith(".jpg"):
                os.remove(file_path)


# Entry Point
if __name__ == "__main__":
    convert_to_jpg("./Objects")
    convert_to_jpg("./Panorama")
    # convert_to_jpg("./Scenes")
