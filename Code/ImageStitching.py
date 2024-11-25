# IMPORTS
import cv2 as cv
import numpy as np
import os

# CONSTANTS
DIRECTORY = "./Panorama"

def remove_borders(stitched_img):
    stitched_img = cv.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, (0, 0, 0))
    gray = cv.cvtColor(stitched_img, cv.COLOR_BGR2GRAY)
    _, thresh_img = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)
    cropped_img = stitched_img[y:y + h, x:x + w]
    cv.imwrite("stitchedOutputProcessed.png", cropped_img)
    return cropped_img

if __name__ == "__main__":
    panorama_input = []
    for input_image in sorted(os.listdir(DIRECTORY)):
        if input_image.endswith(('.png', '.jpg', '.jpeg')) and "Panorama" not in input_image:
            file_path = os.path.join(DIRECTORY, input_image)
            img = cv.imread(file_path)
            img = cv.resize(img, (2000, 1000))
            panorama_input.append(img)

    number_of_images = len(panorama_input)
    if number_of_images < 2:
        print("Not enough images to create a panorama. At least two images are required.")
        exit()
    
    stitcher = cv.Stitcher_create()
    error, output_image = stitcher.stitch(panorama_input)

    if error != cv.Stitcher_OK:
        print(f"Error during stitching: {error}")
        exit()

    output_path = os.path.join(DIRECTORY, "Panorama.jpg")
    output_image = remove_borders(output_image)
    cv.imwrite(output_path, output_image)
    print("Panorama created successfully:", output_path)
