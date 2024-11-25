# IMPORTS
import cv2 as cv
import numpy as np
import os

# CONSTANTS
DIRECTORY = "./Panorama"

def remove_borders(stitched_img):
    """
    -------------------------------------------------------
    Removes unnecessary borders from the stitched image.
    Adds a small border to assist in finding the largest contour
    and crops the image to the bounding box of the largest contour.
    Use: cropped_img = remove_borders(stitched_img)
    -------------------------------------------------------
    Parameters:
        stitched_img - the stitched panorama image (numpy.ndarray)
    Returns:
        cropped_img - the cropped panorama image with borders removed (numpy.ndarray)
    -------------------------------------------------------
    """
    # Add a border around the image to avoid cutting off contours at the edges
    stitched_img = cv.copyMakeBorder(
        stitched_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, (0, 0, 0)
    )
    # Convert the image to grayscale
    gray = cv.cvtColor(stitched_img, cv.COLOR_BGR2GRAY)
    # Threshold the grayscale image to create a binary image
    _, thresh_img = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
    # Find the largest contour
    largest_contour = max(contours, key=cv.contourArea)
    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv.boundingRect(largest_contour)
    # Crop the image to the bounding rectangle
    cropped_img = stitched_img[y:y + h, x:x + w]
    # Save the cropped image for reference
    cv.imwrite("stitchedOutputProcessed.png", cropped_img)
    return cropped_img

if __name__ == "__main__":
    """
    -------------------------------------------------------
    Main block for creating a panorama image from multiple input images.
    - Reads images from the specified directory.
    - Resizes images to a consistent size for stitching.
    - Uses OpenCV's Stitcher class to create the panorama.
    - Removes borders from the final stitched image.
    Use: Run the script directly to execute the functionality.
    -------------------------------------------------------
    """
    # List to store input images for the panorama
    panorama_input = []
    # Read images from the directory
    for input_image in sorted(os.listdir(DIRECTORY)):
        # Check if the file is a valid image and exclude the output panorama
        if input_image.endswith(('.png', '.jpg', '.jpeg')) and "Panorama" not in input_image:
            file_path = os.path.join(DIRECTORY, input_image)
            # Read and resize the image
            img = cv.imread(file_path)
            img = cv.resize(img, (2000, 1000))  # Resize to a uniform size for consistency
            panorama_input.append(img)

    # Check if there are enough images to create a panorama
    number_of_images = len(panorama_input)
    if number_of_images < 2:
        print("Not enough images to create a panorama. At least two images are required.")
        exit()
    stitcher = cv.Stitcher_create()
    error, output_image = stitcher.stitch(panorama_input)

    # Check if stitching was successful
    if error != cv.Stitcher_OK:
        print(f"Error during stitching: {error}")
        exit()

    output_path = os.path.join(DIRECTORY, "Panorama.jpg")
    # Remove borders from the stitched image
    output_image = remove_borders(output_image)
    
    # Save the final panorama image
    cv.imwrite(output_path, output_image)
    print("Panorama created successfully:", output_path)
