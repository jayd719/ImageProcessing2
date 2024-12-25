# Image Stitching and Object Detection


### Overview
This project explores advanced computer vision techniques to develop a system for image stitching and object detection. The primary goal is to create a seamless panoramic image from multiple overlapping input images and to identify and localize objects within the stitched panorama. This report details the algorithms used, implementation steps, and results, demonstrating the project's applications in fields like surveillance, virtual tours, and academic research.

### Introduction
The Image Stitching and Object Detection project combines state-of-the-art computer vision algorithms to address two key tasks: image stitching and object detection. Using SIFT (Scale-Invariant Feature Transform) for feature detection, homography for image alignment, and DBSCAN clustering for object localization, this project provides a modular approach to creating detailed panoramic views while identifying and labeling objects. By dynamically handling user inputs and generating intermediate visualizations, the project is suitable for academic exploration and practical deployments.

### Objectives
- Develop a robust system to stitch multiple overlapping images into a seamless panorama.
- Implement object detection to locate and label objects in the stitched panorama.
- Provide intermediate visualizations for debugging and understanding algorithmic steps.
- Ensure scalability and adaptability of the system for various use cases.

### Methodology
#### 1. Image Stitching
- **Feature Detection**: SIFT is used to detect and describe keypoints in the input images.
- **Keypoint Matching**: FLANN-based matching and Lowe's ratio test are applied to find correspondences between images.
- **Homography Calculation**: RANSAC is used to compute the homography matrix, aligning overlapping images.
- **Stitching**: The homography is applied to warp and blend images into a single panoramic view.

#### 2. Object Detection
- **Feature Matching**: Object images are matched against the panorama using SIFT and FLANN-based matching.
- **Clustering**: DBSCAN groups matched points to identify object locations in the panorama.
- **Bounding Box Calculation**: Dynamic bounding boxes are computed around detected clusters.
- **Visualization**: Labels and bounding boxes are drawn on the panorama for clarity.

### Implementation
1. Input directories for panorama images and object images are specified by the user.
2. The system iteratively processes the images for stitching, saving intermediate keypoint visualizations.
3. The stitched panorama is analyzed for object detection, with results saved as an annotated image.
4. Outputs include the stitched panorama (`Panorama.jpg`) and the annotated panorama (`Panorama_bb.jpg`).

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/enhanced-image-stitching.git
   cd enhanced-image-stitching
   ```
2. Install the required Python libraries:
   ```bash
   pip install opencv-python-headless numpy scikit-learn
   ```
#### Requirements
- Python 3.7+
- Libraries:
  - OpenCV
  - NumPy
  - scikit-learn

### Usage
1. Place the input images for stitching in a directory (e.g., `./Panorama`).
2. Place the object images in another directory (e.g., `./Objects`).
3. Run the script:
   ```bash
   python image_stitching_enhanced.py
   ```
4. Follow the prompts to specify the directories for the panorama and objects. The default directories are `./Panorama` and `./Objects`.
5. The output files will be saved in the `./Panorama` directory:
   - `Panorama.jpg`: The stitched panorama.
   - `Panorama_bb.jpg`: The panorama with bounding boxes and labels for detected objects.
   
## Project Structure
- `image_stitching_enhanced.py`: The main script for stitching images and detecting objects.
- `./Panorama`: Default directory for input panorama images and output stitched panoramas.
- `./Objects`: Default directory for input object images.
- `./Keypoints`: Directory for intermediate visualization of keypoint matches.


### Results



### Future Work
- Add support for non-planar and 360-degree panoramas using video stream.

### Conclusion
This project successfully combines image stitching and object detection into a unified system. The use of advanced computer vision algorithms and modular implementation ensures versatility and robustness. The system demonstrates significant potential for real-world applications and serves as a foundation for future enhancements.


### Acknowledgements
- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)


