#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

const std::string DIRECTORY = "./Panorama";

cv::Mat remove_borders(cv::Mat stitched_img) {
    cv::copyMakeBorder(stitched_img, stitched_img, 10, 10, 10, 10, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::Mat gray;
    cv::cvtColor(stitched_img, gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat thresh_img;
    cv::threshold(gray, thresh_img, 10, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        throw std::runtime_error("No contours found in the image.");
    }

    // Find the largest contour
    auto largest_contour = *std::max_element(contours.begin(), contours.end(), 
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    
    cv::Rect bounding_rect = cv::boundingRect(largest_contour);
    cv::Mat cropped_img = stitched_img(bounding_rect);
    
    cv::imwrite("stitchedOutputProcessed.png", cropped_img);
    return cropped_img;
}

int main() {
    std::vector<cv::Mat> panorama_input;
    
    // Read images from the directory
    for (const auto& entry : fs::directory_iterator(DIRECTORY)) {
        std::string input_image = entry.path().string();
        
        if (input_image.ends_with(".png") || input_image.ends_with(".jpg") || input_image.ends_with(".jpeg")) {
            cv::Mat img = cv::imread(input_image);
            cv::resize(img, img, cv::Size(2000, 1000));  // Resize to 2000x1000
            panorama_input.push_back(img);
        }
    }

    int number_of_images = panorama_input.size();
    if (number_of_images < 2) {
        std::cerr << "Not enough images to create a panorama. At least two images are required." << std::endl;
        return -1;
    }

    // Stitch the images
    cv::Mat output_image;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
    cv::Stitcher::Status status = stitcher->stitch(panorama_input, output_image);

    if (status != cv::Stitcher::OK) {
        std::cerr << "Error during stitching: " << status << std::endl;
        return -1;
    }

    // Remove borders
    output_image = remove_borders(output_image);

    // Save the result
    std::string output_path = DIRECTORY + "/Panorama.jpg";
    cv::imwrite(output_path, output_image);
    std::cout << "Panorama created successfully: " << output_path << std::endl;

    return 0;
}
