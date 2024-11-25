import cv2 as cv
import numpy as np
import os

DIRECTORY = "./Panorama"
MIN_MATCH_COUNT =10


image1 = cv.imread(os.path.join(DIRECTORY,"S1.jpg"))
image2 = cv.imread(os.path.join(DIRECTORY,"S2.jpg"))
# Feature Extraction

# convertImage to grayScale
img1_gray = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(image2,cv.COLOR_BGR2GRAY)
cv.imshow("Image 1 Gray",img1_gray)
cv.imshow("Image 2 Gray",img2_gray)

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1_gray,None)
kp2, des2 = sift.detectAndCompute(img2_gray,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good_matches = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)

if len(good_matches)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

    result = cv.warpPerspective(image1, H,(image1.shape[1] + image2.shape[1], image1.shape[0]))
    result[0:image2.shape[0], 0:image2.shape[1]] = image2

   
else:
    print( "Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT) )







cv.imshow("ds",result)
cv.waitKey(0)
cv.destroyAllWindows()