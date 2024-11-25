import cv2
import os
import numpy as np

LOWES = 0.6
THRESHOLD = 34

# Function to apply Object detection
def match(OKeypoints, ODescriptors, SKeypoints, SDescriptors):
    # Matching process, using Euclidean distance and only checking one direction
    distance = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    #Finding the top two matches between Oject and Scene descriptors
    matches = distance.knnMatch(ODescriptors, SDescriptors, k=2)
    #Matches has two elements, first is top match, second is second best match

    #Used to hold to good matches
    best = []

    for firstMatch, secondMatch in matches:
        #Checking to see if the distance between the two top matches 
        #If low, first match is good otherwise it is rejected
        if firstMatch.distance < (LOWES * secondMatch.distance):
            best.append(firstMatch)
    
    #Check if the number of good matches is greater than the minimum required to form a detection
    #Even if the match is good, as specified above, it must be significant (as significant as the threshold)
    if len(best) >= THRESHOLD:
        return True
    else:
        return False


os.chdir("//Users//jashan//Desktop//Project//code")
sift = cv2.SIFT_create()

OArray = [
    '../Objects/O1.jpg', 
    '../Objects/O2.jpg', 
    '../Objects/O3.jpg',
    '../Objects/O4.jpg', 
    '../Objects/O5.jpg', 
    '../Objects/O6.jpg',
    '../Objects/O7.jpg', 
    '../Objects/O8.jpg', 
    '../Objects/O9.jpg',
    '../Objects/O10.jpg'
]

SArray = [
    '../Scenes/S1_front.jpg', 
    '../Scenes/S1_right.jpg', 
    '../Scenes/S1_left.jpg',
    '../Scenes/S2_front.jpg', 
    '../Scenes/S2_right.jpg', 
    '../Scenes/S2_left.jpg',
    '../Scenes/S3_front.jpg', 
    '../Scenes/S3_right.jpg', 
    '../Scenes/S3_left.jpg',
    '../Scenes/S4_front.jpg', 
    '../Scenes/S4_right.jpg', 
    '../Scenes/S4_left.jpg',
    '../Scenes/S5_front.jpg', 
    '../Scenes/S5_right.jpg', 
    '../Scenes/S5_left.jpg',
    '../Scenes/S6_front.jpg', 
    '../Scenes/S6_right.jpg', 
    '../Scenes/S6_left.jpg',
    '../Scenes/S7_front.jpg', 
    '../Scenes/S7_right.jpg', 
    '../Scenes/S7_left.jpg',
    '../Scenes/S8_front.jpg', 
    '../Scenes/S8_right.jpg', 
    '../Scenes/S8_left.jpg',
    '../Scenes/S9_front.jpg', 
    '../Scenes/S9_right.jpg', 
    '../Scenes/S9_left.jpg',
    '../Scenes/S10_front.jpg', 
    '../Scenes/S10_right.jpg', 
    '../Scenes/S10_left.jpg'
]

#Arrays for the keypoints and descriptors
OArrayKey = []
OArrayDes = []
#Computing for each object once for reuse
for O in OArray:
    Oimg = cv2.imread(O, cv2.IMREAD_GRAYSCALE)
    OKeypoints, ODescriptors = sift.detectAndCompute(Oimg, None)
    OArrayKey.append(OKeypoints)
    OArrayDes.append(ODescriptors)

#False positives, False negatives, and Positives count
FP = 0
FN = 0
P = 0

#Counts used for checking which objects should be in which scene
SCount = 0
num = 0
OCount = 0

for S in SArray:
    print(S)

    #Each scene has three angles
    #SCount stores the current scene number
    if num == 3:
        SCount += 1
        num = 0

    #Getting keypoints and descriptors of the current scene
    Simg = cv2.imread(S, cv2.IMREAD_GRAYSCALE)
    SKeypoints, SDescriptors = sift.detectAndCompute(Simg, None)

    #Loop through all the objects
    for O in OArray:
        print(O)

        #Return true of false if the object O is in scene S
        result = match(OArrayKey[OCount], OArrayDes[OCount], SKeypoints, SDescriptors)
        print("Object Number:", OCount, "Scene Number:", SCount, "Result", result)

        if OCount <= SCount:
            if result:
                P = P + 1
            else:
                print(OCount, SCount, "Was Incorrect")
                FN = FN + 1
        else:
            if not result:
                P = P + 1
            else:
                print(OCount, SCount, "Was Incorrect")
                FP = FP + 1
        
        OCount += 1

    OCount = 0
    num += 1

    #Display current counts
    print("Running Counts")
    print("FN", FN)
    print("FP", FP)
    print("P", P)
    print()


print("Total")
print("FN", FN)
print("FP", FP)
print("P", P)
print("Accuracy", P / (FN + FP + P))