import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load the images in grayscale
img1 = cv.imread('photo_2_query.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('photo_2_train.jpg', cv.IMREAD_GRAYSCALE)

# Initiate SIFT detector
sift = cv.SIFT_create()  # Create a SIFT object to detect keypoints and compute descriptors

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)  # Detect keypoints and compute descriptors in the query image
kp2, des2 = sift.detectAndCompute(img2, None)  # Detect keypoints and compute descriptors in the training image

# FLANN parameters
FLANN_INDEX_KDTREE = 1  # Define the algorithm to use for FLANN indexing
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)  # Specify the indexing parameters, using 5 KD-trees
search_params = dict(checks = 50)  # Specify the number of checks to perform, the more checks the better the results but slower

# Create the FLANN matcher
flann = cv.FlannBasedMatcher(index_params, search_params)  # Create the FLANN matcher object

# Perform knn matching
matches = flann.knnMatch(des1, des2, k=2)  # Find the 2 nearest neighbors

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]  # Initialize the mask for drawing matches

# Ratio test as per Lowe's paper
good = []
for i, (m, n) in enumerate(matches):  # Loop through matches and apply Lowe's ratio test
    if m.distance < 0.8 * n.distance:  # If the distance ratio between the nearest and second nearest neighbor is less than 0.8
        good.append(m)  # Append to good matches
        matchesMask[i] = [1, 0]  # Mark the match as good in the mask

# Display only the top 50 matches
good = sorted(good, key=lambda x: x.distance)[:50]  # Sort matches by distance and take the top 50

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    # If sufficient matches are found, proceed to find the homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)  # Extract point coordinates from the query image
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)  # Extract point coordinates from the train image

    # Find the homography matrix
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()  # Convert the mask to a list for drawing

    # Draw a polygon around the detected object in the train image
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    # If not enough matches are found, print a message
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

# Draw the matches
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

# Combine images with drawn matches
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

# Display the final image with matches
plt.imshow(img3, 'gray'), plt.show()

# Save the final image
cv.imwrite("output_program2.png", img3)
