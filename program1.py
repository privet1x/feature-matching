import cv2 as cv
import numpy as np

# Check if new corner sufficiently far away from all previously chosen corners
def is_far_enough(new_corner, chosen_corners, min_distance=10):
    x_new, y_new = new_corner[:2]
    for x_chosen, y_chosen, _ in chosen_corners:
        if np.sqrt((x_new - x_chosen) ** 2 + (y_new - y_chosen) ** 2) < min_distance:
            return False
    return True

# Load the image and convert to grayscale
img = cv.imread('photo_1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_float32 = np.float32(gray)  # Ensure the image is float32

# Harris Corner Detection
dst = cv.cornerHarris(gray_float32, 2, 3, 0.001)
dst = cv.dilate(dst, None)  # Dilate to enhance corner markings

# Normalize for visualization and thresholding
dst_norm = np.empty(dst.shape, dtype=np.float32)
cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
dst_scaled = np.uint8(np.round(dst_norm))

# Create a list of corners above the threshold with their response values
corners = []
for y in range(dst.shape[0]):
    for x in range(dst.shape[1]):
        response = dst[y, x]
        if response > 29144.8675:  # Use your computed threshold
            corners.append((x, y, response))

# Sort corners based on the Harris response (corner strength)
corners_sorted = sorted(corners, key=lambda x: x[2], reverse=True)

# Select the top 4 distinct strongest corners
top_corners = []
for corner in corners_sorted:
    if is_far_enough(corner, top_corners):
        top_corners.append(corner)
        if len(top_corners) == 4:
            break

# Create a copy of the original image to draw the corners
img_corners = np.copy(img)

# Draw only the top 4 strongest corners
for x, y, strength in top_corners:
    cv.circle(img_corners, (x, y), radius=5, color=(0, 255, 0), thickness=2)

# Initialize SIFT detector
sift = cv.SIFT_create()
keypoints = sift.detect(gray, None)
keypoints, des = sift.detectAndCompute(gray,None)

# Draw keypoints on the original color image
img_sift = cv.drawKeypoints(img, keypoints, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the results
cv.imshow('Top 4 Harris Corners', img_corners)
cv.imshow('SIFT Key Points on Original Image', img_sift)
cv.waitKey()
cv.destroyAllWindows()
