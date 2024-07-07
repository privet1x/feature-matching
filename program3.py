import cv2 as cv
import numpy as np

# Load the reference image in grayscale
reference_image = cv.imread('photo_3_train.jpg', cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()

# Detect keypoints and compute descriptors in the reference image
# Keypoints are distinctive features, and descriptors characterize them
kp1, des1 = sift.detectAndCompute(reference_image, None)

# Set up FLANN-based matcher parameters for efficient feature matching
FLANN_INDEX_KDTREE = 1  # Algorithm index for KD-Tree
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Number of times to recurse for each match
flann = cv.FlannBasedMatcher(index_params, search_params)

# Minimum number of good matches required to compute a reliable homography
MIN_MATCH_COUNT = 10

# Open the video file
cap = cv.VideoCapture('video_3_query.mp4')
if not cap.isOpened():
    print('Error: Cannot open video file')
    exit()

# Initialize an tracker (e.g., CSRT for high accuracy)
tracker = cv.TrackerCSRT_create()
tracking_initialized = False

# Process each frame of the video
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale for feature detection
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors in the current video frame
    kp2, des2 = sift.detectAndCompute(frame_gray, None)

    # Proceed only if there are enough descriptors to match against
    if des2 is not None and len(des2) > 1:
        # Perform FLANN-based matching between the reference and current frame
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test to filter out poor matches
        good_matches = [m for m, n in matches if m.distance < 0.65 * n.distance]

        # If there are enough good matches, compute the homography matrix
        if len(good_matches) > MIN_MATCH_COUNT:
            # Extract the coordinates of the good matches in both images
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # This matrix maps the reference image points to the corresponding frame points
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            if M is not None:
                # Define the rectangle corners of the reference image
                h, w = reference_image.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                # Transform the rectangle corners to the current frame's coordinate system
                dst = cv.perspectiveTransform(pts, M)

                # Compute a bounding box that contains the transformed rectangle
                bbox = cv.boundingRect(np.int32(dst))

                # Initialize the tracker with the detected bounding box
                tracker.init(frame, bbox)
                tracking_initialized = True

    # Update the tracker
    if tracking_initialized:
        success, bbox = tracker.update(frame)
        if success:
            # Draw the tracking rectangle on the current frame
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(frame, p1, p2, (0, 255, 0), 3, cv.LINE_AA)

    # Display the current frame with the tracking rectangle
    cv.imshow('Object Tracking', frame)

    # Stop processing if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture resource and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
