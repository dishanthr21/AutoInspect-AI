import cv2
import numpy as np

def align_images(image, reference):
    # Convert images to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(5000)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray_img, None)
    kp2, des2 = orb.detectAndCompute(gray_ref, None)

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep top 15% matches
    good_matches = matches[:int(len(matches) * 0.15)]
    
    if len(good_matches) < 4:
        print("Warning: Not enough matches for alignment.")
        return image

    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Warp image
    height, width, channels = reference.shape
    aligned_img = cv2.warpPerspective(image, h, (width, height))

    return aligned_img