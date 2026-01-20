import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_images(master_img, test_img):
    # Convert to grayscale
    gray_master = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    score, diff = ssim(gray_master, gray_test, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = test_img.copy()
    missing_items_count = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100: # Filter small noise
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            missing_items_count += 1

    # Color Histogram
    hist_score = cv2.compareHist(
        cv2.calcHist([master_img], [0], None, [256], [0, 256]),
        cv2.calcHist([test_img], [0], None, [256], [0, 256]),
        cv2.HISTCMP_CORREL
    )

    return {
        "similarity_score": round(score * 100, 2),
        "color_match_score": round(hist_score * 100, 2),
        "anomalies_detected": missing_items_count,
        "processed_image": result_img
    }