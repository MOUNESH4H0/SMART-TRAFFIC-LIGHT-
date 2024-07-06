import numpy as np
import cv2

# Load the video file
cap = cv2.VideoCapture('Andhra Pradesh.mp4')

# Create morphological kernel and background subtractor
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjusted kernel size
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = fgbg.apply(frame)

    # Perform morphological opening to remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Show the resulting foreground mask
    cv2.imshow('Foreground Mask', fg_mask)

    # Exit on 'ESC' key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
