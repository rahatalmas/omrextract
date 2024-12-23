import cv2 as cv
import numpy as np

# Read the image
img = cv.imread("omrfull.jpg")
if img is None:
    print("Image not found!")
    exit()

# Get image dimensions
height, width, _ = img.shape

# Calculate the dividing point
divide_point = height // 3

# Split the image
top_part = img[:divide_point, :]
bottom_part = img[divide_point:, :]

# Show the two parts
cv.imshow("Top Part (1:4)", top_part)
cv.imshow("Bottom Part (3:4)", bottom_part)

# Wait for a key press and close the windows
cv.waitKey(0)
cv.destroyAllWindows()

# Optionally save the parts
cv.imwrite("top_part.jpg", top_part)
cv.imwrite("bottom_part.jpg", bottom_part)
