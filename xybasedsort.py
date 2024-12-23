import cv2 as cv
import numpy as np

# Function to filter circular contours
def circularContour(contours):
    circular_contours = []
    for contour in contours:
        # Calculate contour area and perimeter
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        if perimeter == 0:
            continue  # Avoid division by zero

        # Calculate circularity: 4π * (Area / Perimeter²)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # A perfect circle has circularity close to 1
        if 0.8 < circularity < 1.2:  # Adjust thresholds as needed
            circular_contours.append(contour)
    return circular_contours

# Load, preprocess, and detect contours
img = cv.imread("omr2.png")
img = cv.resize(img, (700, 700))
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv.Canny(imgBlur, 10, 50)

contours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Filter circular contours
circleCons = circularContour(contours)

# Sort the contours based on both x and y coordinates
# Sort by x first, then by y
circleCons.sort(key=lambda c: (cv.boundingRect(c)[0], cv.boundingRect(c)[1]))

# Select the first 100 contours
top_100_contours = circleCons[:3]

# Draw the first 100 contours on the original image
imgContours = img.copy()
cv.drawContours(imgContours, top_100_contours, -1, (0, 255, 0), 3)

# Display the image with the top 100 circular contours
cv.imshow("Top 100 Circles", imgContours)
cv.waitKey(0)
cv.destroyAllWindows()
