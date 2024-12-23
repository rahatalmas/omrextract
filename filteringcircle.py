import cv2 as cv
import numpy as np

# Function to filter circular contours
def circularContour(contours, min_area=150):
    circular_contours = []
    for contour in contours:
        # Calculate contour area and perimeter
        area = cv.contourArea(contour)
        if area < min_area:
            continue  # Skip contours with area less than min_area
        
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            continue  # Avoid division by zero

        # Calculate circularity: 4π * (Area / Perimeter²)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # A perfect circle has circularity close to 1
        if 0.8 < circularity < 1.2:  # Adjust thresholds as needed
            circular_contours.append(contour)
    return circular_contours


# Function to display thresholded image of a column
def getThresholdedImage(img, column_idx, num_columns=5):
    # Get the width of the image to define the columns
    img_width = img.shape[1]
    column_width = img_width // num_columns
    
    # Define column region for thresholding
    start_x = column_idx * column_width
    end_x = (column_idx + 1) * column_width

    # Create the thresholded image for the specified column
    column_img = img[:, start_x:end_x]
    _, thresholded = cv.threshold(column_img, 150, 255, cv.THRESH_BINARY_INV)  # Invert the binary image
    
    return thresholded

# Load the image
img = cv.imread("filled.png")
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv.Canny(imgBlur, 10, 50)

# Find contours
contours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Filter circular contours
circleCons = circularContour(contours)

# Reverse the circular contours for the required order
circleCons.reverse()

# Get the width of the image to define the columns
img_width = img.shape[1]

# Split the image into 5 equal columns
num_columns = 5
column_width = img_width // num_columns

# Create a copy of the image to draw contours
imgContours = img.copy()

# Variables to store filled circle count and their areas
circle_count = 0
circle_areas = []

# For each column, count the circles and their areas
column_results = []
for column_idx in range(1):  # Only process the first column
    first_column_contours = []
    
    # Check which contours belong to the current column
    for contour in circleCons:
        x, y, w, h = cv.boundingRect(contour)
        if x < (column_idx + 1) * column_width and x + w > column_idx * column_width:  # Column's bounds
            first_column_contours.append(contour)

    # Count the circles and calculate their areas in the first column
    circle_count = len(first_column_contours)  # Count total circles
    areas = [cv.contourArea(contour) for contour in first_column_contours]  # Calculate areas

    # Append the areas for this column to the result
    circle_areas.extend(areas)  # Store all the areas found

    column_results.append({
        "column": column_idx + 1,
        "circle_count": circle_count,
        "circle_areas": areas,
        "thresholded_image": getThresholdedImage(imgGray, column_idx)
    })

    # Draw the contours for the selected column
    cv.drawContours(imgContours, first_column_contours, -1, (0, 255, 0), 3)

    # Display the thresholded image for the column
    cv.imshow(f"Thresholded Image - Column {column_idx + 1}", column_results[-1]["thresholded_image"])



# Display the original image with the drawn contours
cv.imshow("Circles with All Circles in First Column", imgContours)

# Output column-wise results and circle areas
for result in column_results:
    print(f"  Total Circles: {result['circle_count']}")



cv.waitKey(0)
cv.destroyAllWindows()
