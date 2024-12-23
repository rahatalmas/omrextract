import cv2 as cv
import numpy as np

# Function to filter circular contours
def circularContour(contours, min_area=150):
    circular_contours = []
    for contour in contours:
        # Calculate contour area
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

# Function to count white and black pixels inside the contour
def countBlackWhite(img, contour):
    # Create a mask for the current contour
    mask = np.zeros_like(img)
    cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)

    # Apply the mask to the image and count white pixels
    masked_img = cv.bitwise_and(img, img, mask=mask)
    white_pixels = np.sum(masked_img == 255)
    total_pixels = cv.contourArea(contour)  # Total pixels inside the contour (approximated by area)

    # Calculate percentage of white pixels
    white_percentage = (white_pixels / total_pixels) * 100
    return white_percentage

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

# Function to sort contours top to bottom
def sortContoursTopToBottom(contours):
    bounding_boxes = [cv.boundingRect(c) for c in contours]
    return [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][1])]

# Load the image
img = cv.imread("filled2.png")
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv.Canny(imgBlur, 10, 50)

# Find contours
contours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Filter circular contours
circleCons = circularContour(contours)

# Get the width of the image to define the columns
img_width = img.shape[1]

# Split the image into 5 equal columns
num_columns = 5
column_width = img_width // num_columns

# Process each column one by one
for column_idx in range(num_columns):
    print(f"\nProcessing Column {column_idx + 1}...\n")

    # Get the thresholded image for the current column
    thresholded_image = getThresholdedImage(imgGray, column_idx, num_columns)

    # Check which contours belong to the current column
    column_contours = []
    for contour in circleCons:
        x, y, w, h = cv.boundingRect(contour)
        if x < (column_idx + 1) * column_width and x + w > column_idx * column_width:  # Column's bounds
            column_contours.append(contour)

    # Sort contours top to bottom
    #column_contours = sortContoursTopToBottom(column_contours)
    #column_contours.sort(key=lambda c: sum(cv.boundingRect(c)[:2]))

    # Count the circles and calculate their areas in the current column
    circle_count = len(column_contours)  # Count total circles
    areas = [cv.contourArea(contour) for contour in column_contours]  # Calculate areas

    # Variables to store filled circle count and indices
    filled_circles = 0
    filled_circle_indices = []

    for i, contour in enumerate(column_contours):
        white_percentage = countBlackWhite(thresholded_image, contour)
        if white_percentage >= 90:
            filled_circles += 1  # Count filled circles
            filled_circle_indices.append(i + 1)  # Store 1-based index of the filled circle
            # Draw the filled contour outline on the image
            cv.drawContours(img, [contour], -1, (0, 255, 0), thickness=2)  # Green outline

    # Output results for the current column
    print(f"Total Circles in Column {column_idx + 1}: {circle_count}")
    print(f"Filled Circles in Column {column_idx + 1}: {filled_circles}")
    print(f"Filled Circle Indices in Column {column_idx + 1}: {filled_circle_indices}")

# Save the final output image
cv.imwrite("outlined_filled_circles_all_columns.png", img)

# Display the final image
cv.imshow("Outlined Filled Circles (All Columns)", img)

cv.waitKey(0)
cv.destroyAllWindows()
