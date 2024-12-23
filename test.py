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

# New Function: Draw First Column Circles (1-145)
def drawFirstColumnCircles(img, contours, start_idx=1, end_idx=5):
    """
    Function to draw and output the first column circles from 1 to 145.

    Args:
        img (numpy array): Original image.
        contours (list): List of filtered circular contours.
        start_idx (int): Start index (1-based).
        end_idx (int): End index (1-based).

    Returns:
        None: Outputs the image with drawn contours.
    """
    # Create a copy of the image to draw the circles
    output_image = img.copy()

    # Check bounds and process only the required contours
    for i, contour in enumerate(contours, start=1):  # Start indexing at 1
        if start_idx <= i <= end_idx:
            cv.drawContours(output_image, [contour], -1, (255, 0, 0), thickness=2)  # Blue outline

    # Display the output image with the selected circles
    cv.imshow(f"Circles {start_idx} to {end_idx}", output_image)

    # Save the output image
    cv.imwrite(f"circles_{start_idx}_to_{end_idx}.png", output_image)

    # Wait for a key press and destroy windows
    cv.waitKey(0)
    cv.destroyAllWindows()

# Load the image
img = cv.imread("filled.png")
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

# Variables to store filled circle count and their areas
circle_count = 0
filled_circles = 0
empty_circles = 0
circle_areas = []
filled_circle_indices = []  # List to store indices of filled circles

# Create a blank image to draw filled contours
output_image = img.copy()

# Process only the first column
column_idx = 0  # First column index
thresholded_image = getThresholdedImage(imgGray, column_idx, num_columns)

# Check which contours belong to the first column
first_column_contours = []
for contour in circleCons:
    x, y, w, h = cv.boundingRect(contour)
    if x < (column_idx + 1) * column_width and x + w > column_idx * column_width:  # Column's bounds
        first_column_contours.append(contour)

# Count the circles and calculate their areas in the first column
circle_count = len(first_column_contours)  # Count total circles
areas = [cv.contourArea(contour) for contour in first_column_contours]  # Calculate areas

# Check if the circle is filled or not (based on white percentage)
for i, contour in enumerate(first_column_contours):
    white_percentage = countBlackWhite(thresholded_image, contour)
    if white_percentage >= 90:
        filled_circles += 1  # Count filled circles
        filled_circle_indices.append(i+1)  # Store 1-based index of the filled circle
        # Draw the filled contour outline on the output image
        cv.drawContours(output_image, [contour], -1, (0, 255, 0), thickness=2)  # Green outline

# Output results
print(f"Total Circles in Column 1: {circle_count}")
print(f"Filled Circles: {filled_circles}")
print(f"Empty Circles: {empty_circles}")
#print(f"Circle Areas: {areas}")
print(f"Filled Circle Numbers: {filled_circle_indices}")

# Display the output image with filled contours
cv.imshow("Outlined Filled Circles", output_image)

# Save the output image
cv.imwrite("outlined_filled_circles.png", output_image)

# Draw the first column's 1–145 circles using the new function
drawFirstColumnCircles(img, circleCons, start_idx=100, end_idx=146)
