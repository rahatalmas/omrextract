import cv2 as cv
import numpy as np
def process(block,Q):
    grey_block_image = cv.cvtColor(block, cv.COLOR_BGR2GRAY)
    img_blur_block = cv.GaussianBlur(grey_block_image, (5, 5), 1)
    img_canny_block = cv.Canny(img_blur_block, 10, 50)
    # Find contours
    contours_block, hierarchy_block = cv.findContours(img_canny_block, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # Filter circular contours
    circleCons_block = circularContour(contours_block)
    #circleCons_block.reverse()
    #print(len(circleCons_block))


    # List to store the centroid coordinates and corresponding contours
    centroid_contour_pairs = []
    # Iterate over each contour to find centroids and store them with contours
    for contour in circleCons_block:
        # Calculate moments of the contour
        M = cv.moments(contour)
        
        # Calculate the centroid coordinates (cx, cy)
        if M["m00"] != 0:  # Check to avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid_contour_pairs.append(((cx, cy), contour))  # Store centroid with contour
        else:
            centroid_contour_pairs.append(((float('inf'), float('inf')), contour))  # Handle case where contour has no area

    # Sort the list based on the x-coordinate (cx) of the centroids (ascending order)
    centroid_contour_pairs.sort(key=lambda x: x[0][0])  # Sort by the first element of the tuple (cx)
    # Extract the sorted contours into a new list
    sorted_contours = [contour for (centroid, contour) in centroid_contour_pairs]
    # # Print the sorted contours and their centroids
    # print("Sorted contours based on centroid x (cx):")
    # for idx, contour in enumerate(sorted_contours):
    #     M = cv.moments(contour)
    #     cx = int(M["m10"] / M["m00"])
    #     cy = int(M["m01"] / M["m00"])
    #     print(f"Contour {idx + 1}: Centroid = ({cx}, {cy})")
    print(len(sorted_contours))

    #answer checking ...
    if len(sorted_contours)==5:
        _, thresholded_block_img = cv.threshold(grey_block_image, 150, 255, cv.THRESH_BINARY_INV)
        filled_circles, filled_circle_indices = processColumnContours(sorted_contours, thresholded_block_img)
        print(f"Question: {Q} Answer: {filled_circle_indices} : filled circles {filled_circles}")

        for idx in filled_circle_indices:
            cv.drawContours(block, [sorted_contours[idx - 1]], -1, (0, 255, 0), thickness=2)  # Green for filled circles
            cv.imshow("block img", block)
            cv.waitKey(0)

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
        if 0.7 < circularity < 1.2:  # Adjust thresholds as needed
            circular_contours.append(contour)
    return circular_contours

# Function to check 

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

# Function to get a thresholded image for a column
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

# Function to process contours within each column
def processColumnContours(column_contours, thresholded_image):
    # Sort the contours within the column by (x + y)
    #sorted_column_contours = sortContoursByXYSum(column_contours)

    # Variables to store filled circle count and indices
    filled_circles = 0
    filled_circle_indices = []

    # Check each contour in the sorted column
    for i, contour in enumerate(column_contours):
        white_percentage = countBlackWhite(thresholded_image, contour)
        if white_percentage >= 90:
            filled_circles += 1  # Count filled circles
            filled_circle_indices.append(i + 1)  # Store 1-based index of the filled circle
            # Draw the filled contour outline on the image
            cv.drawContours(img, [contour], -1, (0, 255, 0), thickness=2)  # Green outline

    return filled_circles, filled_circle_indices


# Load the image
img = cv.imread("filled0.png")
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv.Canny(imgBlur, 10, 50)

# Find contours
contours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Filter circular contours
circleCons = circularContour(contours)

# Get the width of the image to define the columns
img_height, img_width = img.shape[:2]
# Split the image into 5 equal columns
num_columns = 5
column_width = img_width // num_columns
num_rows = 31
row_height = img_height // num_rows  # Height of each row
# Process each column one by one

for column_idx in range(num_columns):
    print(f"\nProcessing Column {column_idx + 1}...\n")

    # Define the column's horizontal section
    start_x = column_idx * column_width
    end_x = (column_idx + 1) * column_width
    column_img = img[:, start_x:end_x]  # Extract column part of the image
    
    # Split column into 31 rows
    for row_idx in range(num_rows):
        # Define the vertical section of the current row
        start_y = row_idx * row_height
        end_y = (row_idx + 1) * row_height if row_idx < num_rows - 1 else img_height  # Ensure the last row reaches the bottom
        row_img = column_img[start_y:end_y, :]
        
        #approach(50)(working...)(sorting based on x axis)
        process(row_img,row_idx)

        ##approach (100)
        #answer blocks
        # num_columns_in_row = 6
        # row_height_b, row_width = row_img.shape[:2]
        # block_width = row_width // num_columns_in_row
        # for blocks in range(num_columns_in_row):
        #     # Define the horizontal section of the current column
        #     start_x = blocks * block_width
        #     end_x = (blocks + 1) * block_width if blocks < num_columns_in_row - 1 else row_width  # Ensure the last column reaches the right side

        #     column_img_in_row = row_img[:, start_x:end_x]  # Extract the row image from the column
        #     cv.imshow("col in row",column_img_in_row)
        #     cv.waitKey(0)
        # process(column_img_in_row)

    #approach(10)
    #find contour check circle or not then check filled or not....
    

cv.waitKey(0)
cv.destroyAllWindows()
