import cv2 as cv
import numpy as np

# Load the image
img = cv.imread("filled0.png")
img_height, img_width = img.shape[:2]

# Define number of rows per column
num_rows = 30
row_height = img_height // num_rows  # Height of each row

# Process each column one by one
num_columns = 5
column_width = img_width // num_columns

for column_idx in range(num_columns):
    print(f"\nProcessing Column {column_idx + 1}...\n")

    # Define the column's horizontal section
    start_x = column_idx * column_width
    end_x = (column_idx + 1) * column_width
    column_img = img[:, start_x:end_x]  # Extract column part of the image

    # Split column into 30 rows
    for row_idx in range(num_rows):
        # Define the vertical section of the current row
        start_y = row_idx * row_height
        end_y = (row_idx + 1) * row_height if row_idx < num_rows - 1 else img_height  # Ensure the last row reaches the bottom

        row_img = column_img[start_y:end_y, :]  # Extract the row image from the column

        # Process the row image (example: thresholding)
        # For example, convert row image to grayscale and apply threshold
        gray_row_img = cv.cvtColor(row_img, cv.COLOR_BGR2GRAY)
        _, thresholded_row_img = cv.threshold(gray_row_img, 150, 255, cv.THRESH_BINARY_INV)

        # You can now process each row (e.g., detect circles, contours, etc.)

        # Show or process the row image
        cv.imshow(f"Row {row_idx + 1} in Column {column_idx + 1}", row_img)
        cv.waitKey(0)

# Close all OpenCV windows
cv.destroyAllWindows()
