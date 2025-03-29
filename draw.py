# OpenCV Drawing Demo
# This script demonstrates various drawing functions in OpenCV
# Each step builds on the previous one to create a composite image

import cv2 as cv
import numpy as np

# Create a blank canvas - black image with 3 color channels (BGR)
# Size: 500x500 pixels, data type: unsigned 8-bit integer (0-255 values)
blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Blank Canvas', blank)

# 1. Fill a region with color
# Syntax: image[y_start:y_end, x_start:x_end] = B,G,R
# This creates a red rectangle in the middle-right area
blank[200:300, 300:400] = 0, 0, 255  # BGR format: Red color (0,0,255)
cv.imshow('Red Rectangle Region', blank)

# 2. Draw a filled rectangle
# Syntax: cv.rectangle(image, start_point, end_point, color, thickness)
# thickness=-1 means fill the rectangle
# This draws a green rectangle in the top-left quadrant
cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), 
             (0, 255, 0), thickness=-1)  # Green color
cv.imshow('Green Rectangle Added', blank)

# 3. Draw a filled circle
# Syntax: cv.circle(image, center_point, radius, color, thickness)
# This adds a red circle at the center of the image
center = (blank.shape[1]//2, blank.shape[0]//2)
cv.circle(blank, center, 40, (0, 0, 255), thickness=-1)  # Red circle
cv.imshow('Red Circle Added', blank)

# 4. Draw a line
# Syntax: cv.line(image, start_point, end_point, color, thickness)
# This adds a white diagonal line across the bottom half
cv.line(blank, (100, 250), (400, 400), (255, 255, 255), thickness=3)
cv.imshow('White Line Added', blank)

# 5. Add text to the image
# Syntax: cv.putText(image, text, position, font, scale, color, thickness)
# This adds text in green near the top of the image
font = cv.FONT_HERSHEY_TRIPLEX
cv.putText(blank, 'OpenCV Drawing Demo', (50, 100), font, 
           1.0, (0, 255, 0), thickness=2)
cv.imshow('Final Composite Image', blank)

# Wait for a key press to close all windows
cv.waitKey(0)
cv.destroyAllWindows()