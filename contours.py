"""
Contour Detection Example
-------------------------
This script demonstrates how to detect and draw contours in an image using OpenCV.
It processes an image through several steps: grayscale conversion, blurring, 
edge detection, and finally contour detection and visualization.
"""
import cv2 as cv
import numpy as np

# Load the image from file
img = cv.imread('/Users/rynduma/learningvision/learningvision/Resources/Photos/cats.jpg')
cv.imshow('Cats', img)  # Display the original image

# Create a blank image with the same dimensions as the original
# This will be used to draw the contours
blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)  # Display the blank canvas

# Convert the image to grayscale
# Contour detection works better on grayscale images
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)  # Display the grayscale image

# Apply Gaussian blur to reduce noise and improve edge detection
# The (5,5) is the kernel size - larger values create more blur
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)  # Display the blurred image

# Detect edges using Canny edge detector
# Parameters 125 and 175 are the lower and upper thresholds
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)  # Display the edges

# Alternative method: Thresholding (currently commented out)
# This would convert the image to binary (black and white) based on a threshold - if the intensity of an image is above a certain threshold, it is white, otherwise it is black
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)

# Find contours in the edge-detected image
# RETR_LIST retrieves all contours
# CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

# Draw the contours on the blank image
# Parameters: image, contours, contour index (-1 means all), color (BGR), thickness
cv.drawContours(blank, contours, -1, (0,0,255), 1)  # Red color, thickness 1
cv.imshow('Contours Drawn', blank)  # Display the image with contours

# Wait for a key press to close all windows
cv.waitKey(0)