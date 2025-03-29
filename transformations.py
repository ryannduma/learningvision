import cv2 as cv
import numpy as np

img = cv.imread('/Users/rynduma/learningvision/learningvision/Resources/Photos/park.jpg')
cv.imshow('Park', img)

# Translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]]) # to translate an image you must make a translation matrix - very linear algebra i know
    # the translation matrix is a 2x3 matrix that contains the translation values
    # the first row contains the translation values for the x-axis
    # the second row contains the translation values for the y-axis
    # the third row is the padding row and is not used
    # the translation matrix is a 2x3 matrix that contains the translation values
    # the first row contains the translation values for the x-axis
    # the second row contains the translation values for the y-axis
    # the third row is the padding row and is not used
    dimensions = (img.shape[1], img.shape[0]) # the dimensions of the image
    return cv.warpAffine(img, transMat, dimensions) # the warpAffine function is used to translate the image

# -x --> Left
# -y --> Up
# x --> Right
# y --> Down

translated = translate(img, -100, 100)
cv.imshow('Translated', translated)

# Rotation
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

rotated_rotated = rotate(img, -90)
cv.imshow('Rotated Rotated', rotated_rotated)

# Resizing
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Flipping
flip = cv.flip(img, -1)
cv.imshow('Flip', flip)

# Cropping
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)


cv.waitKey(0)