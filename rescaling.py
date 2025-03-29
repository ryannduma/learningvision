import cv2 as cv

img = cv.imread('../Resources/Photos/cat_large2.jpg')
cv.imshow('Cat', img)

cv.waitKey(0)

# Resizing - takes the width and height of the image and resizes it to the new dimensions
def rescaleFrame(frame, scale=0.75): # scale is the factor by which the image is resized
    width = int(frame.shape[1] * scale) # shape[1] is the width of the image
    height = int(frame.shape[0] * scale) # shape[0] is the height of the image
    dimensions = (width, height) # dimensions is a tuple of the width and height of the image
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) # resize the image to the new dimensions


resized = rescaleFrame(img)
cv.imshow('Resized', resized)

cv.waitKey(0)  

# Reading Videos
capture = cv.VideoCapture('../Resources/Videos/dog.mp4') # this scales the window where the image is displayed

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)
    cv.imshow('Video', frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

# alternate method using capture.set()

def changeRes(width, height): # change res - pass in the width and height of the new resolution - you can use this for images, videos, and live videos
    capture.set(3, width) # 3 is the width of the image
    capture.set(4, height) # 4 is the height of the image 

changeRes(600, 600) # but the change res will only work for live videos - not for images or videos that are already stored 



