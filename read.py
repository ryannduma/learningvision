import cv2 as cv

img = cv.imread('../Resources/Photos/cats.jpg') # returns a numpy array matrix of the image and stores it in img
cv.imshow('Cats', img) # displays the image as a new window named 'Cats' and the pixels to show are stored in img

cv.waitKey(0) # waits for a key press before closing the window

cv.destroyAllWindows() # destroys all the windows created

# Reading Videos
capture = cv.VideoCapture('../Resources/Videos/dog.mp4') # opens the video file and stores it in capture - Storing the video in a variable 
# you can reference your webcam by using 0, 1, 2, etc. - but for this example we are using a video file - in this case dog.mp4


while True: # while loop to read the video frame by frame - this is a loop that will continue to run until the video is finished
    isTrue, frame = capture.read() # reads the video frame by frame - isTrue is a boolean value that is true if the frame is read successfully
    cv.imshow('Video', frame) # displays the video frame by frame
    if cv.waitKey(20) & 0xFF==ord('d'): # waits for a key press before closing the window - 0xFF is a bitwise AND operation that is used to check if the key pressed is 'd'
        break # breaks the loop if the key pressed is 'd'

capture.release() # releases the video capture object
cv.destroyAllWindows() # destroys all the windows created

