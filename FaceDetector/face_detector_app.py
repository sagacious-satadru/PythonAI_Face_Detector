import cv2
from random import randrange
# import numpy as np

# we load the pre trained data on face frontals from opencv (haar cascade algorithm)
# trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # for some reason this line doesn't work
trained_face_data = cv2.CascadeClassifier(r'C:\Users\fcbsa\OneDrive\Documents\PythonAI\FaceDetector\haarcascade_frontalface_default.xml')

# Remember 'haarcascade_frontalface_default.xml' is containing already trained data. If we decide to train it ourselves on data we've collected it can take a fairly long time

# the image is made up of pixels. The pixels in the end are just numbers. So in essence an image is represented as a 2D array of numbers. We load that image (or rather the numbers that rep the image) into a 2D numpy array
# brad_img = cv2.imread(r'C:\Users\fcbsa\OneDrive\Documents\PythonAI\FaceDetector\Bradley_Cooper_png.png')
# emma_img  = cv2.imread(r'C:\Users\fcbsa\OneDrive\Documents\PythonAI\FaceDetector\Emma_Stone.jpg')


alex_img = cv2.imread(r'C:\Users\fcbsa\OneDrive\Documents\PythonAI\FaceDetector\alexandra_daddario.jpg') # alex_img will be a numpy ndarray
img = cv2.imread(r'C:\Users\fcbsa\OneDrive\Documents\PythonAI\FaceDetector\MSN.jpg')


# convert the image to greyscale
# greyscaled_emma = cv2.cvtColor(emma_img, cv2.COLOR_BGR2GRAY) # in openCV the color channels are not RGB, they're BGR (exactly backwards of what they're usually elsewhere)


grey_alex = cv2.cvtColor(alex_img,cv2.COLOR_BGR2GRAY) # grey_alex is a numpy ndarray. It'll have the original image's grayscaled form though
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# getting the coords of the face
coords_alex = trained_face_data.detectMultiScale(grey_alex)
face_coords = trained_face_data.detectMultiScale(grayscaled_img)
# print(coords_alex)

# cv2.imshow("Alex",grey_alex)
# cv2.waitKey()


# draw rectangles around the faces
# (x, y, w, h) =  coords_alex[0] # don't do this unless you're sure that there's just one face, and you would like to draw the rectangle around that face only. 
# In case more than 1 face has been detected then we'll be having a numpy ndarray of arrays, so if we index out the first array (at the 0th index) and only take the coords and the width the height from that array, then we're only taking the data of the 1st face that was detected

# for (x, y, w, h) in coords_alex:
#     cv2.rectangle(alex_img, (x,y), (x+w, y+h), (0,255,0), 2) 

for (x, y, w, h) in face_coords:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('MSN',img)
cv2.waitKey()

# cv2.imshow('Alex', alex_img)
# cv2.waitKey()

# To capture video from webcam
# webcam = cv2.VideoCapture(r"C:\Users\fcbsa\OneDrive\Documents\PythonAI\FaceDetector\AndrewXEmma.mp4") # if an argument of 0 is given it reads from the current attached webcam on the device.
# if we want any other video file to be opened then we've to pass in the path of that video file
webcam = cv2.VideoCapture(0)

# Iterate forever over the frames
while True:

    # getting the image out from the video
    # read the current frame
    successful_frame_read, frame = webcam.read() # getting the current frame (in real time)
    grayscl_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces (now in real time)
    face_coordinates = trained_face_data.detectMultiScale(grayscl_img)

    # Draw rectangles over the detected faces in the current frame
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)

    # cv2.imshow('Face Detector',grayscl_img)
    cv2.imshow('Face Detector',frame)
    key = cv2.waitKey(1) # the job of waitkey is to wait, or rather pause the execution until a certain key is pressed
    # Here what it does is, it captures a frame from the webcam, then waits for 1 ms before capturing and showing the next frame
    # this is essential since if we're moving and therefore the frames are changing, the imshow might indefinitely wait on one frame before moving onto the next.
    # the iteration moves on to the next frame only through the help of the while loop. So if the waitKey() wait time is not specified, it'll capture the posture in which we're sitting in front of the webcam when the while loop begins, then stay on that frame until a key is pressed (since the time to wait is not specified waitKey waits until any key is pressed). Then when a key is pressed the control can move on to the next iteration in the while loop, then another frame is captured ans shown, and again the execution stalls until a key is pressed
    
    #### Stop when and if the Q key is pressed (Q->Quit)
    if key == 81 or key == 113: # ASCII of Q = 81, q = 113
        break

## Release the VideoCapture object
webcam.release()






# the detectMultiScale makes the method detect the face irrespective of the scale of the image, i.e. whether it's a big one, a small one, or there are multiple faces. # Documentation : https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498

# cv2.imshow("Satadru's face detector",emma_img)
# cv2.waitKey() # this is necessary o/w the opened image will close immediately. If it is not used then the image is opened for a split second and then closed immediately. We don't want that. We want the image window to stay open until we manually close it. So the waitKey() makes it wait for us to manually close the window.
# as a matter of fact we can hit ANY key on the keyboard and it'll close the window

# print(coords_alex)