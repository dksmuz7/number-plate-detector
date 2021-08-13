# This project is for the detection of number plates on any vehicle, so lets get started

# importing the required modules
import numpy as np
import cv2 as cv

# defining some constraints or variables
frameWidth = 680
frameHeight = 420
brightness = 100

# setting up webcam
webcam = cv.VideoCapture(0) #camera no
webcam.set(3,frameWidth)
webcam.set(4,frameHeight)
webcam.set(10,brightness)

# Creating path of data and cascading
dataPath = "/media/deepaksagar/Study Materials/Programming/python/OpenCV/resources/cascade_number_plate.xml"
numberPlateCascade = cv.CascadeClassifier(dataPath)
bBoxColor = (0,255,255)

while True:
    success,img = webcam.read()

    grayedImg = cv.imread(img,cv.COLOR_BGR2GRAY)

    numberPlate = numberPlateCascade.detectMultiscale(grayedImg,1.3,4)

    # creating bounding box arround the detected number plate
    for (x,y,w,h) in numberPlate:
        cv.rectangle(img,(x,y),(x+w,y+h),bBoxColor),2

    cv.imshow("Camera 01",img)


    if cv.waitKey(1) & 0xFF == ord('s'):
        break



