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
webcam.set(100,brightness)

# Creating path of data and cascading
dataPath = "resources/cascade_number_plate.xml"
numberPlateCascade = cv.CascadeClassifier(dataPath)
bBoxColor = (0,255,255)
minArea = 500
color = (255,0,0)
count = 0

while True:
    success,img = webcam.read()

    grayedImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    numberPlate = numberPlateCascade.detectMultiScale(grayedImg,1.3,4)

    # creating bounding box arround the detected number plate
    for (x,y,w,h) in numberPlate:
        area = w*h
        if area > minArea:
            cv.rectangle(img,(x,y),(x+w,y+h),bBoxColor,2)
            cv.putText(img,"Number Plate",(x,y-5),cv.FONT_HERSHEY_SIMPLEX,1,color,2)

            numRimage = img[y:y+h,x:x+w]
            cv.imshow("Detected Number Plate",numRimage)

    cv.imshow("Camera 01",img)


    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite("output/scanned/numPlate_"+str(count)+".jpg",numRimage)
        cv.rectangle(img,(0,200),(640,300),(0,255,0),cv.FILLED)
        cv.putText(img,"Saved",(150,265),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
        cv.imshow("Result",img)
        cv.waitKey(500)
        count+=1



