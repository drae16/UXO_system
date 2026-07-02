from ultralytics import YOLO
import cv2 as cv
import time
camera = cv.VideoCapture(2)
camera.set(cv.CAP_PROP_BUFFERSIZE, 1)
model = YOLO(r'/home/drl/poi_dog/poidog_robotarm/src/nav_search/models/pipes.pt')

while True:
    time.sleep(1.5)
    ret,img = camera.read()
    ret,img = camera.read()
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    model.predict(img,show=True)
    # Display the resulting frame
    cv.imshow('frame', img)
    if cv.waitKey(2) == ord('q'):
        break
   
   
