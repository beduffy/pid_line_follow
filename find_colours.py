# from: https://toptechboy.com/tracking-an-object-based-on-color-in-opencv/

import numpy as np
import cv2
print(cv2.__version__)


def onTrack1(val):
    global hueLow
    hueLow = val
    print('Hue Low', hueLow)


def onTrack2(val):
    global hueHigh
    hueHigh = val
    print('Hue High', hueHigh)


def onTrack3(val):
    global satLow
    satLow = val
    print('Sat Low', satLow)


def onTrack4(val):
    global satHigh
    satHigh = val
    print('Sat High', satHigh)


def onTrack5(val):
    global valLow
    valLow = val
    print('Val Low', valLow)


def onTrack6(val):
    global valHigh
    valHigh = val
    print('Val High', valHigh)


width = 640
height = 360
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# cam.set(cv2.CAP_PROP_FPS, 30)
# cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

cv2.namedWindow('myTracker')
cv2.moveWindow('myTracker', width, 0)

# hueLow = 93
# hueHigh = 112
# satLow = 170
# satHigh = 255
# valLow = 0
# valHigh = 90

hueLow = 0
hueHigh = 179
satLow = 27
satHigh = 255
valLow = 200
valHigh = 226

cv2.createTrackbar('Hue Low', 'myTracker', hueLow, 179, onTrack1)
cv2.createTrackbar('Hue High', 'myTracker', hueHigh, 179, onTrack2)
cv2.createTrackbar('Sat Low', 'myTracker', satLow, 255, onTrack3)
cv2.createTrackbar('Sat High', 'myTracker', satHigh, 255, onTrack4)
cv2.createTrackbar('Val Low', 'myTracker', valLow, 255, onTrack5)
cv2.createTrackbar('Val High', 'myTracker', valHigh, 255, onTrack6)


frame = cv2.imread('data/img-28-02-2024-19:30:49-snapshot-night-images-nb9.jpg')
frame = cv2.imread('data/img-28-02-2024-20:24:35-snapshot-night-images-nb9.jpg')
frame = cv2.imread('data/img-28-02-2024-20:11:42-snapshot-night-images-nb9.jpg')



# TODO only look on below part of screen, then plot on main image and go through all images. 
# TODO build my own track?


while True:
    # ignore,  frame = cam.read()
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerBound = np.array([hueLow, satLow, valLow])
    upperBound = np.array([hueHigh, satHigh, valHigh])
    myMask = cv2.inRange(frameHSV, lowerBound, upperBound)
    # myMask=cv2.bitwise_not(myMask)
    myObject = cv2.bitwise_and(frame, frame, mask=myMask)
    myObjectSmall = cv2.resize(myObject, (int(width/2), int(height/2)))
    cv2.imshow('My Object', myObjectSmall)
    cv2.moveWindow('My Object', int(width/2), int(height))
    myMaskSmall = cv2.resize(myMask, (int(width/2), int(height/2)))
    cv2.imshow('My Mask', myMaskSmall)
    cv2.moveWindow('My Mask', 0, height)

    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam', 0, 0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# cam.release()
    
'''
good values for yellow and white

Hue low 0 hue high 179
sat low 27 (bring to 0 to get white) sat high 255
val low 200 val high 226


'''