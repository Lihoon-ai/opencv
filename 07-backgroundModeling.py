# -*- coding: utf-8 -*-
"""
Created on Fri Nov.22 2019

Test09-background modeling

@author:Lihoon
"""


import cv2 as cv
import numpy as np
import matplotlib as plt

# import pets video and check
videoName = "/home/lihoon/code/lesson5/vtest.avi"
cap = cv.VideoCapture(videoName)
if cap.isOpened():
    print("Lihoon, read video successful!")
else:
    print("Error!! read video failed!")

# cap = cv.VideoCapture(0)                  # it's also fun to use the computer camera.

fgbg = cv.createBackgroundSubtractorMOG2()  # create a background object.
areaThresh = 200                            # set area threshold for feature.

while True:
    ret, frame = cap.read()     # capture frame by frame.
    if not ret:                 # ret = True or False represent capture sucess of fail.
        break

    # frame = frame.resize((int(frame.size*0.5)), cv.INTER_LINEAR)
    fgmask = fgbg.apply(frame)  # get foreground mask
    _, fgmask = cv.threshold(fgmask, 30, 255, cv.THRESH_BINARY)

    bgImage = fgbg.getBackgroundImage()
    _, cnts, _ = cv.findContours(fgmask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    count = 0
    c_min = []
    for c in cnts:
        area = cv.contourArea(c)
        if (area < areaThresh):
            continue
        count += 1
        c_min.append(c)

        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 97, 255) , 2)
        cv.putText(frame, str(count), (x,y), cv.FONT_HERSHEY_PLAIN, 2, (0, 97, 255), 2)

    print("there are {} targets detected!".format(count))

    # display the new frame with rectangle and the latest background.
    # cv.drawContours(frame, c_min, -1, (0, 97, 255), thickness=-1)
    cv.imshow("frame", frame)
    cv.imshow("Background", bgImage)
    cv.imshow("Foreground", fgmask)

    key = cv.waitKey(30)        # wait for 30ms for each frame
    if key == 27:               # quit while press ESC
        break

# when everything done, release the capture is important.
cap.release()
cv.destroyAllWindows()
