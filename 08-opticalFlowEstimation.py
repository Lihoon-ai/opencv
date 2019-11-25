# -*- coding: utf-8 -*-
"""
Created on Sat Nov.23 2019

Test10-optical flow estimation

@author:Lihoon
"""


import cv2 as cv
import numpy as np
import matplotlib as plt


# define rand corlor function
def randCorlor():
    corlor = np.random.randint(0, 255, (3))

    r = hex(corlor[0])
    g = hex(corlor[1])
    b = hex(corlor[2])
    return r, g, b
# 问题：为什么我生成的随机数组，无法直接用于cv.drawContours()函数？
# 会报错：Scalar value for argument 'color' is not numeric

# import pets video and check
videoName = "/home/lihoon/code/lesson5/vtest.avi"
cap = cv.VideoCapture(videoName)
# cap = cv.VideoCapture(0)                  # it's also fun to use the computer camera.

if cap.isOpened():
    print("Lihoon, read video successful!")
else:
    print("Error!! read video failed!")


# parameters for corners detection
featureParams = dict(maxCorners = 100, qualityLevel = 0.3,
                     minDistance = 7, blockSize = 7)
# parameters for optical flow estimation
lkParams = dict(winSize = (15, 15), maxLevel = 2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

ret, prev = cap.read()
prevGray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(prevGray, mask=None, **featureParams)

while True:
    ret, frame = cap.read()     # capture frame by frame.
    if not ret:                 # ret = True or False represent capture sucess of fail.
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(prevGray, gray, p0, None, **lkParams)
    '''
    p1 - optical flow points
    st - confidence level
    '''
    goodPoints = p1[st >= 0.7]
    goodPrevPoints = p0[st >= 0.7]

    res = frame.copy()

    for i, (cur, prev) in enumerate(zip(goodPoints, goodPrevPoints)):
        x0, y0 = cur.ravel()
        x1, y1 = prev.ravel()
        cv.line(res, (x0, y0), (x1, y1), (0, 97, 255))
        cv.circle(res, (x0 , y0), 3, (0, 97, 255))

    prevGray = gray.copy()
    p0 = goodPoints.reshape(-1, 1, 2)

    cv.imshow("source", frame)
    cv.imshow("result", res)


    key = cv.waitKey(30)        # wait for 30ms for each frame
    if key == 27:               # quit while press ESC
        break

# when everything done, release the capture is important.
cap.release()
cv.destroyAllWindows()
