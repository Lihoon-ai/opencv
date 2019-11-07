
#Evironment：Ubuntu18.04+Anaconda3+OpenCV3.4.2+python3.7.4


'''Test01:first write cv pro. use python'''

###The first test pro. "Hello world!"

#include opencv.lib
import cv2 as cv

#read the image use the function:imread（）
img = cv.imread("./lena.jpg")

#show the imge use the function:imshow（）
cv.imshow('Hello world!', img)

#waitKey()--wait for any press.这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下 键,则接续等待(循环)
cv.waitKey()

#close all opened windows
cv.destroyAllWindows()
