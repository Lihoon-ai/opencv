
'''Test02:lena.jpg to RGB and HSV'''

#include opencv.lib
import cv2 as cv

#read image
filename = './lena.jpg'
img1 = cv.imread("./lena.jpg")

#change image space from BGR to GRAY
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

#show the source image and gray image
cv.imshow("source image", img1)
cv.imshow("gray", gray)
#cv.waitKey()

#change image space from BGR to HSV
hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)

#show the HSV image * 3
cv.imshow("Hue", hsv[:, :, 0])
cv.imshow("Saturation", hsv[:, :, 1])
cv.imshow("Value", hsv[:, :, 2])
#cv.waitKey()

#show the BGR image * 3
cv.imshow("Blue", img1[:, :, 0])
cv.imshow("Green", img1[:, :, 1])
cv.imshow("Red", img1[:, :, 2])
cv.waitKey()

#close all windows
cv.destroyAllWindows()
