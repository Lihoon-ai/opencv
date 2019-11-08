'''
edge detection 
dege detection using Sobel methond, Laplace method and Canny method.
'''

#test-05 edge detection

import cv2 as cv

#read image
img = cv.imread("./opencv/lena.jpg")
cv.imshow("source", img)

#change image space for BGR to GRAY
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
cv.imwrite('./opencv/gray.jpg', gray)

#source image edge detection with Sobel
sobel = cv.Sobel(img, cv.CV_8U, 1, 1, 3)
cv.imshow("sobel", sobel)
cv.imwrite('./opencv/sobel.jpg', sobel)

#gray image dege detection with Sobel
graysobel = cv.Sobel(gray, cv.CV_8U, 1, 1, ksize = 3)
cv.imshow("graysobel", graysobel)
cv.imwrite('./opencv/sobelgray.jpg', graysobel)

#gray image edge detection with Laplace
laplace = cv.Laplacian(gray, cv.CV_8U, (3, 3))
cv.imshow('laplace', laplace)
cv.imwrite('./opencv/laplace.jpg', laplace)

#gray image dege detection with Canny
canny = cv.Canny(gray, 100, 200, (3, 3))
cv.imshow('canny', canny)
cv.imwrite('./opencv/canny.jpg', canny)

cv.waitKey()
