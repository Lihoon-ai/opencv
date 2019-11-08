'''
image pocessing with different filters, 
linear filter include mean filter and Gaussian filter,
and non-linear filter include median filter
'''

#test-03:different filters

import cv2 as cv

#read source image
img = cv.imread("./lena.jpg")

#show source image
cv.imshow('source', img)

#box filter
box = cv.boxFilter(img, -1, (3, 3))
cv.imshow('box filter', box)

#mean filter
mean = cv.blur(img, (3, 3))
cv.imshow('mean filter',mean)

#median filter
median = cv.medianBlur(img, 3)
cv.imshow('median filter', median)

#Gaussian filter
gaussian = cv.GaussianBlur(img, (3,3), 0, 0)
cv.imshow('Gaussion filter', gaussian)

cv.waitKey()
