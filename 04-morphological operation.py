'''
morphological open operation is eroding after dilating
and close operation is eroding before dilating
'''

import cv2 as cv

#read source image
img1 = cv.imread("./lena.jpg")

#show source image
cv.imshow('source', img1)

#change image form BGR to GRAY model
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)


#get structure for morphologic
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
'''
getStructuringElement(shape, ksize, anchor=None)
where, 
shape include：MORPH_RECT(for rectangle), MORPH_CROSS(for cross) and MORPH_ELLIPSE(for ellipse)
ksize is the size of kernel
anchor is the location of anchor
'''

#get open morph and close maoph
dst1 = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel)
dst2 = cv.morphologyEx(dst1, cv.MORPH_CLOSE, kernel)
cv.imshow("open", dst1)
cv.imshow("open and close", dst2)
cv.waitKey()

