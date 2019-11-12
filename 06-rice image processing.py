# encoding:utf-8

#####################################################################
#test-06
#        grayscale histogram of image and threshold with OTSU-method.
#        find contours and calculate the area and length of contours.
#####################################################################

import cv2 as cv
import  numpy as np
import copy
from matplotlib import pyplot as plt

############################
### 01 read image and cheak.
############################

# name the image for read.
img_name = "/home/lihoon/code/opencvImg/threshold/米粒图片.png"
img = cv.imread(img_name)
if img is None:
    print("image read error!")
else:
    print("image read correctly!")

#change image space for BGR to GRAY
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

############################
### 02-1 calculate grayscale histogram of image with calcHish().
############################
img_histogram = cv.calcHist(img, [0], None, [256], [0,256])

# plot grayscale histogram use matplot
plt.figure(num = 1, figsize = (5, 5))
plt.title("grayscale histogram")
plt.xlabel("bins")
plt.ylabel("number of pixels")
plt.plot(img_histogram)
plt.xlim([0,256])
plt.show()

############################
### 02-2 hist() function in matplotlib is used to calculate.
############################
plt.hist(img.ravel(), 256, [0, 255])
# ravel() function is used to reduce multidimensional arrays to one demension.
plt.title("hist used matplot")
plt.show()

############################
### 03 image threshold with OTSU method
############################
ret1, threshold1 = cv.threshold(gray, 140, 255, cv.THRESH_BINARY)
ret2, threshold2 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)

# the picture showed by matplot is so big that I don't want to use.
# plt.imshow(threshold1)
# plt.show()
# plt.imshow(threshold2)
# plt.show()

cv.imshow("binary", threshold1)
cv.imshow("otsu", threshold2)

############################
### 04 floodfill method
############################
img_copy = gray.copy()
h,w = img.shape[:2]
print(h, w)
mask = np.zeros([h + 2, w + 2], np.uint8)
cv.floodFill(img_copy, mask, (125, 129), 0, 80, 15, cv.FLOODFILL_FIXED_RANGE)
cv.imshow("floodfill",img_copy)

############################
### 05 morphological operation
############################
element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
img1 = cv.morphologyEx(threshold2, cv.MORPH_OPEN, element)

############################
### 06 find contours from binary image
############################
seg = copy.deepcopy(img1)
bin, cnts, hier = cv.findContours(seg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

count = 0            #number of correct counters
area_list = []       # save area of contours
length_list = []     #save length of contours

# judge whether the contours is correct
for i in range(len(cnts), 0, -1):
    con_num = cnts[i -1]                     #contour index
    area = cv.contourArea(con_num)           #calculate the area of current contour
    length = cv.arcLength(con_num, True)     #calculate the length of current contour
    if area < 10 or length < 10:
        continue
    count = count +1
    area_list.append(area)          #area list（append one by one）
    length_list.append(length)      #length list（append one by one）
    print("blob", i, " area is : ", area, " length is : ", length)

    #drawing the bounding rectangle and showing the contours number
    x, y, w, h = cv.boundingRect(con_num)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv.putText(img, str(count), (x, y), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0))

# calculating the mean value and standard deviation value
mean_area = np.mean(area_list)
mean_length = np.mean(length_list)
std_area = np.std(area_list, ddof = 1)
std_length = np.std(length_list, ddof = 1)

#print contours information
print("number of rice is: ", count)
print("mean of area is : ", mean_area)
print("mean of length is : ", mean_length)
print("standard deviation of area is : ", std_area)
print("standard deviation of length is : ", std_length)

#showing the source image and contours with number
cv.imshow("source", img)
cv.imshow("threshold", threshold2)

############################
### 07 calculating the number of rice in 3-sigma range
############################
area_3sigma = []        #save the area value
length_3sigma = []      #save the length value

#find the value in three-sigma range
for i in range(0, count, 1):
    if area_list[i] > (mean_area - 3 * std_area) or area_list[i] < (mean_area + 3 * std_area):
        area_3sigma.append(area_list[i])
    else:
        continue

for i in range(0, count, 1):
    if length_list[i] > (mean_length -3 * std_length) or length_list[i] < (mean_length + 3 * std_area):
        length_3sigma.append(length_list[i])
    else:
        continue

print("result of area for three-sigma is : {}".format(len(area_3sigma)))
print("result of length for length-sigma is : {}".format(len(length_3sigma)))

cv.waitKey()
cv.destroyAllWindows()