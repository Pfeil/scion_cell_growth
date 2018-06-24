import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline

path = "/home/verpfeilt/Git/Scion/Brightfield-time-series-proliferation/darktable_exported/"
img = cv.imread(path + "Brightfield-time-series-proliferation_t01.tif")
images = []
for i in range(1, 98):
    #print path + "Brightfield-time-series-proliferation_t{:02d}.tif".format(i)
    images.append(cv.imread(path + "Brightfield-time-series-proliferation_t{:02d}.tif".format(i)))

means = images[0]
for i in range(1,97):
    means += images[i]

means / float(len(images))

deriv = (means - images[0])
for i in range(1,97):
    deriv += (means - images[i])

cv.imwrite("deriv.tif", deriv)
exit(0)

img1 = cv.GaussianBlur(images[35], (25,25), 0)
img2 = cv.GaussianBlur(images[36], (25,25), 0)
img2_x = cv.Sobel(img2,cv.CV_64F,1,0,ksize=5)
img2_y = cv.Sobel(img2,cv.CV_64F,0,1,ksize=5)
img2_xy = np.abs(img2_x) + np.abs(img2_y)
kernel = np.ones((5,5),np.uint8)
img2_xy = cv.dilate(img2_xy,kernel,iterations = 1)
diff = img1 - img2
alpha = 0.5
#img1[:,:,0] = alpha*img[:,:,0] + (1-alpha)*img1[:,:,0]
img2[:,:,0] = img2_xy[:,:,0]
#cv.imshow("Fenster", img)
cv.imwrite("output.tif", img2)
