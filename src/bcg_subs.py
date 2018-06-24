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

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

result = np.zeros_like(images[0][:,:,0])
fgbg = cv.createBackgroundSubtractorMOG2()
for i in range(0, 97):
    frame = images[i][:,:,0]
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    result += fgmask
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cv.destroyAllWindows()

cv.imwrite("output.tif", result)
