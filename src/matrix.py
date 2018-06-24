
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
#%matplotlib inline

def invert(imagem):
    return (255-imagem)

path = "/home/verpfeilt/Git/Scion/Brightfield-time-series-proliferation/darktable_exported/"

def sobelThresh(img2):
    img2 = cv.GaussianBlur(img2, (25,25), 0)

    img2_x = cv.Sobel(img2,cv.CV_64F,1,0,ksize=5)
    img2_y = cv.Sobel(img2,cv.CV_64F,0,1,ksize=5)
    img2_xy = np.abs(img2_x) + np.abs(img2_y)

    img2_xy = invert(img2_xy)

    width, height, channels = img2_xy.shape
    thresh = 184
    ret,img2_xy = cv.threshold(img2_xy, thresh, 255,  cv.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    #img2_xy = cv.erode(img2_xy,kernel,iterations = 5)
    img2_xy = cv.morphologyEx(img2_xy, cv.MORPH_OPEN, kernel, iterations = 6)
    return img2_xy

def compute_cells(image):
    counter = 0
    width, height, channels = image.shape
    for x in range(width):
        for y in range(height):
            if image[x,y,0] < 200:
                counter += 1
    return counter

def plot_timeline(cellcount):
    t = np.arange(0, len(cellcount), 1)
    s = cellcount
    plot(t, s)

    xlabel('Zeit')
    ylabel('Zellflaeche')
    title('Zellwachstum')
    grid(True)
    show()

cellcount = []
for i in range(1, 98, 10):
    #print path + "Brightfield-time-series-proliferation_t{:02d}.tif".format(i)
    print i
    image = sobelThresh(cv.imread(path + "Brightfield-time-series-proliferation_t{:02d}.tif".format(i)))
    cv.imwrite("out/out_{:02d}.tif".format(i), image)
    cellspace = compute_cells(image)
    cellcount.append(cellspace)

#print cellcount
plot_timeline(cellcount)

#result_images = [sobelThresh(img2) for img2 in images]


#diff = img1 - img2
#alpha = 0.5
#img1[:,:,0] = alpha*img[:,:,0] + (1-alpha)*img1[:,:,0]
#img2[:,:,0] = img2_xy[:,:,0]
#cv.imshow("Fenster", img)
#cv.imwrite("output.tif", img2_xy)
