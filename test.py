import cv2 as cv
import numpy as np
# Merge pictures
classname = ['bikes_blur', 'graffiti_viewpoints', 'bark_zoom_rotation', 'cars_light']
method = ['hcd', 'dog', 'fast', 'mser']

for i in range(4):
    for j in range(4):
        filename = 'result/' + classname[i] +'/' + method[j] 
        img = cv.imread(filename + '_1.jpg')
        for m in range(2):
            temp = cv.imread( filename + '_' + str(2*m+3) +'.jpg')
            img = np.hstack((img, temp))
        cv.imwrite(filename + '.jpg', img)
