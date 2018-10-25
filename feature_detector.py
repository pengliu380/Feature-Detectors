import numpy as np
import cv2 as cv
# four feature detector to use
def Harris_corner(filename,classname,order):
    img = cv.imread(filename)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    resultname = 'result/' + classname + '/hcd_' + order + '.jpg'
    cv.imwrite(resultname,img)
    '''
    cv.imshow('dst',img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
    '''

def DOG_of_SIFT(filename,classname,order):
    img = cv.imread(filename)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,0,0))
    resultname = 'result/' + classname + '/dog_' + order + '.jpg'
    cv.imwrite(resultname,img)
    '''
    cv.imshow('dst',img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
        '''

def FAST(filename,classname,order):
    img = cv.imread(filename,0)
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    # find and draw the keypoints
    kp = fast.detect(img,None)
    img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
    # Print all default params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    #cv.imwrite('fast_true.png',img2)
    resultname = 'result/' + classname + '/fast_' + order + '.jpg'
    cv.imwrite(resultname,img2)
    '''
    cv.imwrite('result/bikes_blur_fast_1.jpg',img2)
    cv.imshow('img2',img2)
    '''

def MSER(filename,classname,order):
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    mser = cv.MSER_create(_min_area=300)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    regions, boxes = mser.detectRegions(gray)

    for box in boxes:
        x, y, w, h = box
        cv.rectangle(img, (x,y),(x+w, y+h), (0, 255, 0), 1)

    resultname = 'result/' + classname + '/mser_' + order + '.jpg'
    cv.imwrite(resultname,img)
    '''
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

if __name__ == '__main__':
    for i in range(6):
        classname = 'bikes_blur'
        #classname = 'graffiti_viewpoints'
        #classname = 'bark_zoom_rotation'
        #classname = 'cars_light'
        #classname = 'bricks_compression'
        filename = 'data/' + classname +'/img' + str(i+1) +'.ppm'
        #print (filename)
        order = str(i+1)
        Harris_corner(filename,classname,order)
        DOG_of_SIFT(filename,classname,order)
        FAST(filename,classname,order)
        MSER(filename,classname,order)
    print('Finished.')

