import cv2 as cv
import numpy as np
def rectContour(contours):
    rectCons = []
    for i in contours:
        area = cv.contourArea(i)
        if area>50000:
            peri =cv.arcLength(i,True)
            #approx = cv.approxPolyDP(i,0.02*peri,True)
            rectCons.append(i)
    return rectCons

def getCorner(conts):
     peri =cv.arcLength(conts,True)
     approx = cv.approxPolyDP(conts,0.02*peri,True)
     return approx


def reorder(myPoints):
     pass


def splitBoxes(img):
    rows = np.vsplit(img,10)
    cv.imshow("split",rows[0])