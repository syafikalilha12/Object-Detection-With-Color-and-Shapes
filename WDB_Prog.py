from matplotlib.pyplot import contour
import numpy as np 
import cv2
import imutils
from itertools import count
import pandas as pd 

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

cap=cv2.VideoCapture(0)
while(1):
    centers=[]
    gbr = cv2.imread('bluecircle.png')
    hsv = cv2.cvtColor(gbr,cv2.COLOR_BGR2HSV)
    low=np.array([20,100,100])
    high= np.array([40,255,255])
    mask = cv2.inRange(hsv,low,high)
    cv2.imshow('ling',mask)
    kernal = np.ones((5,5),"uint8")
    blue = cv2.erode(mask,kernal,iterations=1)
    gbr1 = cv2.bitwise_and(hsv, hsv, mask=blue)
    mask1 = np.sum(mask)

    contours = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if mask1 >0:
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                img_gray = cv2.cvtColor(gbr1, cv2.COLOR_BGR2GRAY)
                img_gray_blured = cv2.blur(img_gray, (3,3))

                detected_circles = cv2.HoughCircles(img_gray_blured,cv2.HOUGH_GRADIENT, 1,20, param1=60, param2=30, minRadius=5, maxRadius=80)
                if detected_circles is not None:
                    detected_circles = np.uint16(np.around(detected_circles))
                    for pt in detected_circles[0, :]:
                        a, b, r = pt[0], pt[1], pt[2]
                        cv2.rectangle(gbr, (a-(1*r),b-(1*r)), (a+(1*r),b+(1*r)), (0,128,0), 2)
                        M = cv2.moments(contour)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        center = (cx,cy)
                        cv2.circle(gbr, (cx, cy), 7, (255, 255, 255), -1)
                        cv2.putText(gbr,"Center", (center[0]+10,center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    else:
        print("Warna tidak sesuai") 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("final",gbr)
cap.release()
cv2.destroyAllWindows()