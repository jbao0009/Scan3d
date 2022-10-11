#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:40:55 2022

@author: yimu
"""

import cv2
import numpy as np

img=cv2.imread('testcirc13.bmp') 

img= cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

kernel = np.ones((5,5),np.uint8)

blurred = cv2.GaussianBlur(img, (3, 3), 0)

blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)  

blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)



ret, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)



#contours then polydp 
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)

sorted_contours=sorted_contours[:15]

thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  

for contour in sorted_contours:
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    
    # finding center point of shape
    center, wh, rotate = cv2.minAreaRect(contour)

  
cv2.drawContours(thresh, sorted_contours, -1, (0, 0, 255), 1)


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
blurred = cv2.medianBlur(img_gray, 5)

img_gray = blurred



circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1.6,  120,
                               param1=1001, param2=20,
                               minRadius=30, maxRadius=50)

cir=np.empty((0,3), int)




if circles is not None:
    circles = np.uint16(np.around(circles))
    # Draw the circles
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
        
        cc=[[i[0],i[1], i[2]]]
        cir=np.append(cir,cc,axis=0)



cir = cir[cir[:, 0].argsort()[::-1]]

print(cir)

dis=[]
dia=[]
dia2=[]


for j in range(len(cir)):
    print(j)
    if (j % 2 == 0)or(j<3):
        a=np.array((cir[j][0],cir[j][1]))
        #print(a)
        b=np.array((cir[j+1][0],cir[j+1][1]))
        #print(b)
        dist = np.linalg.norm(a-b)
        
        dist=(dist/10)
         
        d1= (2*cir[j][2]/10)
        d2=(2*cir[j+1][2]/10)
        
        dis=np.append(dis,dist) 
        dia=np.append(dia,d1)
        dia2=np.append(dia2,d2)
    else:
        c=1
    
print(dis,dia,dia2)
    
cv2.imshow('detected circles',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('13.png',img)
