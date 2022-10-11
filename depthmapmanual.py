#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:40:55 2022

@author: yimu
"""
#this file checks all the bmp n the fomder and finds the
# lenght, width and hole centers  of the parts
# importing the library

from math import atan2, cos, sin, sqrt, pi
from PIL import Image
import os
import cv2
import numpy as np
import math
import imutils
from scipy.interpolate import splprep, splev
# giving directory name
dirname = '/home/yimu/Downloads/PCD_scans/20220928/pcd-0928/test/'
 
# giving file extension
ext = ('.png')

 

def drawlineangle(center, angle,length):
    #idunno cant math right now
    if angle <45:
        angle=angle +90
    x1=length*math.sin(np.deg2rad(-angle))
    y1=length*math.cos(np.deg2rad(-angle))
    start=[center[0]+x1,center[1]+y1]
    end=[center[0]-x1,center[1]-y1]
    start=np.int0(start)
    end=np.int0(end)
    return start,end


 
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return angle


def controller(img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
 
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
 
    if brightness != 0:
 
        if brightness > 0:
 
            shadow = brightness
 
            max = 255
 
        else:
 
            shadow = 0
            max = 255 + brightness
 
        al_pha = (max - shadow) / 255
        ga_mma = shadow
 
        # The function addWeighted
        # calculates the weighted sum
        # of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
 
    else:
        cal = img
 
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
 
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)
 
    # putText renders the specified
    # text string in the image.
    cv2.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)
 
    return cal
        
#downsample?
voxelsize=0.05

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1200,1200)   

gt=[79.06,78.95,98.34,15.84,98.34,15.84]
vari=[0,0,0,0,0,0]
measured=[] 
# iterating over all files

for files in os.listdir(dirname):
    if files.endswith(ext):
            
        data=[]
        print (files)
        img=cv2.imread(dirname+files) 
        raw=img
         
        img=img.astype(np.uint8) 
        shp=img.shape
        blank_image = np.zeros(shape=shp, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
         
        
        
        
            
        img = cv2.medianBlur(img, 5)
        
        img = img
        
        contours, hierarchies = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        raw = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
        
        
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        
        
        
        ccontours=sorted_contours[2:4]#only 2 largest contours
        
        cv2.drawContours(raw, ccontours, -1,   (255, 255, 0), 1)
        
        
        c1=[]
         
        for i in ccontours:
            M = cv2.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.drawContours(raw, [i], -1, (0, 255, 0), 2)
                cv2.circle(raw, (cx, cy), 7, (0, 0, 255), -1)
                c1=np.append(c1,(cx,cy))
                print(c1)
        
        
        cv2.imshow('image', raw)
         
        cv2.waitKey()
        cv2.destroyAllWindows()     
        
        dis=[]
        dia=[]
        dia2=[]
         
        dis=math.dist((c1[0],c1[1]),(c1[2],c1[3]))/20
        #print(dis)
        data=np.append(data,dis) 
        
                #contours then polydp 
        thresh = cv2.bilateralFilter(img,9,75,75)
        
      
        sorted_contours=sorted_contours[:15]
        
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  
        
        sorted_contours=sorted_contours[:1]#only 2 largest contours
        
 
        for contour in sorted_contours:
             
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            h,w=rect[1]
            print("legnth")
            print(max(h,w)/10)
            data=np.append(data,max(h,w)/10) 
            p1=box[0]
            p2=box[1]
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,255),1)
            start,end=drawlineangle(rect[0],rect[2],200)
            print (rect[2])
            shape=img.shape
            
            blank=np.zeros(shape=shape, dtype=np.uint8)
            blank3=np.zeros(shape=shape, dtype=np.uint8)
            line1=cv2.line(blank,start,end,(255,255,255),1)
            #cv2.imshow('lin1',line1)
            
            blank2=np.zeros(shape=shape, dtype=np.uint8)
            testcnt=cv2.drawContours(blank2,contour,-1,(255,255,255),1)
            blank3=cv2.line(blank3,start,end,(0,0,255),1)
            cv2.drawContours(blank3,contour,-1,(255,0,0),1)
            cv2.drawContours(blank3,[box],0,(0,255,0),1)
            
            
            intersect=cv2.bitwise_and(blank,blank2)
            
            #cv2.imshow('testc',intersect)
            pixelpoints = np.transpose(np.nonzero(intersect))
            
            if pixelpoints.size>3:
                width=math.dist(pixelpoints[0],pixelpoints[3])
            #print(width/10)
            else :
                width=1
            data=np.append(data,width/10)     
            
        #diff=[data-gt]
        diff=data
        #vari=np.vstack((vari,diff))
        data=np.append(data,files)     
        print(measured)
        print(data)
        if measured == []:
            measured=np.append(measured, data)
        else:
            
            measured=np.vstack((measured, data))
        
        cv2.namedWindow('detected', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('detected', 1200,1200)   
        cv2.imshow('detected',img)
        
        
print(vari)    
print(measured)  


np.savetxt("dmap.txt", measured,fmt="%s")