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
dirname = '/home/yimu/Downloads/PCD_scans/20220928/pcd-0928/'
 
# giving file extension
ext = ('.png')

def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result

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

gt=[79.06,78.95,98.34,15.84,98.34,15.84]
vari=[0,0,0,0,0,0]
measured=[0,0,0,0,0,0,0] 
# iterating over all files

for files in os.listdir(dirname):
    if files.endswith(ext):
            
        data=[]
        print (files)
        img=cv2.imread(dirname+files) 
        raw=img
         
        
        #img= cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        shp=img.shape
        blank_image = np.zeros(shape=shp, dtype=np.uint8)
        
        num=10
        
        array = get_gradient_3d(shp[0], shp[1], (0, 0, 0), (num, num,num), (True, True, True))
        array=array.astype(np.uint8) 
        array= imutils.rotate(array, angle=180)
   
 
       
        kernel = np.ones((5,5),np.uint8)
        
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        
        blurred = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
        
        blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        
        edges = cv2.Canny(image=blurred, threshold1=10, threshold2=1000)
        
        
        
        laplacian = cv2.Laplacian(img,cv2.CV_64F)

        
       
        laplacian=laplacian.astype(np.uint8) 
         
        laplacian = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)
        
        img = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        kernel = np.ones((7, 7), 'uint8')

        dilate_img = cv2.dilate(laplacian, kernel, iterations=2)
        
        dilate_img = cv2.erode(dilate_img, kernel, iterations=5)   

        
 
        
        contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        contours= sorted(contours, key=cv2.contourArea, reverse= True)

        cv2.drawContours(img, contours,0, (0, 0, 255), 2)
            # Find the orientation of each shape
        angle=getOrientation(contours[0], img)
            
        #print(angle)    
        
        array=cv2.rotate(array,int(angle))
        
        
        contrast=20
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
 
        # The function addWeighted calculates
        # the weighted sum of two arrays
        raw = cv2.addWeighted(raw, Alpha,  raw, 0, Gamma)
        
        
        bright=255
        contrast=250
        bg=array+raw
        
        
        #effect = controller(raw,bright,contrast)
        raw = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)  
        raw = cv2.equalizeHist(raw)
        
        ret, thresh = cv2.threshold(raw,60,255, cv2.THRESH_BINARY)
        
        
        
        
        

       

        
        img=raw
        img_gray = raw
            
            
        blurred = cv2.medianBlur(img_gray, 5)
        
        img_gray = blurred
        
        
        
        circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1.6,  250,
                                       param1=1050, param2=20,
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
        
        #print(cir)
        
        
        cv2.resizeWindow('image', 1200,1200)       
        cv2.imshow('image', img)
         
        cv2.waitKey()
        cv2.destroyAllWindows()     
        
        dis=[]
        dia=[]
        dia2=[]
        
        try:
            
            for j in range(len(cir)):
                #print(j)
                if (j % 2 == 0)and(j<3):
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
        except:
            print('nocirc')
        
        #print(dis)
        data=np.append(data,dis) 
        
                #contours then polydp 
        thresh = cv2.bilateralFilter(thresh,9,75,75)
        
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        
        sorted_contours=sorted_contours[:15]
        
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  
        
        sorted_contours=sorted_contours[:2]#only 2 largest contours
        
 
        for contour in sorted_contours:
             
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            h,w=rect[1]
            #print(max(h,w)/10)
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
            
            cv2.imshow('testc',blank3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
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
        measured=np.append(measured,data)
        
        
        cv2.imshow('detected circles',img)
        
        
print(vari)    
print(measured)  


np.savetxt("dmap.txt", measured,fmt="%s")