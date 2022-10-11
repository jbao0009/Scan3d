import open3d as o3d
import numpy as np
import copy
import cv2 
from PIL import Image as pil
#import matplotlib.pyplot as plt
 
 
def get_flattened_pcds2(source,A,B,C,D,x0,y0,z0): 
    x1 = np.asarray(source.points)[:,0] 
    y1 = np.asarray(source.points)[:,1] 
    z1 = np.asarray(source.points)[:,2] 
    x0 = x0 * np.ones(x1.size) 
    y0 = y0 * np.ones(y1.size) 
    z0 = z0 * np.ones(z1.size) 
    r = np.power(np.square(x1-x0)+np.square(y1-y0)+np.square(z1-z0),0.5) 
    a = (x1-x0)/r 
    b = (y1-y0)/r 
    c = (z1-z0)/r 
    t = -1 * (A * np.asarray(source.points)[:,0] + B * np.asarray(source.points)[:,1] + C * np.asarray(source.points)[:,2] + D) 
    t = t / (a*A+b*B+c*C) 
    np.asarray(source.points)[:,0] = x1 + a * t 
    np.asarray(source.points)[:,1] = y1 + b * t 
    np.asarray(source.points)[:,2] = z1 + c * t 
    return source


#how fast do you want it to run?
voxelsize=0.1

dpath="/home/yimu/Downloads/PCD_scans/20220908/448920-0005T-1-3.pcd"
 
 
pcd = o3d.io.read_point_cloud(dpath)
 

pcd = pcd.voxel_down_sample(voxelsize)
 

print('downsampleed')

plane_model, inliers1 = pcd.segment_plane(distance_threshold=0.5,ransac_n=3,
                                         num_iterations=1000)
 



 
inlier_cloud1 = pcd.select_by_index(inliers1)
 
newcloud = pcd.select_by_index(inliers1, invert=True)

print('plane1done')

plane_model, inliers = newcloud.segment_plane(distance_threshold=0.4,ransac_n=3,
                                         num_iterations=1000)

[a, b, c, d] = plane_model


print(plane_model)

print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")


inlier_cloud = newcloud.select_by_index(inliers)

 

 

plane=get_flattened_pcds2(inlier_cloud,a,b,c,d,100,0,0)

#o3d.io.write_point_cloud("plane.pcd", plane)

plane = copy.deepcopy(plane).translate((40, 0, 0))



def get_contour_areas(contours):

    all_areas= []

    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas


 
 
 
pcd_r = plane
 

 
#o3d.io.write_point_cloud("rotatedsurface.ply", pcd_r, write_ascii=write_text)

np_val=np.array(pcd_r.points)

npconv=[]


#print(np_val.shape)

np_val = np.delete(np_val, 2, 1)
 
 
data = pil.fromarray(np_val)
 
scale=10

x=np_val[:, 0]
y=np_val[:, 1]

img = np.zeros((200*scale,200*scale),dtype=np.uint8)

for i in range(np_val.shape[0]):
    a=round(np_val[i,0]*scale+10*scale)
    #print(a)
    b=round(np_val[i,1]*scale)
    img[a,b]=255
    

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  
cv2.imwrite('testcirc13.bmp',img)

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
                               param1=1001, param2=22,
                               minRadius=30, maxRadius=80)

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

cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

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
    



 