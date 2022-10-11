#6.00450303263566
#6.03678024030668
#6.021143009136065
#6.018174616258902
#5.986958283712583
#5.9947126957995485
import numpy as np
import math
import os
    
    
def generate_specific_rows(filePath, row_indices=[]):
    with open(filePath) as f:
        # using enumerate to track line no.
        for i, line in enumerate(f):
            #if line no. is in the row index list, then return that line
            if i in row_indices:
                yield line
                
 

 
    #for line in o:
readall=np.loadtxt("circICP.txt") 
#print (readall)


dirname = '/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010'
 
# giving file extension
ext = ('.txt')


for files in os.listdir(dirname):
    if files.endswith(ext):
            
        readall=np.loadtxt(files) 
                 
        p1a=readall[0]
        p2a=readall[1]
        p3a=readall[2]
        p4a=readall[3]
        p5a=readall[4]
        p6a=readall[5]
        p7a=readall[6]
        p8a=readall[7]
        p9a=readall[8]
        p10a=readall[9]
        
         
        a1=math.dist(p1a,p2a)
        a2=math.dist(p3a,p4a)
        a3=math.dist(p5a,p6a)
        a4=math.dist(p7a,p8a)
        a5=math.dist(p9a,p10a) 
        
        
        # print(a1)
        # print(a2)
        # print(a3)
        # print(a4)
        # print(a5)
         
        d=[]
        c=[a1,a2,a3,a4,a5]
        for a in c:
            b=abs(50-a)
            d=np.append(d, b)
        me=np.mean(d)
        print(files+" " + str(me))
        