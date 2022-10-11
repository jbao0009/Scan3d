
path1='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/01.pcd'
path2='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/02.pcd'
path3='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/03.pcd'
path4='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/04.pcd'
path5='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/05.pcd'
path6='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/06.pcd'
      
import open3d as o3d
import numpy as np
import copy

#0929scan position
a1=[-236.059,46.278,50.837]

a2=[-236.283,60.519,41.155]

print (np.subtract(a1,a2))

p1off=[83.123, 11.791 ,0]
p2off=[81.834, 26.109 ,0]
p3off=[89.897, 26.109,0]
p4off=[80.547, 0.211 ,0]
p5off=[83.532, -4.357,0]
p6off=[85.777, 7.933,0]

temp=p1off[0]
p1off[0]=-p1off[0]+2*p1off[0]
p2off[0]=-p2off[0]+2*p1off[0]
p3off[0]=-p3off[0]+2*p1off[0]
p4off[0]=-p4off[0]+2*p1off[0]
p5off[0]=-p5off[0]+2*p1off[0]
p6off[0]=-p6off[0]+2*p1off[0]
p1off[0]=temp


# temp=p1off[2]

# p1off[2]=-p1off[2]+2*p1off[2]
# p2off[2]=-p2off[2]+2*p1off[2]
# p3off[2]=-p3off[2]+2*p1off[2]
# p4off[2]=-p4off[2]+2*p1off[2]
# p5off[2]=-p5off[2]+2*p1off[2]
# p6off[2]=-p6off[2]+2*p1off[2]
# p1off[2]=temp


#swap cause its backwards

p1off[0], p1off[1] = p1off[1], p1off[0]
p2off[0], p2off[1] = p2off[1], p2off[0]
p3off[0], p3off[1] = p3off[1], p3off[0]
p4off[0], p4off[1] = p4off[1], p4off[0]
p5off[0], p5off[1] = p5off[1], p5off[0]
p6off[0], p6off[1] = p6off[1], p6off[0]

print(p1off)
print(p2off)
print(p3off)
print(p4off)
print(p5off)
print(p6off)

 



pcd_r1 = o3d.io.read_point_cloud(path1)
pcd_r2 = o3d.io.read_point_cloud(path2)
pcd_r3 = o3d.io.read_point_cloud(path3)
pcd_r4 = o3d.io.read_point_cloud(path4)
pcd_r5 = o3d.io.read_point_cloud(path5)
pcd_r6 = o3d.io.read_point_cloud(path6)


mesh1 = copy.deepcopy(pcd_r1).translate((p1off))
mesh2 = copy.deepcopy(pcd_r2).translate((p2off))
mesh3 = copy.deepcopy(pcd_r3).translate((p3off))
mesh4 = copy.deepcopy(pcd_r4).translate((p4off))
mesh5 = copy.deepcopy(pcd_r5).translate((p5off))
mesh6 = copy.deepcopy(pcd_r6).translate((p6off))

o3d.io.write_point_cloud("mesh1.pcd", mesh1)
o3d.io.write_point_cloud("mesh2.pcd", mesh2)
o3d.io.write_point_cloud("mesh3.pcd", mesh3)
o3d.io.write_point_cloud("mesh4.pcd", mesh4)
o3d.io.write_point_cloud("mesh5.pcd", mesh5)
o3d.io.write_point_cloud("mesh6.pcd", mesh6)


o3d.visualization.draw_geometries([
    mesh1, mesh2,mesh3,mesh4,mesh5,mesh6])


     

    