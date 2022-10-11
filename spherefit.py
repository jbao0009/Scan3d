import numpy as np
import open3d as o3d


def sphere_fit(point_cloud):
    """
    input
        point_cloud: xyz of the point clouds　numpy array
    output
        radius : radius of the sphere
        sphere_center : xyz of the sphere center
    """

    A_1 = np.zeros((3,3))
    #A_1 : 1st item of A
    v_1 = np.array([0.0,0.0,0.0])
    v_2 = 0.0
    v_3 = np.array([0.0,0.0,0.0])
    # mean of multiplier of point vector of the point_clouds
    # v_1, v_3 : vector, v_2 : scalar

    N = len(point_cloud)
    #N : number of the points
    
    """Calculation of the sum(sigma)"""
    for v in point_cloud:
        v_1 += v
        v_2 += np.dot(v, v)
        v_3 += np.dot(v, v) * v
        
        A_1 += np.dot(np.array([v]).T, np.array([v]))
        
    v_1 /= N
    v_2 /= N
    v_3 /= N
    A = 2 * (A_1 / N - np.dot(np.array([v_1]).T, np.array([v_1])))
    # formula ②
    b = v_3 - v_2 * v_1
    # formula ③
    sphere_center = np.dot(np.linalg.inv(A), b)
    #　formula ①
    radius = (sum(np.linalg.norm(np.array(point_cloud) - sphere_center, axis=1))
              /len(point_cloud))
    
    return(radius, sphere_center)
    
    
dpath1="/home/yimu/Downloads/PCD_scans/6pnts_rotated_pcd_0826/mesh1.ply"
dpath2="/home/yimu/Downloads/PCD_scans/6pnts_rotated_pcd_0826/mesh2.ply"
dpath3="/home/yimu/Downloads/PCD_scans/6pnts_rotated_pcd_0826/mesh3.ply"
dpath4="/home/yimu/Downloads/PCD_scans/6pnts_rotated_pcd_0826/mesh4.ply"
dpath5="/home/yimu/Downloads/PCD_scans/6pnts_rotated_pcd_0826/mesh5.ply"
dpath6="/home/yimu/Downloads/PCD_scans/6pnts_rotated_pcd_0826/mesh6.ply"

pcd1 = o3d.io.read_point_cloud(dpath1)
pcd2 = o3d.io.read_point_cloud(dpath2)
pcd3 = o3d.io.read_point_cloud(dpath3)
pcd4 = o3d.io.read_point_cloud(dpath4)
pcd5 = o3d.io.read_point_cloud(dpath5)
pcd6 = o3d.io.read_point_cloud(dpath6)

pntcld1=np.asarray(pcd1.points)
pntcld2=np.asarray(pcd2.points)
pntcld3=np.asarray(pcd3.points)
pntcld4=np.asarray(pcd4.points)
pntcld5=np.asarray(pcd5.points)
pntcld6=np.asarray(pcd6.points)
  
           
r1,c1=sphere_fit(pntcld1)
r2,c2=sphere_fit(pntcld2)
r3,c3=sphere_fit(pntcld3)
r4,c4=sphere_fit(pntcld4)
r5,c5=sphere_fit(pntcld5)
r6,c6=sphere_fit(pntcld6)

newline1=""+str(r1)+","+str(c1)
newline2=""+str(r2)+","+str(c2)
newline3=""+str(r3)+","+str(c3)
newline4=""+str(r4)+","+str(c4)
newline5=""+str(r5)+","+str(c5)
newline6=""+str(r6)+","+str(c6)


with open("OUT.txt", "w") as o:
	o.write("{}\n".format(newline1.rstrip()))
	o.write("{}\n".format(newline2.rstrip()))
	o.write("{}\n".format(newline3.rstrip()))
	o.write("{}\n".format(newline4.rstrip()))
	o.write("{}\n".format(newline5.rstrip()))
	o.write("{}\n".format(newline6.rstrip()))

