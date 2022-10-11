#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:11:04 2022

@author: yimu
"""
import open3d as o3d
import copy
import numpy as np
import time

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            4000, 500))
        
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def prepare_dataset(voxel_size,source,target):
    print(":: Load two point clouds and disturb initial pose.")
    

 
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    #source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999))
    return result

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


voxel_size = 0.3#smallest errror from trial


for i in range(1, 2):
    voxel_size = 0.3
    disp=0
     
    dpath="/home/yimu/Downloads/ballo.ply"
    
    
    
    dpath1='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/mesh1.pcd'
    dpath2='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/mesh2.pcd'
    dpath3='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/mesh3.pcd'
    dpath4='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/mesh4.pcd'
    dpath5='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/mesh5.pcd'
    dpath6='/home/yimu/Downloads/PCD_scans/pcd_1010Ball/pcd-1010/mesh6.pcd'
     
    plist=[dpath1,dpath2,dpath3,dpath4,dpath5,dpath6]
          
    cent=np.asarray([])
    
    for pat in plist:
    
        source = o3d.io.read_point_cloud(pat)
        target = o3d.io.read_point_cloud(dpath)
        
        source=source.voxel_down_sample(voxel_size)
        target=target.voxel_down_sample(voxel_size)
        
        
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        
        
         # means 5cm for this dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
            voxel_size,source,target)
        
        #print("Statistical oulier removal")
        #source, ind = source.remove_statistical_outlier(nb_neighbors=20,
        #                                                    std_ratio=2.0)
        
        
        start = time.time()
        
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        
        
        print("global registration took %.3f sec.\n" % (time.time() - start))
        print(result_ransac)
        
        result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                         voxel_size)
        
        if disp==1:
            draw_registration_result(source_down, target_down, result_ransac.transformation)
        
        
        print(result_icp)
        mat=(result_icp.transformation)
        mat=np.asarray(mat)
        mat=np.linalg.inv(mat)
        x=mat[0][3]
        y=mat[1][3]
        z=mat[2][3]
        print (" ")
        print (pat)#print path
        
        print (x,y,z)
        c1=[x,y,z]
        print (" ")
        
        if cent.shape[0] !=0:
            cent=np.vstack([cent, c1])
        else :
            cent=np.append(cent, c1)
        print(cent)
        
        
        #draw_registration_result(source, target, result_icp.transformation)
        
        ttarg=copy.deepcopy(target).transform(mat)
        
        axis_aligned_bounding_box = ttarg.get_axis_aligned_bounding_box()
        
        idx = axis_aligned_bounding_box.get_point_indices_within_bounding_box(source.points)
        
        outlier_cloud = source.select_by_index(idx, invert=True)
         
        #o3d.visualization.draw_geometries([outlier_cloud])
        #
        #repeat again for second ball
        source=outlier_cloud
        
             # means 5cm for this dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
            voxel_size,source,target)
        
        #print("Statistical oulier removal")
        #source, ind = source.remove_statistical_outlier(nb_neighbors=20,
        #                                                    std_ratio=2.0)
        
        
        start = time.time()
        
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        
        
        print("global registration took %.3f sec.\n" % (time.time() - start))
        print(result_ransac)
        #
        
        result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                         voxel_size)
        
        if disp==1:
            draw_registration_result(source_down, target_down, result_ransac.transformation)
        
        mat=(result_icp.transformation)
        mat=np.asarray(mat)
        mat=np.linalg.inv(mat)
        x=mat[0][3]
        y=mat[1][3]
        z=mat[2][3]
        print (" ")
        print (pat)#print path
        
        print (x,y,z)
        c1=[x,y,z]
        print (" ")
        
        if cent.shape[0] !=0:
            cent=np.vstack([cent, c1])
        else :
            cent=np.append(cent, c1)
        print(cent)
        
        
        
       
        
        
    
    add=voxel_size*10
               
    np.savetxt("circICP"+str(add)+".txt", cent,fmt="%s")
    
     
         