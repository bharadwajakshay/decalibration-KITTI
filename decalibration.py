#!/usr/bin/env python
import os
import sys
import glob
import cv2
import numpy as np
from helperfunctions import *
from scipy.spatial.transform import Rotation as rot
import tqdm


'''
Project the Lidar points on to the image to make sure that the calibration
between the Point cloud and the image are good. 
Steps to achieve it
1. Read the input image
2. Read the input pontcloud
3. Read the calibration file
4. Convert the 3D points on to 2D points similar to the Homogeneous camera calibration 
'''

def readimgfromfile(path_to_file):
    image = cv2.imread(path_to_file)
    return([image.shape[1],image.shape[0],image])

def readvelodynepointcloud(path_to_file):
    '''
    The velodyne data that is presented is in the form of a np array written as binary data.
    So to read the file, we use the inbuilt fuction form the np array to rad from the file
    '''

    pointcloud = np.fromfile(path_to_file, dtype=np.float32).reshape(-1, 4)
    
    # Return points ignoring the reflectivity 
    return(pointcloud[:,:3])

def readveltocamcalibrationdata(path_to_file):
    ''' 
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info 
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    '''
    with open(path_to_file, "r") as f:
        file = f.readlines()    
        
        for line in file:
            (key, val) = line.split(':',1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)

    return R, T

def readcamtocamcalibrationdata(path_to_file, mode='02'):
    """
    If your image is 'rectified image' :
        get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)
        
    in this code, I'll get P matrix since I'm using rectified image
    """
    with open(path_to_file, "r") as f:
        file = f.readlines()
        
        for line in file:
            (key, val) = line.split(':',1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
    return P_


def projpointcloud2imgplane(points, v_fov, h_fov, velcamR, velcamT, camProj, mode='02'):

    """
    xyz_v - 3D velodyne points corresponding to h, v FOV in the velodyne coordinates
    c_    - color value(HSV's Hue) corresponding to distance(m)
    
             [x_1 , x_2 , .. ]
    xyz_v =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
             [ 1  ,  1  , .. ]
    """  
    xyz_v, c_ = velo_points_filter(points, v_fov, h_fov)
    
    """
    RT_ - rotation matrix & translation matrix
        ( velodyne coordinates -> camera coordinates )
    
            [r_11 , r_12 , r_13 , t_x ]
    RT_  =  [r_21 , r_22 , r_23 , t_y ]   
            [r_31 , r_32 , r_33 , t_z ]
    """
    RT_ = np.concatenate((velcamR, velcamT),axis = 1)
    
    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c) 
    for i in range(xyz_v.shape[1]):
        xyz_v[:3,i] = np.matmul(RT_, xyz_v[:,i])
        
    """
    xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
             [x_1 , x_2 , .. ]
    xyz_c =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
    """ 
    xyz_c = np.delete(xyz_v, 3, axis=0)

    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y) 
    for i in range(xyz_c.shape[1]):
        xyz_c[:,i] = np.matmul(camProj, xyz_c[:,i])    

    """
    xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
    ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
             [s_1*x_1 , s_2*x_2 , .. ]
    xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]  
             [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
    """
    xy_i = xyz_c[::]/xyz_c[::][2]
    ans = np.delete(xy_i, 2, axis=0)
    
    """
    width = 1242
    height = 375
    w_range = in_range_points(ans[0], width)
    h_range = in_range_points(ans[1], height)

    ans_x = ans[0][np.logical_and(w_range,h_range)][:,None].T
    ans_y = ans[1][np.logical_and(w_range,h_range)][:,None].T
    c_ = c_[np.logical_and(w_range,h_range)]

    ans = np.vstack((ans_x, ans_y))
    """
    
    return ans, c_


def main():
    '''
    The main function where the images are read, point clouds are read and calibration data
    '''
    if len(sys.argv) != 5:
        print("Incorrect usage")
        exit(-1)
    camvelocalibfilename = sys.argv[1]
    cameracalibfilename = sys.argv[2]
    imagefiledir = sys.argv[3]
    pointcloudfiledir = sys.argv[4]

    # Move into image fdirectory

    os.chdir(imagefiledir)
    
    with tqdm.tqdm(total=len(glob.glob('*.png'))) as img_bar:

        for img_file in glob.glob('*.png'):
            filename = img_file.split('.')[0]
            agumenteddatapath = pointcloudfiledir+'agumenteddata/'+filename
            pointcloudfilename = pointcloudfiledir+filename+'.bin'
            if(not os.path.exists(agumenteddatapath)):
                os.makedirs(agumenteddatapath)

            [width,height,img] = readimgfromfile(img_file)
            pointcloud = readvelodynepointcloud(pointcloudfilename)

            # read cam projection matrix
            P = readcamtocamcalibrationdata(cameracalibfilename)
            # read vel - cam Roation and Translation matrix
            [R,T] = readveltocamcalibrationdata(camvelocalibfilename)

        
            # project the velodyne points on to the image
            # H-FOV = 360deg
            # V-FOV = 26.9deg

            samplesize = 30
            [R_euler,t_decalib] = generaterandomRT(samplesize)

            [points,color] = projpointcloud2imgplane(pointcloud,(-24.9,2),(-45,45), R, T, P)
            ptsnimg = displayprojectedptsonimg(points,color,img)
            cv2.imwrite(agumenteddatapath+'/original.png',ptsnimg)

            with tqdm.tqdm(total=samplesize) as rotation_bar:
                for idx in range(samplesize):
                    R_obj = rot.from_euler('zyx',R_euler[:,idx].T,degrees=True)
                    R_decalib = R_obj.as_matrix()

                    decalib_pts = np.matmul(pointcloud,R_decalib)
                    decalib_pts = decalib_pts + t_decalib[:,idx]

                    decalib_pts.astype(decalib_pts.dtype).tofile(agumenteddatapath+'/'+str(idx)+'.bin')

                    [points_decal,color] = projpointcloud2imgplane(decalib_pts,(-24.9,2),(-45,45), R, T, P)
                    ptsnimg_decal = displayprojectedptsonimg(points_decal,color,img)
                    cv2.imwrite(agumenteddatapath+'/'+str(idx)+'.png',ptsnimg_decal)

                    # Write the R|t to the file 
                    fs = open(agumenteddatapath+"/"+str(idx)+".txt","w")
                    data = "R: "+ np.array2string(R_decalib, precision=6, separator=' ')+'\n'
                    fs.write(data)
                    data = "T: "+ np.array2string(t_decalib[:,idx], precision=6, separator=' ')
                    fs.write(data)
                    fs.close
                    rotation_bar.update(1)
            img_bar.update(1)

    return(0)

if __name__ == '__main__':
    main()