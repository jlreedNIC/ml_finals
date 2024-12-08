# ------------------------
# @file     mv_algos.py
# @date     December 5, 2024
# @author   Jordan Reed
# @email    reed5204@vandals.uidaho.edu
# @brief    doing convolution and other mv algos on point clouds
#           put point clouds into grid and then do algos on grid
# ------------------------
import cv2 as cv
from data_manip import load_data, test_file, load_points_from_stl
import numpy as np
import open3d as o3d
from voxel_custom import Custom_Voxel
from occupancy_grid import Occupancy_Grid

# test_data, test_labels, test_mask = load_data(test_file)
# point_cloud = test_data[0]

point_cloud = load_points_from_stl("data/tape.stl")
# perform random sampling of object
indices = np.array(range(len(point_cloud)))
indices = np.random.choice(indices, 2048, False)
object_model = point_cloud[indices]
print(f"from {point_cloud.shape} to {object_model.shape}")
point_cloud = object_model


def show_point_cloud_from_grids(grids:list):
    # assuming each grid in grids list is an occupancy grid: 2d numpy array of custom voxels
    point_clouds = []
    for grid in grids:
        # get point clouds from grid
        point_clouds.append(grid.get_point_cloud())

    # define colors for each point cloud
    colors = np.array([[252, 15, 192], [0, 150, 255]]) / 255

    o3d_point_clouds = []
    i=0
    for pc in point_clouds:
        # create open3d point cloud from each list of points and apply a color
        new_pc = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(pc)
        new_pc.paint_uniform_color(colors[i])
        # print(colors[i])
        
        i += 1
        o3d_point_clouds.append(new_pc)
    
    o3d.visualization.draw_geometries(
        o3d_point_clouds,
    )

def apply_3d_convolution_filter(occupancy_grid, filter_3d):
    print(f'Applying convolution filter...')
    
    new_occgrid = Occupancy_Grid()
    new_occgrid.init_grid(occupancy_grid.shape, voxel_size)
    filter_size = int((len(filter_3d) - 1) / 2)
    for i in range(1, occupancy_grid.shape[0]-filter_size):
        for j in range(1, occupancy_grid.shape[1]-filter_size):
            for k in range(1, occupancy_grid.shape[2]-filter_size):
                # l,m,n for iterating through filter
                value = 0
                for l in range(len(filter_3d)):
                    for m in range(len(filter_3d[l])):
                        for n in range(len(filter_3d[m])):
                            ihat = i+(l-filter_size)
                            jhat = j+(m-filter_size)
                            khat = k+(n-filter_size)
                            cur_occ_val = occupancy_grid[ihat][jhat][khat].value
                            cur_sobel_val = filter_3d[l][m][n]
                            value += cur_sobel_val * cur_occ_val
                if value != 0:
                    new_occgrid[i][j][k].add_points(occupancy_grid[i][j][k].point_list)
                    new_occgrid[i][j][k].value = value

    return new_occgrid

# ------ main ----------
# voxel_size = .015
voxel_size = .05
print('creating occupancy grid')
occupancy_grid = Occupancy_Grid(point_cloud, voxel_size)
print(f'number of voxels: {occupancy_grid.shape}')

# define sobel filters for x, y, z directions
sobel_3dz_filter = [
    [[-1, -1, -1], 
     [-1, -2, -1], 
     [-1, -1, -1]],
    [[ 0,  0,  0], 
     [ 0,  0,  0], 
     [ 0,  0,  0]],
    [[ 1,  1,  1], 
     [ 1,  2,  1], 
     [ 1,  1,  1]],
]
sobel_3dy_filter = [
    [[-1, -1, -1], 
     [ 0,  0,  0], 
     [ 1,  1,  1]],
    [[-1, -2, -1], 
     [ 0,  0,  0], 
     [ 1,  2,  1]],
    [[-1, -1, -1], 
     [ 0,  0,  0], 
     [ 1,  1,  1]],
]
sobel_3dx_filter = [
    [[ 1,  0, -1], 
     [ 1,  0, -1], 
     [ 1,  0, -1]],
    [[ 1,  0, -1], 
     [ 2,  0, -2], 
     [ 1,  0, -1]],
    [[ 1,  0, -1], 
     [ 1,  0, -1], 
     [ 1,  0, -1]],
]

# 3d convolution

sobel_z = apply_3d_convolution_filter(occupancy_grid, sobel_3dz_filter)
sobel_y = apply_3d_convolution_filter(occupancy_grid, sobel_3dy_filter)
sobel_x = apply_3d_convolution_filter(occupancy_grid, sobel_3dx_filter)
# show_point_cloud_from_grids([occupancy_grid, sobel_z]) # this isn't showing 2 grids at once
# show_point_cloud_from_grids([occupancy_grid, sobel_y]) # this isn't showing 2 grids at once
# show_point_cloud_from_grids([occupancy_grid, sobel_x]) # this isn't showing 2 grids at once

sobel_x.abs()
sobel_y.abs()
sobel_z.abs()
combined_sobels = sobel_x + sobel_y + sobel_z
# combined_sobels.threshold_grid(0)
print(combined_sobels.shape)

show_point_cloud_from_grids([occupancy_grid, combined_sobels])

