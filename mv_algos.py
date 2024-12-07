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

# test_data, test_labels, test_mask = load_data(test_file)
# point_cloud = test_data[0]

point_cloud = load_points_from_stl("data/tape.stl")


def get_bounds_along_axis(point_cloud, coord):
    maximum = np.max(point_cloud[:,coord], axis=0)
    minimum = np.min(point_cloud[:,coord], axis=0)

    return minimum, maximum

def create_occupancy_grid_of_voxels(point_cloud, voxel_size):
    # get bounds of x, y, z
    min_x, max_x = get_bounds_along_axis(point_cloud, 0)
    min_y, max_y = get_bounds_along_axis(point_cloud, 1)
    min_z, max_z = get_bounds_along_axis(point_cloud, 2)

    # remove negative numbers from point cloud
    point_cloud[:,0] -= min_x
    point_cloud[:,1] -= min_y
    point_cloud[:,2] -= min_z

    # get new bounds of x,y,z
    min_x, max_x = get_bounds_along_axis(point_cloud, 0)
    min_y, max_y = get_bounds_along_axis(point_cloud, 1)
    min_z, max_z = get_bounds_along_axis(point_cloud, 2)

    # turn to occupancy voxel grid
    num_x_voxels = int(max_x / voxel_size) + 1
    num_y_voxels = int(max_y / voxel_size) + 1
    num_z_voxels = int(max_z / voxel_size) + 1

    # create blank occupancy grid
    vox_grid = [[[Custom_Voxel(voxel_size) for i in range(num_z_voxels)] for j in range(num_y_voxels)] for k in range(num_x_voxels)]

    # loop through voxel grid and assign 1 if there is a point in there
    for point in point_cloud:
        # convert point to voxel index
        voxel_index = np.array(point/voxel_size, dtype=int)
        vox_grid[voxel_index[0]][voxel_index[1]][voxel_index[2]].add_points([point])



    occupancy_grid = np.array(vox_grid)
    print(f'{(occupancy_grid==1).sum()} occupied voxels in grid')
    print(f'{(occupancy_grid==0).sum()} unoccupied voxels in grid')
    # print(occupancy_grid[0][0][0].point_list)

    return occupancy_grid

def get_point_cloud(occupancy_grid):
    # occupancy grid is 2d numpy array of voxels
    print(f'in get point cloud: {occupancy_grid[0][0][0]}')
    point_cloud = []
    for i in range(len(occupancy_grid)):
        for j in range(len(occupancy_grid[i])):
            for k in range(len(occupancy_grid[i][j])):
                for point in occupancy_grid[i][j][k].point_list:
                    point_cloud.append(point)
    
    return point_cloud

def show_point_cloud_from_grids(grids:list, voxel_size):
    # assuming each grid in grids list is an occupancy grid: 2d numpy array of custom voxels
    point_clouds = []
    for grid in grids:
        # get point clouds from grid
        point_clouds.append(get_point_cloud(grid))

    # define colors for each point cloud
    colors = np.array([[252, 15, 192], [0, 150, 255]]) / 255

    o3d_point_clouds = []
    i=0
    for pc in point_clouds:
        # create open3d point cloud from each list of points and apply a color
        new_pc = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(pc)
        # pc.scale( 1 / np.max(pc.get_max_bound() - pc.get_min_bound()), center = pc.get_center())
        new_pc.paint_uniform_color(colors[i])
        # print(colors[i])

        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(points, voxel_size)
        
        i += 1
        o3d_point_clouds.append(new_pc)
    
    o3d.visualization.draw_geometries(
        o3d_point_clouds,
    )

def apply_3d_convolution_filter(occupancy_grid, filter_3d):
    print(f'Applying convolution filter...')
    # new_array = np.zeros(occupancy_grid.shape)
    num_z_voxels = occupancy_grid.shape[2]
    num_y_voxels = occupancy_grid.shape[1]
    num_x_voxels = occupancy_grid.shape[0]
    new_array = [[[Custom_Voxel(voxel_size) for i in range(num_z_voxels)] for j in range(num_y_voxels)] for k in range(num_x_voxels)]
    filter_size = int((len(filter_3d) - 1) / 2)
    for i in range(1, len(occupancy_grid)-filter_size):
        for j in range(1, len(occupancy_grid[i])-filter_size):
            for k in range(1, len(occupancy_grid[i][j])-filter_size):
                # l,m,n for iterating through filter
                value = 0
                # print('values', occupancy_grid[i-1], occupancy_grid[i], occupancy_grid[i+1])
                for l in range(len(filter_3d)):
                    for m in range(len(filter_3d[l])):
                        for n in range(len(filter_3d[m])):
                            ihat = i+(l-filter_size)
                            jhat = j+(m-filter_size)
                            khat = k+(n-filter_size)
                            # print(f'k {k} n {n} khat {khat}')
                            # print(f'len {len(occupancy_grid[j])}')
                            cur_occ_val = occupancy_grid[ihat][jhat][khat].value
                            cur_sobel_val = filter_3d[l][m][n]
                            # print(f' sobel {sobel_3dx_filter[l][m][n]} * occ {occupancy_grid[ihat][jhat][khat]} = {sobel_3dx_filter[l][m][n] * occupancy_grid[ihat][jhat][khat]}')
                            # print(f'      value {value} + res {sobel_3dx_filter[l][m][n] * occupancy_grid[ihat][jhat][khat]} = {value + sobel_3dx_filter[l][m][n] * occupancy_grid[ihat][jhat][khat]}')
                            value += cur_sobel_val * cur_occ_val
                    # print(f'value after x{l}: {value}')
                # print(value)
                if value != 0:
                    new_array[i][j][k].add_points(occupancy_grid[i][j][k].point_list)
                    new_array[i][j][k].value = value
                # exit(0)

    # print(new_array)
    # don't think we need this right now
    # threshold = new_array < np.max(new_array)/2
    # new_array[threshold] = 0
    # print('newarray', new_array)
    # print(f'edges detected: {(new_array!=0).sum()}')

    return new_array

# ------ main ----------
# voxel_size = .015
voxel_size = .01
print('creating occupancy grid')
occupancy_grid = create_occupancy_grid_of_voxels(point_cloud, voxel_size)
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
# show_point_cloud_from_grid([occupancy_grid], voxel_size)
# show_point_cloud_from_grid([sobel_z], voxel_size)
# show_point_cloud_from_grids([occupancy_grid, sobel_z], voxel_size) # this isn't showing 2 grids at once
# show_point_cloud_from_grids([occupancy_grid, sobel_y], voxel_size) # this isn't showing 2 grids at once
# show_point_cloud_from_grids([occupancy_grid, sobel_x], voxel_size) # this isn't showing 2 grids at once

# num_z_voxels = occupancy_grid.shape[2]
# num_y_voxels = occupancy_grid.shape[1]
# num_x_voxels = occupancy_grid.shape[0]
# combined_sobels = [[[Custom_Voxel(voxel_size) for i in range(num_z_voxels)] for j in range(num_y_voxels)] for k in range(num_x_voxels)]
# # combine sobel grids
# for i in range(len(sobel_x)):
#     for j in range(len(sobel_x[i])):
#         for k in range(len(sobel_x[i][j])):
#             combined_sobels[i][j][k].add_points(sobel_x[i][j][k].point_list)
#             combined_sobels[i][j][k].add_points(sobel_y[i][j][k].point_list)
#             combined_sobels[i][j][k].add_points(sobel_z[i][j][k].point_list)

#             # combined_sobels[i][j][k].value = 0
#             combined_sobels[i][j][k].value = sobel_x[i][j][k].value
#             combined_sobels[i][j][k].value += sobel_y[i][j][k].value
#             combined_sobels[i][j][k].value += sobel_z[i][j][k].value

combined_sobels = sobel_x + sobel_y + sobel_z
combined_sobels = np.add(sobel_x, sobel_y)
combined_sobels = np.add(combined_sobels, sobel_z)
# threshold = np.max(combined_sobels)*.75
# combined_sobels = combined_sobels[combined_sobels>=threshold]
# combined_sobels = np.array(combined_sobels)
print(combined_sobels.shape)
# print(combined_sobels[0][0][0])

show_point_cloud_from_grids([occupancy_grid, combined_sobels], voxel_size)



# # ---- open3d voxels exploration -----
# color = [252, 15, 192]
# color = np.array(color) / 255
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(test_data[0])
# point_cloud.scale( 1 / np.max(point_cloud.get_max_bound() - point_cloud.get_min_bound()), center = point_cloud.get_center())
# point_cloud.paint_uniform_color(color)

# # # showing point cloud
# # o3d.visualization.draw_geometries(
# #     [point_cloud],
# # )

# # showing voxel grid
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, .01)
# # voxel_grid = voxel_grid.create_from_point_cloud(point_cloud, .05)
# print(voxel_grid)
# print(voxel_grid.VoxelGrid)
# print(voxel_grid.is_empty())
# vox_list = voxel_grid.get_voxels()
# print(type(vox_list))
# print(vox_list[0])
# print(type(vox_list[0]))
# vox_list = np.array(vox_list)
# print(len(vox_list.shape))
# print(voxel_grid.get_max_bound())

# # o3d.visualization.draw_geometries(
# #     [voxel_grid],
# # )

# # --------------------------
