# ------------------------
# @file     mv_algos.py
# @date     December 5, 2024
# @author   Jordan Reed
# @email    reed5204@vandals.uidaho.edu
# @brief    doing convolution and other mv algos on point clouds
#           put point clouds into grid and then do algos on grid
# ------------------------
import cv2 as cv
from data_manip import load_data, test_file, show_point_clouds
import numpy as np
import open3d as o3d
from voxel_custom import Custom_Voxel

test_data, test_labels, test_mask = load_data(test_file)

point_cloud = test_data[0]

def get_bounds_along_axis(point_cloud, coord):
    maximum = np.max(point_cloud[:,coord], axis=0)
    minimum = np.min(point_cloud[:,coord], axis=0)

    return minimum, maximum

def create_occupancy_grid(point_cloud, voxel_size):
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
    # is that the right term??
    num_x_voxels = int(max_x / voxel_size) + 1
    num_y_voxels = int(max_y / voxel_size) + 1
    num_z_voxels = int(max_z / voxel_size) + 1

    # create blank occupancy grid
    occupancy_grid = np.zeros((num_x_voxels, num_y_voxels, num_z_voxels))

    # loop through voxel grid and assign 1 if there is a point in there
    for point in point_cloud:
        # convert point to voxel index
        voxel_index = np.array(point/voxel_size, dtype=int)
        occupancy_grid[voxel_index[0], voxel_index[1], voxel_index[2]] = 1

    print(f'{(occupancy_grid==1).sum()} occupied voxels in grid')
    print(f'{(occupancy_grid==0).sum()} unoccupied voxels in grid')

    return occupancy_grid

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
    # occupancy_grid = np.zeros((num_x_voxels, num_y_voxels, num_z_voxels))

    print(vox_grid[8][2][1])
    # loop through voxel grid and assign 1 if there is a point in there
    for point in point_cloud:
        # convert point to voxel index
        voxel_index = np.array(point/voxel_size, dtype=int)
        # print(voxel_index)
        # occupancy_grid[voxel_index[0], voxel_index[1], voxel_index[2]] = 1
        vox_grid[voxel_index[0]][voxel_index[1]][voxel_index[2]].add_points([point])


    # print(f'{(occupancy_grid==1).sum()} occupied voxels in grid')
    # print(f'{(occupancy_grid==0).sum()} unoccupied voxels in grid')

    occupancy_grid = np.array(vox_grid)
    # print(occ_grid)
    print(f'{(occupancy_grid==1).sum()} occupied voxels in grid')
    print(f'{(occupancy_grid==0).sum()} unoccupied voxels in grid')

    return occupancy_grid

def get_point_cloud(occupancy_grid):
    # occupancy grid is 2d numpy array of voxels
    point_cloud = []
    for i in range(len(occupancy_grid)):
        for j in range(len(occupancy_grid[i])):
            for k in range(len(occupancy_grid[i][j])):
                point_cloud.append(occupancy_grid[i][j][k].point_list)
    
    return point_cloud

def calculate_point_cloud(occupancy_grid):
    # occupancy grid is 2d numpy array of values: 1 is there is a point there
    point_clouds = []
    for grid in occupancy_grid:
        points = np.zeros(((grid!=0).sum(), 3))
        point_idx = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                for k in range(len(grid[i][j])):
                    if grid[i][j][k] != 0:
                        x_coord = i*voxel_size
                        y_coord = j*voxel_size
                        z_coord = k*voxel_size
                        
                        p = [x_coord, y_coord, z_coord]
                        points[point_idx] = p
                        point_idx += 1
        # print(points)
        point_clouds.append(points)

def show_point_cloud_from_grid(occupancy_grid, voxel_size):
    point_clouds = calculate_point_cloud(occupancy_grid)
    # point_clouds = get_point_cloud(occupancy_grid)
    # print(point_clouds)
    # convert numpy list of points to open3d vector and open3d voxelgrid and show
    # paint one color
    colors = np.array([[252, 15, 192], [0, 150, 255]]) / 255
    # print(colors)

    o3d_voxel_grid = []
    i=0
    for pc in point_clouds:
        points = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(test_data[0])
        points.points = o3d.utility.Vector3dVector(pc)
        # pc.scale( 1 / np.max(pc.get_max_bound() - pc.get_min_bound()), center = pc.get_center())
        points.paint_uniform_color(colors[i])
        print(colors[i])

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(points, voxel_size)
        
        i += 1
        o3d_voxel_grid.append(voxel_grid)
    
    o3d.visualization.draw_geometries(
        o3d_voxel_grid,
    )

def apply_3d_convolution_filter(occupancy_grid, filter_3d):
    new_array = np.zeros(occupancy_grid.shape)
    filter_size = int((len(filter_3d) - 1) / 2)
    print(filter_size, len(occupancy_grid))
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
                new_array[i][j][k] = value
                # exit(0)

    print(new_array)
    threshold = new_array < np.max(new_array)/2
    # new_array[threshold] = 0
    # print('newarray', new_array)
    # print(f'edges detected: {(new_array!=0).sum()}')

    return new_array

# ------ main ----------
# voxel_size = .015
voxel_size = .05
occupancy_grid = create_occupancy_grid_of_voxels(point_cloud, voxel_size)
# occupancy_grid = create_occupancy_grid(point_cloud, voxel_size)
# print(occupancy_grid)
print(occupancy_grid.shape)

sobel_3dz_filter = [
    [[0, -1,  0], 
     [0, -2,  0], 
     [0, -1,  0]],
    [[1,  0, -1], 
     [2,  0, -2], 
     [1,  0, -1]],
    [[0,  1,  0], 
     [0,  2,  0], 
     [0,  1,  0]],
]

# 3d convolution

sobel_z = apply_3d_convolution_filter(occupancy_grid, sobel_3dz_filter)
# show_point_cloud_from_grid([occupancy_grid], voxel_size)
# show_point_cloud_from_grid([sobel_z], voxel_size)
# show_point_cloud_from_grid([occupancy_grid, sobel_z], voxel_size) # this isn't showing 2 grids at once



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
