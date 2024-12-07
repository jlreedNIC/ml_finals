# ------------------------
# @file     occupancy_grid.py
# @date     December 7, 2024
# @author   Jordan Reed
# @email    reed5204@vandals.uidaho.edu
# @brief    a custom class for occupancy grid of voxels
# ------------------------

from voxel_custom import Custom_Voxel
import numpy as np

class Occupancy_Grid():
    def __init__(self, point_cloud=None, voxel_size=.01):
        print(f'Initializing occupancy grid...')
        self.grid = None
        self.shape = None
        self.voxel_size = voxel_size

        self._minx = None
        self._maxx = None
        self._miny = None
        self._maxy = None
        self._minz = None
        self._maxz = None

        if point_cloud is not None:
            self.create_occupancy_grid(point_cloud, voxel_size)
    
    def init_grid(self, shape, voxel_size):
        self.voxel_size = voxel_size
        self.shape = shape
        print(f'Occupancy grid is of size: {self.shape}')

        # initialize a blank occupancy grid of size shape
        self.grid = [[[Custom_Voxel(self.voxel_size) for i in range(self.shape[2])] for j in range(self.shape[1])] for k in range(self.shape[0])]
    
    def create_occupancy_grid(self, point_cloud, voxel_size):
        # creating occupancy grid from point cloud
        print(f'Creating occupancy grid for point cloud...')
        # self.voxel_size = voxel_size
        self._get_bounds_point_cloud(point_cloud)
        point_cloud[:,0] -= self._minx
        point_cloud[:,1] -= self._miny
        point_cloud[:,2] -= self._minz
        self._get_bounds_point_cloud(point_cloud)

        num_x_voxels = int(self._maxx / voxel_size) + 1
        num_y_voxels = int(self._maxy / voxel_size) + 1
        num_z_voxels = int(self._maxz / voxel_size) + 1
        # self.shape = (num_x_voxels, num_y_voxels, num_z_voxels)

        
        # create blank grid of size shape
        self.init_grid((num_x_voxels, num_y_voxels, num_z_voxels), voxel_size)
        # self.grid = [[[Custom_Voxel(self.voxel_size) for i in range(self.shape[2])] for j in range(self.shape[1])] for k in range(self.shape[0])]

        # loop through each point in point cloud and assign it to a voxel in the grid
        for point in point_cloud:
            # convert point to voxel index
            vidx = np.array(point/voxel_size, dtype=int)
            self.grid[vidx[0]][vidx[1]][vidx[2]].add_points([point])
        
        self.grid = np.array(self.grid)

        print(f'{(self.grid==1).sum()} occupied voxels in grid')
        print(f'{(self.grid==0).sum()} unoccupied voxels in grid')


    def get_point_cloud(self):
        point_cloud = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    point_cloud += self.grid[i][j][k]
        
        return point_cloud

    def _get_bounds_point_cloud(self, point_cloud):
        self._minx = np.min(point_cloud[:,0], axis=0)
        self._maxx = np.max(point_cloud[:,0], axis=0)

        self._miny = np.min(point_cloud[:,1], axis=0)
        self._maxy = np.max(point_cloud[:,1], axis=0)

        self._minz = np.min(point_cloud[:,2], axis=0)
        self._maxz = np.max(point_cloud[:,2], axis=0)
    
    def __getitem__(self, keyval):
        return self.grid[keyval]


def main():
    point_cloud = np.array([[1,2,3],[5,6,9],[4,5,6],[7,8,9]])

    occgrid = Occupancy_Grid(point_cloud, 1)
    # occgrid.create_occupancy_grid(point_cloud)

    print(occgrid[0][0][0])



if __name__ == "__main__":
    main()