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
    """
    class to create an occupancy grid of voxels that keep track of points in voxels.
    designed for converting point clouds to voxel grid and back
    """
    def __init__(self, point_cloud=None, voxel_size=.01):
        # print(f'Initializing occupancy grid...')
        self.grid = None                # numpy list
        self.shape = None               # tuple
        self.voxel_size = voxel_size    # float

        self._minx = None
        self._maxx = None
        self._miny = None
        self._maxy = None
        self._minz = None
        self._maxz = None

        if point_cloud is not None:
            self.create_occupancy_grid(point_cloud, voxel_size)
    
    def init_grid(self, shape, voxel_size):
        """
        Initialize a blank occupancy grid of the given shape and size

        :param shape: tuple of (x,y,z) shape of grid
        :param voxel_size: size of each voxel
        """
        self.voxel_size = voxel_size
        self.shape = shape
        # print(f'Occupancy grid is of size: {self.shape} with voxel size: {voxel_size}')

        # initialize a blank occupancy grid of size shape
        self.grid = [[[Custom_Voxel(self.voxel_size) for i in range(self.shape[2])] for j in range(self.shape[1])] for k in range(self.shape[0])]
        self.grid = np.array(self.grid)
    
    def create_occupancy_grid(self, point_cloud, voxel_size):
        """
        Create an occupancy grid based on the point cloud and voxel size

        :param point_cloud: numpy list of points (x,y,z)
        :param voxel_size: size of each voxel for point cloud
        """
        # creating occupancy grid from point cloud
        # print(f'Creating occupancy grid for point cloud...')
        self._get_bounds_point_cloud(point_cloud)
        point_cloud[:,0] -= self._minx
        point_cloud[:,1] -= self._miny
        point_cloud[:,2] -= self._minz
        self._get_bounds_point_cloud(point_cloud)

        num_x_voxels = int(self._maxx / voxel_size) + 1
        num_y_voxels = int(self._maxy / voxel_size) + 1
        num_z_voxels = int(self._maxz / voxel_size) + 1
        
        # create blank grid of size shape
        self.init_grid((num_x_voxels, num_y_voxels, num_z_voxels), voxel_size)

        # loop through each point in point cloud and assign it to a voxel in the grid
        for point in point_cloud:
            # convert point to voxel index
            vidx = np.array(point/voxel_size, dtype=int)
            self.grid[vidx[0]][vidx[1]][vidx[2]].add_points([point])
        
        self.grid = np.array(self.grid)

        print(f'{(self.grid==1).sum()} occupied voxels in grid')
        print(f'{(self.grid==0).sum()} unoccupied voxels in grid')

    def get_point_cloud(self):
        """
        Return a numpy list of points for point cloud represented in grid. Points contained in voxels of grid

        :return: numpy list of points (x,y,z)
        """
        point_cloud = []
        point_list = self.grid[self.grid != 0]
        point_list = np.array(point_list)
        for vox in point_list:
            point_cloud += vox.point_list
        point_cloud=np.array(point_cloud)
        return point_cloud
    
    def threshold_grid(self, threshold):
        """
        Any value less than or equal to the threshold passed turns to 0

        :param threshold: number
        """
        self.grid[self.grid<=threshold] = Custom_Voxel(self.voxel_size)

    def __str__(self):
        """
        return string of grid

        :return: string repr. of grid values
        """
        output = ""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                output += '[ '
                for k in range(self.shape[2]):
                    output += f'{self.grid[i][j][k]} '
                output += "]\n"
            output += " \n"
        
        return output

    def _get_bounds_point_cloud(self, point_cloud):
        """
        Get the minimum and maximum of a point cloud in all directions in order to create a grid

        :param point_cloud: list of points (x,y,z)
        """
        self._minx = np.min(point_cloud[:,0], axis=0)
        self._maxx = np.max(point_cloud[:,0], axis=0)

        self._miny = np.min(point_cloud[:,1], axis=0)
        self._maxy = np.max(point_cloud[:,1], axis=0)

        self._minz = np.min(point_cloud[:,2], axis=0)
        self._maxz = np.max(point_cloud[:,2], axis=0)
    
    def __getitem__(self, keyval):
        """
        Allows array access of class for grid values

        :param keyval: index
        :return: grid value at index
        """
        return self.grid[keyval]
    
    def __add__(self, object):
        """
        Add 2 grids together by adding each voxel in grid element wise

        :param object: occupancy grid
        :return: occupancy grid
        """
        if type(object) != type(self):
            print(f'Cannot add object type {type(object)} to type {type(self)}')
            exit(1)
        
        if self.voxel_size != object.voxel_size:
            print(f'Voxel size must be the same: {self.voxel_size} != {object.voxel_size}')

        new_grid = Occupancy_Grid()
        new_grid.init_grid(self.shape, self.voxel_size)
        new_grid.grid = np.add(self.grid, object.grid)
        return new_grid
    
    def maximum(self):
        """
        return the maximum value of the grid

        :return: max value of grid
        """
        return np.max(self.grid)
    
    def __lt__(self, object):
        return self.grid < object.grid
    
    def __gt__(self, object):
        return self.grid > object.grid
    
    def abs(self):
        """
        make sure every value in grid is above 0 by performing absolute value of each value
        """
        # loop through and do abs function
        # np.abs
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self.grid[i][j][k].abs()
        return
    

# ----------- testing -----------------
def main():
    point_cloud = np.array([[1,2,3],[5,6,9],[4,5,6],[7,8,9]])

    occgrid = Occupancy_Grid(point_cloud, 1)
    # occgrid.create_occupancy_grid(point_cloud)

    print(occgrid[0][0][0])
    occgrid[0][0][0].value = -5
    occgrid[0][1][2].value = 2
    print(occgrid)

    # occgrid.threshold_grid(4)
    print(occgrid.shape)
    print(occgrid)
    pc = occgrid.get_point_cloud()
    print(pc)

    maximum = occgrid.maximum()
    print(maximum)
    print(type(maximum))
    print(int(maximum))

    occgrid2 = Occupancy_Grid(np.array([[1,2,3],[5,6,9],[1,2,4],[7,8,9]]), 1)

    occgrid3 = occgrid + occgrid2
    print(occgrid3)

    occgrid3.abs()
    print('abs', occgrid3)




if __name__ == "__main__":
    main()