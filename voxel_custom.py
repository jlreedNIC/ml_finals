# ------------------------
# @file     voxel_custom.py
# @date     December 7, 2024
# @author   Jordan Reed
# @email    reed5204@vandals.uidaho.edu
# @brief    a custom voxel class that keeps track of the points within the voxel
# ------------------------


class Custom_Voxel():
    def __init__(self, vox_size=.1):
        # print("initializing voxel...")

        self.value = 0
        self.point_list = []    # should this be numpy array?

        self.isEmpty = True     # if value == 1 and pointlist is not empty; probably not necessary
        self.voxel_size = vox_size
    
    def add_points(self, new_points:list):
        # print(f"Adding points to voxel...")
        if len(new_points) != 0:
            for pt in new_points:
                self.point_list.append(pt)

            if self.value == 0:
                self.value = 1
                self.isEmpty = False
    
    def remove_points(self, points:list):
        # print(f"Removing points from voxel...")
        if len(points) != 0:
            for pt in points:
                self.point_list.remove(pt)
            
            if len(self.point_list) != 0:
                self.value = 0
                self.isEmpty = True
    
    def show_points(self):
        print(f'\nPoint List in Voxel of size {self.voxel_size}')
        print(f'-----------------------------------')
        for point in self.point_list:
            print(f'  {point}')
        print()
    
    def __str__(self):
        return f'{self.value}'
    
    def __eq__(self, value):
        return self.value == value
    
    def __lt__(self, value):
        return self.value < value
    
    def __gt__(self, value):
        return self.value > value
    
    def __ge__(self, value):
        return self.value >= value
    
    def __le__(self, value):
        return self.value <= value
    
    
    def __get__(self, instance, owner):
        return self.value
    
    def __add__(self, object):
        if type(object) != type(self):
            print(f'Cannot add object type {type(object)} to type {type(self)}')
            exit(1)

        if self.voxel_size != object.voxel_size:
            print(f'Voxel size must be the same: {self.voxel_size} != {object.voxel_size}')
        new_voxel = Custom_Voxel(vox_size=self.voxel_size)
        new_voxel.value = self.value + object.value
        if new_voxel.value != 0:
            new_voxel.add_points(self.point_list)
            new_voxel.add_points(object.point_list)
        

        return new_voxel
    
    def abs(self):
        self.value = np.abs(self.value)
        return
    
    # def __mul__(self, value):
    #     self.value *= value
    #     return self

import numpy as np
def main():
    # for testing class code
    vox = Custom_Voxel()

    vox.add_points([[1,2,3],[2,3,4],[5,6,8]])
    vox.show_points()
    print(f'vox: {vox}')
    vox.remove_points([[1,2,3],[2,3,4],[5,6,8]])
    print(f'vox: {vox}')

    print(f'vox == 2: {vox==2}')
    print(f'vox <= 5: {vox<=5}')
    print(f'vox >= 1: {vox >= 1}')

    testing = np.array([[vox]])
    print(testing)

    print(testing[0])
    print((testing==1).sum())

    vox2 = Custom_Voxel()
    vox2.add_points([[1,2,3],[2,3,5],[5,7,8]])

    vox3 = vox + vox2
    print('vox3', vox3)
    vox3.show_points()



if __name__ == "__main__":
    main()