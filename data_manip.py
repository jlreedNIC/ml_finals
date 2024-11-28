# ------------------------
# @file     data_manip.py
# @date     November 20, 2024
# @author   Jordan Reed
# @email    reed5204@vandals.uidaho.edu
# @brief    data manipulation for final project
# ------------------------

# data is in h5 files

import os
import h5py as hp
import numpy as np
import open3d as o3d
import random 
import torch
import torch.utils as tu
from panda3d_viewer import Viewer, ViewerConfig

# folder is located:
data_dir = "/mnt/d/school/dataset/h5_files/main_split"
file = 'objectdataset_augmented25rot.h5'
test_file = f'{data_dir}/test_{file}'
train_file = f'{data_dir}/training_{file}'

# print(f'test: {test_file}')
# print(f'train: {train_file}')

def investigate_files(dir):
    print("\nStarting file investigation...")
    file_list = os.listdir(dir)
    print(file_list)

    single_file = os.listdir(f"{dir}/{file_list[0]}")
    print(single_file[0])

    full_file_path = f'{dir}/{file_list[0]}/{single_file[0]}'
    print(full_file_path)
    print("File investigation concluded.\n")

def investigate_data(filename):
    print("\nStarting data investigation...")
    f = hp.File(filename, 'r')
    keys = f.keys()
    print(f'Keys in data: {keys}')

    print("SHAPES of keys")
    for key in keys:
        print(f"   {key}: {f[key].shape}")
    
    print(f"{f['data'].shape[0]} objects in dataset with {f['data'].shape[1]} points each")
    print(f"There are {np.unique(f['label']).shape[0]} labels.")
    print(f"  Labels: {np.unique(f['mask'])}")

    f.close()

    print("Data investigation concluded.\n")

def load_data(filename):
    print(f"Starting load data for {filename}")
    with hp.File(filename) as f:
        data = np.array(f['data'][:])
        labels = np.array(f['label'][:])
        mask = np.array(f['mask'][:])
    
    mask[mask==-1] = 15
    print(np.unique(mask))
    return data, labels, mask

def input_dataloader(data, labels, batch_size=16):
    # put data into dataloader for pytorch implementation
    inputs = torch.tensor(data)
    outputs = torch.LongTensor(labels)
    dataset = tu.data.TensorDataset(inputs, outputs)
    dataloader = tu.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def show_point_clouds(clouds=[]):
    open3d_clouds = []
    for cloud in clouds:
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(cloud)
        open3d_clouds.append(new_cloud)

    for cloud in open3d_clouds:
        color = [random.randrange(0,100)/100, random.randrange(0,100)/100, random.randrange(0,100)/100] 
        # print(f'Color: {color}')
        cloud.paint_uniform_color(color)
    
    try:
        print('trying draw geometries')
        o3d.visualization.draw_geometries(
            open3d_clouds,
            zoom = 0.4459,
            front=[0.9288, -0.2951, -0.2242],
            lookat=[1.6784, 2.0612, 1.4451],
            up=[-0.3402, -0.9189, -0.1996]
        )
    except Exception as e:
        print(f'issue with open3d: {e}')

    try:
        print('try plotly')
        o3d.visualization.draw_plotly(
            open3d_clouds,
            zoom = 0.4459,
            front=[0.9288, -0.2951, -0.2242],
            lookat=[1.6784, 2.0612, 1.4451],
            up=[-0.3402, -0.9189, -0.1996]
        )
    except Exception as e:
        print(f'issue with open3d: {e}')

def show_point_cloud_panda(clouds:list):
    print('showing point cloud')

    colors = []
    for cloud in clouds:
        # convert array to float32 then uint32 to ensure viewer processes correctly
        cloud = np.array(cloud, np.float32)
        # cloud = np.view(dtype=np.uint32)    

        color = [random.randrange(0,100)/100, random.randrange(0,100)/100, random.randrange(0,100)/100] 
        colors.append(color)

    # create window
    with Viewer(show_grid=False) as viewer:
        viewer.reset_camera((10, 10, 15), look_at=(0, 0, 0))
        viewer.append_group('root')
        viewer.append_cloud('root', 'cloud', thickness=4)

        while True:
            for i in range(0, len(clouds)):
                viewer.set_cloud_data('root', 'cloud', cloud[i], colors[i])
            # time.sleep(0.03)
def main():
    investigate_files(data_dir)
    investigate_data(train_file)

    # data is actual points
    # label is the class label, what is in the 'scene' (sofa, chair, table, etc.)
    # mask is -1 for a point if it's in the background

    # test_data, test_labels, test_mask = load_data(test_file)
    # train_data, train_labels, train_mask = load_data(train_file)

if __name__ == "__main__":
    main()




