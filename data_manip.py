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
# import open3d as o3d
import random 
from stl import mesh
# import torch
# import torch.utils as tu
from panda3d_viewer import Viewer, ViewerConfig

# folder is located:
# data_dir = "/mnt/d/school/dataset/h5_files/main_split"
data_dir = "data"
file = 'objectdataset_augmented25rot.h5'
# file = 'objectdataset_augmentedrot_scale75.h5'
test_file = f'{data_dir}/test_{file}'
train_file = f'{data_dir}/training_{file}'

# print(f'test: {test_file}')
# print(f'train: {train_file}')

class_dict = {
    '0': 'bag',
    '1': 'bin',
    '2': 'box', 
    '3': 'cabinet', 
    '4': 'chair', 
    '5': 'desk', 
    '6': 'display', 
    '7': 'door', 
    '8': 'shelf', 
    '9': 'table', 
    '10': 'bed', 
    '11': 'pillow', 
    '12': 'sink', 
    '13': 'sofa', 
    '14': 'toilet'
}

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
    
    # mask[mask==-1] = 15
    # print(labels)
    
    # only care about background and foreground
    mask[mask!=-1] = 0
    mask[mask==-1] = 1
    print(np.unique(mask))
    return data, labels, mask

def load_points_from_stl(filename, round=False):
    print('Loading stl file...')
    model_mesh = mesh.Mesh.from_file(filename)
    point_cloud = np.copy(model_mesh.data['vectors'])

    # make 1x3 dimensional list instead of 3x3
    point_cloud = np.reshape(point_cloud, (point_cloud.shape[0]*point_cloud.shape[1], 3))

    if round:
        # rounds points to 2 decimal places
        # does remove some accuracy
        point_cloud *= 10
        point_cloud = np.trunc(point_cloud)
        point_cloud /= 10

    # remove duplicate values
    print("Removing duplicate points...")
    point_cloud = np.unique(point_cloud, axis=0)

    # min = np.min(point_cloud)
    # if min < 0:
    #     print(f'getting rid of negative numbers by {min}')
    #     point_cloud -= min

    return point_cloud

def input_dataloader(data, labels, batch_size=16):
    # put data into dataloader for pytorch implementation
    inputs = torch.tensor(data)
    outputs = torch.LongTensor(labels)
    dataset = tu.data.TensorDataset(inputs, outputs)
    dataloader = tu.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def one_hot_encode(test_labels, train_data):
    
    # labels = np.expand_dims(labels, -1)

    background = np.zeros(test_labels.shape)
    background[test_labels==1] = 1

    foreground = np.zeros(test_labels.shape)
    foreground[test_labels==0] = 1
    print(f'shapes: {test_labels.shape} {background.shape} {foreground.shape}')

    labels = np.squeeze(train_data).copy()
    print('label shape', labels.shape)
    labels = np.hstack((labels, background, foreground))
    # labels[3] = np.array([background, foreground])
    print('labels reshaped', labels.shape)
    # print(f'num background in testmask: {(test_labels==1).sum()}')
    # print(f'num foreground in testmask: {(test_labels==0).sum()}')

    # print(f'num background in background: {(background==1).sum()}')
    # print(f'num foreground in foreground: {(foreground==1).sum()}')
    # print(f'shapes: {test_labels.shape} {background.shape} {foreground.shape}')

    # labels = np.array([foreground, background])
    # # print(labels.shape)
    # labels = np.reshape(labels, (labels.shape[1], labels.shape[2], labels.shape[0]))
    # print(f'reshaped onehot labels: {labels.shape}')

    return labels

def show_point_clouds(clouds=[]):
    open3d_clouds = []
    for cloud in clouds:
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(cloud)
        open3d_clouds.append(new_cloud)

    colors = np.array([[252, 15, 192], [0, 150, 255]], dtype=np.float64) # bright pink, bright blue

    for i, cloud in enumerate(open3d_clouds):
        # assign colors. if more than 2 clouds are specified, start assigning random colors
        if i >= 2:
            color = [random.randrange(0,100)/100, random.randrange(0,100)/100, random.randrange(0,100)/100]
            color = np.array(color, dtype=np.float32)
        else:
            color = colors[i]/255

        cloud.paint_uniform_color(color)
    
    try:
        print('trying draw geometries')
        o3d.visualization.draw_geometries(
            open3d_clouds,
        )
    except Exception as e:
        print(f'issue with open3d: {e}')

    # try:
    #     print('try plotly')
    #     o3d.visualization.draw_plotly(
    #         open3d_clouds,
    #     )
    # except Exception as e:
    #     print(f'issue with open3d: {e}')

def show_point_cloud_panda(clouds:list):
    print('showing point cloud')

    colors = []
    for cloud in clouds:
        # convert array to float32 then uint32 to ensure viewer processes correctly
        print(type(cloud))
        cloud = np.array(cloud, np.float16)#.astype(np.uint32)
        # cloud = np.array(cloud, dtype=np.uint32)
        cloud = np.view(dtype=np.uint32)    

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




