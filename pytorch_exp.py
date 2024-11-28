print('processing imports...')
from learning3d.models import PointNet, create_pointconv, MaskNet, DGCNN
from learning3d.models import Segmentation
import h5py as hp
import numpy as np
import torch.utils as tu
import torch
import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from pytorch_models import Custom_Model


# folder is located:
data_dir = "/mnt/d/school/dataset/h5_files/main_split"
file = 'objectdataset_augmented25rot.h5'
test_file = f'{data_dir}/test_{file}'
train_file = f'{data_dir}/training_{file}'

def load_data(filename):
    print(f"Starting load data for {filename}")
    with hp.File(filename) as f:
        data = np.array(f['data'][:])
        labels = np.array(f['label'][:])
        mask = np.array(f['mask'][:])

    # mask[mask==-1] = 15
    
    # only care about background and foreground
    mask[mask!=-1] = 0
    mask[mask==-1] = 1
    print(np.unique(mask))

    return data, labels, mask

def input_dataloader(data, labels, batch_size=16):
    # put data into dataloader
    inputs = torch.tensor(data)
    outputs = torch.LongTensor(labels)
    dataset = tu.data.TensorDataset(inputs, outputs)
    dataloader = tu.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# check for cuda
try:
    print(f' cuda available: {torch.cuda.is_available()}')
    print(f' device count: {torch.cuda.device_count()}')
    print(f' current device: {torch.cuda.current_device()}')
    print(f' cuda device: {torch.cuda.device(0)}')
    print(f' get device name: {torch.cuda.get_device_name(0)}')
except:
    print('issue when accessing cuda')


# load data
print('loading data...')
train_data, train_labels, train_mask = load_data(train_file)
test_data, test_labels, test_mask = load_data(test_file)

# create validation set from 10% of train set
train_data, valid_data, train_mask, valid_mask = train_test_split(train_data, train_mask, test_size=.1)

# put into dataloader for model
train_loader = input_dataloader(train_data, train_mask, 16)
test_loader = input_dataloader(test_data, test_mask, 16)
valid_loader = input_dataloader(valid_data, valid_mask, 16)


# create models
# following is to train multiple models
# PointConv = create_pointconv(classifier=False, pretrained=None)
# # ptconv = PointConv(emb_dims=1024, input_shape='bnc', input_channel_dim=3, classifier=True)
# models = [
#     Custom_Model(DGCNN(), "DGCNN"),
#     Custom_Model(MaskNet(), "MaskNet"),
#     Custom_Model(PointConv(), "pointconv"),
#     Custom_Model(PointNet(), "pointnet"), 
    
#     ]

# for model in models:
#     try:
#         model.train(200, train_loader, valid_loader)
#     except Exception as e:
#         print(f'\nencountered error: {e}\n')

# load model

model = PointNet(emb_dims=1024, input_shape='bnc', use_bn=True)
model = Custom_Model(model, "pointnet")
model.model.load_state_dict(torch.load("./models/pointnet_seg_20241127_194009_21"))

# predict on test set?
result = model.predict(test_data[5])

object_points = test_data[5][result == 0]
background_points = test_data[5][result == 1]

from data_manip import show_point_cloud_panda
show_point_cloud_panda([object_points, background_points])
