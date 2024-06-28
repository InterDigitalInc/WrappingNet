##################################################
#
#   Copyright (c) 2010-2024, InterDigital
#   All rights reserved. 
#
#   See LICENSE under the root folder.
#
##################################################

import os
import os.path as osp
import glob
import tqdm
import open3d
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from pytorch_lightning import LightningDataModule


def get_data_lightning(args):
    if args.data_name == 'manifold40':
        return manifold40(data_dir=f'{args.data_root}/Manifold40')

def get_base_mesh(data, n_iter=3):
    # assumes faces 
    pos, faces,  y = data.pos, data.face.T,  data.y

    for _ in range(n_iter):
        nf = faces.shape[0] // 4 # output number of faces
        faces = torch.stack((
            faces[0:nf,0],
            faces[nf:2*nf,0],
            faces[2*nf:3*nf,0]
        ), dim=1)

        num_nodes = len(torch.unique(faces.flatten()))
        pos = pos[0:num_nodes]
    data = torch_geometric.data.Data(pos=pos, face=faces.T,  y=y)
    return data

def preprocess_mesh(data):
    pos = data.pos
    # centralize
    centroid = torch.mean(pos, dim=0)
    pos = pos - centroid
    # normalize in ball of radius 10
    m = torch.max(torch.sqrt(torch.sum(pos**2, dim=1)))
    pos = 10 * pos / m # scaling

    data.pos = pos
    return data

def normalize(data, ntype='mean-0-std-1',return_minmax=False):
    # data of shape (timesteps, vertices, number of features)
    if ntype=='range-0,1':
        # data between 0 and std 1 for each mesh at different timesteps
        vval_min = np.min(data)
        vval_max = np.max(data)
        return (data - vval_min) / (vval_max - vval_min) 
    if ntype=='range-0,1-mean-0':
        vval_mean = np.repeat(np.reshape(np.mean(data,axis=(1)), (-1,1,3)), data.shape[1], axis=1)
        data = (data - vval_mean)
        vval_min = np.min(data)
        vval_max = np.max(data)
        if return_minmax:
            return (data - vval_min) / (vval_max - vval_min), vval_min, vval_max
        else:
            return (data - vval_min) / (vval_max - vval_min) 
    elif ntype=='mean-0-std-1':
        # mean 0 and std 1 for each mesh at different timesteps and features
        vval_mean = np.repeat(np.reshape(np.mean(data,axis=(1)), (-1,1,3)), data.shape[1], axis=1)
        vval_std = np.repeat(np.reshape(np.std(data-vval_mean,axis=(1)), (-1,1,3)), data.shape[1], axis=1)
        return (data - vval_mean) / vval_std
    else:
        return data


class manifold40_dset(InMemoryDataset):
    def __init__(self, root, transform=None, train=True, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        print(self.raw_dir)
        categories = glob.glob(osp.join(self.raw_dir, '*'))
        categories = sorted([x.split(os.sep)[-1] for x in categories])

        data_list = []

        for category in tqdm.tqdm(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob(f'{folder}/*.obj')
            N = 5
            for path in paths:
                mesh = open3d.io.read_triangle_mesh(path)
                pos, faces = torch.tensor(np.asarray(mesh.vertices)).float(), torch.tensor(np.asarray(mesh.triangles)).long()
                data = torch_geometric.data.Data(pos=pos, face=faces.T)
                # TO DO: generate V_list and F_list
                #data.pos = torch.tensor(V_list[-1]).float()
                #data.face = torch.tensor(F_list[-1]).long().T
                #data = preprocess_mesh(data)
                #data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'


class manifold40(LightningDataModule):
    def __init__(self, data_dir, batch_size=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def train_dataloader(self):
        dset = manifold40_dset(root=self.data_dir, train=True)
        return torch_geometric.loader.DataLoader(dset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dset = manifold40_dset(root=self.data_dir, train=False)
        torch.manual_seed(0)
        dset = dset.shuffle()
        return torch_geometric.loader.DataLoader(dset, batch_size=self.batch_size)

