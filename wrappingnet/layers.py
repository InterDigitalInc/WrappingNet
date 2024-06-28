##################################################
#
#   Copyright (c) 2010-2024, InterDigital
#   All rights reserved. 
#
#   See LICENSE under the root folder.
#
##################################################

import wrappingnet.utils as utils

import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.nn.models import MLP


class FaceConv(torch.nn.Module):
    """
    Face convolution layer, like SubdivNet.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_size = 4
        self.conv2d = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=True)

    def forward(self, faces, face_features):
        # face_features: [num_faces, num_channels]
        CKP = utils.compute_face_adjacency(faces) # conv kernel pattern: [num_faces, num_neighbors]
        num_neighbors = CKP.shape[1]
        conv_feats = face_features[CKP].permute(2, 0, 1) # [num_channels, num_faces, num_neighbors]
        conv_feats = conv_feats.unsqueeze(0)
        y0 = face_features.T.unsqueeze(0)
        # we gather the 4 neighbor-index-invariant features
        features = [y0, 
                    conv_feats.sum(dim=-1),
                    torch.abs(conv_feats[...,[num_neighbors-1] + list(range(num_neighbors-1))] - conv_feats).sum(dim=-1),
                    torch.abs(y0.unsqueeze(-1) - conv_feats).sum(dim=-1)]
        features = torch.stack(features, dim=-1)
        output_features = self.conv2d(features)[0,:,:,0] # [num_channels, num_faces]
        return output_features.T # [num_faces, num_channels]


class Face2Node(torch.nn.Module):
    """
    Converts face features into node position updates, while also updating face features.
    For node j, this will pool face features from all faces containing j to form update for node j's position.
    """
    def __init__(self, in_channels, out_channels=29):
        super().__init__()
        # self.faceconv = FaceConv(in_channels, 9)
        self.mlp = MLP(in_channels=in_channels+6, hidden_channels=in_channels, out_channels=out_channels+3, num_layers=2, batch_norm=False)

    def forward(self, pos, faces, face_features):
        # Concatenate local face features (in order relative to the node to be aggregated) containing edge vectors to the input face features (which are order-invariant).
        f0 = torch.cat((pos[faces[:,1]]-pos[faces[:,0]], pos[faces[:,2]]-pos[faces[:,0]], face_features), dim=1)
        f1 = torch.cat((pos[faces[:,2]]-pos[faces[:,1]], pos[faces[:,0]]-pos[faces[:,1]], face_features), dim=1)
        f2 = torch.cat((pos[faces[:,0]]-pos[faces[:,2]], pos[faces[:,1]]-pos[faces[:,2]], face_features), dim=1)
        # Forward pass concatenated features through MLP, for each of 3 nodes to be aggregated. 
        g0 = self.mlp(f0) # faces[:,0] is the center node
        g1 = self.mlp(f1)
        g2 = self.mlp(f2)
        face_features = (g0[:,3:] + g1[:,3:] + g2[:,3:]) / 3 # update face features are just averaged among the three 
        
        # Perform aggregation, such that the face features aggregated are relative to the node to be aggregated. 
        delta_pos = torch_scatter.scatter_mean(torch.cat((g0[:,0:3], g1[:,0:3], g2[:,0:3])), faces.T.flatten().unsqueeze(1), dim=0) 
        pos = pos + delta_pos # update position
        return delta_pos, pos, face_features


class LoopPool(torch.nn.Module):
    """
    Loop Pooling layer.
    Does the inverse of Loop for one iteration, followed by mean/max pooling of face features.
    """
    def __init__(self, pooling_type='mean'):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, faces, face_features):
        # face_features: [num_faces, channels]
        # assumes faces 
        nf = faces.shape[0] // 4 # output number of faces
        faces_new = torch.stack((
            faces[0:nf,0],
            faces[nf:2*nf,0],
            faces[2*nf:3*nf,0]
        ), dim=1)

        face_features = torch.stack((
            face_features[0:nf],
            face_features[nf:2*nf],
            face_features[2*nf:3*nf],
            face_features[3*nf:]
        ), dim=-1) # [num_faces / 4, channels, 4]

        if self.pooling_type == 'mean':
            face_features = torch.mean(face_features, dim=-1)
        elif self.pooling_type == 'max':
            face_features = torch.max(face_features, dim=-1)[0]
        else:
            raise Exception('Invalid pooling type')
        return faces_new, face_features


class LoopUnPool(torch.nn.Module):
    """
    Loop Unpooling layer.
    Applies one iteration of Loop subdivision, followed by interpolation of face features. 
    """
    def __init__(self):
        super().__init__()

    def loop_subdivision(self, pos, faces):
        """
        Loop subdivision indexing consistent with Neural Subdivision.
        """
        num_nodes = pos.shape[0]
        num_faces = faces.shape[0]
        hE = torch.cat((faces[:,[0,1]], faces[:, [1,2]], faces[:, [2,0]]), dim=0)
        hE = torch.sort(hE, dim=1)[0]
        E, hE2E = torch.unique(hE, dim=0, return_inverse=True)
        newV = (pos[E[:,0],:]+pos[E[:,1],:]) / 2.0
        pos = torch.cat((pos, newV), dim=0)

        E2 = num_nodes + torch.arange(num_faces).to(pos.device)
        E0 = num_nodes + num_faces + torch.arange(num_faces).to(pos.device)
        E1 = num_nodes + 2*num_faces + torch.arange(num_faces).to(pos.device)

        faces_new = torch.cat((
            torch.stack((faces[:,0], E2, E1), dim=1),
            torch.stack((faces[:,1], E0, E2), dim=1),
            torch.stack((faces[:,2], E1, E0), dim=1),
            torch.stack((E0, E1, E2), dim=1),
        ), dim=0)
        hE2E = torch.cat((torch.arange(num_nodes).to(pos.device), hE2E+num_nodes), dim=0)
        faces = hE2E[faces_new]

        return pos, faces

    def loop_subdivision2(self, pos, faces):
        """
        Loop subdivision indexing consistent with SubdivNet's implementation.
        """
        num_nodes = faces.max()+1
        num_faces = faces.shape[0]
        current_edges = torch.cat((faces[:,[0,1]], faces[:, [1,2]], faces[:, [2,0]]), dim=0)
        unique_edges = current_edges.min(dim=1)[0] * current_edges.max()  + current_edges.max(dim=1)[0]

        E2F = torch.argsort(unique_edges)
        F2E = torch.zeros_like(E2F)
        F2E[E2F] = torch.div(torch.arange(unique_edges.shape[0], device=faces.device), 2, rounding_mode='floor')

        E2 = num_nodes + F2E[:num_faces]
        E0 = num_nodes + F2E[num_faces:num_faces*2]
        E1 = num_nodes + F2E[num_faces*2:]

        faces_new = torch.cat((
            torch.stack((faces[:,0], E2, E1), dim=1),
            torch.stack((faces[:,1], E0, E2), dim=1),
            torch.stack((faces[:,2], E1, E0), dim=1),
            torch.stack((E0, E1, E2), dim=1),
        ), dim=0)

        nodes_to_add = len(torch.unique(F2E))

        # using Midpoint subdivision updates
        pos = torch.cat((pos, torch.zeros(nodes_to_add, pos.shape[1], device=pos.device)))
        pos[E1] = 0.5*(pos[faces[:,0]] + pos[faces[:,2]]) 
        pos[E0] = 0.5*(pos[faces[:,1]] + pos[faces[:,2]]) 
        pos[E2] = 0.5*(pos[faces[:,0]] + pos[faces[:,1]]) 

        return pos, faces_new

    def forward(self, pos, faces, face_features, mode='nearest'):
        pos, faces = self.loop_subdivision(pos, faces)
        if mode == 'nearest':
            face_features = torch.cat((
                [face_features]*4
            ), dim=0)
        elif mode == 'bilinear':
            FAF = utils.compute_face_adjacency(faces)
            neighbor_feats = face_features[FAF].permute(0,2,1) # [num_faces, channels, neighbors]
            face_features = torch.cat((
                (face_features * 2 + neighbor_feats[...,1] + neighbor_feats[...,2]) / 4,
                (face_features * 2 + neighbor_feats[...,2] + neighbor_feats[...,0]) / 4,
                (face_features * 2 + neighbor_feats[...,0] + neighbor_feats[...,1]) / 4,
                face_features
            ), dim=0) # concatenate along face dimension
        else:
            raise Exception("Invalid interpolation mode")
        return pos, faces, face_features.float()

