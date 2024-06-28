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
from botorch.utils.sampling import sample_hypersphere

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../nndistance'))
from modules.nnd import NNDModule
nndistance = NNDModule()


def sample_points(n, pos, face):
    """
    Adapted from pytorch-geometric's uniform sampling
    """
    face = face.T
    pos_max = pos.abs().max()
    pos = pos / pos_max

    area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
    area = area.norm(p=2, dim=1).abs() / 2

    prob = (area / area.sum()).double()
    
    sample = torch.multinomial(prob, n, replacement=True)
    face = face[:, sample]

    frac = torch.rand(n, 2, device=pos.device)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    vec1 = pos[face[1]] - pos[face[0]]
    vec2 = pos[face[2]] - pos[face[0]]

    pos_sampled = pos[face[0]]
    pos_sampled += frac[:, :1] * vec1
    pos_sampled += frac[:, 1:] * vec2

    pos_sampled = pos_sampled * pos_max

    return pos_sampled

def nndistance_simple(rec, data):
    """
    A simple nearest neighbor search, not very efficient, just for reference
    """
    rec_sq = torch.sum(rec * rec, dim=2, keepdim=True) # (B,N,1)
    data_sq = torch.sum(data * data, dim=2, keepdim=True) # (B,M,1)
    cross = torch.matmul(data, rec.permute(0, 2, 1)) # (B,M,N)
    dist = data_sq - 2 * cross + rec_sq.permute(0, 2, 1) # (B,M,N)
    data_dist, data_idx = torch.min(dist, dim=2)
    rec_dist, rec_idx = torch.min(dist, dim=1)
    return data_dist, rec_dist, data_idx, rec_idx

def chamfer_forward(p1, p2):
    _, rec_dist, _, _ = nndistance_simple(p1.unsqueeze(0), p2.unsqueeze(0))
    rec_dist = torch.mean(rec_dist, 1)
    loss_pos = torch.mean(rec_dist)
    return loss_pos

def chamfer_forward_idx(p1, p2):
    _, _, _, rec_idx = nndistance(p1.unsqueeze(0), p2.unsqueeze(0))
    return rec_idx

def chamfer(p1, p2):
    data_dist, rec_dist, _, _ = nndistance_simple(p1.unsqueeze(0), p2.unsqueeze(0))
    data_dist, rec_dist = torch.mean(data_dist, 1), torch.mean(rec_dist, 1)
    loss_pos = torch.mean(data_dist + rec_dist)
    return loss_pos

def base_loss(pos_base, scale=10):
    pos_sphere = scale*sample_hypersphere(d=3, n=pos_base.shape[0]*4**3).to(pos_base.device)
    return chamfer_forward(pos_sphere, pos_base)

def multiscale_chamfer(pos_list, face_list, p_true, f_true, n=20000):
    d_multi = torch.stack([chamfer_face(pos, p_true, face, f_true) for pos, face in zip(pos_list, face_list)])
    return torch.mean(d_multi)

def l2(pos_list, face_list, p_true, f_true):
    return torch.mean(torch.linalg.norm(pos_list[-1] - p_true[0:pos_list[-1].shape[0], :], dim=1)**2)

def multiscale_l2(pos_list, face_list, p_true, f_true):
    d_multi = torch.stack([torch.mean(torch.linalg.norm(pos - p_true[0:pos.shape[0], :], dim=1)**2) for pos in pos_list])
    return torch.mean(d_multi)

def chamfer_face(pos, pos_hat, triangles, triangles_hat):
    bary1 = utils.get_barycenters(pos, triangles)
    bary2 = utils.get_barycenters(pos_hat, triangles_hat)
    p1 = torch.cat((pos, bary1), dim=0)
    p2 = torch.cat((pos_hat, bary2), dim=0)
    data_dist, rec_dist, _, _ = nndistance_simple(p1.unsqueeze(0), p2.unsqueeze(0))
    data_dist, rec_dist = torch.mean(data_dist, 1), torch.mean(rec_dist, 1)
    loss_pos = torch.mean(data_dist + rec_dist)
    return loss_pos

def get_distortion_loss(loss_func):
    if loss_func == 'MSL2':
        return multiscale_l2
    elif loss_func == 'MSChamfer':
        return multiscale_chamfer
    elif loss_func == 'L2':
        return l2
    else:
        sys.exit('Need to specify loss function')

