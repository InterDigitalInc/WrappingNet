##################################################
#
#   Copyright (c) 2010-2024, InterDigital
#   All rights reserved. 
#
#   See LICENSE under the root folder.
#
##################################################

import torch
import numpy as np


def get_base_mesh(pos, faces, n_iter=3):
    # assumes faces 
    for _ in range(n_iter):
        nf = faces.shape[0] // 4 # output number of faces
        faces = torch.stack((
            faces[0:nf,0],
            faces[nf:2*nf,0],
            faces[2*nf:3*nf,0]
        ), dim=1)

        num_nodes = len(torch.unique(faces.flatten()))
        pos = pos[0:num_nodes]
    return pos, faces

def compute_face_adjacency(faces):
    F = faces.shape[0]
    FAF = torch.zeros_like(faces)
    current_edges = torch.cat((faces[:,[1,2]], faces[:, [2,0]], faces[:, [0,1]]), dim=0)
    unique_edges = current_edges.min(dim=1)[0] * current_edges.max()  + current_edges.max(dim=1)[0]
    S = torch.argsort(unique_edges)
    S = S.reshape(-1, 2) # S contains half-flap index pairs, so the 3 neighbors are simply the three half-flaps
    
    FAF[S[:, 0] % F, torch.div(S[:,0], F, rounding_mode='floor')] = S[:, 1] % F
    FAF[S[:, 1] % F, torch.div(S[:,1], F, rounding_mode='floor')] = S[:, 0] % F
    return FAF

def extract_features(pos, faces):
    # order invariant features
    vecs1 = pos[faces[:,1]] - pos[faces[:,0]]
    vecs2 = pos[faces[:,2]] - pos[faces[:,1]]
    vecs3 = pos[faces[:,0]] - pos[faces[:,2]]
    a = torch.linalg.norm(vecs1, dim=1)
    b = torch.linalg.norm(vecs2, dim=1)
    c = torch.linalg.norm(vecs3, dim=1)
    s = 0.5*(a+b+c)
    area_sq = s*(s-a)*(s-b)*(s-c)
    face_normals = torch.cross(vecs1, vecs2, dim=1)
    center_pos = (pos[faces[:,0]] + pos[faces[:,1]] + pos[faces[:,2]]) / 3
    FAF = compute_face_adjacency(faces)
    pos0, pos1, pos2 = pos[faces[:,0]], pos[faces[:,1]], pos[faces[:,2]]
    center_neigh0 = (pos0[FAF[:,0]] + pos1[FAF[:,0]] + pos2[FAF[:,0]]) / 3
    center_neigh1 = (pos0[FAF[:,1]] + pos1[FAF[:,1]] + pos2[FAF[:,1]]) / 3
    center_neigh2 = (pos0[FAF[:,2]] + pos1[FAF[:,2]] + pos2[FAF[:,2]]) / 3
    center_neigh = (center_neigh0 + center_neigh1 + center_neigh2) / 3
    curve_vec = center_pos - center_neigh
    # bary = get_barycenters(pos, faces)
    feats = torch.cat((area_sq.unsqueeze(1),
                        face_normals,
                        curve_vec,
                        # bary
    ), dim=1)
    return feats

def extract_features_local(pos, faces):
    # order invariant features
    vecs1 = pos[faces[:,1]] - pos[faces[:,0]]
    vecs2 = pos[faces[:,2]] - pos[faces[:,1]]
    vecs3 = pos[faces[:,0]] - pos[faces[:,2]]
    a = torch.linalg.norm(vecs1, dim=1)
    b = torch.linalg.norm(vecs2, dim=1)
    c = torch.linalg.norm(vecs3, dim=1)
    s = 0.5*(a+b+c)
    area_sq = s*(s-a)*(s-b)*(s-c)
    face_normals = torch.cross(vecs1, vecs2, dim=1)
    center_pos = (pos[faces[:,0]] + pos[faces[:,1]] + pos[faces[:,2]]) / 3
    FAF = compute_face_adjacency(faces)
    pos0, pos1, pos2 = pos[faces[:,0]], pos[faces[:,1]], pos[faces[:,2]]
    center_neigh0 = (pos0[FAF[:,0]] + pos1[FAF[:,0]] + pos2[FAF[:,0]]) / 3
    center_neigh1 = (pos0[FAF[:,1]] + pos1[FAF[:,1]] + pos2[FAF[:,1]]) / 3
    center_neigh2 = (pos0[FAF[:,2]] + pos1[FAF[:,2]] + pos2[FAF[:,2]]) / 3
    center_neigh = (center_neigh0 + center_neigh1 + center_neigh2) / 3
    curve_vec = center_pos - center_neigh
    # bary = get_barycenters(pos, faces)
    feats = torch.cat((area_sq.unsqueeze(1),
                        face_normals,
                        curve_vec,
                        # bary
    ), dim=1)
    return feats

def normalize_pos(pos, out_norm=10):
    # centralize
    centroid = torch.mean(pos, dim=0)
    pos = pos - centroid
    # normalize in ball of radius 10
    m = torch.max(torch.sqrt(torch.sum(pos**2, dim=1)))
    pos = out_norm * pos / m # scaling
    return pos

def gen_sphere_samples(n):
    indices = torch.arange(n) + 0.5

    phi = torch.arccos(1 - 2*indices/n)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
    return torch.stack((x,y,z), dim=1)

def get_barycenters(pos, face_list):
    """
    Given list of faces, return barycenters of vertex positions.
    pos: [num_nodes, 3]
    face_list: [num_faces, 3]
    """
    tri_pos = torch.stack((pos[face_list[:,0]], pos[face_list[:,1]], pos[face_list[:,2]]), dim=2) # [num_faces, 3, 3]
    bary = torch.mean(tri_pos, dim=2) # [num_faces, 3]
    return bary

