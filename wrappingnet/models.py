##################################################
#
#   Copyright (c) 2010-2024, InterDigital
#   All rights reserved. 
#
#   See LICENSE under the root folder.
#
##################################################

import wrappingnet.utils as utils
import wrappingnet.losses as losses
from wrappingnet.layers import LoopPool, LoopUnPool, FaceConv, Face2Node

import torch
from torch_geometric.nn.models import MLP


def get_model(args):
    input_dim = 7
    if args.model_name == 'LC':
        return WrappingNet_sphere_LC(input_dim, args.latent_dim, 3)
    elif args.model_name == 'global_basesup3':
        return WrappingNet_global_basesup3(input_dim, args.latent_dim, 3)
    else:
        return Autoencoder(input_dim, args.latent_dim, 3)


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim=7, feature_dim=16, num_loop=3):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, feature_dim=feature_dim, num_layers=num_loop)
        self.decoder = Decoder(input_dim=input_dim, feature_dim=feature_dim, num_layers=num_loop)

    def forward(self, pos, faces):
        face_base, features = self.encoder(pos, faces)
        num_nodes = face_base.max() + 1 # len(torch.unique(face_base.flatten()))
        pos_base = pos[0:num_nodes]
        pos_list, face_list = self.decoder(pos_base, face_base, features)
        return pos_list, face_list


class WrappingNet_sphere_LC(torch.nn.Module):
    """
    WrappingNet-LC with SphereNet.
    """
    def __init__(self, input_dim=7, feature_dim=16, num_loop=3):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, feature_dim=feature_dim*2, num_layers=num_loop, hidden_dim=feature_dim // 4)
        self.make_sphere = MakeSphere(input_dim=input_dim)
        self.decoder = Decoder_basesup3(input_dim=input_dim, feature_dim=feature_dim, num_layers=num_loop, hidden_dim=feature_dim // 2)
        self.mlp = MLP(in_channels=feature_dim*2, hidden_channels=feature_dim*2, out_channels=feature_dim*2, num_layers=2, batch_norm=False)
        self.mlp2 = MLP(in_channels=feature_dim*2, hidden_channels=feature_dim, out_channels=feature_dim, num_layers=3, batch_norm=False)

    def forward(self, pos, faces, pos_base):
        face_base, features = self.encoder(pos, faces)
        with torch.no_grad():
            pos_base = self.make_sphere(pos_base, face_base)
            pos_sphere = 10*utils.gen_sphere_samples(10000).to(pos_base.device)
            idx = losses.chamfer_forward_idx(pos_sphere, pos_base).squeeze()
            pos_base = pos_sphere[idx]
        features = self.mlp(features)
        latent_code = torch.max(features, dim=0)[0].unsqueeze(0)
        latent_code = self.mlp2(latent_code).squeeze() 
        features = latent_code.repeat(features.shape[0], 1)
        pos_list, face_list = self.decoder(pos_base, face_base, features)
        return pos_list, face_list, pos_base


class WrappingNet_global_basesup3(torch.nn.Module):
    def __init__(self, input_dim=7, feature_dim=16, num_loop=3):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, feature_dim=feature_dim*2, num_layers=num_loop, hidden_dim=int(feature_dim/4))
        self.make_sphere = MakeSphere(input_dim=input_dim)
        self.decoder = Decoder_basesup2(input_dim=input_dim, feature_dim=feature_dim, num_layers=num_loop, hidden_dim=int(feature_dim/4))
        self.mlp = MLP(in_channels=feature_dim*2, hidden_channels=feature_dim*2, out_channels=feature_dim*2, num_layers=2, batch_norm=False)
        self.mlp2 = MLP(in_channels=feature_dim*2, hidden_channels=feature_dim, out_channels=feature_dim, num_layers=3, batch_norm=False)

    def forward(self, pos, faces, pos_base):
        face_base, features = self.encoder(pos, faces)
        with torch.no_grad():
            pos_base = self.make_sphere(pos_base, face_base)
            pos_sphere = 10*utils.gen_sphere_samples(pos_base.shape[0]).to(pos_base.device)
            idx = utils.matching(pos_base, pos_sphere)
            pos_base = pos_sphere[idx]
        features = self.mlp(features)
        latent_code = torch.max(features, dim=0)[0].unsqueeze(0)
        latent_code = self.mlp2(latent_code).squeeze() 
        features = latent_code.repeat(features.shape[0], 1)
        pos_list, face_list = self.decoder(pos_base, face_base, features)
        return pos_list, face_list, pos_base


class Encoder(torch.nn.Module):
    def __init__(self, input_dim=7, feature_dim=16, num_layers=4, hidden_dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        # hidden_dim = 64
        self.pool = LoopPool(pooling_type='mean')
        self.conv1 = FaceConv(input_dim, hidden_dim)
        self.conv2 = FaceConv(hidden_dim, hidden_dim)
        self.conv3 = FaceConv(hidden_dim, hidden_dim)
        self.conv4 = FaceConv(hidden_dim, self.feature_dim)

    def forward(self, pos, faces):
        face_features = torch.relu(self.conv1(faces, utils.extract_features(pos, faces)))
        faces, face_features = self.pool(faces, face_features)
        face_features = torch.relu(self.conv2(faces, face_features))
        faces, face_features = self.pool(faces, face_features)
        face_features = torch.relu(self.conv3(faces, face_features))
        faces, face_features = self.pool(faces, face_features)
        face_features = self.conv4(faces, face_features)
        return faces, face_features


class MakeSphere(torch.nn.Module):
    def __init__(self, input_dim=7, feature_dim=16, num_layers=4, hidden_dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        # hidden_dim = 64
        self.conv1 = FaceConv(input_dim, hidden_dim)
        self.conv1a = FaceConv(hidden_dim, hidden_dim)
        self.f2n1 = Face2Node(hidden_dim, hidden_dim)
        self.conv2 = FaceConv(hidden_dim, hidden_dim)
        self.conv2a = FaceConv(hidden_dim, hidden_dim)
        self.f2n2 = Face2Node(hidden_dim, hidden_dim)
        self.conv3 = FaceConv(hidden_dim, hidden_dim)
        self.conv3a = FaceConv(hidden_dim, hidden_dim)
        self.f2n3 = Face2Node(hidden_dim, 0)

    def forward(self, pos, faces):
        face_features = torch.relu(self.conv1(faces, utils.extract_features_local(pos, faces)))
        face_features = torch.relu(self.conv1a(faces, face_features))
        _, pos, face_features = self.f2n1(pos, faces, face_features)

        face_features = torch.relu(self.conv2(faces, face_features))
        face_features = torch.relu(self.conv2a(faces, face_features))
        _, pos, face_features = self.f2n2(pos, faces, face_features)

        face_features = torch.relu(self.conv3(faces, face_features))
        face_features = torch.relu(self.conv3a(faces, face_features))
        _, pos, _ = self.f2n3(pos, faces, face_features)
        return pos


class Decoder_basesup3(torch.nn.Module):
    def __init__(self, input_dim=7, feature_dim=16, num_layers=4, hidden_dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.interp_mode = 'nearest'
        self.feature_dim = feature_dim
        # hidden_dim = 64
        hidden_dim2 = hidden_dim // 2
        self.unpool = LoopUnPool()
        self.conv1 = FaceConv(hidden_dim2 + feature_dim, hidden_dim2)
        self.f2n1 = Face2Node(hidden_dim2, hidden_dim2)
        self.conv2 = FaceConv(hidden_dim2, hidden_dim2)
        self.f2n2 = Face2Node(hidden_dim2, hidden_dim2)
        self.conv3 = FaceConv(hidden_dim2, hidden_dim2)
        self.f2n3 = Face2Node(hidden_dim2, 0)

        self.conv1_sphere = FaceConv(feature_dim+input_dim, hidden_dim)
        self.f2n1_sphere = Face2Node(hidden_dim, hidden_dim)
        # self.mid1 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv2_sphere = FaceConv(hidden_dim, hidden_dim)
        self.f2n2_sphere = Face2Node(hidden_dim, hidden_dim)
        # self.mid2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv3_sphere = FaceConv(hidden_dim, hidden_dim)
        self.f2n3_sphere = Face2Node(hidden_dim, hidden_dim2)
        # self.conv4_sphere = FaceConv(hidden_dim+feature_dim, hidden_dim)
        # self.f2n4_sphere = Face2Node(hidden_dim, hidden_dim)
        # # self.mid4 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        # self.conv5_sphere = FaceConv(hidden_dim, hidden_dim)
        # self.f2n5_sphere = Face2Node(hidden_dim, hidden_dim)
        # # self.mid5 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        # self.conv6_sphere = FaceConv(hidden_dim, hidden_dim)
        # self.f2n6_sphere = Face2Node(hidden_dim, hidden_dim2)

    def forward(self, pos, faces, input_feature=None):
        # pos: [num_pos_base, 3]
        # faces: [num_face_base, 3]
        # input_features: optional, [num_face_base, feature_dim]
        pos_list = []
        face_list = []

        # if input_feature is None:
        #     input_feature = torch.ones((faces.shape[0], self.feature_dim)).to(pos.device)
        face_features = torch.cat((utils.extract_features(pos, faces), input_feature), dim=1)
        # face_features = input_feature

        # unmake sphere
        face_features = torch.relu(self.conv1_sphere(faces, face_features))
        _, pos, face_features = self.f2n1_sphere(pos, faces, face_features)
        # face_features = self.mid1(face_features)
        face_features = torch.relu(self.conv2_sphere(faces, face_features))
        _, pos, face_features = self.f2n2_sphere(pos, faces, face_features)
        # face_features = self.mid2(face_features)
        face_features = torch.relu(self.conv3_sphere(faces, face_features))
        _, pos, face_features = self.f2n3_sphere(pos, faces, face_features)
        # face_features = self.mid3(face_features)
        # face_features = torch.relu(self.conv4_sphere(faces, face_features))
        # _, pos, face_features = self.f2n4_sphere(pos, faces, face_features)
        # # face_features = self.mid4(face_features)
        # face_features = torch.relu(self.conv5_sphere(faces, face_features))
        # _, pos, face_features = self.f2n5_sphere(pos, faces, face_features)
        # # face_features = self.mid5(face_features)
        # face_features = torch.relu(self.conv6_sphere(faces, face_features))
        # _, pos, face_features = self.f2n6_sphere(pos, faces, face_features)
        pos_list.append(pos)
        face_list.append(faces)

        # face_features = torch.cat((utils.extract_features(pos, faces), face_features), dim=1)
        face_features = torch.cat((face_features, input_feature), dim=1)
        pos, faces, face_features = self.unpool(pos, faces, face_features, mode=self.interp_mode)
        face_features = torch.relu(self.conv1(faces, face_features))
        _, pos,  face_features = self.f2n1(pos, faces, face_features)
        # pos_list.append(pos)
        # face_list.append(faces)
        # from_nodes = self.n2f1(pos, faces)

        # print(face_features)
        # face_features = torch.cat((utils.extract_features(pos, faces), face_features), dim=1)
        # print(face_features)
        pos, faces, face_features = self.unpool(pos, faces, face_features, mode=self.interp_mode)
        face_features = torch.relu(self.conv2(faces, face_features))
        _, pos, face_features = self.f2n2(pos, faces, face_features)
        # pos_list.append(pos)
        # face_list.append(faces)
        # from_nodes = self.n2f2(pos, faces)

        # face_features = torch.cat((utils.extract_features(pos, faces), face_features), dim=1)
        pos, faces, face_features = self.unpool(pos, faces, face_features, mode=self.interp_mode)
        face_features = torch.relu(self.conv3(faces, face_features))
        _, pos, _ = self.f2n3(pos, faces, face_features)
        pos_list.append(pos)
        face_list.append(faces)

        return pos_list, face_list


class Decoder_basesup2(torch.nn.Module):
    def __init__(self, input_dim=7, feature_dim=16, num_layers=4, hidden_dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.interp_mode = 'nearest'
        self.feature_dim = feature_dim
        # hidden_dim = 64
        self.unpool = LoopUnPool()
        self.conv1 = FaceConv(hidden_dim + input_dim, hidden_dim)
        self.f2n1 = Face2Node(hidden_dim, hidden_dim)
        self.conv2 = FaceConv(hidden_dim, hidden_dim)
        self.f2n2 = Face2Node(hidden_dim, hidden_dim)
        self.conv3 = FaceConv(hidden_dim, hidden_dim)
        self.f2n3 = Face2Node(hidden_dim, 0)

        self.conv1_sphere = FaceConv(feature_dim + input_dim, hidden_dim)
        self.f2n1_sphere = Face2Node(hidden_dim, hidden_dim)
        self.conv2_sphere = FaceConv(hidden_dim, hidden_dim)
        self.f2n2_sphere = Face2Node(hidden_dim, hidden_dim)
        self.conv3_sphere = FaceConv(hidden_dim, hidden_dim)
        self.f2n3_sphere = Face2Node(hidden_dim, hidden_dim)
        # self.conv4_sphere = FaceConv(hidden_dim, hidden_dim)
        # self.f2n4_sphere = Face2Node(hidden_dim, hidden_dim)
        # self.conv5_sphere = FaceConv(hidden_dim, hidden_dim)
        # self.f2n5_sphere = Face2Node(hidden_dim, hidden_dim)
        # self.conv6_sphere = FaceConv(hidden_dim, hidden_dim)
        # self.f2n6_sphere = Face2Node(hidden_dim, hidden_dim)

    def forward(self, pos, faces, input_feature=None):
        # pos: [num_pos_base, 3]
        # faces: [num_face_base, 3]
        # input_features: optional, [num_face_base, feature_dim]
        pos_list = []
        face_list = []

        face_features = torch.cat((utils.extract_features(pos, faces), input_feature), dim=1)

        # unmake sphere
        face_features = torch.relu(self.conv1_sphere(faces, face_features))
        _, pos, face_features = self.f2n1_sphere(pos, faces, face_features)
        face_features = torch.relu(self.conv2_sphere(faces, face_features))
        _, pos, face_features = self.f2n2_sphere(pos, faces, face_features)
        face_features = torch.relu(self.conv3_sphere(faces, face_features))
        _, pos, face_features = self.f2n3_sphere(pos, faces, face_features)
        pos_list.append(pos)
        face_list.append(faces)

        face_features = torch.cat((utils.extract_features(pos, faces), face_features), dim=1)

        pos, faces, face_features = self.unpool(pos, faces, face_features, mode=self.interp_mode)
        face_features = torch.relu(self.conv1(faces, face_features))
        _, pos,  face_features = self.f2n1(pos, faces, face_features)
        pos_list.append(pos)
        face_list.append(faces)
        # from_nodes = self.n2f1(pos, faces)

        # face_features = torch.cat((utils.extract_features(pos, faces), face_features), dim=1)
        pos, faces, face_features = self.unpool(pos, faces, face_features, mode=self.interp_mode)
        face_features = torch.relu(self.conv2(faces, face_features))
        _, pos, face_features = self.f2n2(pos, faces, face_features)
        pos_list.append(pos)
        face_list.append(faces)
        # from_nodes = self.n2f2(pos, faces)

        # face_features = torch.cat((utils.extract_features(pos, faces), face_features), dim=1)
        pos, faces, face_features = self.unpool(pos, faces, face_features, mode=self.interp_mode)
        face_features = torch.relu(self.conv3(faces, face_features))
        _, pos, _ = self.f2n3(pos, faces, face_features)
        pos_list.append(pos)
        face_list.append(faces)

        return pos_list, face_list


class Decoder(torch.nn.Module):
    def __init__(self, input_dim=7, feature_dim=16, num_layers=4, hidden_dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.interp_mode = 'nearest'
        self.feature_dim = feature_dim
        # hidden_dim = 64
        self.unpool = LoopUnPool()
        self.conv1 = FaceConv(feature_dim + input_dim, hidden_dim)
        self.f2n1 = Face2Node(hidden_dim, hidden_dim)
        self.conv2 = FaceConv(hidden_dim, hidden_dim)
        self.f2n2 = Face2Node(hidden_dim, hidden_dim)
        self.conv3 = FaceConv(hidden_dim, hidden_dim)
        self.f2n3 = Face2Node(hidden_dim, 0)

    def forward(self, pos, faces, input_feature=None):
        # pos: [num_pos_base, 3]
        # faces: [num_face_base, 3]
        # input_features: optional, [num_face_base, feature_dim]
        pos_list = []
        face_list = []

        if input_feature is None:
            input_feature = torch.ones((faces.shape[0], self.feature_dim)).to(pos.device)
        face_features = torch.cat((utils.extract_features(pos, faces), input_feature), dim=1)
        pos, faces, face_features = self.unpool(pos, faces, face_features, mode=self.interp_mode)
        face_features = torch.relu(self.conv1(faces, face_features))
        _, pos,  face_features = self.f2n1(pos, faces, face_features)
        pos_list.append(pos)
        face_list.append(faces)
        # from_nodes = self.n2f1(pos, faces)

        # print(face_features)
        # face_features = torch.cat((utils.extract_features(pos, faces), face_features), dim=1)
        # print(face_features)
        pos, faces, face_features = self.unpool(pos, faces, face_features, mode=self.interp_mode)
        face_features = torch.relu(self.conv2(faces, face_features))
        _, pos, face_features = self.f2n2(pos, faces, face_features)
        pos_list.append(pos)
        face_list.append(faces)
        # from_nodes = self.n2f2(pos, faces)

        # face_features = torch.cat((utils.extract_features(pos, faces), face_features), dim=1)
        pos, faces, face_features = self.unpool(pos, faces, face_features, mode=self.interp_mode)
        face_features = torch.relu(self.conv3(faces, face_features))
        _, pos, _ = self.f2n3(pos, faces, face_features)
        pos_list.append(pos)
        face_list.append(faces)

        return pos_list, face_list

