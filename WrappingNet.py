##################################################
#
#   Copyright (c) 2010-2024, InterDigital
#   All rights reserved. 
#
#   See LICENSE under the root folder.
#
##################################################

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from argparse import ArgumentParser
import os
import torch
torch.set_num_threads(4)
torch.set_float32_matmul_precision('high')

# These imports have their own get() functions which will get the correct data/loss func/model depending on imput args.
import wrappingnet.dataloaders as dataloaders
import wrappingnet.losses as losses
import wrappingnet.models as models
import wrappingnet.utils as utils


class WrappingNetLightning(LightningModule):

    def __init__(
        self,
        model,
        lr=1e-3,
        lmbda=1.,
        batch_size=1,
        epochs_sphere=200,
        norm=False,
        loss_func='',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.batch_size = batch_size
        self.model = model
        self.lr=lr
        self.lmbda = lmbda
        self.norm = norm
        self.epochs_sphere = epochs_sphere
        self.distortion_func = losses.get_distortion_loss(loss_func)
        self.distortion_eval = losses.l2

    def training_step(self, data, batch_idx):
        if self.norm:
            data.pos = utils.normalize_pos(data.pos, 1)
        # n_base = data.face_base.max()+1
        # pos_base = data.pos[0:n_base]
        pos_base, faces_base = utils.get_base_mesh(data.pos, data.face.T)
        if self.current_epoch <self.epochs_sphere:
            pos_sphere = self.model.make_sphere(pos_base, faces_base)
            loss = losses.base_loss(pos_sphere, scale=10)
            self.log_dict(
                {
                    "train_base": loss.item(),
                    "step" : self.current_epoch*1.
                },
                sync_dist=True, on_step=False, on_epoch=True, batch_size=1
            )
        else:
            pos_list, face_list, _ = self.model(data.pos, data.face.T, pos_base)
            rate = torch.tensor(0.)
            distortion_loss = self.distortion_func(pos_list, face_list, data.pos, data.face.T)
            chamfer_loss = losses.chamfer(pos_list[-1], data.pos)
            loss = distortion_loss #+ 0.5*chamfer_loss
            self.log_dict(
                {
                    "train_rate" : rate.item(),
                    "train_distortion" : distortion_loss.item(),
                    "train_chamfer" : chamfer_loss.item(),
                    "train_loss" : loss.item(),
                    "step" : self.current_epoch*1.
                },
                sync_dist=True, on_step=False, on_epoch=True, batch_size=1
            )
        return loss

    def validation_step(self, data, batch_idx):
        if self.norm:
            data.pos = utils.normalize_pos(data.pos, 1)
        # n_base = data.face_base.max()+1
        # pos_base = data.pos[0:n_base]
        pos_base, faces_base = utils.get_base_mesh(data.pos, data.face.T)
        if self.current_epoch < self.epochs_sphere:
            pos_sphere = self.model.make_sphere(pos_base, faces_base)
            loss = losses.base_loss(pos_sphere, scale=10)
            self.log_dict(
                {
                    "val_base": loss.item(),
                    "step" : self.current_epoch*1.
                },
                sync_dist=True, on_step=False, on_epoch=True, batch_size=1
            )
        else:
            pos_list, face_list, _ = self.model(data.pos, data.face.T, pos_base)
            rate = torch.tensor(0.)
            distortion_loss = self.distortion_eval(pos_list, face_list, data.pos, data.face.T)
            chamfer_loss = losses.chamfer(pos_list[-1], data.pos)
            loss = rate + self.lmbda*distortion_loss
            self.log_dict(
                {
                    "val_rate" : rate.item(),
                    "val_distortion" : distortion_loss.item(),
                    "val_chamfer" : chamfer_loss.item(),
                    "val_loss" : loss.item(),
                    "step" : self.current_epoch*1.
                },
                sync_dist=True, on_step=False, on_epoch=True, batch_size=1
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return {"optimizer":optimizer, "lr_scheduler": lr_scheduler}


def main(args) -> None:
    """
    Training command examples:
    - python WrappingNet.py --gpus 0 --epochs 1000 --epochs_sphere 200 --latent_dim 512 --lr 1e-5 --data_name "manifold40" --model_name "LC"
    - python WrappingNet.py --gpus 0 --epochs 500 --latent_dim 512 --data_name "manifold40" --model_name "global_basesup3"
    """
    # Init lightning model
    extra=''
    extra += args.model_name
    model = models.get_model(args)
    if args.pretrain:
        saved = torch.load(f'trained/MeshAE_{args.loss_func}_{args.data_name}_d{args.latent_dim}{extra}.ckpt', map_location='cpu')
        model.load_state_dict(saved)
    elif args.load_make_sphere:
        print("LOAD MAKESPHERE")
        saved = torch.load(f'trained_make_sphere/make_sphere_{args.data_name}.ckpt', map_location='cpu')
        # saved = torch.load(f'trained_make_sphere/make_sphere_manifold40.ckpt', map_location='cpu')
        model.make_sphere.load_state_dict(saved)
    args.model = model
    lightning_model = WrappingNetLightning(**vars(args))

    trainer = Trainer(accelerator='gpu',
                      devices=len(args.gpus[0]), 
                    #   strategy='ddp',
                      max_epochs=args.epochs,
                      callbacks=[LearningRateMonitor(logging_interval='epoch')],
                      strategy='ddp_find_unused_parameters_true',
                    #   strategy=DDPPlugin(find_unused_parameters=False)
                    #   accumulate_grad_batches=2
                     )

    datamodule = dataloaders.get_data_lightning(args)
    trainer.fit(lightning_model, datamodule)

    if not os.path.exists('trained/'):
        os.makedirs('trained/')
    torch.save(lightning_model.model.state_dict(), f'trained/MeshAE_{args.loss_func}_{args.data_name}_d{args.latent_dim}{extra}.ckpt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-g','--gpus', type=int, nargs='+', action='append', help='gpu_list')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--epochs_sphere', type=int, default=200, help='number of epochs for sphere training')
    parser.add_argument('--latent_dim', type=int, default=512, help='bottleneck dimension')
    parser.add_argument('--data_name', type=str, default='shrec11_loop', help='data name')
    parser.add_argument('--pretrain', dest='pretrain', action='store_true', default=False)
    parser.add_argument('--load_make_sphere', dest='load_make_sphere', action='store_true', default=False)
    parser.add_argument('--norm10', dest='norm10', action='store_true', default=False)
    parser.add_argument('--norm', dest='norm', action='store_true', default=False)
    parser.add_argument('--loss_func', type=str, default='MSL2', help='Loss function')
    parser.add_argument('--model_name', type=str, default='', help='Model Type (see models')
    parser.add_argument('--data_root', type=str, default='./datasets', help='data root')

    args = parser.parse_args()

    main(args)

