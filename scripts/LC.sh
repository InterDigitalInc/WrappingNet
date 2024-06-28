#!/bin/bash
python WrappingNet.py --gpus 0 --epochs 1000 --epochs_sphere 200 --latent_dim 512 --lr 1e-5 --data_name "manifold40" --model_name "LC"
