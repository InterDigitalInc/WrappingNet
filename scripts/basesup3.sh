#!/bin/bash
python WrappingNet.py --gpus 0 --epochs 500 --latent_dim 512 --data_name "manifold40" --model_name "global_basesup3"
