#!/bin/bash
python fm_example.py --dataset_name movielens1M \
--dataset_path ml-1m/ratings.dat --model_name ncf --epoch 15 \
--learning_rate 0.001 --batch_size 2048 \
--weight_decay 1e-6 --device cpu --save_dir chkpt
