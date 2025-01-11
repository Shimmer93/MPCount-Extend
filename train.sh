#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py -g 1 -n 1 -w 8 -b 16 --data_dir data/sta --exp_name sta -c configs/mpcount_sta.yml --version $(date +'%Y%m%d_%H%M%S')