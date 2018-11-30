#!/bin/bash

python train.py -data ../data/profile.6133.filtered.pt -save_model trained -save_mode best -proj_share_weight
