#!/bin/bash

python pix2pix.py \
  --mode train \
  --checkpoint face2face-model \
  --output_dir face2face-model \
  --max_epochs 2 \
  --input_dir photos/combined/train \
  --which_direction AtoB
