#!/bin/bash

# 1) Install Singularity on the computer.
# 2) Place glaunet_predict.py in /worksite
# 3) Place the model files (model_N.h5, N = [1..6]) in /worksite
# 4) Place the Singularity image (gn.img) in the directory
#    with this script. (Suggested directory: ~)
# 5) Place one or more TIF fundus images in /incoming
# 6) Run this program. The output will be /outgoing/C_D_ratio.txt
# 7) Repeat steps 5 and 6 as needed.

sudo singularity run --writable --bind /outgoing:/outgoing,/worksite:/worksite,/incoming:/incoming gn.img

echo 'DONE'
