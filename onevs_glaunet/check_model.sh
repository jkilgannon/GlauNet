#!/bin/bash

echo "START"

echo $1 $2

cp /outgoing/best_model_incremental-cls$1-run$2.h5 ./best_model_incremental.h5
echo "onevsmany_loadnet.py"
python3 onevsmany_loadnet.py
echo "onevsmany_unhot.py"
python3 onevsmany_unhot.py
echo "onevsmany_color_predicted.py"
python3 onevsmany_color_predicted.py

echo $1 $2

echo "DONE"
