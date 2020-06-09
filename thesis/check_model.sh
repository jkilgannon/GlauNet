#!/bin/bash

echo "START"
echo '--- check_model ---'
echo $0
echo $1
echo $2
echo '------'

cp /outgoing/best_model_incremental-cls1-run$1.h5 ./best_model_incremental.h5

echo "onevsmany_loadnet.py"
python3 onevsmany_loadnet.py
echo "onevsmany_unhot.py"
python3 onevsmany_unhot.py
echo "onevsmany_color_predicted.py"
python3 onevsmany_color_predicted.py $2

echo "DONE: $1"
