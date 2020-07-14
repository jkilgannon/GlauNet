#!/bin/bash

set -x

# Make directories
sudo mkdir /worksite
sudo mkdir /incoming
sudo mkdir /outgoing

# Copy files
sudo cp /local/repository/thesis/* /users/jk880380/
sudo cp /local/repository/thesis/glaunet_predict.py /worksite/
sudo cp /worksite/fundus/image333* /incoming/test_333.tif

# Update permissions
sudo chmod 777 /users/jk880380/* || true
sudo chmod 777 /incoming || true
sudo chmod 777 /outgoing || true
sudo chmod 777 /worksite || true
sudo chmod 777 /incoming/* || true
sudo chmod 777 /outgoing/* || true
sudo chmod 777 /worksite/* || true

## Install software into the machine
sudo apt-get -y update
sudo apt-get -y install wget python3 nano unzip
sudo apt-get -y install python3-pip
#sudo apt-get -y install singularity
#sudo apt install -y singularity-container

### Install TensorFlow via pip
sudo pip3 install tensorflow
sudo pip3 install pillow
sudo pip3 install SciPy
sudo pip3 install matplotlib
