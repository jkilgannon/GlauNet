#!/bin/bash

set -x

## Install software into the container
sudo apt-get -y update
#sudo apt-get -y install wget python3 nano unzip
#sudo apt-get -y install python3-pip
sudo apt-get -y install singularity

## Install TensorFlow via pip
#sudo pip3 install tensorflow
#sudo pip3 install pillow
#sudo pip3 install SciPy
#sudo pip3 install matplotlib

# Make directories
sudo mkdir /worksite
#sudo mkdir /worksite/fundus/
#sudo mkdir /worksite/fundusvalidate/
#sudo mkdir /worksite/mask0/
#sudo mkdir /worksite/maskvalidate0/
#sudo mkdir /worksite/mask1/
#sudo mkdir /worksite/maskvalidate1/
#sudo mkdir /worksite/mask2/
#sudo mkdir /worksite/maskvalidate2/
sudo mkdir /incoming/
sudo mkdir /outgoing/

sudo chmod 777 /worksite
#sudo chmod 777 /worksite/fundus
#sudo chmod 777 /worksite/fundusvalidate
#sudo chmod 777 /worksite/mask0
#sudo chmod 777 /worksite/maskvalidate0
#sudo chmod 777 /worksite/mask1
#sudo chmod 777 /worksite/maskvalidate1
#sudo chmod 777 /worksite/mask2
#sudo chmod 777 /worksite/maskvalidate2
sudo chmod 777 /incoming
sudo chmod 777 /outgoing

# Copy files
#sudo cp /local/repository/trimmed_imgs/fundus/* /worksite/fundus/
#sudo cp /local/repository/trimmed_imgs/fundusvalidate/* /worksite/fundusvalidate/
#sudo cp /local/repository/trimmed_imgs/mask0/* /worksite/mask0/
#sudo cp /local/repository/trimmed_imgs/mask1/* /worksite/mask1/
#sudo cp /local/repository/trimmed_imgs/mask2/* /worksite/mask2/
#sudo cp /local/repository/trimmed_imgs/maskvalidate0/* /worksite/maskvalidate0/
#sudo cp /local/repository/trimmed_imgs/maskvalidate1/* /worksite/maskvalidate1/
#sudo cp /local/repository/trimmed_imgs/maskvalidate2/* /worksite/maskvalidate2/

sudo cp /local/repository/thesis/* /users/jk880380/
sudo cp /local/repository/thesis/glaunet_predict.py /worksite/

#sudo cp /worksite/fundus/image333* /users/jk880380/test.tif
sudo cp /worksite/fundus/image333* /incoming/test_333.tif

#sudo chmod 777 /worksite/fundus/*
#sudo chmod 777 /worksite/fundusvalidate/*
#sudo chmod 777 /worksite/mask0/*
#sudo chmod 777 /worksite/maskvalidate0/*
#sudo chmod 777 /worksite/mask1/*
#sudo chmod 777 /worksite/maskvalidate1/*
#sudo chmod 777 /worksite/mask2/*
#sudo chmod 777 /worksite/maskvalidate2/*
sudo chmod 777 /users/jk880380/*
sudo chmod 777 /incoming/*
