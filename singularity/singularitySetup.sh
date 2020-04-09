#!/bin/bash

set -x

# Install software into the container
sudo apt-get -y update
sudo apt-get -y install wget python3 nano unzip
sudo apt-get -y install python3-pip

# Install TensorFlow via pip
sudo pip3 install tensorflow
sudo pip3 install pillow
sudo pip3 install SciPy
sudo pip3 install matplotlib

# Make directories
sudo mkdir /worksite
sudo mkdir /worksite/fundus/
sudo mkdir /worksite/fundusvalidate/
sudo mkdir /worksite/mask/
sudo mkdir /worksite/maskvalidate/
sudo mkdir /outgoing/

sudo chmod 777 /worksite
sudo chmod 777 /worksite/fundus
sudo chmod 777 /worksite/fundusvalidate
sudo chmod 777 /worksite/mask
sudo chmod 777 /worksite/maskvalidate
sudo chmod 777 /outgoing

# Copy files
sudo cp /local/repository/inprime/* /worksite/fundus/
sudo cp /local/repository/inprimeval/* /worksite/fundusvalidate/
sudo cp /local/repository/inseg/* /worksite/mask/
sudo cp /local/repository/insegval/* /worksite/maskvalidate/
sudo cp /local/repository/nn/* /users/jk880380/

sudo chmod 777 /worksite/fundus/*
sudo chmod 777 /worksite/fundusvalidate/*
sudo chmod 777 /worksite/mask/*
sudo chmod 777 /worksite/maskvalidate/*
sudo chmod 777 /users/jk880380/*
