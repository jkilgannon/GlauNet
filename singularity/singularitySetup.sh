#!/bin/bash

set -x

# Install software into the container
sudo apt-get -y update
sudo apt-get -y install wget python3 nano unzip
sudo apt-get -y install python-pip3

# Install TensorFlow via pip
sudo pip3 install tensorflow
sudo pip3 install pillow
sudo pip3 install SciPy

# Make directories
sudo mkdir /worksite
sudo mkdir /worksite/inprime/
sudo mkdir /worksite/inprimeval/
sudo mkdir /worksite/inseg/
sudo mkdir /worksite/insegval/
sudo mkdir /worksite/inprime/all_data/
sudo mkdir /worksite/inprimeval/all_data/
sudo mkdir /worksite/inseg/all_data/
sudo mkdir /worksite/insegval/all_data/
sudo mkdir /outgoing/

sudo chmod 777 /worksite
sudo chmod 777 /worksite/inprime
sudo chmod 777 /worksite/inprimeval
sudo chmod 777 /worksite/inseg
sudo chmod 777 /worksite/insegval
sudo chmod 777 /worksite/inprime/all_data/
sudo chmod 777 /worksite/inprimeval/all_data/
sudo chmod 777 /worksite/inseg/all_data/
sudo chmod 777 /worksite/insegval/all_data/
sudo chmod 777 /outgoing

# Copy files
sudo cp /local/repository/inprime/* /worksite/inprime/all_data/
sudo cp /local/repository/inprimeval/* /worksite/inprimeval/all_data/
sudo cp /local/repository/inseg/* /worksite/inseg/all_data/
sudo cp /local/repository/insegval/* /worksite/insegval/all_data/
sudo cp /local/repository/nn/* /users/jk880380/

sudo chmod 777 /worksite/inprime/all_data/*
sudo chmod 777 /worksite/inprimeval/all_data/*
sudo chmod 777 /worksite/inseg/all_data/*
sudo chmod 777 /worksite/insegval/all_data/*
sudo chmod 777 /users/jk880380/*
