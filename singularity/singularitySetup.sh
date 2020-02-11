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

# Copy files
sudo cp -r /local/repository/inprime/all_data /worksite/inprime
sudo cp -r /local/repository/inprimeval/all_data /worksite/inprimeval/
sudo cp -r /local/repository/inseg/all_data /worksite/inseg/
sudo cp -r /local/repository/insegval/all_data /worksite/insegval/

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

sudo chmod 777 /worksite/inprime/all_data/*.tif
sudo chmod 777 /worksite/inprimeval/all_data/*.tif
sudo chmod 777 /worksite/inseg/all_data/*.tif
sudo chmod 777 /worksite/insegval/all_data/*.bmp
