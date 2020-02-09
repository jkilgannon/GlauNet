#!/bin/bash

set -x

# Install software into the container
sudo apt-get -y update
sudo apt-get -y install wget python3 nano unzip
sudo apt-get -y install python-pip

# Install TensorFlow via pip
sudo pip install tensorflow
sudo pip install pillow

# Make directories
sudo mkdir /worksite
sudo mkdir /worksite/inprime/
sudo mkdir /worksite/inprimeval/
sudo mkdir /worksite/inseg/
sudo mkdir /worksite/insegval/
sudo mkdir /outgoing/

# Copy files
sudo cp -r /local/repository/inprime /worksite/
sudo cp -r /local/repository/inprimeval /worksite/
sudo cp -r /local/repository/inseg /worksite/
sudo cp -r /local/repository/insegval /worksite/

sudo chmod 777 /worksite/inprime
sudo chmod 777 /worksite/inprimeval
sudo chmod 777 /worksite/inseg
sudo chmod 777 /worksite/insegval
sudo chmod 777 /outgoing

sudo chmod 777 /worksite/inprime/*.bmp
sudo chmod 777 /worksite/inprimeval/*.bmp
sudo chmod 777 /worksite/inseg/*.bmp
sudo chmod 777 /worksite/insegval/*.bmp
