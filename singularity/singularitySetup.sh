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
sudo cp /worksite/inprime /inprime
sudo cp /worksite/inprimeval /inprimeval
sudo cp /worksite/inseg /inseg
sudo cp /worksite/insegval /insegval

sudo chmod 755 /worksite/inprime/*
sudo chmod 755 /worksite/inprimeval/*
sudo chmod 755 /worksite/inseg/*
sudo chmod 755 /worksite/insegval/*
