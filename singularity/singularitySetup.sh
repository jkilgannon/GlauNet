#!/bin/bash

set -x

# Install software into the container
sudo apt-get -y update
sudo apt-get -y install wget python3 nano unzip
sudo apt-get -y install python-pip

# Install TensorFlow via pip
sudo pip install tensorflow
sudo pip install pillow

sudo mkdir /worksite || true
sudo mkdir /outgoing || true
