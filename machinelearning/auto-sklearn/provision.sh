#/usr/bin/env bash

# System requirements
apt-get update && apt-get install -y \
  build-essential \
  curl \
  python3-pip \
  swig3.0 \
  && rm -rf /var/lib/apt/lists/*

ln -sf /usr/bin/swig3.0 /usr/bin/swig

# Upgrade pip
pip3 install --upgrade pip
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt \
  | xargs -n 1 -L 1 pip3 install

# Install
pip3 install auto-sklearn jupyter
