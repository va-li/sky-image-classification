#!/bin/bash

# copy webapp files to Raspberry Pi and set up virtual environment

HOST=$1

if [ -z $HOST ]; then
  echo "Usage: $0 <hostname>"
  exit 1
fi

TARGET_DIR=~/sky-image-classification/webapp

# create the folders on the Raspberry Pi if they don't exist
ssh $HOST "mkdir -p $TARGET_DIR"

# exclude __pycache__, .git, venv, .vscode folders
rsync -av --exclude '__pycache__' --exclude '.git' --exclude 'venv' --exclude '.vscode' ./webapp/ $HOST:${TARGET_DIR}
