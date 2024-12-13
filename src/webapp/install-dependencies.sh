#!/bin/bash

TARGET_DIR=.

# resolve the TARGET_DIR path to an absolute path
TARGET_DIR=$(realpath $TARGET_DIR)

echo "Installing dependencies into '$TARGET_DIR'" 

# create a virtual environment on the Raspberry Pi if it doesn't exist
cd "${TARGET_DIR}" && python3.11 -m venv venv

# install the required Python packages
cd "${TARGET_DIR}" && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

echo "Dependencies installed into '$TARGET_DIR'"
echo "To start the webapp, execute 'run.sh'"
