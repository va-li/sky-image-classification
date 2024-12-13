#!/bin/bash

# check that the virtual environment exists
if [ ! -d "venv" ]; then
    VENV_DIR=$(realpath venv)
    echo "Python virtual environment not found at '${VENV_DIR}'."
    echo "Run 'install-dependencies.sh' first."
    exit 1
fi

# source the virtual environment
source venv/bin/activate

# run the webapp
python server.py
