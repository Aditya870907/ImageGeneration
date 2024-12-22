#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate sd_env

# Start the server
python client.py
