#!/bin/bash
venv_path=$(pwd)/hf_venv
if [ ! -d $venv_path ]; then
    python3 -m venv hf_venv
    hf_venv/bin/python -m pip install -U pip
    hf_venv/bin/python -m pip install -r requirements.txt
fi
source $venv_path/bin/activate
export LD_LIBRARY_PATH=$venv_path/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
