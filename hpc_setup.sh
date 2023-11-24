#!/bin/bash

module load modenv/hiera GCCcore/11.3.0 Python/3.10.4

virtualenv --system-site-packages hybridqa_env

source hybridqa_env/bin/activate 
pip install -r requirements.txt
