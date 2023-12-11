#!/bin/bash
#taurus
#module load modenv/hiera GCCcore/11.3.0 Python/3.10.4

#barnard
module load release/23.04  GCCcore/11.3.0 Python/3.10.4

virtualenv --system-site-packages hybridqa_env

source hybridqa_env/bin/activate 

#install torch version that fits to loaded CUDA Version
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
