#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc

export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0;8.6"   # 3090:8.6; a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1;


pip install openmim # OK
mim install mmcv-full # pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
mim install mmdet
mim install mmsegmentation

pip install "git+https://github.com/open-mmlab/mmdetection3d.git"
# git clone https://github.com/open-mmlab/mmdetection3d.git
# maybe not correct commit
# cd mmdetection3d
# pip install -r requirements.txt
# pip install .

pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# on windows before installing pytorch3d, set the following environment variables. Otherwise, it will install without GPU support.
# set TORCH_CUDA_ARCH_LIST=8.6 #rtx 3060: 8.6
# set FORCE_CUDA=1
# for CUDA 11.7 need to replace included cub in CUDA\v11.7\include\cub with separate cub library version 1.16.0

# download openpoints
git submodule update --init --recursive

# install cpp extensions, the pointnet++ library
# set CUDA_HOME correctly before running this command
# use cuda 11.7 and pytorch 2.0
# modified F:\PythonVenvs\3.8venvTANnet\Lib\site-packages\torch\utils\cpp_extension.py to parse cuda version differently
#     might not be necessary
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
cd subsampling
python setup.py build_ext --inplace
cd ..


# # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
cd pointops/
python setup.py install
cd ..


# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../

# added strict version for some of the packages in requirements.txt, because newer didn't work
pip install -r requirements.txt
