#!/bin/bash
set -x
set -e


rm -rf build
mkdir -p build
cd build

PYTORCH_PATH=~/Documents/pytorch/pytorch

cmake .. -DCMAKE_CXX_FLAGS="-isystem ${PYTORCH_PATH}/third_party/eigen -isystem ${PYTORCH_PATH}/third_party/cub -isystem ${PYTORCH_PATH}/third_party/protobuf/src -isystem ${PYTORCH_PATH}" -DCaffe2_DIR=/usr/local/lib/python3.6/dist-packages/torch/share/cmake/Caffe2 -DPYTORCH_PATH=${PYTORCH_PATH}

make -j4
