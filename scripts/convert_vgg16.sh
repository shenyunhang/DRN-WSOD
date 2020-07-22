#!/bin/bash
set -x
set -e

~/Documents/cpg/caffe-wsl/build/tools/upgrade_net_proto_binary \
	~/Dataset/model/VGG_ILSVRC_16_layers.caffemodel \
	~/Dataset/model/VGG_ILSVRC_16_layers_v1.caffemodel

~/Documents/cpg/caffe-wsl/build/tools/upgrade_net_proto_text \
	~/Dataset/model/VGG_ILSVRC_16_layers_deploy.prototxt \
	~/Dataset/model/VGG_ILSVRC_16_layers_deploy_v1.prototxt

python ./tools/pickle_caffe_blobs.py \
	--prototxt ~/Dataset/model/VGG_ILSVRC_16_layers_deploy_v1.prototxt \
	--caffemodel ~/Dataset/model/VGG_ILSVRC_16_layers_v1.caffemodel \
	--output ~/Dataset/model/VGG_ILSVRC_16_layers.pkl
