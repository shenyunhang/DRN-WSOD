import datetime
import json
import os
import sys
import os.path as osp
import cPickle as pickle
import numpy as np
from os.path import expanduser
home = expanduser("~")

from PIL import Image
# from pycococreatortools import pycococreatortools

import detectron.datasets.dataset_catalog as dataset_catalog


def load_pickle(path):
    """Check and load pickle object.
    According to this post: https://stackoverflow.com/a/41733927, cPickle and 
    disabling garbage collector helps with loading speed."""
    print path
    assert osp.exists(path)
    # gc.disable()
    with open(path, 'rb') as f:
        ret = pickle.load(f)
        # gc.enable()
    return ret


dataset_name = 'voc_2007'
dataset_split = 'test'
root_dir = osp.join(home, 'Dataset')

pkl_path = sys.argv[1]
with open(pkl_path, 'r') as f:
    detections = pickle.load(f)
all_boxes = detections['all_boxes']
print(detections.keys())
print(len(all_boxes))
print(len(all_boxes[1]))

num_classes = len(all_boxes) - 1
num_images = len(all_boxes[1])

ann_fn = dataset_catalog.get_ann_fn(dataset_name + '_' + dataset_split)
with open(ann_fn, 'r') as f:
    json_data = json.load(f)
print(json_data.keys())
print(len(json_data['annotations']))

for i in range(0, len(json_data['images'])):
    print(json_data['images'][i])
    break

for i in range(0, len(json_data['annotations'])):
    print(json_data['annotations'][i])
    break

anns = []
ann_id = 1
for i in range(0, num_images):
    image_id = json_data['images'][i]['id']
    for c in range(1, num_classes):
        boxes = all_boxes[c][i]
        for j in range(0, len(boxes)):
            b = boxes[j][0:4]
            b = b.astype(np.int)
            # https://github.com/cocodataset/cocoapi/blob/master/MatlabAPI/CocoUtils.m#L336
            b[2:4] = b[2:4] - b[0:2] + 1
            x1 = b[0]
            x2 = b[0] + b[2]
            y1 = b[1]
            y2 = b[1] + b[3]

            S = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            a = b[2] * b[3]

            ann = dict()
            ann[u'segmentation'] = S
            ann[u'area'] = a
            ann[u'iscrowd'] = 0
            ann[u'image_id'] = image_id
            ann[u'bbox'] = b.tolist()
            ann[u'category_id'] = c
            ann[u'id'] = ann_id
            ann[u'ignore'] = 0

            ann_id = ann_id + 1
            anns.append(ann)

json_data['annotations'] = anns

for i in range(0, len(json_data['images'])):
    print(json_data['images'][i])
    break

for i in range(0, len(json_data['annotations'])):
    print(json_data['annotations'][i])
    break

print(ann_id)

json_path = os.path.join(root_dir,
                         dataset_name + '_' + dataset_split + '_pgt.json')
with open(json_path, 'w') as output_json_file:
    json.dump(json_data, output_json_file)
