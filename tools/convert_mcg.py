"""Script to convert Mutiscale Combinatorial Grouping proposal boxes into the Detectron proposal
file format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import scipy.io as sio
import sys
import os

from detectron.datasets.json_dataset_wsl import JsonDataset

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    dir_in = sys.argv[2]
    file_out = sys.argv[3]

    ds = JsonDataset(dataset_name)
    roidb = ds.get_roidb()
    print(len(roidb))

    boxes = []
    scores = []
    ids = []
    for i in range(len(roidb)):
        if i % 1000 == 0:
            print('{}/{}'.format(i + 1, len(roidb)))

        index = os.path.splitext(os.path.basename(roidb[i]['image']))[0]
        box_file = os.path.join(dir_in, '{}.mat'.format(index))
        mat_data = sio.loadmat(box_file)
        if i == 0:
            print(mat_data.keys())
        # boxes_data = mat_data['bboxes']
        # scores_data = mat_data['bboxes_scores']
        boxes_data = mat_data['boxes']
        scores_data = mat_data['scores']
        # selective search boxes are 1-indexed and (y1, x1, y2, x2)
        # Boxes from the MCG website are in (y1, x1, y2, x2) order
        # boxes_data = boxes_data[:, (1, 0, 3, 2)] - 1
        boxes_data_ = boxes_data.astype(np.uint16) - 1
        boxes_data = boxes_data_[:, (1, 0, 3, 2)]

        boxes.append(boxes_data.astype(np.uint16))
        scores.append(scores_data.astype(np.float32))
        ids.append(roidb[i]['id'])

        # boxes.append(boxes_data.astype(np.int16))
        # scores.append(np.squeeze(scores_data.astype(np.float32)))
        # ids.append(index)

    with open(file_out, 'wb') as f:
        pickle.dump(
            dict(boxes=boxes, scores=scores, indexes=ids), f,
            pickle.HIGHEST_PROTOCOL)
