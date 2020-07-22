from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import scipy.io as sio
import sys
import os
from time import strftime

from detectron.datasets.json_dataset_wsl import JsonDataset

import tsnecuda
from tsnecuda import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

from detectron.utils.io import save_object
from detectron.utils.io import load_object

# tsnecuda.test()


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))

    # custom
    # c4 = np.copy(color_list[4])
    # c6 = np.copy(color_list[6])
    # c14 = np.copy(color_list[14])

    # color_list[4] = c14
    # color_list[14] = c6
    # color_list[6] = c4

    # color_list = np.insert(color_list, 0, [0, 0, 0, 1], 0)

    cmap_name = base.name + str(N)
    return mpl.colors.LinearSegmentedColormap.from_list(
        cmap_name, color_list, N)


def tsne(X, n_components, perplexity, learning_rate, N, gt_classes, dir_out,
         suffix, has_label):

    X_embedded = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
    ).fit_transform(X)

    if has_label:
        fig = plt.figure(figsize=(20, 15))
    else:
        fig = plt.figure(figsize=(15, 15))

    # cmap = plt.cm.jet
    # cmap = plt.cm.rainbow
    cmap = discrete_cmap(N, base_cmap='tab20')

    scat = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=gt_classes[:],
        cmap=cmap,
    )

    # bg_inds = np.where(gt_classes == 0)[0]
    # fg_inds = np.where(gt_classes > 0)[0]
    # plt.scatter(
    # X_embedded[bg_inds, 0],
    # X_embedded[bg_inds, 1],
    # c=gt_classes[bg_inds],
    # cmap=cmap,
    # )

    if has_label:
        plt.colorbar(ticks=range(N))
        plt.clim(-0.5, N - 0.5)

    plt.axis('off')
    plt.tick_params(axis='both',
                    left='off',
                    top='off',
                    right='off',
                    bottom='off',
                    labelleft='off',
                    labeltop='off',
                    labelright='off',
                    labelbottom='off')
    plt.tight_layout(pad=0)

    # now = strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        dir_out, '{}_{}_{}.png'.format(
            perplexity,
            learning_rate,
            suffix,
        ))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close('all')


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    dir_in = sys.argv[2]
    dir_out = sys.argv[3]

    has_fc = True
    has_fc = False
    has_label = True
    has_label = False

    N = 21  # Number of labels

    save_path = os.path.join(dir_out, 'data.pkl')
    if True:
        roi_feats = []
        if has_fc:
            fc6s = []
            fc7s = []
        max_classes = []
        gt_classes = []
        gt_classes_cnt = np.zeros(N, dtype=np.int32)
        cnt = 0
        cnt2 = 0
        for root, dirs, files in os.walk(dir_in, topdown=False):
            for name in files:
                if name.endswith('pkl'):
                    pass
                else:
                    continue
                print(cnt, os.path.join(root, name))
                path = os.path.join(root, name)
                pkl = load_object(path)

                roi_feat = pkl['roi_feats']
                if has_fc:
                    fc6 = pkl['fc6']
                    fc7 = pkl['fc7']

                gt_class = pkl['gt_classes']

                print(cnt, cnt2, 'roi_feat: ', roi_feat.shape)

                t = []
                for i in range(roi_feat.shape[0]):
                    if gt_class[i] in t:
                        # continue
                        pass
                    else:
                        pass

                    if gt_class[i] == 0:
                        continue
                        if np.random.random() >= 0.0:
                            continue
                    else:
                        if gt_classes_cnt[gt_class[i]] >= 600:
                            continue

                    roi_feats.append(roi_feat[i])
                    if has_fc:
                        fc6s.append(fc6[i])
                        fc7s.append(fc7[i])

                    gt_classes.append(gt_class[i])
                    gt_classes_cnt[gt_class[i]] += 1

                    t.append(gt_class[i])

                    cnt2 += 1

                cnt += 1
                # if cnt > 1000:
                # break
                if cnt2 > 13500:
                    break

            for name in dirs:
                print(os.path.join(root, name))

        roi_feats = np.array(roi_feats)
        print('roi_feats: ', roi_feats.shape)
        if has_fc:
            fc6s = np.array(fc6s)
            fc7s = np.array(fc7s)
            print('fc6s: ', fc6s.shape)
            print('fc7s: ', fc7s.shape)

        roi_feats = np.reshape(roi_feats, (roi_feats.shape[0], -1))
        if has_fc:
            fc6s = np.reshape(fc6s, (fc6s.shape[0], -1))
            fc7s = np.reshape(fc7s, (fc7s.shape[0], -1))

        gt_classes = np.array(gt_classes)

        # save_object(dict(
        # roi_feats=roi_feats,
        # max_classes=max_classes,
        # gt_classes=gt_classes,
        # ),
        # save_path,
        # pickle_format=4)
    else:
        print('Loading ', save_path)
        pkl = load_object(save_path)
        roi_feats = pkl['roi_feats']
        max_classes = pkl['max_classes']
        gt_classes = pkl['gt_classes']

    print('roi_feats: ', roi_feats.shape)
    if has_fc:
        print('fc6s: ', fc6s.shape)
        print('fc7s: ', fc7s.shape)

    print(np.unique(gt_classes).shape)

    n_components = 2
    for perplexity in range(100, 100 + 10, 10):
        for learning_rate in range(100, 1000 + 100, 100):
            # n_components = 2
            # perplexity = 105
            # learning_rate = 810

            print('n_components: ', n_components, ' perplexity: ', perplexity,
                  ' learning_rate: ', learning_rate)

            tsne(roi_feats, n_components, perplexity, learning_rate, N - 1,
                 gt_classes - 1, dir_out, 'roi_feat', has_label)

            if has_fc:
                tsne(fc6s, n_components, perplexity, learning_rate, N - 1,
                     gt_classes - 1, dir_out, 'fc6', has_label)
                tsne(fc7s, n_components, perplexity, learning_rate, N - 1,
                     gt_classes - 1, dir_out, 'fc7', has_label)

            # exit(0)
