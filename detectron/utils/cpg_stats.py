from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python import workspace

from detectron.core.config import cfg


class Statistic(object):
    def __init__(self, model, prefix):
        self.LOG_PERIOD = int(1280 / cfg.NUM_GPUS)

        self.label_name = 'labels_oh'
        self.pred_name = 'cls_prob'
        self.pred_pos_name = prefix + 'cls_prob_pos'
        self.pred_neg_name = prefix + 'cls_prob_neg'
        self.csc_name = prefix + 'csc'

        self.num_classes = model.num_classes - 1

        self.ori_label = [0.0 for c in range(self.num_classes)]
        self.ori_pred = [0.0 for c in range(self.num_classes)]
        self.ori_num_roi = [0.0 for c in range(self.num_classes)]
        self.ori_acm_label = 0.0
        self.ori_acm_pred = 0.0
        self.ori_acm_num_roi = 0.0

        self.csc_label = [0.0 for c in range(self.num_classes)]
        self.csc_pred = [0.0 for c in range(self.num_classes)]
        self.csc_pred_pos = [0.0 for c in range(self.num_classes)]
        self.csc_pred_neg = [0.0 for c in range(self.num_classes)]
        self.csc_num_roi = [0.0 for c in range(self.num_classes)]
        self.csc_roi_pos = [0.0 for c in range(self.num_classes)]
        self.csc_roi_neg = [0.0 for c in range(self.num_classes)]
        self.csc_roi_zero = [0.0 for c in range(self.num_classes)]
        self.csc_acm_label = 0.0
        self.csc_acm_pred = 0.0
        self.csc_acm_pred_pos = 0.0
        self.csc_acm_pred_neg = 0.0
        self.csc_acm_num_roi = 0.0
        self.csc_acm_roi_pos = 0.0
        self.csc_acm_roi_neg = 0.0
        self.csc_acm_roi_zero = 0.0

        self.num_img = 0

    def UpdateIterStats(self):
        for i in range(cfg.NUM_GPUS):
            label_val = workspace.FetchBlob('gpu_{}/{}'.format(
                i, self.label_name))
            pred_val = workspace.FetchBlob('gpu_{}/{}'.format(
                i, self.pred_name))
            pred_pos_val = workspace.FetchBlob('gpu_{}/{}'.format(
                i, self.pred_pos_name))
            pred_neg_val = workspace.FetchBlob('gpu_{}/{}'.format(
                i, self.pred_neg_name))
            csc_val = workspace.FetchBlob('gpu_{}/{}'.format(i, self.csc_name))

            label_val = np.squeeze(label_val)
            pred_val = np.squeeze(pred_val)
            pred_pos_val = np.squeeze(pred_pos_val)
            pred_neg_val = np.squeeze(pred_neg_val)
            csc_val = np.squeeze(csc_val)

            for c in range(self.num_classes):
                if label_val[c] <= 0.5:
                    continue

                self.ori_label[c] += 1
                self.ori_pred[c] += pred_val[c]
                self.ori_num_roi[c] += csc_val.shape[0]

                self.ori_acm_label += 1
                self.ori_acm_pred += pred_val[c]
                self.ori_acm_num_roi += csc_val.shape[0]

                if pred_val[c] < cfg.WSL.CPG_TAU:
                    continue

                self.csc_label[c] += 1
                self.csc_pred[c] += pred_val[c]
                self.csc_pred_pos[c] += pred_pos_val[c]
                self.csc_pred_neg[c] += pred_neg_val[c]
                self.csc_num_roi[c] += csc_val.shape[0]

                self.csc_acm_label += 1
                self.csc_acm_pred += pred_val[c]
                self.csc_acm_pred_pos += pred_pos_val[c]
                self.csc_acm_pred_neg += pred_neg_val[c]
                self.csc_acm_num_roi += csc_val.shape[0]

                for r in range(csc_val.shape[0]):
                    if csc_val[r, c] > 0:
                        self.csc_roi_pos[c] += 1
                        self.csc_acm_roi_pos += 1
                    elif csc_val[r, c] < 0:
                        self.csc_roi_neg[c] += 1
                        self.csc_acm_roi_neg += 1
                    else:
                        self.csc_roi_zero[c] += 1
                        self.csc_acm_roi_zero += 1

            self.num_img += 1

    def LogIterStats(self, cur_iter):
        if cur_iter % self.LOG_PERIOD > 0:
            return

        print(
            '#class\tpred\t#roi\t#class\tpred\tpos\tneg\t#roi\t#pos\t%\t#neg\t%\t#zero\t%\tclass'
        )

        for c in range(self.num_classes):

            if self.ori_label[c] > 0:
                info_ori = '{}\t{:.5f}\t{}\t'.format(
                    int(self.ori_label[c]),
                    self.ori_pred[c] / self.ori_label[c],
                    int(self.ori_num_roi[c] / self.ori_label[c]))
            else:
                info_ori = '0\t0.0000\t0\t'

            self.ori_label[c] = 0.0
            self.ori_pred[c] = 0.0
            self.ori_num_roi[c] = 0.0

            if self.csc_label[c] > 0:
                info_csc = '{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}\t{}\t{:.5f}\t{}\t{:.5f}\t{}\t{:.5f}\t{}'.format(
                    int(self.csc_label[c]),
                    self.csc_pred[c] / self.csc_label[c],
                    self.csc_pred_pos[c] / self.csc_label[c],
                    self.csc_pred_neg[c] / self.csc_label[c],
                    int(self.csc_num_roi[c] / self.csc_label[c]),
                    int(self.csc_roi_pos[c] / self.csc_label[c]),
                    1.0 * self.csc_roi_pos[c] / self.csc_num_roi[c],
                    int(self.csc_roi_neg[c] / self.csc_label[c]),
                    1.0 * self.csc_roi_neg[c] / self.csc_num_roi[c],
                    int(self.csc_roi_zero[c] / self.csc_label[c]),
                    1.0 * self.csc_roi_zero[c] / self.csc_num_roi[c], c)
            else:
                info_csc = '0\t0.0000\t0.0000\t0.0000\t0\t0\t0.0000\t0\t0.0000\t0\t0.0000\t{}'.format(
                    c)

            self.csc_label[c] = 0.0
            self.csc_pred[c] = 0.0
            self.csc_pred_pos[c] = 0.0
            self.csc_pred_neg[c] = 0.0
            self.csc_num_roi[c] = 0.0
            self.csc_roi_pos[c] = 0.0
            self.csc_roi_neg[c] = 0.0
            self.csc_roi_zero[c] = 0.0

            print(info_ori, info_csc)

        if self.ori_acm_label > 0 and self.csc_acm_label > 0:
            info_acm = '{}\t{:.5f}\t{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}\t{}\t{:.5f}\t{}\t{:.5f}\t{}\t{:.5f}\t{}'.format(
                int(self.ori_acm_label),
                self.ori_acm_pred / self.ori_acm_label,
                int(self.ori_acm_num_roi / self.ori_acm_label),
                self.csc_acm_label, self.csc_acm_pred / self.csc_acm_label,
                self.csc_acm_pred_pos / self.csc_acm_label,
                self.csc_acm_pred_neg / self.csc_acm_label,
                int(self.csc_acm_num_roi / self.csc_acm_label),
                int(self.csc_acm_roi_pos / self.csc_acm_label),
                1.0 * self.csc_acm_roi_pos / self.csc_acm_num_roi,
                int(self.csc_acm_roi_neg / self.csc_acm_label),
                1.0 * self.csc_acm_roi_neg / self.csc_acm_num_roi,
                int(self.csc_acm_roi_zero / self.csc_acm_label),
                1.0 * self.csc_acm_roi_zero / self.csc_acm_num_roi,
                self.num_img)
        else:
            info_acm = ''

        self.ori_acm_label = 0.0
        self.ori_acm_pred = 0.0
        self.ori_acm_num_roi = 0.0

        self.csc_acm_label = 0.0
        self.csc_acm_pred = 0.0
        self.csc_acm_pred_pos = 0.0
        self.csc_acm_pred_neg = 0.0
        self.csc_acm_num_roi = 0.0
        self.csc_acm_roi_pos = 0.0
        self.csc_acm_roi_neg = 0.0
        self.csc_acm_roi_zero = 0.0

        self.num_img = 0

        print(info_acm)
