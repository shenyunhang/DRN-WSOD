from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest
import cv2
import math

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace

import detectron.utils.c2 as c2_utils


class DenseCRFOpTest(unittest.TestCase):
    def _run_dense_crf_op(self, U, I):
        op = core.CreateOperator('DenseCRF', ['U', 'I'], ['M'], max_iter=5)
        workspace.FeedBlob('U', U)
        workspace.FeedBlob('I', I)
        workspace.RunOperatorOnce(op)
        M = workspace.FetchBlob('M')
        return M

    def _run_dense_crf_op_gpu(self, U, I):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('DenseCRF', ['U', 'I'], ['M'], max_iter=5)
            workspace.FeedBlob('U', U)
            workspace.FeedBlob('I', I)
        workspace.RunOperatorOnce(op)
        M = workspace.FetchBlob('M')
        return M

    def _test_handles_arrays(self, idx, fun):
        im = cv2.imread('detectron/ops/densecrf/examples/im' + idx + '.ppm')
        # process image
        H, W, C = im.shape
        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im = im.reshape((1, H, W, 3))
        channel_swap = (0, 3, 1, 2)
        im = im.transpose(channel_swap)

        anno = cv2.imread('detectron/ops/densecrf/examples/anno' + idx +
                          '.ppm')

        # Get label
        label = {
            255: [0, np.array([255, 0, 0])],
            -1: [-1, np.array([0, 0, 0])]
        }
        H, W, C = anno.shape
        lbl = np.zeros((H, W), dtype=np.int)
        for h in range(H):
            for w in range(W):
                color = anno[h, w, :]
                key = color[0] + color[1] * 256 + color[2] * 256 * 256
                if key:
                    if key in label.keys():
                        l = label[key][0]
                    else:
                        l = len(label.keys())
                        label[key] = [l, color]
                else:
                    l = -1

                lbl[h, w] = l
        print(label)

        # computeUnary
        unary = np.zeros((21, H, W), dtype=np.float32)
        u_energy = -math.log(1.0 / 21.0)
        n_energy = -math.log((1.0 - 0.5) / (21.0 - 1.0))
        p_energy = -math.log(0.5)
        print(u_energy, n_energy, p_energy)
        for h in range(H):
            for w in range(W):
                l = lbl[h, w]
                if l >= 0:
                    unary[:, h, w] = n_energy
                    unary[l, h, w] = p_energy
                else:
                    unary[:, h, w] = u_energy
        unary = unary.reshape(1, 21, H, W)
        # unary = -1.0 * unary

        M = fun(unary, im)
        print(M.shape, M.max(), M.min(), M.mean())
        print(M.shape, M.max(), M.min(), M.mean())

        aM = np.argmax(M, axis=1)
        res = np.zeros((H, W, 3), dtype=np.uint8)
        for h in range(H):
            for w in range(W):
                l = aM[0, h, w]
                # l = lbl[h, w]
                # print(M[0, :, h, w])
                for key in label.keys():
                    if l == label[key][0]:
                        color = label[key][1]
                res[h, w, :] = color
        cv2.imwrite('build/im' + idx + '.png', res)

    def test_handles_arrays(self):
        self._test_handles_arrays('1', self._run_dense_crf_op)
        self._test_handles_arrays('2', self._run_dense_crf_op)

    def test_handles_arrays_gpu(self):
        self._test_handles_arrays('3', self._run_dense_crf_op_gpu)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    c2_utils.import_custom_ops()
    assert 'DenseCRF' in workspace.RegisteredOperators()
    unittest.main()
