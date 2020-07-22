from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import math

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir


def vis_training(cur_iter):
    if not (cfg.WSL.DEBUG or
            (cfg.WSL.SAMPLE and cur_iter % cfg.WSL.SAMPLE_ITER == 0)):
        return

    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    sample_dir = os.path.join(output_dir, 'sample')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for gpu_id in range(cfg.NUM_GPUS):
        data_ids = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'data_ids'))
        ims = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'data'))
        cpg = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'cpg'))
        labels_oh = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, 'labels_oh'))

        if cfg.WSL.DEBUG:
            print('gpu_id: ', gpu_id, 'cpg: ', cpg.shape, cpg.max(), cpg.min(),
                  cpg.mean())

        prefix = 'iter_' + str(cur_iter) + '_gpu_' + str(gpu_id)
        save_im(ims, cfg.PIXEL_MEANS, prefix, sample_dir)
        save_cpg(cpg, labels_oh, prefix, sample_dir)

        im_scores = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'cls_prob'))
        roi_scores = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, 'rois_pred'))
        rois = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'rois'))
        if cfg.WSL.CSC and (not cfg.MODEL.MASK_ON or True):
            csc = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'csc'))
            if cfg.WSL.DEBUG:
                print('gpu_id: ', gpu_id, 'csc: ', csc.shape, csc.max(),
                      csc.min(), csc.mean())

            save_csc(csc, labels_oh, im_scores, roi_scores, ims, rois,
                     cfg.PIXEL_MEANS, prefix, '', sample_dir)

        if cfg.WSL.CENTER_LOSS and False:
            center_S = workspace.FetchBlob('gpu_{}/S'.format(gpu_id))
            save_center(center_S, labels_oh, roi_scores, ims, rois,
                        cfg.PIXEL_MEANS, prefix, '', sample_dir)

        if not cfg.MODEL.MASK_ON:
            continue

        if cfg.WSL.CSC:
            mask_csc = workspace.FetchBlob('gpu_{}/mask_{}'.format(
                gpu_id, 'csc'))

            if cfg.WSL.DEBUG:
                print('gpu_id: ', gpu_id, 'mask_csc: ', mask_csc.shape,
                      mask_csc.max(), mask_csc.min(), mask_csc.mean())

            save_csc(mask_csc, labels_oh, im_scores, roi_scores, ims, rois,
                     cfg.PIXEL_MEANS, prefix, 'mask', sample_dir)

        if 'deeplab' in cfg.MRCNN.ROI_MASK_HEAD:
            blobs_name = [
                'mask_labels_oh', 'mask_fc8', 'mask_fc8_up', 'mask_fc8_crf_fg',
                'mask_fc8_bg'
            ]
            sigmoids = [0, 1, 1, 0, 0]
            for blob_name, sigmoid in zip(blobs_name, sigmoids):
                data = workspace.FetchBlob('gpu_{}/{}'.format(
                    gpu_id, blob_name))
                if cfg.WSL.DEBUG:
                    print('gpu_id: ', gpu_id, ' ', blob_name, ': ', data.shape,
                          data.max(), data.min(), data.mean())
                if sigmoid:
                    save_sigmoid(data, labels_oh, prefix, blob_name,
                                 sample_dir)
                else:
                    save_common(data, labels_oh, prefix, blob_name, sample_dir)

            continue

            mask_crf = workspace.FetchBlob(
                'gpu_{}/mask_fc8_crf_fg_up'.format(gpu_id))
            save_common(mask_crf, labels_oh, prefix, 'mask_fc8_crf_fg_up',
                        sample_dir)

            mask = workspace.FetchBlob('gpu_{}/mask_fc8_up'.format(gpu_id))
            save_pixels_pkl(data_ids, cpg, mask, mask_crf, labels_oh,
                            sample_dir)


def save_pixels_pkl(data_ids, cpg, mask, mask_crf, labels_oh, output_dir):
    batch_size, _ = data_ids.shape
    for b in range(batch_size):
        data_id = data_ids[b][0]
        data_id = str(data_id).zfill(6)

        file_name = os.path.join(output_dir, data_id + '_cpg.npy')
        data = cpg[b].squeeze()
        np.save(file_name, data)

        file_name = os.path.join(output_dir, data_id + '_mask.npy')
        data = mask[b].squeeze()
        np.save(file_name, data)

        file_name = os.path.join(output_dir, data_id + '_mask_crf.npy')
        data = mask_crf[b].squeeze()
        np.save(file_name, data)


def save_im(ims, pixel_means, prefix, output_dir):
    batch_size, _, _, _ = ims.shape
    for b in range(batch_size):
        im = ims[b, :, :, :].copy()
        channel_swap = (1, 2, 0)
        im = im.transpose(channel_swap)
        im += pixel_means
        im = im.astype(np.uint8)
        file_name = os.path.join(output_dir, prefix + '_b_' + str(b) + '.png')
        cv2.imwrite(file_name, im)


def save_cpg(cpgs, labels_oh, prefix, output_dir):
    batch_size, num_classes, _, _ = cpgs.shape
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0:
                continue
            cpg = cpgs[b, c, :, :].copy()
            max_value = np.max(cpg)
            if max_value > 0:
                max_value = max_value * 0.1
                cpg = np.clip(cpg, 0, max_value)
                cpg = cpg / max_value * 255
            cpg = cpg.astype(np.uint8)
            im_color = cv2.applyColorMap(cpg, cv2.COLORMAP_JET)
            file_name = os.path.join(
                output_dir,
                prefix + '_b_' + str(b) + '_c_' + str(c) + '_cpg.png')
            cv2.imwrite(file_name, im_color)


def save_common(datas, labels_oh, prefix, suffix, output_dir):
    if datas is None:
        return
    if len(datas.shape) == 3:
        datas = datas[np.newaxis, :]
    batch_size, num_classes, _, _ = datas.shape
    # print(datas.shape, labels_oh.shape)
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0 and num_classes > 1:
                continue
            data = datas[b, c, :, :].copy()
            data = data * 255
            data = data.astype(np.uint8)
            im_color = cv2.applyColorMap(data, cv2.COLORMAP_JET)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_color)


def save_sigmoid(datas, labels_oh, prefix, suffix, output_dir):
    batch_size, num_classes, _, _ = datas.shape
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0:
                continue
            data = datas[b, c, :, :].copy()
            data = np.reciprocal(1 + np.exp(-data))
            data = data * 255
            data = data.astype(np.uint8)
            im_color = cv2.applyColorMap(data, cv2.COLORMAP_JET)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_color)


def save_csc(csc, labels_oh, im_scores, roi_scores, ims, rois, pixel_means,
             prefix, suffix, output_dir):
    num_rois, num_classes = roi_scores.shape
    batch_size, _, height, weight = ims.shape
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0:
                continue
            if im_scores[b][c] < cfg.WSL.CPG_TAU:
                continue
            im = ims[b, :, :, :].copy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            im += pixel_means
            im = im.astype(np.uint8)
            im_pW = im.copy()
            im_nW = im.copy()
            im_S = im.copy()
            im_pWS = im.copy()
            im_nWS = im.copy()

            argsort = np.argsort(np.abs(csc[:, c]))
            scale_p = 1.0 / csc[:, c].max()
            scale_n = -1.0 / csc[:, c].min()
            for n in range(num_rois):
                roi = rois[argsort[n]]
                if roi[0] != b:
                    continue
                if csc[argsort[n]][c] < 0.0:
                    jet = gray2jet(-csc[argsort[n]][c] * scale_n)
                    cv2.rectangle(im_nW, (roi[1], roi[2]), (roi[3], roi[4]),
                                  jet, 1)
                else:
                    jet = gray2jet(csc[argsort[n]][c] * scale_p)
                    cv2.rectangle(im_pW, (roi[1], roi[2]), (roi[3], roi[4]),
                                  jet, 1)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_pW_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_pW)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_nW_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_nW)

            argsort = np.argsort(np.abs(roi_scores[:, c]))
            scale_p = 1.0 / roi_scores[:, c].max()
            for n in range(num_rois):
                roi = rois[argsort[n]]
                if roi[0] != b:
                    continue
                jet = gray2jet(roi_scores[argsort[n]][c] * scale_p)
                cv2.rectangle(im_S, (roi[1], roi[2]), (roi[3], roi[4]), jet, 1)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_S_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_S)

            ws = np.multiply(roi_scores[:, c], csc[:, c])
            argsort = np.argsort(np.abs(ws))
            scale_p = 1.0 / ws.max()
            scale_n = -1.0 / ws.min()
            for n in range(num_rois):
                roi = rois[argsort[n]]
                if roi[0] != b:
                    continue
                if ws[argsort[n]] < 0.0:
                    jet = gray2jet(-ws[argsort[n]] * scale_n)
                    cv2.rectangle(im_nWS, (roi[1], roi[2]), (roi[3], roi[4]),
                                  jet, 1)
                else:
                    jet = gray2jet(ws[argsort[n]] * scale_p)
                    cv2.rectangle(im_pWS, (roi[1], roi[2]), (roi[3], roi[4]),
                                  jet, 1)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_pWS_'
                + suffix + '.png')
            cv2.imwrite(file_name, im_pWS)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_nWS_'
                + suffix + '.png')
            cv2.imwrite(file_name, im_nWS)


def save_center(center_S, labels_oh, roi_scores, ims, rois, pixel_means,
                prefix, suffix, output_dir):
    num_rois, num_classes = roi_scores.shape
    batch_size, _, height, weight = ims.shape
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0:
                continue
            # if im_scores[b][c] < cfg.WSL.CPG_TAU:
            # continue
            im = ims[b, :, :, :].copy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            im += pixel_means
            im = im.astype(np.uint8)

            argsort = np.argsort(-np.abs(roi_scores[:, c]))
            scale_p = 1.0 / roi_scores[:, c].max()
            for n in range(10):
                roi = rois[argsort[n]]
                if roi[0] != b:
                    continue
                im_roi = im[int(roi[2]):int(roi[4]) + 1,
                            int(roi[1]):int(roi[3]) + 1, :].copy()
                file_name = os.path.join(
                    output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) +
                    '_m_' + str(int(
                        center_S[c])) + '_' + str(n) + '_' + suffix + '.png')
                cv2.imwrite(file_name, im_roi)


def dump_proto_files(model, output_dir):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    with open(os.path.join(output_dir, model.net.Proto().name), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(
            os.path.join(output_dir,
                         model.param_init_net.Proto().name), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))


def gray2jet(f):
    # plot short rainbow RGB
    a = f / 0.25  # invert and group
    X = math.floor(a)  # this is the integer part
    Y = math.floor(255 * (a - X))  # fractional part from 0 to 255
    Z = math.floor(128 * (a - X))  # fractional part from 0 to 128

    if X == 0:
        r = 0
        g = Y
        b = 128 - Z
    elif X == 1:
        r = Y
        g = 255
        b = 0
    elif X == 2:
        r = 255
        g = 255 - Z
        b = 0
    elif X == 3:
        r = 255
        g = 128 - Z
        b = 0
    elif X == 4:
        r = 255
        g = 0
        b = 0
    # opencv is bgr, not rgb
    return (b, g, r)


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if len(pts) == 0:
        return

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(
        img,
        pts,
        color,
        thickness=1,
        style='dotted',
):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)
