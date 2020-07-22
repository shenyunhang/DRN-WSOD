# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""PASCAL VOC dataset evaluation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import shutil
import uuid

from detectron.core.config import cfg
from detectron.datasets.dataset_catalog import get_devkit_dir
from detectron.datasets.voc_eval import voc_eval
from detectron.datasets.voc_eval import voc_eval_corloc
from detectron.utils.io import save_object

logger = logging.getLogger(__name__)


def evaluate_boxes(
    json_dataset,
    all_boxes,
    output_dir,
    use_salt=True,
    cleanup=True,
    use_matlab=False
):
    salt = '_{}'.format(str(uuid.uuid4())) if use_salt else ''
    filenames = _write_voc_results_files(json_dataset, all_boxes, salt)
    _do_python_eval(json_dataset, salt, output_dir)
    _do_python_eval_corloc(json_dataset, salt, output_dir)
    if use_matlab:
        _do_matlab_eval(json_dataset, salt, output_dir)
    if cleanup:
        for filename in filenames:
            shutil.copy(filename, output_dir)
            os.remove(filename)
    return None


def _write_voc_results_files(json_dataset, all_boxes, salt):
    filenames = []
    image_set_path = voc_info(json_dataset)['image_set_path']
    assert os.path.exists(image_set_path), \
        'Image set path does not exist: {}'.format(image_set_path)
    with open(image_set_path, 'r') as f:
        image_index = [x.strip() for x in f.readlines()]
    # Sanity check that order of images in json dataset matches order in the
    # image set
    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        index = os.path.splitext(os.path.split(entry['image'])[1])[0]
        assert index == image_index[i]
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        logger.info('Writing VOC results for: {}'.format(cls))
        filename = _get_voc_results_file_template(json_dataset,
                                                  salt).format(cls)
        filenames.append(filename)
        assert len(all_boxes[cls_ind]) == len(image_index)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind][im_ind]
                if type(dets) == list:
                    assert len(dets) == 0, \
                        'dets should be numpy.ndarray or empty list'
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.9f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
    return filenames


def _get_voc_results_file_template(json_dataset, salt):
    info = voc_info(json_dataset)
    year = info['year']
    image_set = info['image_set']
    devkit_path = info['devkit_path']
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = 'comp4' + salt + '_det_' + image_set + '_{:s}.txt'
    return os.path.join(devkit_path, 'results', 'VOC' + year, 'Main', filename)


def _do_python_eval(json_dataset, salt, output_dir='output'):
    info = voc_info(json_dataset)
    year = info['year']
    anno_path = info['anno_path']
    image_set_path = info['image_set_path']
    devkit_path = info['devkit_path']
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(
            json_dataset, salt).format(cls)
        rec, prec, ap = voc_eval(
            filename, anno_path, image_set_path, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        logger.info('AP for {} = {:.4f}'.format(cls, ap))
        res_file = os.path.join(output_dir, cls + '_pr.pkl')
        save_object({'rec': rec, 'prec': prec, 'ap': ap}, res_file)
    logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('Results:')
    for ap in aps:
        logger.info('{:.3f}'.format(ap))
    logger.info('{:.3f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info('----------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info('Results should be very close to the official MATLAB code.')
    logger.info('Use `./tools/reval.py --matlab ...` for your paper.')
    logger.info('-- Thanks, The Management')
    logger.info('----------------------------------------------------------')


    print('Results:')
    for ap in aps:
        print('{:.2f}&'.format(ap * 100.0), end = '')
    print('{:.2f}'.format(np.mean(aps) * 100.0))


def _do_python_eval_corloc(json_dataset, salt, output_dir='output'):
    info = voc_info(json_dataset)
    year = info['year']
    anno_path = info['anno_path']
    image_set_path = info['image_set_path']
    devkit_path = info['devkit_path']
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    corlocs = []
    too_min_rates = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(
            json_dataset, salt).format(cls)
        corloc, too_min_rate = voc_eval_corloc(
            filename, anno_path, image_set_path, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        corlocs += [corloc]
        too_min_rates += [too_min_rate]
        logger.info('CorLoc for {} = {:.4f}'.format(cls, corloc))
        logger.info('too_min_rate for {} = {:.4f}'.format(cls, too_min_rate))
        res_file = os.path.join(output_dir, cls + '_corloc.pkl')
        save_object({'corloc': corloc}, res_file)
    logger.info('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))
    logger.info('Mean too_min_rate = {:.4f}'.format(np.mean(too_min_rates)))
    logger.info('~~~~~~~~')
    logger.info('Results:')
    for corloc in corlocs:
        logger.info('{:.3f}'.format(corloc))
    logger.info('{:.3f}'.format(np.mean(corlocs)))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info('----------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info('Results should be very close to the official MATLAB code.')
    logger.info('Use `./tools/reval.py --matlab ...` for your paper.')
    logger.info('-- Thanks, The Management')
    logger.info('----------------------------------------------------------')

    print('Results:')
    for corloc in corlocs:
        print('{:.2f}&'.format(corloc * 100.0), end = '')
    print('{:.2f}'.format(np.mean(corlocs) * 100.0))


def _do_matlab_eval(json_dataset, salt, output_dir='output'):
    import subprocess
    logger.info('-----------------------------------------------------')
    logger.info('Computing results with the official MATLAB eval code.')
    logger.info('-----------------------------------------------------')
    info = voc_info(json_dataset)
    path = os.path.join(
        cfg.ROOT_DIR, 'detectron', 'datasets', 'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
       .format(info['devkit_path'], 'comp4' + salt, info['image_set'],
               output_dir)
    logger.info('Running:\n{}'.format(cmd))
    subprocess.call(cmd, shell=True)


def voc_info(json_dataset):
    year = json_dataset.name[4:8]
    image_set = json_dataset.name[9:]
    devkit_path = get_devkit_dir(json_dataset.name)
    assert os.path.exists(devkit_path), \
        'Devkit directory {} not found'.format(devkit_path)
    anno_path = os.path.join(
        devkit_path, 'VOC' + year, 'Annotations', '{:s}.xml')
    image_set_path = os.path.join(
        devkit_path, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt')
    return dict(
        year=year,
        image_set=image_set,
        devkit_path=devkit_path,
        anno_path=anno_path,
        image_set_path=image_set_path)


def evaluate_masks(
    json_dataset,
    all_boxes,
    all_segms,
    output_dir,
    use_salt=False,
    cleanup=False,
    use_matlab=False
):
    return None
    _write_voc_segms_results_files(json_dataset, all_boxes, all_segms, output_dir)

    print("Coding unfinish...")
    return None
    _do_python_eval_segms(json_dataset, output_dir)
    if use_matlab:
        _do_matlab_eval(json_dataset, salt, output_dir)
    if cleanup:
        for filename in filenames:
            shutil.copy(filename, output_dir)
            os.remove(filename)
    return None


def _write_voc_segms_results_files(json_dataset, all_boxes, all_segms, result_dir):
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        print('Writing {} VOC results file'.format(cls))
        filename = os.path.join(result_dir, cls + '_det.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(all_boxes[cls_ind], f, pickle.HIGHEST_PROTOCOL)
        filename = os.path.join(result_dir, cls + '_seg.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(all_segms[cls_ind], f, pickle.HIGHEST_PROTOCOL)


def _do_python_eval_segms(json_dataset, output_dir):
        info_str = ''
        gt_dir = self.data_path
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        cache_dir = os.path.join(self.devkit_path, 'annotations_cache')
        output_dir = os.path.join(self.result_path, 'results')
        aps = []
        # define this as true according to SDS's evaluation protocol
        use_07_metric = True
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        info_str += 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += '\n'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        print('~~~~~~ Evaluation use min overlap = 0.5 ~~~~~~')
        info_str += '~~~~~~ Evaluation use min overlap = 0.5 ~~~~~~'
        info_str += '\n'
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = voc_eval_sds(det_filename, seg_filename, gt_dir,
                              imageset_file, cls, cache_dir, self.classes, self.mask_size, self.binary_thresh, ov_thresh=0.5)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
            info_str += 'AP for {} = {:.2f}\n'.format(cls, ap*100)
        print('Mean AP@0.5 = {:.2f}'.format(np.mean(aps)*100))
        info_str += 'Mean AP@0.5 = {:.2f}\n'.format(np.mean(aps)*100)
        print('~~~~~~ Evaluation use min overlap = 0.7 ~~~~~~')
        info_str += '~~~~~~ Evaluation use min overlap = 0.7 ~~~~~~\n'
        aps = []
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = voc_eval_sds(det_filename, seg_filename, gt_dir,
                              imageset_file, cls, cache_dir, self.classes, self.mask_size, self.binary_thresh, ov_thresh=0.7)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
            info_str += 'AP for {} = {:.2f}\n'.format(cls, ap*100)
        print('Mean AP@0.7 = {:.2f}'.format(np.mean(aps)*100))
        info_str += 'Mean AP@0.7 = {:.2f}\n'.format(np.mean(aps)*100)

        return info_str
