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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging
import numpy as np
import os
import scipy.sparse

# Must happen before importing COCO API (which imports matplotlib)
import detectron.utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from detectron.core.config import cfg
from detectron.utils.timer import Timer
import detectron.datasets.dataset_catalog as dataset_catalog
import detectron.utils.boxes as box_utils
from detectron.utils.io import load_object
import detectron.utils.segms as segm_utils

logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        assert dataset_catalog.contains(name), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(dataset_catalog.get_im_dir(name)), \
            'Im dir \'{}\' not found'.format(dataset_catalog.get_im_dir(name))
        assert os.path.exists(dataset_catalog.get_ann_fn(name)), \
            'Ann fn \'{}\' not found'.format(dataset_catalog.get_ann_fn(name))
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = dataset_catalog.get_im_dir(name)
        self.image_prefix = dataset_catalog.get_im_prefix(name)
        self.COCO = COCO(dataset_catalog.get_ann_fn(name))
        self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self._init_keypoints()

        logger.info(self.classes)
        logger.info(self.json_category_id_to_contiguous_id)
        logger.info(self.contiguous_category_id_to_json_id)

    def get_roidb(
        self,
        gt=False,
        proposal_file=None,
        min_proposal_size=20,
        proposal_limit=-1,
        crowd_filter_thresh=0
    ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        for entry in roidb:
            self._prep_roidb_entry(entry)
        if gt:
            # Include ground-truth object annotations
            self.debug_timer.tic()
            for entry in roidb:
                self._add_gt_annotations(entry)
            logger.debug(
                '_add_gt_annotations took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )

        if cfg.USE_PSEUDO and 'test' not in self.name:
            pgt_roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
            for entry in pgt_roidb:
                self._prep_roidb_entry(entry)
            self._add_pseudo_gt_annotations(pgt_roidb, roidb)
            roidb = pgt_roidb

        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)
        if gt:
            roidb = _filter_no_class(self.name, roidb)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name']
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['obn_scores'] = np.empty((0, 1), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        all_diffcult_truncated = True
        for obj in objs:
            # crowd regions are RLE encoded
            if segm_utils.is_poly(obj['segmentation']):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue

            if 'diffcult' in obj:
                if obj['diffcult'] == 0:
                    all_diffcult_truncated = False
            else:
                all_diffcult_truncated = False
            if 'truncated' in obj:
                if obj['truncated'] == 0:
                    all_diffcult_truncated = False
            else:
                all_diffcult_truncated = False

            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])

        if all_diffcult_truncated:
            valid_objs = []
            valid_segms = []

        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        obn_scores = np.zeros((num_valid_objs, 1), dtype=entry['obn_scores'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['obn_scores'] = np.append(entry['obn_scores'], obn_scores, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_pseudo_gt_annotations(self, roidb, gt_roidb):
        """
        Return the database of pseudo ground-truth regions of interest from detect result.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # gt_roidb = copy.deepcopy(roidb)
        # for entry in roidb:
            # self._add_gt_annotations(entry)


        assert 'train' in self.name or 'val' in self.name, 'Only trainval dataset has pseudo gt.'

        # detection.pkl is 0-based indices
        # the VOC result file is 1-based indices

        cache_files = cfg.PSEUDO_PATH
        if isinstance(cache_files, str):
            cache_files = (cache_files, )
        all_dets = None
        for cache_file in cache_files:
            if self.name not in cache_file:
                continue
            assert os.path.exists(cache_file), cache_file
            # with open(cache_file, 'rb') as fid:
                # res = cPickle.load(fid)
            res = load_object(cache_file)
            print('{} pseudo gt roidb loaded from {}'.format(self.name, cache_file))
            if all_dets is None:
                all_dets = res['all_boxes']
            else:
                for i in range(len(all_dets)):
                    all_dets[i].extend(res['all_boxes'][i])

        assert len(all_dets[1]) == len(roidb), len(all_dets[1])
        if len(all_dets) == self.num_classes:
            cls_offset = 0
        elif len(all_dets) + 1 == self.num_classes:
            cls_offset = -1
        else:
            raise Exception('Unknown mode.')

        threshold = 1.0

        for im_i, entry in enumerate(roidb):
            if im_i % 1000 == 0:
                print('{:d} / {:d}'.format(im_i + 1, len(roidb)))
            num_valid_objs = 0
            if len(gt_roidb[im_i]['gt_classes']) == 0:
                print(gt_roidb[im_i])
            if len(gt_roidb[im_i]['is_crowd']) == sum(gt_roidb[im_i]['is_crowd']):
                print(gt_roidb[im_i])

            # when cfg.WSL = False, background class is in.
            # detection.pkl only has 20 classes
            # fast_rcnn need 21 classes
            for cls in range(1, self.num_classes):
                # TODO(YH): we need threshold the pseudo label
                # filter the pseudo label

                # self._gt_class has 21 classes
                # if self._gt_classes[ix][cls] == 0:
                if cls not in gt_roidb[im_i]['gt_classes']:
                    continue
                dets = all_dets[cls + cls_offset][im_i]
                if dets.shape[0] <= 0:
                    continue

                # TODO(YH): keep only one box
                # if dets.shape[0] > 0:
                # num_valid_objs += 1

                max_score = 0
                num_valid_objs_cls = 0
                for i in range(dets.shape[0]):
                    det = dets[i]

                    score = det[4]
                    if score > max_score:
                        max_score = score
                    if score < threshold:
                        continue
                    num_valid_objs += 1
                    num_valid_objs_cls += 1

                if num_valid_objs_cls == 0:
                    if max_score > 0:
                        num_valid_objs += 1

            boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
            # obn_scores = np.zeros((num_valid_objs, 1), dtype=entry['obn_scores'].dtype)
            gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
            gt_overlaps = np.zeros(
                (num_valid_objs, self.num_classes),
                dtype=entry['gt_overlaps'].dtype
            )
            seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
            is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
            box_to_gt_ind_map = np.zeros(
                (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
            )


            obj_i = 0
            valid_segms = []
            for cls in range(1, self.num_classes):
                # filter the pseudo label
                # self._gt_class has 21 classes
                # if self._gt_classes[ix][cls] == 0:
                if cls not in gt_roidb[im_i]['gt_classes']:
                    continue
                dets = all_dets[cls + cls_offset][im_i]
                if dets.shape[0] <= 0:
                    continue

                max_score = 0
                max_score_bb = []
                num_valid_objs_cls = 0
                for i in range(dets.shape[0]):
                    det = dets[i]
                    x1 = det[0]
                    y1 = det[1]
                    x2 = det[2]
                    y2 = det[3]

                    score = det[4]
                    if score > max_score:
                        max_score = score
                        max_score_bb = [x1, y1, x2, y2]
                    if score < threshold:
                        continue

                    assert x1 >= 0
                    assert y1 >= 0
                    assert x2 >= x1
                    assert y2 >= y1
                    assert x2 < entry['width']
                    assert y2 < entry['height']

                    boxes[obj_i, :] = [x1, y1, x2, y2]
                    gt_classes[obj_i] = cls
                    seg_areas[obj_i] = (x2 - x1 + 1) * (y2 - y1 + 1)
                    is_crowd[obj_i] = 0
                    box_to_gt_ind_map[obj_i] = obj_i
                    gt_overlaps[obj_i, cls] = 1.0
                    valid_segms.append([])
                    

                    obj_i += 1
                    num_valid_objs_cls += 1

                if num_valid_objs_cls == 0:
                    x1, y1, x2, y2 = max_score_bb[:]

                    assert x1 >= 0
                    assert y1 >= 0
                    assert x2 >= x1
                    assert y2 >= y1
                    assert x2 < entry['width']
                    assert y2 < entry['height']

                    boxes[obj_i, :] = [x1, y1, x2, y2]
                    gt_classes[obj_i] = cls
                    seg_areas[obj_i] = (x2 - x1 + 1) * (y2 - y1 + 1)
                    is_crowd[obj_i] = 0
                    box_to_gt_ind_map[obj_i] = obj_i
                    gt_overlaps[obj_i, cls] = 1.0
                    valid_segms.append([])

                    obj_i += 1

            assert obj_i == num_valid_objs

            # Show Pseudo GT boxes
            if True:
            # if False:
                import cv2
                im = cv2.imread(entry['image'])
                for obj_i in range(num_valid_objs):
                    cv2.rectangle(im, (boxes[obj_i][0], boxes[obj_i][1]),
                                  (boxes[obj_i][2], boxes[obj_i][3]), (255, 0,
                                                                       0), 5)
                save_dir = os.path.join(cfg.OUTPUT_DIR, 'pgt')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, str(im_i) + '.png')
                # print(save_path)
                cv2.imwrite(save_path, im)

                # cv2.imshow('im', im)
                # cv2.waitKey()

            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            # entry['obn_scores'] = np.append(entry['obn_scores'], obn_scores, axis=0)
            entry['segms'].extend(valid_segms)
            # To match the original implementation:
            # entry['boxes'] = np.append(
            #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
            entry['gt_overlaps'] = np.append(
                entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
            )
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
            entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'], box_to_gt_ind_map
            )

    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        proposals = load_object(proposal_file)

        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix

        # _remove_proposals_not_in_roidb(proposals, roidb, id_field)
        _sort_proposals(proposals, id_field)
        box_list = []
        score_list = []
        total_roi = 0
        up_1024 = 0
        up_2048 = 0
        up_3072 = 0
        up_4096 = 0
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            scores = proposals['scores'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            # boxes = box_utils.clip_boxes_to_image(
                # boxes, entry['height'], entry['width']
            # )
            assert (boxes[:, 0] >= 0).all()
            assert (boxes[:, 1] >= 0).all()
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            assert (boxes[:, 3] >= boxes[:, 1]).all()
            assert (boxes[:, 2] < entry['width']).all(), entry['image']
            assert (boxes[:, 3] < entry['height']).all(), entry['image']

            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            scores = scores[keep]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            scores = scores[keep]

             # sort by confidence
            sorted_ind = np.argsort(-scores.flatten())
            boxes = boxes[sorted_ind, :]
            scores = scores[sorted_ind, :]

            if top_k > 0:
                boxes = boxes[:top_k, :]
                scores = scores[:top_k]

            total_roi += boxes.shape[0]
            if boxes.shape[0] > 1024:
                up_1024 += 1
            if boxes.shape[0] > 2048:
                up_2048 += 1
            if boxes.shape[0] > 3072:
                up_3072 += 1
            if boxes.shape[0] > 4096:
                up_4096 += 1

            box_list.append(boxes)
            score_list.append(scores)

        print('total_roi: ', total_roi, ' ave roi: ', total_roi / len(box_list))
        print('up_1024: ', up_1024)
        print('up_2048: ', up_2048)
        print('up_3072: ', up_3072)
        print('up_4096: ', up_4096)

        _merge_proposal_boxes_into_roidb(roidb, box_list, score_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            self.keypoint_flip_map = {
                'left_eye': 'right_eye',
                'left_ear': 'right_ear',
                'left_shoulder': 'right_shoulder',
                'left_elbow': 'right_elbow',
                'left_wrist': 'right_wrist',
                'left_hip': 'right_hip',
                'left_knee': 'right_knee',
                'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps


def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list, score_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        scores = score_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['obn_scores'] = np.append(
            entry['obn_scores'],
            scores.astype(entry['obn_scores'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _filter_no_class(name, roidb):
    if 'test' in name:
        return roidb
    new_roidb = []
    for entry in roidb:
        if sum(entry['max_classes']) == 0:
            continue
        new_roidb.append(entry)
    print('number of original roidb: ', len(roidb))
    print('number of new roidb: ', len(new_roidb))
    return new_roidb
        

def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]


def _remove_proposals_not_in_roidb(proposals, roidb, id_field):
    # fix proposals so they don't contain entries for images not in the roidb
    roidb_ids = set({entry["id"] for entry in roidb})
    keep = [i for i, id in enumerate(proposals[id_field]) if id in roidb_ids]
    for f in ['boxes', id_field, 'scores']:
        proposals[f] = [proposals[f][i] for i in keep]
