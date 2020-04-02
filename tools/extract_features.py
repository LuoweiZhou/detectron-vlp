#!/usr/bin/env python2

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

"""
Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
Modified by Tina Jiang
Again modified by Luowei Zhou on 12/18/2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import numpy as np
import base64
import csv
import timeit
import json
import h5py
import itertools

from utils.io import cache_url
import utils.c2 as c2_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

from caffe2.python import workspace
import caffe2

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
from utils.boxes import nms
c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

csv.field_size_limit(sys.maxsize)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl',
        type=str
    )
    parser.add_argument(
        '--box-output-dir',
        dest='box_output_dir',
        help='bounding box output dir name',
        required=True,
        type=str
    )
    parser.add_argument(
        '--featcls-output-dir',
        dest='featcls_output_dir',
        help='region feature and class prob output dir name',
        required=True,
        type=str
    )
    parser.add_argument(
        '--output-file-prefix',
        dest='output_file_prefix',
        required=True,
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--min_bboxes',
        help=" min number of bboxes",
        type=int,
        default=100
    )
    parser.add_argument(
        '--max_bboxes',
        help=" min number of bboxes",
        type=int,
        default=100
    )
    parser.add_argument(
        '--feat_name',
        help=" the name of the feature to extract, default: gpu_0/fc6",
        type=str,
        default="gpu_0/fc6"
    )
    parser.add_argument(
        '--dataset',
        help='Support dataset COCO | CC | Flickr30k | SBU',
        type=str,
        default='Flickr30k'
    )
    parser.add_argument(
        '--proc_split',
        help='only process image IDs that match this pattern at the end',
        type=str,
        default='000'
    )
    parser.add_argument(
        '--data_type',
        help='default float32, set to float16 to save storage space (e.g., for CC and SBU)',
        type=str,
        default='float32'
    )
    parser.add_argument(
        'im_or_folder',
        help='image or folder of images',
        default=None
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_detections_from_im(cfg, model, im, image_id, featmap_blob_name, feat_blob_name ,MIN_BOXES, MAX_BOXES, conf_thresh=0.2, bboxes=None):

    assert conf_thresh >= 0.
    with c2_utils.NamedCudaScope(0):
        scores, cls_boxes, im_scale = infer_engine.im_detect_bbox(model, im,cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=bboxes)
        num_rpn = scores.shape[0]
        region_feat = workspace.FetchBlob(feat_blob_name)
        max_conf = np.zeros((num_rpn,), dtype=np.float32)
        max_cls = np.zeros((num_rpn,), dtype=np.int32)
        max_box = np.zeros((num_rpn, 4), dtype=np.float32)

        for cls_ind in range(1, cfg.MODEL.NUM_CLASSES):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes[:, (cls_ind*4):(cls_ind*4+4)], cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            inds_update = np.where(cls_scores[keep] > max_conf[keep])
            kinds = keep[inds_update]
            max_conf[kinds] = cls_scores[kinds]
            max_cls[kinds] = cls_ind
            max_box[kinds] = dets[kinds][:,:4]

        keep_boxes = np.where(max_conf > conf_thresh)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

        objects = max_cls[keep_boxes]
        obj_prob = max_conf[keep_boxes]
        obj_boxes = max_box[keep_boxes, :]
        cls_prob = scores[keep_boxes, :]

    # print('{} ({}x{}): {} boxes, box size {}, feature size {}, class size {}'.format(image_id,
    #       np.size(im, 0), np.size(im, 1), len(keep_boxes), cls_boxes[keep_boxes].shape,
    #       box_features[keep_boxes].shape, objects.shape))
    # print(cls_boxes[keep_boxes][:10, :], objects[:10], obj_prob[:10])

    assert(np.sum(objects>=cfg.MODEL.NUM_CLASSES) == 0)
    # assert(np.min(obj_prob[:10])>=0.2)
    # if np.min(obj_prob) < 0.2:
        # print('confidence score too low!', np.min(obj_prob[:10]))
    # if np.max(cls_boxes[keep_boxes]) > max(np.size(im, 0), np.size(im, 1)):
    #     print('box is offscreen!', np.max(cls_boxes[keep_boxes]), np.size(im, 0), np.size(im, 1))

    return {
        "image_id": image_id,
        "image_h": np.size(im, 0),
        "image_w": np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': obj_boxes,
        'region_feat': region_feat[keep_boxes, :],
        'object': objects,
        'obj_prob': obj_prob,
        'cls_prob': cls_prob
    }


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    start = timeit.default_timer()

    # extract region bboxes and features from pre-trained models
    count = 0
    if not os.path.exists(args.box_output_dir):
        os.makedirs(args.box_output_dir)
    if not os.path.exists(args.featcls_output_dir):
        os.makedirs(args.featcls_output_dir)

    results = {}

    with h5py.File(os.path.join(args.box_output_dir, args.output_file_prefix+'_bbox'+args.proc_split+'.h5'), "w") as f, \
        h5py.File(os.path.join(args.featcls_output_dir, args.output_file_prefix+'_feat'+args.proc_split+'.h5'), "w") as f_feat, \
        h5py.File(os.path.join(args.featcls_output_dir, args.output_file_prefix+'_cls'+args.proc_split+'.h5'), "w") as f_cls:

        if args.dataset in ('COCO', 'Flickr30k'):
            if os.path.isdir(args.im_or_folder):
                im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
        elif args.dataset == 'CC':
            valid_ids = json.load(open('/mnt/dat/CC/annotations/cc_valid_jpgs.json')) # some images are broken. hard coded for now
            im_list = valid_ids.keys()
            print('number of valid CC images {}'.format(len(im_list)))
        elif args.dataset == 'SBU':
            valid_ids = json.load(open('/z/dat/VLP/dat/SBU/annotations/sbu_valid_jpgs.json')) # some images are broken. hard coded for now
            im_list = valid_ids.keys()
            print('number of valid SBU images {}'.format(len(im_list)))

        for i, im_name in enumerate(im_list):
            im_base_name = os.path.basename(im_name)
            image_id = im_base_name
            if image_id[-4-len(args.proc_split):-4] == args.proc_split:
                im_name = os.path.join(args.im_or_folder, image_id)

                print(im_name)
                im = cv2.imread(im_name)
                result = get_detections_from_im(cfg, model, im, image_id, '', args.feat_name,
                                                           args.min_bboxes, args.max_bboxes)
                # store results
                proposals = np.concatenate((result['boxes'], np.expand_dims(result['object'], axis=1) \
                    .astype(np.float32), np.expand_dims(result['obj_prob'], axis=1)), axis=1)

                f.create_dataset(image_id[:-4], data=proposals.astype(args.data_type))
                f_feat.create_dataset(image_id[:-4], data=result['region_feat'].squeeze().astype(args.data_type))
                f_cls.create_dataset(image_id[:-4], data=result['cls_prob'].astype(args.data_type))

                count += 1
                if count % 10 == 0:
                    end = timeit.default_timer()
                    epoch_time = end - start
                    print('process {:d} images after {:.1f} s'.format(count, epoch_time))


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
