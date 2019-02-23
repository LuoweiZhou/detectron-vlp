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

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
Modified by Tina Jiang
Last modified by Luowei Zhou on 10/26/2018
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

BOTTOM_UP_FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'object']

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='/z/home/luozhou/subsystem/detectron-xinlei/configs/visual_genome_trainval/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/z/home/luozhou/subsystem/detectron-xinlei/checkpoint/visual_genome_train+visual_genome_val/e2e_faster_rcnn_X-101-64x4d-FPN_2x/RNG_SEED#1989_FAST_RCNN.MLP_HEAD_DIM#2048_FPN.DIM#512/model_final.pkl',
        type=str
    )
    parser.add_argument(
        '--split',
        type=int,
        required=True
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output dir name',
        required=True,
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='.jpg',
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
        'im_or_folder', help='image or folder of images', default=None
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_detections_from_im(cfg, model, im, image_id, featmap_blob_name, feat_blob_name ,MIN_BOXES, MAX_BOXES, conf_thresh=0.2, bboxes=None):

    assert conf_thresh >= 0.
    with c2_utils.NamedCudaScope(0):
        start_timer = timeit.default_timer()
        scores, cls_boxes, im_scale = infer_engine.im_detect_bbox(model, im,cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=bboxes)
        num_rpn = scores.shape[0]
        # region_feat = workspace.FetchBlob(feat_blob_name)
        # cls_prob = workspace.FetchBlob("gpu_0/cls_prob")
        # rois = workspace.FetchBlob("gpu_0/rois")
        max_conf = np.zeros((num_rpn,), dtype=np.float32)
        max_cls = np.zeros((num_rpn,), dtype=np.int32)
        max_box = np.zeros((num_rpn, 4), dtype=np.float32)
        mid_timer = timeit.default_timer()

        for cls_ind in range(1, cfg.MODEL.NUM_CLASSES):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes[:, (cls_ind*4):(cls_ind*4+4)], cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            inds_update = np.where(cls_scores[keep] > max_conf[keep])
            kinds = keep[inds_update]
            max_conf[kinds] = cls_scores[kinds]
            max_cls[kinds] = cls_ind
            max_box[kinds] = dets[kinds][:,:4]

        nms_timer = timeit.default_timer()
        keep_boxes = np.where(max_conf > conf_thresh)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

        objects = max_cls[keep_boxes]
        obj_prob = max_conf[keep_boxes]
        obj_boxes = max_box[keep_boxes, :]
        sort_timer = timeit.default_timer()
        # print('infer: {:.1f}, nms: {:.1f}, sort: {:.1f}'.format(mid_timer-start_timer, nms_timer-mid_timer, \
        #     sort_timer-nms_timer))

    assert(np.sum(objects>=1601) == 0)

    return {
        "image_id": image_id,
        "image_h": np.size(im, 0),
        "image_w": np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': obj_boxes,
        # 'region_feat': region_feat[keep_boxes, :],
        'object': objects,
        'obj_prob': obj_prob
    }


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    start = timeit.default_timer()

    ##extract bboxes from bottom-up attention model
    image_bboxes={}

    count = 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    results = {}

    # split the task into 40 GPUs
    itv = int(np.ceil(220847.0/40))
    list_of_folder = list(range(1+args.split*itv, min(220848, 1+(args.split+1)*itv)))

    for i, folder_name in enumerate(list_of_folder):
      folder_name = str(folder_name)
      print('processing video {}'.format(folder_name))
      vid_path = os.path.join(args.im_or_folder, folder_name)
      fpv = len([j for j in os.listdir(vid_path) if os.path.isfile(os.path.join(vid_path, j))]) # 10
      dets_labels = np.zeros((1, fpv, 100, 6))
      dets_num = np.zeros((1, fpv))
      # dets_feat = np.zeros((1, fpv, 100, 2048))
      nms_num = np.zeros((1, fpv))
      hw = np.zeros((1, 2))

      for j in range(fpv):
        im_name = os.path.join(vid_path, str(j+1).zfill(6)+args.image_ext)

        im = cv2.imread(im_name)
        try:
            result = get_detections_from_im(cfg, model, im, '', '', args.feat_name,
                                                   args.min_bboxes, args.max_bboxes)
        except:
            print('missing frame: ', im_name)
            num_frm = j
            break

        height, width, _ = im.shape
        hw[0, 0] = height
        hw[0, 1] = width

        # store results
        num_proposal = result['boxes'].shape[0]
        proposals = np.concatenate((result['boxes'], np.expand_dims(result['object'], axis=1),
                                    np.expand_dims(result['obj_prob'], axis=1)), axis=1)

        dets_labels[0, j, :num_proposal] = proposals
        dets_num[0, j] = num_proposal
        # dets_feat[0, j] = result['region_feat'].squeeze()
        nms_num[0, j] = num_proposal # for now, treat them the same

      count += 1

      if count % 100 == 0:
          end = timeit.default_timer()
          epoch_time = end - start
          print('process {:d} videos after {:.1f} s'.format(count, epoch_time))

      f = h5py.File(os.path.join(args.output_dir, folder_name+'.h5'), "w")
      f.create_dataset("dets_labels", data=dets_labels)
      f.create_dataset("dets_num", data=dets_num)
      # f.create_dataset("dets_feat", data=dets_feat)
      f.create_dataset("nms_num", data=nms_num)
      f.create_dataset("hw", data=hw)
      f.close()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    print(args)
    main(args)
