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
Last modified by Luowei Zhou on 09/25/2018
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
        default='/private/home/luoweizhou/subsystem/detectron-xinlei/configs/visual_genome_trainval/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/private/home/luoweizhou/subsystem/detectron-xinlei/checkpoint/visual_genome_train+visual_genome_val/e2e_faster_rcnn_X-101-64x4d-FPN_2x/RNG_SEED#1989_FAST_RCNN.MLP_HEAD_DIM#2048_FPN.DIM#512/model_final.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output dir name',
        required=True,
        type=str
    )
    parser.add_argument(
        '--det-output-file',
        dest='det_output_file',
        default='log/flickr30k_detection_vg_thresh0.2_feat.h5',
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
        default=10
    )
    parser.add_argument(
        '--max_bboxes',
        help=" min number of bboxes",
        type=int,
        default=100
    )
    parser.add_argument(
        '--feat_name',
        help=" the name of the feature to extract, default: gpu_0/fc7",
        type=str,
        default="gpu_0/fc6"
    )
    parser.add_argument(
        '--list_of_ids',
        type=str,
        default=''
    )

    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_detections_from_im(cfg, model, im, image_id, featmap_blob_name, feat_blob_name ,MIN_BOXES, MAX_BOXES, conf_thresh=0.2, bboxes=None):

    with c2_utils.NamedCudaScope(0):
        scores, cls_boxes, im_scale = infer_engine.im_detect_bbox(model, im,cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=bboxes)
        # box_features = workspace.FetchBlob(feat_blob_name)
        # featmap = workspace.FetchBlob(featmap_blob_name)
        region_feat = workspace.FetchBlob(feat_blob_name)
        cls_prob = workspace.FetchBlob("gpu_0/cls_prob")
        rois = workspace.FetchBlob("gpu_0/rois")
        # print('feat map size: {}, region feature size: {}'.format(featmap.shape, region_feat.shape))
        max_conf = np.zeros((rois.shape[0]))
        # unscale back to raw image space
        cls_boxes = rois[:, 1:5] / im_scale

        for cls_ind in range(1, cls_prob.shape[1]):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

        keep_boxes = np.where(max_conf >= conf_thresh)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
        objects = np.argmax(cls_prob[keep_boxes], axis=1)
        obj_prob = np.amax(cls_prob[keep_boxes], axis=1) # proposal not in order!

    # return box_features[keep_boxes]
    # print('{} ({}x{}): {} boxes, box size {}, feature size {}, class size {}'.format(image_id,
    #       np.size(im, 0), np.size(im, 1), len(keep_boxes), cls_boxes[keep_boxes].shape,
    #       box_features[keep_boxes].shape, objects.shape))
    # print(cls_boxes[keep_boxes][:10, :], objects[:10], obj_prob[:10])

    assert(np.sum(objects>=1601) == 0)
    # assert(np.min(obj_prob[:10])>=0.2)
    # if np.min(obj_prob) < 0.2:
        # print('confidence score too low!', np.min(obj_prob[:10]))
        # pass
    # if np.max(cls_boxes[keep_boxes]) > max(np.size(im, 0), np.size(im, 1)):
    #     print('box is offscreen!', np.max(cls_boxes[keep_boxes]), np.size(im, 0), np.size(im, 1))

    return {
        "image_id": image_id,
        "image_h": np.size(im, 0),
        "image_w": np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': cls_boxes[keep_boxes],
        'region_feat': region_feat[keep_boxes],
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

    with open(args.list_of_ids) as f:
        list_of_folder = json.load(f)

    N = len(list_of_folder)
    fpv = 10
    dets_labels = np.zeros((N, fpv, 100, 6))
    dets_num = np.zeros((N, fpv))
    nms_num = np.zeros((N, fpv))

    # for i, img_id in enumerate(info['images']):
    for i, folder_name in enumerate(list_of_folder):
        # im_base_name = os.path.basename(im_name)
        # image_id = int(im_base_name.split(".")[0].split("_")[2])   ##for COCO
        # image_id = int(im_base_name.split(".")[0])      ##for visual genome
        # out_name =  "COCO_genome_%012d.jpg"%image_id

      # dets_feat = np.zeros((fpv, 100, 2048))
      dets_feat = []
      for j in range(fpv):
        im_name = os.path.join(args.im_or_folder, folder_name, str(j+1).zfill(2)+args.image_ext)
        # print(im_name)

        im = cv2.imread(im_name)
        # image_id = im_name.split('/')[-1][:-4]
        try:
            result = get_detections_from_im(cfg, model, im, '', '', args.feat_name,
                                                   args.min_bboxes, args.max_bboxes)
        except:
            print('missing frame: ', im_name)
            num_frm = j
            break

        # store results
        num_proposal = result['boxes'].shape[0]
        proposals = np.concatenate((result['boxes'], np.expand_dims(result['object'], axis=1),
                                    np.expand_dims(result['obj_prob'], axis=1)), axis=1)

        dets_feat.append(result['region_feat'].squeeze())

        dets_labels[i, j, :num_proposal] = proposals
        dets_num[i, j] = num_proposal
        nms_num[i, j] = num_proposal # for now, treat them the same

      # save features to individual npy files
      feat_output_file = os.path.join(args.output_dir, folder_name+'.npy')
      if len(dets_feat) > 0:
          dets_feat = np.stack(dets_feat)
          print('Processed clip {}, feature shape {}'.format(folder_name, dets_feat.shape))
          np.save(feat_output_file, dets_feat)
          # np.save(feat_output_file, dets_feat[:num_frm])
          # print(feat_output_file)
      else:
          print('Empty feature file! Skipping {}...'.format(folder_name))

      count += 1

      if count % 10 == 0:
          end = timeit.default_timer()
          epoch_time = end - start
          print('process {:d} videos after {:.1f} s'.format(count, epoch_time))

    f = h5py.File(args.det_output_file, "w")
    f.create_dataset("dets_labels", data=dets_labels)
    f.create_dataset("dets_num", data=dets_num)
    f.create_dataset("nms_num", data=nms_num)
    f.close()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    print(args)
    main(args)
