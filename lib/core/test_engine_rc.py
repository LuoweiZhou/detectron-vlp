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

"""A simplified version of test engine specifically for region classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

from caffe2.python import workspace

from core.config import cfg
from core.config import get_output_dir
from core.test_rc import im_classify_bbox
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
from utils.io import save_object
from utils.timer import Timer
import utils.c2 as c2_utils
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils
import utils.blob as blob_utils

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    child_func = test_net
    parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]
    proposal_file = None

    return dataset_name, proposal_file


def run_inference(
    weights_file, ind_range=None,
    multi_gpu_testing=False, gpu_id=0,
    check_expected_results=False,
):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = get_output_dir(training=False, weights=weights_file)
                results = parent_func(
                    weights_file,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.update(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = get_output_dir(training=False, weights=weights_file)
            return child_func(
                weights_file,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        output_dir = get_output_dir(training=False, weights=weights_file)
        task_evaluation.log_copy_paste_friendly_results(all_results, output_dir)

    return all_results


def test_net_on_dataset(
    weights_file,
    dataset_name,
    proposal_file,
    output_dir,
    multi_gpu=False,
    gpu_id=0
):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        all_scores = multi_gpu_test_net_on_dataset(
            weights_file, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_scores = test_net(
            weights_file, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    results = task_evaluation.evaluate_scores(
        dataset, all_scores, output_dir
    )
    return results


def multi_gpu_test_net_on_dataset(
    weights_file, dataset_name, proposal_file, num_images, output_dir
):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, 'test_net' + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(cfg.CFG_FILE, num_images, binary, output_dir, weights_file)

    # Collate the results from each subprocess
    all_scores = []
    for det_data in outputs:
        all_scores_batch = det_data['all_scores']
        all_scores += all_scores_batch

    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_scores=all_scores,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_scores


def test_net(
    weights_file,
    dataset_name,
    proposal_file,
    output_dir,
    ind_range=None,
    gpu_id=0
):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )

    model = initialize_model_from_cfg(weights_file, gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_scores = empty_results(num_images)
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        # just get the ground truth boxes
        box_proposals = entry['boxes'][entry['gt_classes'] > 0]
        if len(box_proposals) == 0:
            cls_scores_i = blob_utils.zeros((0, cfg.MODEL.NUM_CLASSES))
            extend_results(i, all_scores, cls_scores_i)
            continue

        im = cv2.imread(entry['image'])
        with c2_utils.NamedCudaScope(gpu_id):
            cls_scores_i = im_classify_bbox(
                model, im, box_proposals, timers
            )

        extend_results(i, all_scores, cls_scores_i)

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_classify_bbox'].average_time
            )
            misc_time = (
                timers['misc_bbox'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, det_time, misc_time, eta
                )
            )

        # if cfg.VIS:
        #     im_name = os.path.splitext(os.path.basename(entry['image']))[0]
        #     vis_utils.vis_one_image(
        #         im[:, :, ::-1],
        #         '{:d}_{:s}'.format(i, im_name),
        #         os.path.join(output_dir, 'vis'),
        #         cls_boxes_i,
        #         segms=cls_segms_i,
        #         keypoints=cls_keyps_i,
        #         thresh=cfg.VIS_TH,
        #         box_alpha=0.8,
        #         dataset=dataset,
        #         show_class=True
        #     )

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = cfg.CFG_FILE + '_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_scores=all_scores,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_scores


def initialize_model_from_cfg(weights_file, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    creates the networks in the Caffe2 workspace.
    """
    model = model_builder.create(cfg.MODEL.TYPE, train=False, gpu_id=gpu_id)
    net_utils.initialize_gpu_from_weights_file(
        model, weights_file, gpu_id=gpu_id,
    )
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)

    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb(gt=True)

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_images):
    all_scores = [[] for _ in range(num_images)]
    return all_scores


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    all_res[index] = im_res
