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

"""Test a RetinaNet network on an image database"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging
from collections import defaultdict

from caffe2.python import core, workspace

from core.config import cfg
from utils.timer import Timer

import utils.blob as blob_utils
import utils.boxes as box_utils
from roi_data.rc import _add_multilevel_rois

logger = logging.getLogger(__name__)


def im_classify_bbox(model, im, box_proposals, timers=None):
    """Generate RetinaNet detections on a single image."""
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    inputs = {}
    inputs['data'], im_scale, inputs['im_info'] = \
        blob_utils.get_image_blob(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    # do something to create the rois

    sampled_rois = box_proposals * inputs['im_info'][0, 2]
    repeated_batch_idx = blob_utils.zeros((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))
    inputs['rois'] = sampled_rois
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(inputs)

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)

    workspace.RunNet(model.net.Proto().name)
    if cfg.MODEL.TYPE == 'region_classification':
        cls_prob = core.ScopedName('cls_prob')
    elif cfg.MODEL.TYPE == 'region_memory':
        cls_prob = core.ScopedName('final/cls_prob')
    else:
        raise NotImplementedError
    cls_scores = workspace.FetchBlob(cls_prob)

    timers['im_detect_bbox'].toc()

    # Combine predictions across all levels and retain the top scoring by class
    timers['misc_bbox'].tic()
    timers['misc_bbox'].toc()

    return cls_scores
