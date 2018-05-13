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

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
from utils.net import get_group_gn
import utils.blob as blob_utils

from caffe2.python.modeling import initializers


# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_fast_rcnn_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    model.FC(
        blob_in,
        'bbox_pred',
        dim,
        model.num_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )

    if cfg.MODEL.ATTR and model.train:
        in_dim = dim
        if cfg.MODEL.CLS_EMBED:
            # first slice the fc7 feature
            model.net.SelectFG([blob_in, 'fg_idx'], 'fc7_fg')
            model.create_param(param_name='class_embedding',
                               initializer=initializers.Initializer("GaussianFill", std=0.01),
                               shape=[model.num_classes, 256])
            # op that just takes the class index and returns the corresponding row
            model.net.Embed(['class_embedding', 'labels_int32_fg'], 'embed_fg')
            # then do concatenation
            model.net.Concat(['fc7_fg','embed_fg'],
                            ['concat_attr', 'concat_split'], axis=1)
            in_dim += 256
        else:
            model.net.SelectFG([blob_in, 'fg_idx'], 'concat_attr')

        model.FC('concat_attr',
                'fc_attr',
                in_dim,
                512,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0))
        model.Relu('fc_attr', 'fc_attr')
        model.FC('fc_attr',
                'attr_score',
                512,
                model.num_attributes,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0))

def add_fast_rcnn_outputs_class_only(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')

def add_fast_rcnn_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale()
    )
    # add attribute loss
    if cfg.MODEL.ATTR:
        attr_prob, loss_attr = model.net.SoftmaxAttr(
            ['attr_score', 'attr_labels_int32'],
            ['attr_prob', 'loss_attr'],
            scale=model.GetLossScale() * cfg.MODEL.LOSS_ATTR,
            ignore=-1
        )
    loss_bbox = model.net.SmoothL1Loss(
        [
            'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox',
        scale=model.GetLossScale()
    )
    if cfg.MODEL.ATTR:
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_attr, loss_bbox])
        model.AddLosses(['loss_cls', 'loss_attr', 'loss_bbox'])
    else:
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
        model.AddLosses(['loss_cls', 'loss_bbox'])

    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddMetrics('accuracy_cls')
    return loss_gradients

def add_fast_rcnn_losses_class_only(model):
    """Add losses for RoI classification and bounding box regression."""
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls])
    model.AddLosses(['loss_cls'])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddMetrics('accuracy_cls')
    return loss_gradients

# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim

def add_roi_mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    fc = model.Relu('fc6', 'fc6')

    # add more layers:
    for i in range(1, cfg.FAST_RCNN.NUM_LAYERS):
        name = 'fc%d' % (i+6)
        model.FC(fc, name, hidden_dim, hidden_dim)
        fc = model.Relu(name, name)

    return name, hidden_dim


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim
