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

"""Defines DetectionModelHelper, the class that represents a Detectron model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging
import os.path as osp

from caffe2.python import cnn
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

from core.config import cfg
from ops.collect_and_distribute_fpn_rpn_proposals \
    import CollectAndDistributeFpnRpnProposalsOp
from ops.generate_proposal_labels import GenerateProposalLabelsOp
from ops.generate_proposals import GenerateProposalsOp
import roi_data.fast_rcnn
import utils.c2 as c2_utils

logger = logging.getLogger(__name__)


class DetectionModelHelper(cnn.CNNModelHelper):
    def __init__(self, **kwargs):
        # Handle args specific to the DetectionModelHelper, others pass through
        # to CNNModelHelper
        self.train = kwargs.get('train', False)
        self.num_classes = kwargs.get('num_classes', -1)
        assert self.num_classes > 0, 'num_classes must be > 0'
        for k in ('train', 'num_classes'):
            if k in kwargs:
                del kwargs[k]
        kwargs['order'] = 'NCHW'
        # Defensively set cudnn_exhaustive_search to False in case the default
        # changes in CNNModelHelper. The detection code uses variable size
        # inputs that might not play nicely with cudnn_exhaustive_search.
        kwargs['cudnn_exhaustive_search'] = False
        super(DetectionModelHelper, self).__init__(**kwargs)
        self.roi_data_loader = None
        self.losses = []
        self.metrics = []
        self.do_not_update_params = []  # Param on this list are not updated
        self.net.Proto().type = cfg.MODEL.EXECUTION_TYPE
        self.net.Proto().num_workers = max(cfg.NUM_GPUS * 4, cfg.NUM_CPUS)
        self.prev_use_cudnn = self.use_cudnn
        self.gn_params = []  # Param on this list are GroupNorm parameters

    def AddSummaryHistogram(self, blob_name):
        if self.writer:
            self.writer.append_histogram(blob_name)

    def AddSummaryImage(self, blob_name):
        if self.writer:
            self.writer.append_image(blob_name)

    def AddSummaryImageBoxes(self, im_name, box_name):
        if self.writer:
            self.writer.append_image_boxes(im_name, box_name)

    def AddSummaryMem(self, mem_name):
        if self.writer:
            self.writer.append_mem(mem_name)

    def ResizeMemoryInit(self):
        blobs_in = ['mem_00/spatial', 'im_info', cfg.MEM.REFER]
        blobs_out = ['mem_00/values']
        mem_init = self.net.ResizeMemoryInit(blobs_in, blobs_out, 
                                        spatial_scale=cfg.MEM.SCALE,
                                        e_value = 0.)
        if cfg.MEM.ACT == 'tanh':
            return self.Tanh(mem_init, mem_init)
        elif cfg.MEM.ACT == 'relu':
            return mem_init
        else:
            raise NotImplementedError

    def ResizeMemoryPyramidInit(self, fpn_blobs):
        scale_inv = float((2 ** cfg.FPN.RPN_MAX_LEVEL))
        mem_blobs = []
        for i, fb in enumerate(fpn_blobs):
            lvl = cfg.FPN.RPN_MAX_LEVEL - i
            scale = 1. / scale_inv
            assert str(lvl) in fb._name
            blobs_in = ['mem_00/spatial', 'im_info', c2_utils.UnscopeGPUName(fb._name)]
            blobs_out = ['mem_00/values_%d' % lvl]
            mem = self.net.ResizeMemoryInit(blobs_in, blobs_out, spatial_scale=scale)
            mem_blobs.append(mem)
            scale_inv /= 2.

        return list(reversed(mem_blobs))

    def ResizeNormalizerInit(self):
        blobs_in = ['mem_00/spatial_normalizer', 'im_info', cfg.MEM.REFER]
        blobs_out = ['mem_00/normalizer']
        return self.net.ResizeMemoryInit(blobs_in, blobs_out, 
                                        spatial_scale=cfg.MEM.SCALE,
                                        e_value=1.)

    def SumConvFC(self, blobs_in, blobs_out):
        return self.net.SumConvFC(blobs_in, blobs_out)

    def MulConvFC(self, blobs_in, blobs_out):
        return self.net.MulConvFC(blobs_in, blobs_out)

    def MulConvGate(self, blobs_in, blobs_out):
        return self.net.MulConvGate(blobs_in, blobs_out)

    def DivConvNorm(self, mem, norm):
        blobs = [mem, norm]
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in blobs ]
        dirname = osp.dirname(blobs_in[0])
        blobs_out = [dirname + '/normalized_assemble']
        return self.net.DivConvNorm(blobs_in, blobs_out)

    def CropAndResize(self, rois, feats):
        blobs = [feats, rois]
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in blobs ]

        dirname = osp.dirname(blobs_in[1])
        basename = osp.basename(blobs_in[0])
        output_name = dirname + '/' + basename + '_crop'
        blobs_out = [ output_name ]

        return model.CropAndResize(blobs_in, 
                                    blobs_out, 
                                    spatial_scale=cfg.MEM.SCALE,
                                    pooled_h=cfg.MEM.CROP_SIZE,
                                    pooled_w=cfg.MEM.CROP_SIZE)

    def InvCropAndResize(self, rois, feats, rfeats):
        blobs = [feats, rois, rfeats]
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in blobs ]

        dirname = osp.dirname(blobs_in[1])
        basename = osp.basename(blobs_in[2])

        output_name = dirname + '/' + basename + '_assemble'
        blobs_out = [ output_name ]

        return self.net.InvCropAndResize(
                            blobs_in, 
                            blobs_out, 
                            spatial_scale=cfg.MEM.SCALE)

    def RoIAlign(self, rois, feats):
        blobs = [feats, rois]
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in blobs ]

        dirname = osp.dirname(blobs_in[1])
        basename = osp.basename(blobs_in[0])
        output_name = dirname + '/' + basename + '_crop'
        blobs_out = [ output_name ]

        return self.net.RoIAlign(blobs_in, 
                                blobs_out, 
                                spatial_scale=cfg.MEM.SCALE,
                                pooled_h=cfg.MEM.CROP_SIZE,
                                pooled_w=cfg.MEM.CROP_SIZE,
                                sampling_ratio=0)

    def RoIAlignList(self, rois_list, feats_list):
        crops = []
        for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL+1):
            ind = lvl - cfg.FPN.RPN_MIN_LEVEL
            blobs = [feats_list[ind], rois_list[ind]]
            blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in blobs ]
            assert str(lvl) in blobs_in[0]
            assert str(lvl) in blobs_in[1]
            dirname = osp.dirname(blobs_in[1])
            basename = osp.basename(blobs_in[0])
            output_name = dirname + '/' + basename + '_crop'
            blobs_out = [ output_name ]
            scale = (0.5)**(lvl)
            crops.append(self.net.RoIAlign(blobs_in, 
                                            blobs_out, 
                                            spatial_scale=scale,
                                            pooled_h=cfg.MEM.CROP_SIZE,
                                            pooled_w=cfg.MEM.CROP_SIZE,
                                            sampling_ratio=0))

        # combine features
        crops = [ c2_utils.UnscopeGPUName(b._name) for b in crops ]
        dirname = osp.commonprefix(crops)
        blobs_out = [dirname + 'crop', dirname + 'crop_split']
        crop = self.net.Concat(crops, blobs_out, axis=0)

        return crop[0]

    def InvRoIAlign(self, rois, feats, rfeats):
        blobs = [feats, rois, rfeats]
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in blobs ]

        output_name = blobs_in[2] + '_assemble'
        blobs_out = [ output_name ]

        return self.net.InvRoIAlign(
                            blobs_in, 
                            blobs_out, 
                            spatial_scale=cfg.MEM.SCALE)

    def ResizeMemoryAs(self, mem, blob, scale, layer):
        blobs = [mem, blob]
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in blobs ]

        dirname = osp.dirname(blobs_in[0])
        basename = 'mem%d' % layer
        output_name = dirname + '/' + basename 
        blobs_out = [ output_name ]

        return self.net.ResizeBilinearAs(
                            blobs_in, 
                            blobs_out, 
                            spatial_scale=scale)

    def ConcatAttention(self, attends_to_concat):
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in attends_to_concat ]

        dirname = 'final/'
        basename = osp.basename(blobs_in[0])
        output_name = dirname + basename + '_concat'
        blobs_out = [ output_name ]

        return self.net.ConcatPlusAttention(blobs_in, 
                                            blobs_out)

    def ConcatAttentionRegion(self, attends_to_concat):
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in attends_to_concat ]

        dirname = 'final/'
        basename = osp.basename(blobs_in[0])
        output_name = dirname + basename + '_concat'
        blobs_out = [ output_name ]

        return self.net.ConcatPlusAttentionRegion(blobs_in, 
                                                blobs_out)

    def ConcatAttentionRegionNormal(self, attends_to_concat):
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in attends_to_concat ]
        dirname = 'final/'
        basename = osp.basename(blobs_in[0])

        blobs_out = [dirname + basename + '_concat', dirname + basename + '_split']
        results = self.net.Concat(blobs_in, blobs_out, axis=1)
        return results[0]

    def AddSpatialSoftmax(self, attends):
        blobs_in = [ c2_utils.UnscopeGPUName(attends._name) ]

        dirname = 'final/'
        basename = osp.basename(blobs_in[0])
        output_name = dirname + basename.replace('_concat', '_softmax')
        blobs_out = [ output_name ]

        return self.net.GroupSpatialSoftmax(
                            blobs_in, 
                            blobs_out, 
                            num_classes=cfg.MEM.ITER+1)

    def ReduceWithAttention(self, blobs, attend):
        blobs_in = [ attend ]
        blobs_in.extend(blobs)
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in blobs_in ]

        dirname = 'final/'
        basename = osp.basename(blobs_in[-1]).replace('_nb', '')
        output_name = dirname + basename
        blobs_out = [ output_name ]

        return self.net.ReduceWithAttention(blobs_in, 
                                            blobs_out, 
                                            iter=cfg.MEM.ITER+1)

    def ReduceWithAttentionRegion(self, blobs, attend):
        blobs_in = [ attend ]
        blobs_in.extend(blobs)
        blobs_in = [ c2_utils.UnscopeGPUName(b._name) for b in blobs_in ]

        dirname = 'final/'
        basename = osp.basename(blobs_in[-1]).replace('_nb', '')
        output_name = dirname + basename
        blobs_out = [ output_name ]

        return self.net.ReduceWithAttentionRegion(blobs_in, blobs_out, 
                                                 iter=cfg.MEM.ITER+1)

    def TrainableParams(self, gpu_id=-1):
        """Get the blob names for all trainable parameters, possibly filtered by
        GPU id.
        """
        return [
            p for p in self.params
            if (
                p in self.param_to_grad and   # p has a gradient
                p not in self.do_not_update_params and  # not on the blacklist
                (gpu_id == -1 or  # filter for gpu assignment, if gpu_id set
                 str(p).find('gpu_{}'.format(gpu_id)) == 0)
            )]

    def AffineChannel(self, blob_in, blob_out, dim, inplace=False):
        """Affine transformation to replace BN in networks where BN cannot be
        used (e.g., because the minibatch size is too small).

        The operations can be done in place to save memory.
        """
        blob_out = blob_out or self.net.NextName()
        param_prefix = blob_out

        scale = self.create_param(
            param_name=param_prefix + '_s',
            initializer=initializers.Initializer("ConstantFill", value=1.),
            tags=ParameterTags.WEIGHT,
            shape=[dim, ],
        )
        bias = self.create_param(
            param_name=param_prefix + '_b',
            initializer=initializers.Initializer("ConstantFill", value=0.),
            tags=ParameterTags.BIAS,
            shape=[dim, ],
        )
        if inplace:
            return self.net.AffineChannel([blob_in, scale, bias], blob_in)
        else:
            return self.net.AffineChannel([blob_in, scale, bias], blob_out)

    def GenerateProposals(self, blobs_in, blobs_out, anchors, spatial_scale):
        """Op for generating RPN porposals.

        blobs_in:
          - 'rpn_cls_probs': 4D tensor of shape (N, A, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].
          - 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.
          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.

        blobs_out:
          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        if cfg.TRAIN.CPP_RPN == 'proposals' or (not self.train and cfg.TRAIN.CPP_RPN == 'all'):
            stride = int(1. / spatial_scale)
            if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
                lvl = int(blobs_in[0][-1])
                anchors = 'anchors_%d' % lvl
                return self.GenerateProposalsCppForLabels(blobs_in, blobs_out, anchors, stride)
            else:
                anchors = 'anchors'
                return self.GenerateProposalsCppForLabels(blobs_in, blobs_out, anchors, stride)
        elif cfg.TRAIN.CPP_RPN == 'all':
            stride = int(1. / spatial_scale)
            blobs_out_ims = []
            for im in range(cfg.TRAIN.IMS_PER_BATCH):
                blobs_out_ims.extend([ b + '_%02d' % im for b in blobs_out ])
            if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
                lvl = int(blobs_in[0][-1])
                anchors = 'anchors_%d' % lvl
                return self.GenerateProposalsCpp(blobs_in, blobs_out_ims, anchors, stride)
            else:
                anchors = 'anchors'
                return self.GenerateProposalsCpp(blobs_in, blobs_out_ims, anchors, stride)
        elif cfg.TRAIN.CPP_RPN == 'none':
            return self.GenerateProposalsPython(blobs_in, blobs_out, anchors, spatial_scale)
        else:
            raise NotImplementedError

    def GenerateProposalsPython(self, blobs_in, blobs_out, anchors, spatial_scale):
        name = 'GenerateProposalsOp:' + ','.join([str(b) for b in blobs_in])
        # spatial_scale passed to the Python op is only used in convert_pkl_to_pb
        # only deals with a single layer, which generates multiple proposals
        self.net.Python(
            GenerateProposalsOp(anchors, spatial_scale, self.train).forward
        )(blobs_in, blobs_out, name=name, spatial_scale=spatial_scale)
        return blobs_out

    def GenerateProposalsCpp(self, blobs_in, blobs_out, anchors, stride):
        rpn_scope = 'rpn/'
        num_images = cfg.TRAIN.IMS_PER_BATCH if self.train else 1
        cp = blobs_in[0]
        bp = blobs_in[1]
        info = blobs_in[2]
        
        self.net.CopyGPUToCPU(cp, cp + '_host')
        self.net.CopyGPUToCPU(bp, bp + '_host') 
        self.net.CopyGPUToCPU(info, info + '_host')

        if self.train:
            pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N
            post_nms_topN = cfg.TRAIN.RPN_POST_NMS_TOP_N
            nms_thresh = cfg.TRAIN.RPN_NMS_THRESH
            num_images = cfg.TRAIN.IMS_PER_BATCH
        else:
            pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N
            post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N
            nms_thresh = cfg.TEST.RPN_NMS_THRESH
            num_images = 1

        rois = []
        roi_probs = []
        for im in range(num_images):
            blobs_out_im = [rpn_scope + blobs_out[im*2],
                            rpn_scope + blobs_out[im*2+1]]
            with c2_utils.CpuScope():
                self.net.GenerateProposalsSingleImage([cp + '_host', bp + '_host', anchors, info + '_host'], 
                                                      [blob + '_host' for blob in blobs_out_im], 
                                                      pre_top_n=pre_nms_topN,
                                                      post_top_n=post_nms_topN,
                                                      nms_thresh=nms_thresh,
                                                      im=im,
                                                      stride=stride)
            rois.append(blobs_out_im[0] + '_host')
            roi_probs.append(blobs_out_im[1] + '_host')

        return blobs_out

    def GenerateProposalsCppForLabels(self, blobs_in, blobs_out, anchors, stride):
        rpn_scope = 'rpn/'
        num_images = cfg.TRAIN.IMS_PER_BATCH if self.train else 1
        cp = blobs_in[0]
        bp = blobs_in[1]
        info = blobs_in[2]
        
        self.net.CopyGPUToCPU(cp, cp + '_host')
        self.net.CopyGPUToCPU(bp, bp + '_host') 
        self.net.CopyGPUToCPU(info, info + '_host')

        if self.train:
            pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N
            post_nms_topN = cfg.TRAIN.RPN_POST_NMS_TOP_N
            nms_thresh = cfg.TRAIN.RPN_NMS_THRESH
            num_images = cfg.TRAIN.IMS_PER_BATCH
        else:
            pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N
            post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N
            nms_thresh = cfg.TEST.RPN_NMS_THRESH
            num_images = 1

        rois = []
        roi_probs = []
        for im in range(num_images):
            blobs_out_im = [rpn_scope + blobs_out[0] + '_%02d' % im,
                            rpn_scope + blobs_out[1] + '_%02d' % im]
            with c2_utils.CpuScope():
                self.net.GenerateProposalsSingleImage([cp + '_host', bp + '_host', anchors, info + '_host'], 
                                                      [blob + '_host' for blob in blobs_out_im], 
                                                      pre_top_n=pre_nms_topN,
                                                      post_top_n=post_nms_topN,
                                                      nms_thresh=nms_thresh,
                                                      im=im,
                                                      stride=stride)
            rois.append(blobs_out_im[0] + '_host')
            roi_probs.append(blobs_out_im[1] + '_host')

        # then just combine all of them
        rois_out = [rpn_scope + blobs_out[0] + '_host',
                    rpn_scope + blobs_out[0] + '_split']
        roi_probs_out = [rpn_scope + blobs_out[1] + '_host',
                        rpn_scope + blobs_out[1] + '_split']
        with c2_utils.CpuScope():
            self.net.Concat(rois, rois_out, axis=0)
            self.net.Concat(roi_probs, roi_probs_out, axis=0)

        # copy it back
        self.net.CopyCPUToGPU(rpn_scope + blobs_out[0] + '_host', blobs_out[0])
        self.net.CopyCPUToGPU(rpn_scope + blobs_out[1] + '_host', blobs_out[1])
        # self.net.Alias(rpn_scope + blobs_out[0] + '_host', blobs_out[0])
        # self.net.Alias(rpn_scope + blobs_out[1] + '_host', blobs_out[1])

        return blobs_out

    def GenerateProposalLabels(self, blobs_in):
        """Op for generating training labels for RPN proposals. This is used
        when training RPN jointly with Fast/Mask R-CNN (as in end-to-end
        Faster R-CNN training).

        blobs_in:
          - 'rpn_rois': 2D tensor of RPN proposals output by GenerateProposals
          - 'roidb': roidb entries that will be labeled
          - 'im_info': See GenerateProposals doc.

        blobs_out:
          - (variable set of blobs): returns whatever blobs are required for
            training the model. It does this by querying the data loader for
            the list of blobs that are needed.
        """
        name = 'GenerateProposalLabelsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # The list of blobs is not known before run-time because it depends on
        # the specific model being trained. Query the data loader to get the
        # list of output blob names.
        blobs_out = roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=self.train
        )
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        self.net.Python(GenerateProposalLabelsOp().forward)(
            blobs_in, blobs_out, name=name
        )
        return blobs_out

    def GenerateProposalLabelsCpp(self):
        num_images = cfg.TRAIN.IMS_PER_BATCH
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        # do the label calculation separately, then merge them
        rpn_rois_list = ['rpn/rpn_rois_%02d' % im for im in range(num_images)]
        gt_boxes_list = ['gt_boxes_%02d' % im for im in range(num_images)]
        gt_classes_list = ['gt_classes_%02d' % im for im in range(num_images)]

        rois_list = ['rois_%02d' % im for im in range(num_images)]
        labels_list = ['labels_int32_%02d' % im for im in range(num_images)]
        bbox_targets_list = ['bbox_targets_%02d' % im for im in range(num_images)]
        bbox_inside_weights_list = ['bbox_inside_weights_%02d' % im for im in range(num_images)]
        bbox_outside_weights_list = ['bbox_outside_weights_%02d' % im for im in range(num_images)]
        for im in range(num_images):
            with c2_utils.CpuScope():
                self.net.GenerateProposalLabelsSingleImage([rpn_rois_list[im] + '_host', 
                                                            gt_boxes_list[im], 
                                                            gt_classes_list[im], 
                                                            'im_info_host'], 
                                                          [rois_list[im] + '_host', 
                                                           labels_list[im] + '_host', 
                                                           bbox_targets_list[im] + '_host', 
                                                           bbox_inside_weights_list[im] + '_host', 
                                                           bbox_outside_weights_list[im] + '_host'], 
                                                          num_classes=self.num_classes,
                                                          rois_per_image=rois_per_image,
                                                          fg_rois_per_image=fg_rois_per_image,
                                                          fg_thresh=cfg.TRAIN.FG_THRESH,
                                                          bg_thresh_hi=cfg.TRAIN.BG_THRESH_HI,
                                                          bg_thresh_lo=cfg.TRAIN.BG_THRESH_LO,
                                                          im=im,
                                                          rng_seed=cfg.RNG_SEED)
        with c2_utils.CpuScope():
            rois_out = ['rois_host', 'rois_split_host']
            self.net.Concat([b + '_host' for b in rois_list], rois_out, axis=0)
            labels_out = ['labels_int32_host', 'labels_int32_split_host']
            self.net.Concat([b + '_host' for b in labels_list], labels_out, axis=0)
            bbox_targets_out = ['bbox_targets_host', 'bbox_targets_split_host']
            self.net.Concat([b + '_host' for b in bbox_targets_list], bbox_targets_out, axis=0)
            bbox_inside_weights_out = ['bbox_inside_weights_host', 'bbox_inside_weights_split_host']
            self.net.Concat([b + '_host' for b in bbox_inside_weights_list], bbox_inside_weights_out, axis=0)
            bbox_outside_weights_out = ['bbox_outside_weights_host', 'bbox_outside_weights_split_host']
            self.net.Concat([b + '_host' for b in bbox_outside_weights_list], bbox_outside_weights_out, axis=0)

        self.net.CopyCPUToGPU('rois_host', 'rois')
        self.net.CopyCPUToGPU('labels_int32_host', 'labels_int32')
        self.net.CopyCPUToGPU('bbox_targets_host', 'bbox_targets')
        self.net.CopyCPUToGPU('bbox_inside_weights_host', 'bbox_inside_weights')
        self.net.CopyCPUToGPU('bbox_outside_weights_host', 'bbox_outside_weights')

    def GenerateProposalLabelsCppV2(self, use_gpu=True):
        num_images = cfg.TRAIN.IMS_PER_BATCH
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        # do the label calculation separately, then merge them
        rpn_rois_list = ['rpn/rpn_rois_%02d' % im for im in range(num_images)]
        gt_boxes_list = ['gt_boxes_%02d' % im for im in range(num_images)]
        gt_classes_list = ['gt_classes_%02d' % im for im in range(num_images)]

        if use_gpu:
            for im in range(num_images):
                self.net.CopyCPUToGPU(gt_boxes_list[im], gt_boxes_list[im] + '_gpu')
                self.net.CopyCPUToGPU(gt_classes_list[im], gt_classes_list[im] + '_gpu')

        rois_list = ['rois_%02d' % im for im in range(num_images)]
        labels_list = ['labels_int32_%02d' % im for im in range(num_images)]
        targets_list = ['targets_%02d' % im for im in range(num_images)]
        bbox_targets_list = ['bbox_targets_%02d' % im for im in range(num_images)]
        bbox_inside_weights_list = ['bbox_inside_weights_%02d' % im for im in range(num_images)]
        bbox_outside_weights_list = ['bbox_outside_weights_%02d' % im for im in range(num_images)]
        for im in range(num_images):
            with c2_utils.CpuScope():
                self.net.GenerateProposalLabelsRoIsOnly([rpn_rois_list[im] + '_host', 
                                                        gt_boxes_list[im], 
                                                        gt_classes_list[im]], 
                                                        [rois_list[im] + '_host', 
                                                           labels_list[im] + '_host', 
                                                           targets_list[im] + '_host'], 
                                                          rois_per_image=rois_per_image,
                                                          fg_rois_per_image=fg_rois_per_image,
                                                          fg_thresh=cfg.TRAIN.FG_THRESH,
                                                          bg_thresh_hi=cfg.TRAIN.BG_THRESH_HI,
                                                          bg_thresh_lo=cfg.TRAIN.BG_THRESH_LO,
                                                          im=im,
                                                          rng_seed=cfg.RNG_SEED)
            if use_gpu:
                self.net.CopyCPUToGPU(rois_list[im] + '_host', rois_list[im])
                self.net.CopyCPUToGPU(labels_list[im] + '_host', labels_list[im])
                self.net.CopyCPUToGPU(targets_list[im] + '_host', targets_list[im])
                # then just do the bounding box regression part separately
                self.net.ComputeBoxTargets([rois_list[im], gt_boxes_list[im] + '_gpu', 
                                            labels_list[im], targets_list[im]] + '_gpu',
                                            [bbox_targets_list[im],
                                            bbox_inside_weights_list[im],
                                            bbox_outside_weights_list[im]],
                                            num_classes=self.num_classes)
            else:
                with c2_utils.CpuScope():
                    self.net.ComputeBoxTargets([rois_list[im] + '_host', gt_boxes_list[im] + '_host', 
                                            labels_list[im]  + '_host', targets_list[im]  + '_host'],
                                            [bbox_targets_list[im] + '_host',
                                            bbox_inside_weights_list[im] + '_host',
                                            bbox_outside_weights_list[im] + '_host'],
                                            num_classes=self.num_classes)

        if use_gpu:
            # should do the concatenation here
            rois_out = ['rois', 'rois_split']
            self.net.Concat(rois_list, rois_out, axis=0)
            labels_out = ['labels_int32', 'labels_int32_split']
            self.net.Concat(labels_list, labels_out, axis=0)
            bbox_targets_out = ['bbox_targets', 'bbox_targets_split']
            self.net.Concat(bbox_targets_list, bbox_targets_out, axis=0)
            bbox_inside_weights_out = ['bbox_inside_weights', 'bbox_inside_weights_split']
            self.net.Concat(bbox_inside_weights_list, bbox_inside_weights_out, axis=0)
            bbox_outside_weights_out = ['bbox_outside_weights', 'bbox_outside_weights_split']
            self.net.Concat(bbox_outside_weights_list, bbox_outside_weights_out, axis=0)
        else:
            with c2_utils.CpuScope():
                rois_out = ['rois_host', 'rois_split_host']
                self.net.Concat([b + '_host' for b in rois_list], rois_out, axis=0)
                labels_out = ['labels_int32_host', 'labels_int32_split_host']
                self.net.Concat([b + '_host' for b in labels_list], labels_out, axis=0)
                bbox_targets_out = ['bbox_targets_host', 'bbox_targets_split_host']
                self.net.Concat([b + '_host' for b in bbox_targets_list], bbox_targets_out, axis=0)
                bbox_inside_weights_out = ['bbox_inside_weights_host', 'bbox_inside_weights_split_host']
                self.net.Concat([b + '_host' for b in bbox_inside_weights_list], bbox_inside_weights_out, axis=0)
                bbox_outside_weights_out = ['bbox_outside_weights_host', 'bbox_outside_weights_split_host']
                self.net.Concat([b + '_host' for b in bbox_outside_weights_list], bbox_outside_weights_out, axis=0)

            self.net.CopyCPUToGPU('rois_host', 'rois')
            self.net.CopyCPUToGPU('labels_int32_host', 'labels_int32')
            self.net.CopyCPUToGPU('bbox_targets_host', 'bbox_targets')
            self.net.CopyCPUToGPU('bbox_inside_weights_host', 'bbox_inside_weights')
            self.net.CopyCPUToGPU('bbox_outside_weights_host', 'bbox_outside_weights')

    def CollectAndDistributeFpnRpnProposalsCpp(self):
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL
        num_images = cfg.TRAIN.IMS_PER_BATCH
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        # collect all the rois from different levels
        for im in range(num_images):
            rpn_rois = ['rpn/rpn_rois_fpn%d_%02d_host' % (lvl, im) for lvl in range(k_min, k_max+1)]
            rpn_rois_out = ['rpn_rois_%02d_host' % im, 'rpn_rois_split_%02d_host' % im]
            with c2_utils.CpuScope():
                self.net.Concat(rpn_rois, rpn_rois_out, axis=0)
                # self.net.Concat(rpn_roi_probs, rpn_roi_probs_out, axis=0)

        # do the label calculation separately, then merge them
        rpn_rois_list = ['rpn_rois_%02d' % im for im in range(num_images)]
        gt_boxes_list = ['gt_boxes_%02d' % im for im in range(num_images)]
        gt_classes_list = ['gt_classes_%02d' % im for im in range(num_images)]

        rois_list = ['rois_%02d' % im for im in range(num_images)]
        labels_list = ['labels_int32_%02d' % im for im in range(num_images)]
        bbox_targets_list = ['bbox_targets_%02d' % im for im in range(num_images)]
        bbox_inside_weights_list = ['bbox_inside_weights_%02d' % im for im in range(num_images)]
        bbox_outside_weights_list = ['bbox_outside_weights_%02d' % im for im in range(num_images)]
        for im in range(num_images):
            with c2_utils.CpuScope():
                self.net.GenerateProposalLabelsSingleImage([rpn_rois_list[im] + '_host', 
                                                            gt_boxes_list[im], 
                                                            gt_classes_list[im], 
                                                            'im_info_host'], 
                                                          [rois_list[im] + '_host', 
                                                           labels_list[im] + '_host', 
                                                           bbox_targets_list[im] + '_host', 
                                                           bbox_inside_weights_list[im] + '_host', 
                                                           bbox_outside_weights_list[im] + '_host'], 
                                                          num_classes=self.num_classes,
                                                          rois_per_image=rois_per_image,
                                                          fg_rois_per_image=fg_rois_per_image,
                                                          fg_thresh=cfg.TRAIN.FG_THRESH,
                                                          bg_thresh_hi=cfg.TRAIN.BG_THRESH_HI,
                                                          bg_thresh_lo=cfg.TRAIN.BG_THRESH_LO,
                                                          im=im,
                                                          rng_seed=cfg.RNG_SEED)

        # should do the concatenation here
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL

        rois_final = ['rois_idx_restore_int32']
        for lvl in range(k_min, k_max+1):
            rois_final += ['rois_fpn' + str(lvl)]

        with c2_utils.CpuScope():
            rois_out = ['rois_host', 'rois_split_host']
            self.net.Concat([rois + '_host' for rois in rois_list], rois_out, axis=0)
            self.net.DistributeFPN(['rois_host'],
                                    [rois + '_host' for rois in rois_final],
                                    k_min=k_min,
                                    k_max=k_max,
                                    c_scale=cfg.FPN.ROI_CANONICAL_SCALE,
                                    c_level=cfg.FPN.ROI_CANONICAL_LEVEL)

        for rois in rois_final:
            self.net.CopyCPUToGPU(rois + '_host', rois)
        self.net.CopyCPUToGPU('rois_host', 'rois')

        with c2_utils.CpuScope():
            labels_out = ['labels_int32_host', 'labels_int32_split_host']
            self.net.Concat([b + '_host' for b in labels_list], labels_out, axis=0)
            bbox_targets_out = ['bbox_targets_host', 'bbox_targets_split_host']
            self.net.Concat([b + '_host' for b in bbox_targets_list], bbox_targets_out, axis=0)
            bbox_inside_weights_out = ['bbox_inside_weights_host', 'bbox_inside_weights_split_host']
            self.net.Concat([b + '_host' for b in bbox_inside_weights_list], bbox_inside_weights_out, axis=0)
            bbox_outside_weights_out = ['bbox_outside_weights_host', 'bbox_outside_weights_split_host']
            self.net.Concat([b + '_host' for b in bbox_outside_weights_list], bbox_outside_weights_out, axis=0)

        self.net.CopyCPUToGPU('labels_int32_host', 'labels_int32')
        self.net.CopyCPUToGPU('bbox_targets_host', 'bbox_targets')
        self.net.CopyCPUToGPU('bbox_inside_weights_host', 'bbox_inside_weights')
        self.net.CopyCPUToGPU('bbox_outside_weights_host', 'bbox_outside_weights')

    def CollectAndDistributeFpnRpnProposals(self):
        """Merge RPN proposals generated at multiple FPN levels and then
        distribute those proposals to their appropriate FPN levels. An anchor
        at one FPN level may predict an RoI that will map to another level,
        hence the need to redistribute the proposals.

        This function assumes standard blob names for input and output blobs.

        Input blobs: [rpn_rois_fpn<min>, ..., rpn_rois_fpn<max>,
                      rpn_roi_probs_fpn<min>, ..., rpn_roi_probs_fpn<max>]
          - rpn_rois_fpn<i> are the RPN proposals for FPN level i; see rpn_rois
            documentation from GenerateProposals.
          - rpn_roi_probs_fpn<i> are the RPN objectness probabilities for FPN
            level i; see rpn_roi_probs documentation from GenerateProposals.

        If used during training, then the input blobs will also include:
          [roidb, im_info] (see GenerateProposalLabels).

        Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                       rois_idx_restore]
          - rois_fpn<i> are the RPN proposals for FPN level i
          - rois_idx_restore is a permutation on the concatenation of all
            rois_fpn<i>, i=min...max, such that when applied the RPN RoIs are
            restored to their original order in the input blobs.

        If used during training, then the output blobs will also include:
          [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights].
        """
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL

        # Prepare input blobs
        rois_names = ['rpn_rois_fpn' + str(l) for l in range(k_min, k_max+1)]
        score_names = [
            'rpn_roi_probs_fpn' + str(l) for l in range(k_min, k_max+1)
        ]
        blobs_in = rois_names + score_names
        if self.train:
            blobs_in += ['roidb', 'im_info']
        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'CollectAndDistributeFpnRpnProposalsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # Prepare output blobs
        blobs_out = roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=self.train
        )
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        outputs = self.net.Python(
            CollectAndDistributeFpnRpnProposalsOp(self.train).forward
        )(blobs_in, blobs_out, name=name)

        return outputs

    def DropoutIfTraining(self, blob_in, dropout_rate):
        """Add dropout to blob_in if the model is in training mode and
        dropout_rate is > 0."""
        blob_out = blob_in
        if self.train and dropout_rate > 0:
            blob_out = self.Dropout(
                blob_in, blob_in, ratio=dropout_rate, is_test=False
            )
        return blob_out

    def RoIFeatureTransform(
        self,
        blobs_in,
        blob_out,
        blob_rois='rois',
        method='RoIPoolF',
        resolution=7,
        spatial_scale=1. / 16.,
        sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)
        has_argmax = (method == 'RoIPoolF')
        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                bl_out = blob_out + '_fpn' + str(lvl)
                bl_out_list.append(bl_out)
                bl_argmax = ['_argmax_' + bl_out] if has_argmax else []
                self.net.__getattr__(method)(
                    [bl_in, bl_rois], [bl_out] + bl_argmax,
                    pooled_w=resolution,
                    pooled_h=resolution,
                    spatial_scale=sc,
                    sampling_ratio=sampling_ratio
                )
            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled, _ = self.net.Concat(
                bl_out_list, [blob_out + '_shuffled', '_concat_' + blob_out],
                axis=0
            )
            # Unshuffle to match rois from dataloader
            restore_bl = blob_rois + '_idx_restore_int32'
            xform_out = self.net.BatchPermutation(
                [xform_shuffled, restore_bl], blob_out
            )
        else:
            # Single feature level
            bl_argmax = ['_argmax_' + blob_out] if has_argmax else []
            # sampling_ratio is ignored for RoIPoolF
            xform_out = self.net.__getattr__(method)(
                [blobs_in, blob_rois], [blob_out] + bl_argmax,
                pooled_w=resolution,
                pooled_h=resolution,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio
            )
        # Only return the first blob (the transformed features)
        return xform_out

    def ConvShared(self,
                    blob_in,
                    blob_out,
                    dim_in,
                    dim_out,
                    kernel,
                    weight=None,
                    bias=None,
                    **kwargs):
        """Add conv op that shares weights and/or biases with another conv op.
        """
        use_bias = (
            False if ('no_bias' in kwargs and kwargs['no_bias']) else True
        )

        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

        if use_bias:
            blobs_in = [blob_in, weight, bias]
        else:
            blobs_in = [blob_in, weight]

        if 'no_bias' in kwargs:
            del kwargs['no_bias']

        return self.net.Conv(
            blobs_in, blob_out, kernel=kernel, order=self.order, **kwargs
        )

    def FCShared(self,
                blob_in,
                blob_out,
                dim_in,
                dim_out,
                weight=None,
                bias=None,
                **kwargs):
        """Add conv op that shares weights and/or biases with another conv op.
        """
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

        blobs_in = [blob_in, weight, bias]

        return self.net.FC(blobs_in, blob_out, **kwargs)

    def BilinearInterpolation(
        self, blob_in, blob_out, dim_in, dim_out, up_scale
    ):
        """Bilinear interpolation in space of scale.

        Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale

        Adapted from the CVPR'15 FCN code.
        See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
        """
        assert dim_in == dim_out
        assert up_scale % 2 == 0, 'Scale should be even'

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        kernel = np.zeros(
            (dim_in, dim_out, kernel_size, kernel_size), dtype=np.float32
        )
        kernel[range(dim_out), range(dim_in), :, :] = bil_filt

        blob = self.ConvTranspose(
            blob_in,
            blob_out,
            dim_in,
            dim_out,
            kernel_size,
            stride=int(up_scale),
            pad=int(up_scale / 2),
            weight_init=('GivenTensorFill', {'values': kernel}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        self.do_not_update_params.append(self.weights[-1])
        self.do_not_update_params.append(self.biases[-1])
        return blob

    def ConvAffine(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernel, stride, pad,
        group=1, dilation=1,
        weight_init=None,
        bias_init=None,
        suffix='_bn',
        inplace=False):
        """ConvAffine adds a Conv op followed by a AffineChannel op (which
        replaces BN during fine tuning).
        """
        conv_blob = self.Conv(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernel,
            stride=stride,
            pad=pad,
            group=group,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=1
        )
        blob_out = self.AffineChannel(
            conv_blob, prefix + suffix, dim=dim_out, inplace=inplace
        )
        return blob_out

    def ConvGN(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernel, stride, pad,
        group_gn,  # num of groups in gn
        group=1, dilation=1,
        weight_init=None,
        bias_init=None,
        suffix='_gn',
        no_conv_bias=1):
        """ConvGN adds a Conv op followed by a GroupNorm op,
        including learnable scale/bias (gamma/beta)
        """
        conv_blob = self.Conv(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernel,
            stride=stride,
            pad=pad,
            group=group,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=no_conv_bias)

        if group_gn < 1:
            logger.warning(
                'Layer: {} (dim {}): '
                'group_gn < 1; reset to 1.'.format(prefix, dim_in)
            )
            group_gn = 1

        blob_out = self.SpatialGN(
            conv_blob, prefix + suffix,
            dim_out, num_groups=group_gn,
            epsilon=cfg.GROUP_NORM.EPSILON,)

        self.gn_params.append(self.params[-1])  # add gn's bias to list
        self.gn_params.append(self.params[-2])  # add gn's scale to list
        return blob_out

    def DisableCudnn(self):
        self.prev_use_cudnn = self.use_cudnn
        self.use_cudnn = False

    def RestorePreviousUseCudnn(self):
        prev_use_cudnn = self.use_cudnn
        self.use_cudnn = self.prev_use_cudnn
        self.prev_use_cudnn = prev_use_cudnn

    def UpdateWorkspaceLr(self, cur_iter, new_lr):
        """Updates the model's current learning rate and the workspace (learning
        rate and update history/momentum blobs).
        """
        # The workspace is the one source of truth for the lr
        # The lr is always the same on all GPUs
        cur_lr = workspace.FetchBlob('gpu_0/lr')[0]
        # There are no type conversions between the lr in Python and the lr in
        # the GPU (both are float32), so exact comparision is ok
        if cur_lr != new_lr:
            ratio = _get_lr_change_ratio(cur_lr, new_lr)
            if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
                logger.info(
                    'Changing learning rate {:.6f} -> {:.6f} at iter {:d}'.
                    format(cur_lr, new_lr, cur_iter))
            self._SetNewLr(cur_lr, new_lr)
        return new_lr

    def _SetNewLr(self, cur_lr, new_lr):
        """Do the actual work of updating the model and workspace blobs.
        """
        for i in range(cfg.NUM_GPUS):
            with c2_utils.CudaScope(i):
                workspace.FeedBlob(
                    'gpu_{}/lr'.format(i), np.array([new_lr], dtype=np.float32))
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            self._CorrectMomentum(new_lr / cur_lr)

    def _CorrectMomentum(self, correction):
        """The MomentumSGDUpdate op implements the update V as

            V := mu * V + lr * grad,

        where mu is the momentum factor, lr is the learning rate, and grad is
        the stochastic gradient. Since V is not defined independently of the
        learning rate (as it should ideally be), when the learning rate is
        changed we should scale the update history V in order to make it
        compatible in scale with lr * grad.
        """
        logger.info(
            'Scaling update history by {:.6f} (new lr / old lr)'.
            format(correction))
        for i in range(cfg.NUM_GPUS):
            with c2_utils.CudaScope(i):
                for param in self.TrainableParams(gpu_id=i):
                    op = core.CreateOperator(
                        'Scale', [param + '_momentum'], [param + '_momentum'],
                        scale=correction)
                    workspace.RunOperatorOnce(op)

    def GetLossScale(self):
        """Allow a way to configure the loss scale dynamically.

        This may be used in a distributed data parallel setting.
        """
        return 1.0 / cfg.NUM_GPUS

    def AddLosses(self, losses):
        if not isinstance(losses, list):
            losses = [losses]
        # Conversion to str allows losses to include BlobReferences
        losses = [c2_utils.UnscopeGPUName(str(l)) for l in losses]
        self.losses = list(set(self.losses + losses))

    def AddMetrics(self, metrics):
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = list(set(self.metrics + metrics))


def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio
