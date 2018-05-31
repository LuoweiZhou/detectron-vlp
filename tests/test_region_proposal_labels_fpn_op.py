from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import unittest
import os.path as osp

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace

from core.config import cfg, assert_and_infer_cfg
from core.config import merge_cfg_from_file
from core.test import _get_blobs
import core.test_engine as test_engine
from datasets.json_dataset import JsonDataset
from modeling.generate_anchors import generate_anchors
import utils.c2
import utils.net
import utils.vis
import utils.boxes as box_utils
import utils.c2

class RegionProposalLabelsOpTest(unittest.TestCase):
    def _test_std(self):
        root_dir = osp.join('/private', 'home', 'xinleic', 'pyramid')
        cfg_file = osp.join(root_dir, 'configs', 'visual_genome', 'e2e_faster_rcnn_R-50-FPN_1x.yaml')
        merge_cfg_from_file(cfg_file)
        cfg.NUM_GPUS = 1
        cfg.TEST.RPN_PRE_NMS_TOP_N = 12000
        cfg.TEST.RPN_POST_NMS_TOP_N = 2000
        assert_and_infer_cfg()
        test_weight = osp.join(root_dir, 'outputs', 'train', 'visual_genome_train', 
                            'e2e_faster_rcnn_R-50-FPN_1x', 'RNG_SEED#3', 'model_final.pkl')
        model = test_engine.initialize_model_from_cfg(test_weight, gpu_id=0)
        dataset = JsonDataset('visual_genome_val')
        roidb = dataset.get_roidb(gt=True)
        num_images = len(roidb)
        num_classes = cfg.MODEL.NUM_CLASSES
        entry = roidb[1]
        im = cv2.imread(entry['image'])
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
        # input: rpn_cls_probs_fpn2, rpn_bbox_pred_fpn2
        # output: rpn_rois_fpn2, rpn_roi_probs_fpn2
        with utils.c2.NamedCudaScope(0):
            # let's manually do the testing here
            inputs, im_scale = _get_blobs(im, None, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

            for k, v in inputs.items():
                workspace.FeedBlob(core.ScopedName(k), v)

            workspace.RunNet(model.net.Proto().name)
            cls_probs = [core.ScopedName('rpn_cls_probs_fpn%d' % i) for i in range(min_level, max_level+1)]
            box_preds = [core.ScopedName('rpn_bbox_pred_fpn%d' % i) for i in range(min_level, max_level+1)]
            rpn_rois = [core.ScopedName('rpn_rois_fpn%d' % i) for i in range(min_level, max_level+1)]
            rpn_roi_probs = [core.ScopedName('rpn_roi_probs_fpn%d' % i) for i in range(min_level, max_level+1)]

            cls_probs = workspace.FetchBlobs(cls_probs)
            box_preds = workspace.FetchBlobs(box_preds)
            rpn_rois = workspace.FetchBlobs(rpn_rois)
            rpn_roi_probs = workspace.FetchBlobs(rpn_roi_probs)

        rpn_rois = np.vstack(rpn_rois)
        rpn_roi_probs = np.vstack(rpn_roi_probs)
        # # remove the image dimension
        # rpn_rois = rpn_rois[:, 1:]
        # boxes = np.hstack([rpn_rois, rpn_roi_probs])
        im_name = osp.splitext(osp.basename(entry['image']))[0]
        # utils.vis.vis_one_image(im[:, :, ::-1],
        #                         '{:s}-std-output'.format(im_name),
        #                         osp.join(root_dir, 'tests'),
        #                         boxes,
        #                         segms=None,
        #                         keypoints=None,
        #                         thresh=0.,
        #                         box_alpha=0.8,
        #                         dataset=dataset,
        #                         show_class=False)

        gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_boxes = entry['boxes'][gt_inds, :] * im_scale
        gt_classes = entry['gt_classes'][gt_inds]
        workspace.ResetWorkspace()
        im_info = inputs['im_info'].astype(np.float32)

        return rpn_rois, rpn_roi_probs, gt_boxes, gt_classes, im_info, im, im_name, root_dir, dataset

    def _run_general_op_gpu(self, op_name, blobs_in, values_in, blobs_out, **kwargs):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator(op_name, blobs_in, blobs_out, **kwargs)
            for name, value in zip(blobs_in, values_in):
                workspace.FeedBlob(name, value)

        workspace.RunOperatorOnce(op)
        values_out = workspace.FetchBlobs(blobs_out)
        workspace.ResetWorkspace()

        return values_out

    def _run_general_op_cpu(self, op_name, blobs_in, values_in, blobs_out, **kwargs):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
            op = core.CreateOperator(op_name, blobs_in, blobs_out, **kwargs)
            for name, value in zip(blobs_in, values_in):
                workspace.FeedBlob(name, value)

        workspace.RunOperatorOnce(op)
        values_out = workspace.FetchBlobs(blobs_out)
        workspace.ResetWorkspace()

        return values_out

    def _run_generate_proposal_labels_single_image_op_cpu(self, rpn_rois, gt_boxes, gt_classes, im_info):
        op_name = 'GenerateProposalLabelsSingleImage'
        blobs_in = ['rpn_rois', 'gt_boxes', 'gt_classes', 'im_info']
        values_in = [rpn_rois, gt_boxes, gt_classes, im_info]
        blobs_out = ['rois', 'labels', 'bbox_targets', 'bbox_inside_weights', 'bbox_outside_weights']
        args = {'num_classes': 1601, 
                'rois_per_image' : cfg.TRAIN.BATCH_SIZE_PER_IM, 
                'fg_rois_per_image': int(np.round(cfg.TRAIN.FG_FRACTION * cfg.TRAIN.BATCH_SIZE_PER_IM)),
                'fg_thresh' : cfg.TRAIN.FG_THRESH,
                'bg_thresh_hi': cfg.TRAIN.BG_THRESH_HI,
                'bg_thresh_lo': cfg.TRAIN.BG_THRESH_LO,
                'im': 0,
                'rng_seed': cfg.RNG_SEED}

        return self._run_general_op_cpu(op_name, blobs_in, values_in, blobs_out, **args)

    def _run_cpu_generate_labels(self, rpn_rois, gt_boxes, gt_classes, im_info):
        blobs_out = self._run_generate_proposal_labels_single_image_op_cpu(rpn_rois, 
                                                                            gt_boxes, 
                                                                            gt_classes, 
                                                                            im_info)
        rois = blobs_out[0]
        labels = blobs_out[1]
        bbox_targets = blobs_out[2]
        bbox_inside_weights = blobs_out[3]
        bbox_outside_weights = blobs_out[4]

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def _run_distribute(self, rois):
        op_name = 'DistributeFPN'
        blobs_in = ['rois']
        values_in = [rois]
        blobs_out = ['rois_idx_restore_int32']
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
        for lvl in range(min_level, max_level + 1):
            blobs_out += ['rois_fpn%d' % lvl]

        args = {'k_min': min_level,
                'k_max': max_level,
                'c_scale': cfg.FPN.ROI_CANONICAL_SCALE,
                'c_level': cfg.FPN.ROI_CANONICAL_LEVEL
        }

        values_out = self._run_general_op_cpu(op_name, blobs_in, values_in, blobs_out, **args)
        rois_idx = values_out[0]

        rois_shuffled = np.vstack(values_out[1:])

        return rois_idx, rois_shuffled

    def test_res(self):
        rpn_rois, rpn_roi_probs, gt_boxes, gt_classes, im_info, im, im_name, root_dir, dataset = self._test_std()
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = self._run_cpu_generate_labels(rpn_rois, gt_boxes, gt_classes, im_info)
        labels = labels.reshape(cfg.TRAIN.BATCH_SIZE_PER_IM, 1)
        rois_idx, rois_shuffled = self._run_distribute(rois)
        assert (rois_shuffled[rois_idx] == rois).all()

        boxes_to_viz = np.hstack([rois[:,1:5], labels.astype(np.float32)])
        utils.vis.vis_one_image(im[:, :, ::-1],
                                '{:s}-lb-output'.format(im_name),
                                osp.join(root_dir, 'tests'),
                                boxes_to_viz,
                                segms=None,
                                keypoints=None,
                                thresh=0.,
                                box_alpha=0.8,
                                dataset=dataset,
                                show_class=False)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_detectron_ops()
    utils.c2.import_custom_ops()
    assert 'GenerateProposalsSingleImage' in workspace.RegisteredOperators()
    assert 'GenerateProposalLabelsSingleImage' in workspace.RegisteredOperators()
    assert 'DistributeFPN' in workspace.RegisteredOperators()
    unittest.main()