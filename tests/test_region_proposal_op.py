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

class RegionProposalOpTest(unittest.TestCase):
    def _test_std(self):
        root_dir = osp.join('/private', 'home', 'xinleic', 'pyramid')
        cfg_file = osp.join(root_dir, 'configs', 'visual_genome', 'e2e_faster_rcnn_R-50-FPN_1x.yaml')
        merge_cfg_from_file(cfg_file)
        cfg.NUM_GPUS = 1
        cfg.TEST.RPN_PRE_NMS_TOP_N = 100
        cfg.TEST.RPN_POST_NMS_TOP_N = 20
        assert_and_infer_cfg()
        test_weight = osp.join(root_dir, 'outputs', 'train', 'visual_genome_train', 
                            'e2e_faster_rcnn_R-50-FPN_1x', 'RNG_SEED#3', 'model_final.pkl')
        model = test_engine.initialize_model_from_cfg(test_weight, gpu_id=0)
        dataset = JsonDataset('visual_genome_val')
        roidb = dataset.get_roidb()
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
        # remove the image dimension
        rpn_rois = rpn_rois[:, 1:]
        boxes = np.hstack([rpn_rois, rpn_roi_probs])
        im_name = osp.splitext(osp.basename(entry['image']))[0]
        utils.vis.vis_one_image(im[:, :, ::-1],
                                '{:s}-std-output'.format(im_name),
                                osp.join(root_dir, 'tests'),
                                boxes,
                                segms=None,
                                keypoints=None,
                                thresh=0.,
                                box_alpha=0.8,
                                dataset=dataset,
                                show_class=False) 
        workspace.ResetWorkspace()
        im_info = inputs['im_info'].astype(np.float32)

        return cls_probs, box_preds, im_info, im, im_name, root_dir, dataset

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

    def _run_generate_proposals_single_image_op_cpu(self, cls_prob, bbox_prob, anchor, im_info, stride, im):
        op_name = 'GenerateProposalsSingleImage'
        blobs_in = ['cls_prob', 'bbox_prob', 'anchor', 'im_info']
        values_in = [ cls_prob, bbox_prob, anchor, im_info ]
        blobs_out = [ 'rois', 'roi_probs' ]
        args = {'pre_top_n': 100, 
                'post_top_n' : 20, 
                'nms' : cfg.TEST.RPN_NMS_THRESH,
                'im': im, 
                'stride' : stride}
        return self._run_general_op_cpu(op_name, blobs_in, values_in, blobs_out, **args)

    def _run_cpu_testing(self, cls_probs, bbox_preds, anchors, im_info):
        # run each of the levels separately
        rpn_rois = []
        rpn_roi_probs = []
        stride = 2 ** cfg.FPN.RPN_MIN_LEVEL
        for cp, bp, ac in zip(cls_probs, bbox_preds, anchors):
            blobs_out = self._run_generate_proposals_single_image_op_cpu(cp, bp, ac, im_info, stride, 0)
            rpn_roi = blobs_out[0]
            rpn_roi_prob = blobs_out[1]
            stride *= 2
            rpn_rois.append(rpn_roi)
            rpn_roi_probs.append(rpn_roi_prob)
        rpn_rois = np.vstack(rpn_rois)
        rpn_roi_probs = np.vstack(rpn_roi_probs)
        rpn_rois = rpn_rois[:, 1:]
        boxes = np.hstack([rpn_rois, rpn_roi_probs])

        return boxes

    def test_res(self):
        cls_probs, box_preds, im_info, im, im_name, root_dir, dataset = self._test_std()
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
        # then build the anchors
        anchors = []
        for lvl in range(min_level, max_level+1):
            field_stride = 2 ** lvl
            anchor_sizes = (cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - min_level), )
            anchor_aspect_ratios = cfg.FPN.RPN_ASPECT_RATIOS
            anchor = generate_anchors(
                stride=field_stride,
                sizes=anchor_sizes,
                aspect_ratios=anchor_aspect_ratios
            )
            anchors.append(anchor.astype(np.float32))

        boxes = self._run_cpu_testing(cls_probs, box_preds, anchors, im_info)

        utils.vis.vis_one_image(im[:, :, ::-1],
                                '{:s}-cd-output'.format(im_name),
                                osp.join(root_dir, 'tests'),
                                boxes,
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
    unittest.main()