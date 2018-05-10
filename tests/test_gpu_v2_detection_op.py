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

from core.config import cfg
from core.config import merge_cfg_from_file
from datasets.json_dataset import JsonDataset
from modeling import model_builder
from core.test_retinanet import im_detect_bbox
import utils.c2
import utils.net
import utils.vis
import utils.boxes as box_utils
import utils.c2

class GPUDetectionOpTest(unittest.TestCase):
    def _test_std(self):
        current_dir = osp.dirname(osp.realpath(__file__))
        cfg_file = osp.join(current_dir, '..', 'configs', 'R-50_1x.yaml')
        merge_cfg_from_file(cfg_file)
        cfg.TEST.WEIGHTS = osp.join(current_dir, '..', 'outputs', 'train', 
                            'coco_2014_train+coco_2014_valminusminival', 
                            'R-50_1x', 'default', 'model_final.pkl')
        cfg.RETINANET.INFERENCE_TH = 0.

        dataset = JsonDataset('coco_2014_minival')
        roidb = dataset.get_roidb()
        model = model_builder.create(cfg.MODEL.TYPE, train=False, gpu_id=0)
        utils.net.initialize_gpu_from_weights_file(model, cfg.TEST.WEIGHTS, gpu_id=0)
        model_builder.add_inference_inputs(model)
        workspace.CreateNet(model.net)
        workspace.CreateNet(model.conv_body_net)
        num_images = len(roidb)
        num_classes = cfg.MODEL.NUM_CLASSES
        entry = roidb[0]
        im = cv2.imread(entry['image'])
        with utils.c2.NamedCudaScope(0):
            cls_boxes, cls_preds, cls_probs, box_preds, anchors, im_info = im_detect_bbox(model, im, debug=True)
        cls_boxes = cls_boxes[:, :5]

        im_name = osp.splitext(osp.basename(entry['image']))[0]
        # utils.vis.vis_one_image(im[:, :, ::-1],
        #                         '{:s}-std-output'.format(im_name),
        #                         current_dir,
        #                         cls_boxes,
        #                         segms=None,
        #                         keypoints=None,
        #                         thresh=0.,
        #                         box_alpha=0.8,
        #                         dataset=dataset,
        #                         show_class=False) 
        workspace.ResetWorkspace()

        return cls_preds, cls_probs, box_preds, anchors, im_info, im, im_name, current_dir, dataset

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

    def _run_select_top_n_op_gpu(self, X, top_n, num_images):
        op_name = 'SelectTopN'
        blobs_in = ['X']
        values_in = [ X ]
        blobs_out = []
        for im in range(num_images):
            blobs_out.extend(['yi_%02d' % im, 'yv_%02d' % im])
        args = {'top_n': 1000}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args)

    def _run_select_top_n_single_op_gpu(self, X, top_n, im):
        op_name = 'SelectTopNSingle'
        blobs_in = ['X']
        values_in = [ X ]
        blobs_out = ['yi', 'yv']
        args = {'top_n': 1000, 'im': im}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args)

    def _run_select_top_n_single_op_cpu(self, X, top_n, im):
        op_name = 'SelectTopNSingle'
        blobs_in = ['X']
        values_in = [ X ]
        blobs_out = ['yi', 'yv']
        args = {'top_n': 1000, 'im': im}
        return self._run_general_op_cpu(op_name, blobs_in, values_in, blobs_out, **args)

    def _run_select_bottom_n_single_op_gpu(self, X, top_n, im):
        op_name = 'SelectBottomNSingle'
        blobs_in = ['X']
        values_in = [ X ]
        blobs_out = ['yi', 'yv']
        args = {'top_n': 1000, 'im': im}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args)

    def _run_boxes_only_op_gpu(self, box_pred, anchor,
                                    yi, yv, im_info, lvl):
        op_name = 'BoxesOnly'
        blobs_in = ['box_pred', 'anchor', 'yi', 'yv', 'im_info']
        values_in = [box_pred, anchor, yi, yv, im_info]
        blobs_out = ['boxes', 'stats']        
        args = {'A': 9, 'im': 0, 'level': lvl, 'num_cls': 80}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_class_based_boxes_only_op_gpu(self, stats, boxes, class_id):
        op_name = 'ClassBasedBoxesOnly'
        blobs_in = ['stats', 'boxes']
        values_in = [stats, boxes]
        blobs_out = ['cls_boxes']
        args = {'class_id': class_id} 
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_nms_op_gpu(self, boxes):
        op_name = 'NMSOnly'
        blobs_in = ['boxes']
        values_in = [boxes]
        blobs_out = ['nms_boxes']
        args = {'nms':0.5, 'dpi': 100}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_nms_op_cpu(self, boxes):
        op_name = 'NMSOnly'
        blobs_in = ['boxes']
        values_in = [boxes]
        blobs_out = ['nms_boxes']
        args = {'nms':0.5, 'dpi': 100}
        return self._run_general_op_cpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_reduce_boxes_only_op_gpu(self, boxes):
        op_name = 'ReduceBoxesOnly'
        blobs_in = ['boxes']
        values_in  = [boxes]
        blobs_out = ['rois', 'stats']
        args = {'im':0, 'dpi':100, }
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_reduce_sum_op_gpu(self, values_in):
        op_name = 'ReduceSumGPU'
        blobs_in = [ 'stats_%d' % i for i in range(len(values_in)) ]
        blobs_out = ['stats']
        args = {}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_get_logits_op_gpu(self, cls_preds, frois):
        op_name = 'GetLogits'
        blobs_in = ['frois']
        values_in = [frois]
        for i, cp in enumerate(cls_preds):
            blobs_in.append('cp_%d' % i)
            values_in.append(cp)
        blobs_out = ['feats']
        args = {'num_feats': 80, 'k_min': 3}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_gpu_testing(self, cls_preds, cls_probs, bbox_preds, anchors, im_info):
        # run select top n op
        num_images = 1
        num_classes = 81
        anchors = [ a.astype(np.float32) for a in anchors.values() ]
        im_info = im_info.astype(np.float32)

        boxes = []
        stats = []

        lvl = 3
        for cr, cp, bp, ac in zip(cls_preds, cls_probs, bbox_preds, anchors):
            # pay attention to the negative sign here!!!!
            picked = self._run_select_bottom_n_single_op_gpu(-cp, 1000, 0)
            yi = picked[0]
            yv = picked[1]

            bfs = self._run_boxes_only_op_gpu(bp, ac, yi, -yv, im_info, lvl)
            boxes.append(bfs[0])
            stats.append(bfs[1])

            lvl += 1

        boxes = np.vstack(boxes)

        stats = self._run_reduce_sum_op_gpu(stats)
        stats = stats[0]

        cls_boxes = []
        for cls_id in range(num_classes-1):
            cfs = self._run_class_based_boxes_only_op_gpu(stats, boxes, cls_id)
            cls_boxes.append(cfs[0])

        nms_boxes = []
        for i in range(1, num_classes):
            nms = self._run_nms_op_cpu(cls_boxes[i-1])
            nms_boxes.append(nms[0])
        
        agg_nms_boxes = np.vstack(nms_boxes)
        pim = self._run_reduce_boxes_only_op_gpu(agg_nms_boxes)
        rois = pim[0]
        stats = pim[1]

        # feats = self._run_get_logits_op_gpu(cls_preds, frois)

        return rois[:, :5]

    def test_res(self):
        cls_preds, cls_probs, box_preds, anchors, im_info, im, im_name, current_dir, dataset = self._test_std()
        rois = self._run_gpu_testing(cls_preds, cls_probs, box_preds, anchors, im_info)
        rois = rois / im_info[0, 2]
        scores = rois[:, [0]]
        scores[:] = 1.
        boxes = rois[:, 1:]
        cls_boxes = np.hstack([boxes, scores])

        utils.vis.vis_one_image(im[:, :, ::-1],
                                '{:s}-gd-output'.format(im_name),
                                current_dir,
                                cls_boxes,
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
    assert 'SelectTopN' in workspace.RegisteredOperators()
    assert 'SelectTopNSingle' in workspace.RegisteredOperators()
    assert 'SelectBottomNSingle' in workspace.RegisteredOperators()
    assert 'BoxesOnly' in workspace.RegisteredOperators()
    assert 'ClassBasedBoxesOnly' in workspace.RegisteredOperators()
    assert 'NMSOnly' in workspace.RegisteredOperators()
    assert 'ReduceBoxesOnly' in workspace.RegisteredOperators()
    assert 'ReduceSumGPU' in workspace.RegisteredOperators()
    unittest.main()