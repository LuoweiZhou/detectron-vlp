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
        entry = roidb[5]
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

    def _run_boxes_and_feats_op_gpu(self, cls_pred, box_pred, anchor,
                                    yi, yv, im_info, lvl):
        op_name = 'BoxesAndFeats'
        blobs_in = ['cls_pred', 'box_pred', 'anchor', 'yi', 'yv', 'im_info']
        values_in = [cls_pred, box_pred, anchor, yi, yv, im_info]
        blobs_out = ['boxes', 'feats', 'stats']        
        args = {'A': 9, 'im': 0, 'level': lvl}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_class_based_boxes_op_gpu(self, stats, boxes, feats, num_classes):
        op_name = 'ClassBasedBoxes'
        blobs_in = ['stats', 'boxes', 'feats']
        values_in = [stats, boxes, feats]
        blobs_out = []
        for i in range(1, num_classes):
            blobs_out.extend(['boxes_%02d' % i, 'feats_%02d' % i])
        args = {} 
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_class_based_boxes_single_op_gpu(self, stats, boxes, feats, class_id):
        op_name = 'ClassBasedBoxesSingle'
        blobs_in = ['stats', 'boxes', 'feats']
        values_in = [stats, boxes, feats]
        blobs_out = ['cls_boxes', 'cls_feats']
        args = {'class_id': class_id} 
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_nms_op_gpu(self, boxes, feats):
        op_name = 'NMS'
        blobs_in = ['boxes', 'feats']
        values_in = [boxes, feats]
        blobs_out = ['nms_boxes', 'nms_feats']
        args = {'nms':0.5, 'dpi': 100}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_nms_op_cpu(self, boxes, feats):
        op_name = 'NMS'
        blobs_in = ['boxes', 'feats']
        values_in = [boxes, feats]
        blobs_out = ['nms_boxes', 'nms_feats']
        args = {'nms':0.5, 'dpi': 100}
        return self._run_general_op_cpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_reduce_boxes_and_feats_op_gpu(self, boxes, feats):
        op_name = 'ReduceBoxesAndFeats'
        blobs_in = ['boxes', 'feats']
        values_in  = [boxes, feats]
        blobs_out = ['rois', 'feats_out']
        args = {'im':0, 'dpi':100}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_reduce_sum_op_gpu(self, values_in):
        op_name = 'ReduceSumGPU'
        blobs_in = [ 'stats_%d' % i for i in range(len(values_in)) ]
        blobs_out = ['stats']
        args = {}
        return self._run_general_op_gpu(op_name, blobs_in, values_in, blobs_out, **args) 

    def _run_gpu_testing(self, cls_preds, cls_probs, bbox_preds, anchors, im_info):
        # run select top n op
        num_images = 1
        num_classes = 81
        anchors = [ a.astype(np.float32) for a in anchors.values() ]
        im_info = im_info.astype(np.float32)

        boxes = []
        feats = []
        stats = []

        lvl = 3
        for cr, cp, bp, ac in zip(cls_preds, cls_probs, bbox_preds, anchors):
            picked = self._run_select_top_n_single_op_gpu(-cp, 1000, 0)
            yi = picked[0]
            yv = picked[1]

            # Code for testing the top k selection
            # cls_prob_ravel = cp.ravel()
            # inds = np.argpartition(cls_prob_ravel, -1000)[-1000:]
            # inds_act = np.sort(yi)
            # inds.sort()
            # yvv = cp.ravel()[yi]

            # np.testing.assert_allclose(inds_act, inds)
            # np.testing.assert_allclose(-yvv, yv)

            bfs = self._run_boxes_and_feats_op_gpu(cr, bp, ac, yi, yv, im_info, lvl)
            boxes.append(bfs[0])
            feats.append(bfs[1])
            stats.append(bfs[2])

            lvl += 1

        boxes = np.vstack(boxes)
        feats = np.vstack(feats)

        # stats_exp = np.vstack(stats)
        # stats_exp = np.sum(stats_exp, axis=0)
        # stats_exp = stats_exp.astype(np.int32)

        stats = self._run_reduce_sum_op_gpu(stats)
        stats = stats[0]

        # cfs = self._run_class_based_boxes_op_gpu(stats, boxes, feats, num_classes)
        # cls_boxes = []
        # cls_feats = []
        # for i in range(1, num_classes):
        #     cls_boxes.append(cfs[i*2-2])
        #     cls_feats.append(cfs[i*2-1])

            # Code for testing the class dispatching
            # inds = np.where(boxes[:,5] == i-1)[0]
            # res = boxes[inds,:5]
            # inds = np.argsort(-res[:, 4])
            # res = res[inds,:]
            # res_act = cls_boxes[i-1][:,:5]
            # inds = np.argsort(-res_act[:, 4])
            # res_act = res_act[inds,:]

            # np.testing.assert_allclose(res_act, res)

        cls_boxes = []
        cls_feats = []
        for cls_id in range(num_classes-1):
            cfs = self._run_class_based_boxes_single_op_gpu(stats, boxes, feats, cls_id)
            cls_boxes.append(cfs[0])
            cls_feats.append(cfs[1])

        nms_boxes = []
        nms_feats = []
        for i in range(1, num_classes):
            nms = self._run_nms_op_cpu(cls_boxes[i-1], cls_feats[i-1])
            nms_boxes.append(nms[0])
            nms_feats.append(nms[1])

            # Code for checking the nms algorithm
            # keep = utils.boxes.nms(cls_boxes[i-1][:,:5], 0.5)
            # res = cls_boxes[i-1][keep,:5]
            # inds = np.argsort(-res[:, 4])
            # res = res[inds,:]

            # np.testing.assert_allclose(nms[0][:,:5], res)
        
        agg_nms_boxes = np.vstack(nms_boxes)
        agg_nms_feats = np.vstack(nms_feats)
        pim = self._run_reduce_boxes_and_feats_op_gpu(agg_nms_boxes, agg_nms_feats)
        rois = pim[0]
        feats = pim[1]

        # Code for testing reduce boxes and feats
        # inds = np.argsort(-agg_nms_boxes[:, 4])
        # rois_exp = agg_nms_boxes[inds[0:cfg.MEM.DPI], :4]
        # im_ids = np.zeros((rois_exp.shape[0],1),dtype=np.float32)
        # rois_exp = np.hstack([im_ids, rois_exp])

        # np.testing.assert_allclose(rois_exp, rois)

        return rois

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
    assert 'BoxesAndFeats' in workspace.RegisteredOperators()
    assert 'ClassBasedBoxes' in workspace.RegisteredOperators()
    assert 'ClassBasedBoxesSingle' in workspace.RegisteredOperators()
    assert 'NMS' in workspace.RegisteredOperators()
    assert 'ReduceBoxesAndFeats' in workspace.RegisteredOperators()
    assert 'ReduceSumGPU' in workspace.RegisteredOperators()
    unittest.main()