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

class BoxesAndFeatsOpTest(unittest.TestCase):
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

        workspace.ResetWorkspace()

        return cls_preds, cls_probs, box_preds, anchors, im_info

    def _run_select_top_n_op_gpu(self, X, top_n):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('SelectTopN', ['X'], ['YI', 'YV'], 
                                                    top_n=top_n)
            workspace.FeedBlob('X', X)

        workspace.RunOperatorOnce(op)

        YI = workspace.FetchBlob('YI')
        YV = workspace.FetchBlob('YV')

        workspace.ResetWorkspace()

        return YI, YV

    def _run_boxes_and_feats_op_gpu(self, cls_preds, box_preds, anchors,
                                    YI, YV, im_info, level):
        blobs_in = ['cls_preds', 'box_preds', 'anchors', 'YI', 'YV', 'im_info']
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('BoxesAndFeats', blobs_in, ['boxes', 'feats', 'stats'], 
                                                    A=9,
                                                    im=0,
                                                    level=level)
            workspace.FeedBlob('cls_preds', cls_preds)
            workspace.FeedBlob('box_preds', box_preds)
            workspace.FeedBlob('anchors', anchors)
            workspace.FeedBlob('YI', YI)
            workspace.FeedBlob('YV', YV)
            workspace.FeedBlob('im_info', im_info)

        workspace.RunOperatorOnce(op)

        boxes = workspace.FetchBlob('boxes')
        feats = workspace.FetchBlob('feats')
        stats = workspace.FetchBlob('stats')

        workspace.ResetWorkspace()

        return boxes, feats, stats

    def test_res(self):
        cls_preds, cls_probs, box_preds, anchors, im_info = self._test_std()
        im_info = im_info.astype(np.float32)

        for level in range(7, 8):
            cls_pred = cls_preds[level-3]
            cls_prob = cls_probs[level-3]
            box_pred = box_preds[level-3]
            anchor = anchors[level].astype(np.float32)

            YI, YV = self._run_select_top_n_op_gpu(-cls_prob, 1000)
            boxes_act, feats_act, stats_act = self._run_boxes_and_feats_op_gpu(cls_pred, box_pred, anchor, YI, YV, im_info, level)

            cls_probs_ravel = cls_prob.ravel()
            A = 9
            num_cls = int(cls_prob.shape[1] / A)
            H = cls_prob.shape[2]
            W = cls_prob.shape[3]
            cls_pred = cls_pred.reshape((1, A, num_cls, H, W))
            cls_prob = cls_prob.reshape((1, A, num_cls, H, W))
            box_pred = box_pred.reshape((1, A, 4, H, W))
            inds_5d = np.array(np.unravel_index(YI, cls_prob.shape)).transpose()
            classes = inds_5d[:, 2]
            anchor_ids, y, x = inds_5d[:, 1], inds_5d[:, 3], inds_5d[:, 4]
            feats_exp = cls_pred[:, anchor_ids, :, y, x]
            feats_exp = feats_exp.reshape(-1, num_cls)

            scores = cls_prob[:, anchor_ids, classes, y, x]
            scores = scores.ravel()

            boxes = np.column_stack((x, y, x, y)).astype(dtype=np.float32)
            boxes *= (2**level)
            boxes += anchor[anchor_ids, :]
            box_deltas = box_pred[0, anchor_ids, :, y, x]
            pred_boxes = box_utils.bbox_transform(boxes, box_deltas)
            pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im_info[0, :2])

            boxes_exp = np.zeros((pred_boxes.shape[0], 5), dtype=np.float32)
            boxes_exp[:, 0:4] = pred_boxes
            boxes_exp[:, 4] = scores

            # for i in range(num_cls):
            #     if stats_act[0, i] > 0:
            #         print('cls %d: %d' % (i+1, stats_act[0, i]))

            np.testing.assert_allclose(boxes_act[:, :5], boxes_exp, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(feats_act, feats_exp)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_detectron_ops()
    utils.c2.import_custom_ops()
    assert 'BoxesAndFeats' in workspace.RegisteredOperators()
    unittest.main()
