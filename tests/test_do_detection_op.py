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
from ops.do_detection import DoDetectionOp
import utils.c2
import utils.net
import utils.vis

class DoDetectionOpTest(unittest.TestCase):

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
        utils.vis.vis_one_image(im[:, :, ::-1],
                                '{:s}-std-output'.format(im_name),
                                current_dir,
                                cls_boxes,
                                segms=None,
                                keypoints=None,
                                thresh=0.,
                                box_alpha=0.8,
                                dataset=dataset,
                                show_class=False) 
        workspace.ResetWorkspace()

        return cls_preds, cls_probs, box_preds, anchors, im_info, im, im_name, current_dir, dataset

    def _run_op(self, cls_preds, cls_probs, box_preds, anchors, im_info):
        num_levels = len(cls_preds)
        blobs_in = []
        blobs_dict = {}
        for i in range(num_levels):
            name = 'cls_preds_%d' % i
            blobs_in.append(name)
            blobs_dict[name] = cls_preds[i]
        for i in range(num_levels):
            name = 'cls_probs_%d' % i 
            blobs_in.append(name)
            blobs_dict[name] = cls_probs[i]
        for i in range(num_levels):
            name = 'box_preds_%d' % i
            blobs_in.append(name)
            blobs_dict[name] = box_preds[i]
        for i in range(num_levels):
            name = 'anchors_%d' % i
            blobs_in.append(name)
            blobs_dict[name] = anchors[i+3]
        blobs_in.append('im_info')
        blobs_dict['im_info'] = im_info
        blobs_out = ['rois', 'logits']

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
            op = core.CreatePythonOperator(DoDetectionOp().forward, 
                                            blobs_in, 
                                            blobs_out)
            for key, value in blobs_dict.iteritems():
                workspace.FeedBlob(key, value)
        workspace.RunOperatorOnce(op)
        rois = workspace.FetchBlob('rois')
        return rois

    def test_do_detection(self):
        cls_preds, cls_probs, box_preds, anchors, im_info, im, im_name, current_dir, dataset = self._test_std()
        rois = self._run_op(cls_preds, cls_probs, box_preds, anchors, im_info)
        rois = rois / im_info[0, 2]
        scores = rois[:, [0]]
        scores[:] = 1.
        boxes = rois[:, 1:]
        cls_boxes = np.hstack([boxes, scores])

        utils.vis.vis_one_image(im[:, :, ::-1],
                                '{:s}-dd-output'.format(im_name),
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
    unittest.main()
