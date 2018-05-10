from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from PIL import Image
import unittest
import os.path as osp

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace

import utils.c2

class CropAndResizeOpTest(unittest.TestCase):

    def _run_crop_and_resize_op_gpu(self, X, rois):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('CropAndResize', ['X', 'rois'], ['Y'], 
                                                      spatial_scale=1., 
                                                      pooled_h=200, 
                                                      pooled_w=200)
            workspace.FeedBlob('X', X)
            workspace.FeedBlob('rois', rois)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def test_image(self):
        this_dir = osp.dirname(__file__)
        X = cv2.imread(osp.join(this_dir, 'zerg.jpg')).astype(np.float32)
        # to C, H, W
        X = np.transpose(X, (2, 0, 1))
        X = np.expand_dims(X, 0).astype(np.float32)
        rois = np.array([[0., 10., 20., 400., 200.], 
                         [0., 300., 100., 470., 250.],
                         [0., -100., -20., 500., 300.]], dtype=np.float32)
        Y = self._run_crop_and_resize_op_gpu(X, rois)

        Y_dump1 = np.transpose(Y[0,:], (1, 2, 0))
        Y_dump2 = np.transpose(Y[1,:], (1, 2, 0))
        Y_dump3 = np.transpose(Y[2,:], (1, 2, 0))
        cv2.imwrite(osp.join(this_dir, 'car1-output.jpg'), Y_dump1)
        cv2.imwrite(osp.join(this_dir, 'car2-output.jpg'), Y_dump2)
        cv2.imwrite(osp.join(this_dir, 'car3-output.jpg'), Y_dump3)

        return True

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_custom_ops()
    assert 'CropAndResize' in workspace.RegisteredOperators()
    unittest.main()
