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

class InvCropAndResizeOpTest(unittest.TestCase):
    def _run_crop_and_resize_op_gpu(self, X, R):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('CropAndResize', ['X', 'R'], ['RX'], 
                                                      spatial_scale=1., 
                                                      pooled_h=200, 
                                                      pooled_w=200)
            workspace.FeedBlob('X', X)
            workspace.FeedBlob('R', R)
        workspace.RunOperatorOnce(op)
        RX = workspace.FetchBlob('RX')
        return RX

    def _run_inv_crop_and_resize_op_gpu(self, X, R, RX):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('InvCropAndResize', ['X', 'R', 'RX'], ['Y'], 
                                                      spatial_scale=1.)
            workspace.FeedBlob('X', X)
            workspace.FeedBlob('R', R)
            workspace.FeedBlob('RX', RX)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def test_image(self):
        this_dir = osp.dirname(__file__)
        X = cv2.imread(osp.join(this_dir, 'zerg.jpg')).astype(np.float32)
        # to C, H, W
        X = np.transpose(X, (2, 0, 1))
        X = np.expand_dims(X, 0).astype(np.float32)
        # ignore the extrapolation part
        R = np.array([[0., 10., 20., 400., 200.], 
                      [0., 300., 100., 470., 250.]], dtype=np.float32)
        # do the actual computation
        RX = self._run_crop_and_resize_op_gpu(X, R)
        Y = self._run_inv_crop_and_resize_op_gpu(X, R, RX)
        # compute the divider
        x = np.ones(X.shape, dtype=np.float32)
        rx = self._run_crop_and_resize_op_gpu(x, R)
        y = self._run_inv_crop_and_resize_op_gpu(x, R, rx)
        Y = Y / np.maximum(y, 1e-14)

        output = Y[0,:]
        output = np.transpose(output, [1, 2, 0])
        cv2.imwrite(osp.join(this_dir, 'icar-output.jpg'), output)

        return True

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_custom_ops()
    assert 'CropAndResize' in workspace.RegisteredOperators()
    assert 'InvCropAndResize' in workspace.RegisteredOperators()
    unittest.main()
