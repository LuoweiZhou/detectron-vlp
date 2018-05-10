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

class ResizeMemoryInitOpTest(unittest.TestCase):

    def _run_resize_memory_init_op_gpu(self, X, im_info, data):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('ResizeMemoryInit', ['X', 'im_info', 'data'], ['Y'], spatial_scale=0.5)
            workspace.FeedBlob('X', X)
            workspace.FeedBlob('im_info', im_info)
            workspace.FeedBlob('data', data)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def test_image(self):
        this_dir = osp.dirname(__file__)
        X = cv2.imread(osp.join(this_dir, 'zerg.jpg')).astype(np.float32)
        # to C, H, W
        X = np.transpose(X, (2, 0, 1))
        im_info = np.array([[508., 944., 1.], [127., 236., 1.]], dtype=np.float32)
        data = np.zeros([2, 1, 1000, 1000], dtype=np.float32)
        Y = self._run_resize_memory_init_op_gpu(X, im_info, data)

        Y_dump1 = np.transpose(Y[0,:], (1, 2, 0))
        Y_dump2 = np.transpose(Y[1,:], (1, 2, 0))
        cv2.imwrite(osp.join(this_dir, 'z1-output.jpg'), Y_dump1)
        cv2.imwrite(osp.join(this_dir, 'z2-output.jpg'), Y_dump2)

        return True

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_custom_ops()
    assert 'ResizeMemoryInit' in workspace.RegisteredOperators()
    unittest.main()
