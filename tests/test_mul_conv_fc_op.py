from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from PIL import Image
import unittest
import os.path as osp

import _init_paths
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace

import utils.c2

class MulConvFCOpTest(unittest.TestCase):

    def _run_mul_conv_fc_op_gpu(self, X1, X2):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('MulConvFC', ['X1', 'X2'], ['Y'])
            workspace.FeedBlob('X1', X1)
            workspace.FeedBlob('X2', X2)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def test_res(self):
        X1 = np.random.rand(20, 512, 7, 7).astype(np.float32)
        X2 = np.random.rand(20, 512).astype(np.float32)
        Y_act = self._run_mul_conv_fc_op_gpu(X1, X2)
        Y_exp = X1 * X2.reshape(20, 512, 1, 1)
        np.testing.assert_allclose(Y_act, Y_exp)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_custom_ops()
    assert 'MulConvFC' in workspace.RegisteredOperators()
    unittest.main()
