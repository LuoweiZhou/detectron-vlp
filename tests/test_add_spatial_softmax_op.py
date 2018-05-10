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

class AddSpatialSoftmaxOpTest(unittest.TestCase):

    def _run_add_spatial_softmax_op_gpu(self, X, nc=2):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('AddSpatialSoftmax', 
                                    ['X'], ['Y'], 
                                    num_classes=nc)
            workspace.FeedBlob('X', X)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def _test_single_res(self, X):
        Y_act = self._run_add_spatial_softmax_op_gpu(X)
        n = X.shape[0]
        c = X.shape[1]
        h = X.shape[2]
        w = X.shape[3]
        X = X.reshape(n*c, 1, h, w)
        zerosX = np.zeros_like(X, dtype=np.float32)
        Xhat = np.concatenate([zerosX, X], axis=1)
        maxX = Xhat.max(axis=1, keepdims=True)
        Xhat -= maxX
        Xhat = np.exp(Xhat)
        sumX = Xhat.sum(axis=1, keepdims=True)
        Y_exp = Xhat / sumX
        Y_exp = Y_exp.reshape(n, c*2, h, w)
        np.testing.assert_allclose(Y_act, Y_exp, atol=1e-7)

    def _test_mult_res(self, X, nc):
        Y_act = self._run_add_spatial_softmax_op_gpu(X, nc)
        n = X.shape[0]
        c = X.shape[1]
        nd = nc - 1
        assert c % nd == 0
        a = int(c / nd)
        h = X.shape[2]
        w = X.shape[3]
        X = X.reshape(n*a, nd, h, w)
        zerosX = np.zeros((n*a, 1, h, w), dtype=np.float32)
        Xhat = np.concatenate([zerosX, X], axis=1)
        maxX = Xhat.max(axis=1, keepdims=True)
        Xhat -= maxX
        Xhat = np.exp(Xhat)
        sumX = Xhat.sum(axis=1, keepdims=True)
        Y_exp = Xhat / sumX
        Y_exp = Y_exp.reshape(n, a * (nd + 1), h, w)
        np.testing.assert_allclose(Y_act, Y_exp, atol=1e-7)

    def test_one_res(self):
        X = np.random.rand(2, 1, 7, 7).astype(np.float32)
        self._test_single_res(X)

    def test_mult_res(self):
        X = np.random.rand(20, 6, 8, 8).astype(np.float32)
        self._test_mult_res(X, 3)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_custom_ops()
    assert 'AddSpatialSoftmax' in workspace.RegisteredOperators()
    unittest.main()
