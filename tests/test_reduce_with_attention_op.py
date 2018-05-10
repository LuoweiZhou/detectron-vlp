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

class ReduceWithAttentionOpTest(unittest.TestCase):

    def _run_reduce_with_attention_op_gpu(self, Xs, attention, iter):
        blobs_in = ['attention']
        lx = len(Xs)
        for i in range(lx):
            blobs_in.append('X%d' % i)
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('ReduceWithAttention', 
                                    blobs_in, ['Y'],
                                    iter=iter)
            workspace.FeedBlob('attention', attention)
            for i in range(lx):
                workspace.FeedBlob('X%d' % i, Xs[i])
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def test_res(self):
        iter = 2
        a = 10
        x = 6
        Xs = []
        for i in range(iter):
            Xs.append(np.random.rand(2, a * x, 7, 7).astype(np.float32))
        attention = np.random.rand(2, iter * a, 7, 7).astype(np.float32)
        Y_act = self._run_reduce_with_attention_op_gpu(Xs, attention, iter)
        A = attention.reshape(2, iter, a, 7, 7)
        As = np.split(A, iter, axis=1)
        As = [ sa.reshape(2, a, 1, 7, 7) for sa in As ]
        Xs = [ xs.reshape(2, a, x, 7, 7) for xs in Xs ]
        Y_exp = np.zeros_like(Xs[0])
        for i in range(iter):
            Y_exp += Xs[i] * As[i]
        Y_exp = Y_exp.reshape(2, a * x, 7, 7)
        np.testing.assert_allclose(Y_act, Y_exp, atol=1e-7)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_custom_ops()
    assert 'ReduceWithAttention' in workspace.RegisteredOperators()
    unittest.main()
