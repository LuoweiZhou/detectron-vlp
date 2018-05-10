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

class SelectTopNOpTest(unittest.TestCase):

    def _run_select_top_n_op_gpu(self, X, top_n):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('SelectTopN', ['X'], ['YI', 'YV'], 
                                                    top_n=top_n)
            workspace.FeedBlob('X', X)
        workspace.RunOperatorOnce(op)
        YI = workspace.FetchBlob('YI')
        YV = workspace.FetchBlob('YV')
        return YI, YV

    def test_res(self):
        top_n = 1000
        X = np.random.rand(1, 720, 20, 30).astype(np.float32) * 100.
        YI_act, YV_act = self._run_select_top_n_op_gpu(X, top_n)
        YI_act.sort()
        YV_act.sort()

        Xr = X.ravel()
        YI_exp = np.argpartition(Xr, top_n)[:top_n]
        YV_exp = Xr[YI_exp]
        YI_exp.sort()
        YV_exp.sort()

        np.testing.assert_allclose(YI_act, YI_exp)
        np.testing.assert_allclose(YV_act, YV_exp)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_custom_ops()
    assert 'SelectTopN' in workspace.RegisteredOperators()
    unittest.main()
