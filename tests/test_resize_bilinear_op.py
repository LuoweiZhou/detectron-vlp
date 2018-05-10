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

class ResizeBilinearOpTest(unittest.TestCase):

    def _run_resize_bilinear_op_gpu(self, X, h, w):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('ResizeBilinear', ['X'], ['Y'], height=h, width=w)
            workspace.FeedBlob('X', X)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def _run_resize_cv2(self, X, h, w):
        X_hat = X[0,:]
        X_hat = np.transpose(X_hat, (1, 2, 0))
        Y_hat = cv2.resize(X_hat, (w, h), interpolation=cv2.INTER_LINEAR)
        Y_hat = np.transpose(Y_hat, (2, 0, 1))
        Y_hat = np.expand_dims(Y_hat, 0)
        return Y_hat

    def _run_resize_pil(self, X, h, w):
        X_hat = X[0,:]
        X_hat = np.transpose(X_hat, (1, 2, 0))
        X_hat = Image.fromarray(X_hat.astype(np.uint8))
        Y_hat = X_hat.resize((w, h), Image.BILINEAR)
        Y_hat = np.array(Y_hat)
        Y_hat = np.transpose(Y_hat.astype(np.float32), (2, 0, 1))
        Y_hat = np.expand_dims(Y_hat, 0)
        return Y_hat

    def test_image(self):
        this_dir = osp.dirname(__file__)
        X = cv2.imread(osp.join(this_dir, 'zerg.jpg'))
        X = np.transpose(X, (2, 0, 1))
        X = np.expand_dims(X, 0).astype(np.float32)
        Y = self._run_resize_bilinear_op_gpu(X, 508, 944)
        Y_dump = np.transpose(Y[0,:], (1, 2, 0))
        cv2.imwrite(osp.join(this_dir, 'zerg-output.jpg'), Y_dump)
        # Y_hat = self._run_resize_cv2(X, 508, 944)

        return True

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.c2.import_custom_ops()
    assert 'ResizeBilinear' in workspace.RegisteredOperators()
    unittest.main()
