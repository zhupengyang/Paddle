# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid

np.random.seed(10)


def ref_frobenius_norm(x, axis=None, keepdims=False):
    if axis is None:
        if len(x.shape) == 2:
            axis = (0, 1)
        else:
            raise ValueError('If axis is None, length of x should be 2.')
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.linalg.norm(x, ord='fro', axis=axis, keepdims=keepdims)


class TestFrobeniusNormOp(OpTest):
    def setUp(self):
        self.op_type = "frobenius_norm"
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]
        self.axis = [0, 1]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()

        self.set_in_out()
        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all
        }

    def set_in_out(self):
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.out = ref_frobenius_norm(self.x, self.axis, self.keepdim)

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestFrobeniusNormOpFloat32(TestFrobeniusNormOp):
    def set_attrs(self):
        self.dtype = 'float32'

    def set_in_out(self):
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x[np.abs(self.x) < 0.1] = 0.1
        self.out = ref_frobenius_norm(self.x, self.axis, self.keepdim)


class TestFrobeniusNormOpShape2D(TestFrobeniusNormOp):
    def set_attrs(self):
        self.shape = [10, 12]


class TestFrobeniusNormOpAxisTuple(TestFrobeniusNormOp):
    def set_attrs(self):
        self.axis = (2, 3)


class TestFrobeniusNormOpKeepdim(TestFrobeniusNormOp):
    def set_attrs(self):
        self.keepdim = True


class TestFrobeniusNormOpReduceAll(TestFrobeniusNormOp):
    def set_attrs(self):
        self.shape = [10, 12]
        self.reduce_all = True


def ref_p_norm(x, axis, porder, keepdim=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    if isinstance(porder, int):
        porder = float(porder)

    if porder == 0:
        out = (x != 0).astype(x.dtype)
        out = np.sum(out, axis=axis, keepdims=keepdim)
    elif porder == float('inf'):
        out = np.absolute(x)
        out = np.max(out, axis=axis, keepdims=keepdim)
    elif porder == float('-inf'):
        out = np.absolute(x)
        out = np.min(out, axis=axis, keepdims=keepdim)
    elif isinstance(porder, float) and porder > 0:
        out = np.abs(x)
        out = np.power(out, porder)
        out = np.sum(out, axis=axis, keepdims=keepdim)
        out = np.power(out, 1.0 / porder)
    elif porder == 'fro':
        out = np.power(x, 2.0)
        out = np.sum(out, axis=axis, keepdims=keepdim)
        out = np.power(out, 1.0 / 2.0)
    else:
        raise ValueError('not supported porder')

    return out


class TestPnormOp(OpTest):
    def setUp(self):
        self.op_type = "p_norm"
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.porder = 2
        self.keepdim = False
        self.epsilon = 1e-12
        self.set_attrs()

        self.set_in_out()
        self.inputs = {'X': self.x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': self.porder
        }
        self.outputs = {'Out': self.out}

    def set_in_out(self):
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.out = ref_p_norm(self.x, self.axis, self.porder, self.keepdim)

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestPnormOpFloat32(TestPnormOp):
    def set_attrs(self):
        self.dtype = 'float32'


class TestPnormOpShape1D(TestPnormOp):
    def set_attrs(self):
        self.shape = [100]
        self.axis = 0


class TestPnormOpP0(TestPnormOp):
    def set_attrs(self):
        self.porder = 0

    def set_in_out(self):
        self.x = np.random.randint(0, 2, self.shape).astype(self.dtype)
        self.out = ref_p_norm(self.x, self.axis, self.porder, self.keepdim)


class TestPnormOpInf(TestPnormOp):
    def set_attrs(self):
        self.porder = float('inf')


class TestPnormOpNegativeInf(TestPnormOp):
    def set_attrs(self):
        self.porder = float('-inf')


class TestNormAPI(unittest.TestCase):
    def setUp(self):
        self.place=paddle.CUDAPlace(0) \
            if paddle.fluid.core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def run_static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, self.dtype)
            out = paddle.norm(x, self.p, self.axis, self.keepdim)
            exe = paddle.static.Executor(self.place)
            ret = exe.run(feed={'X': self.x}, fetch_list=[out])
        return ret[0]

    def run_dygraph(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out = paddle.norm(x, self.p, self.axis, self.keepdim)
        out = out.numpy()
        paddle.enable_static()
        return out

    def api_case(self,
                 dtype='float32',
                 shape=[2, 3, 4, 5],
                 p='fro',
                 axis=None,
                 keepdim=False):
        self.dtype = dtype
        self.shape = shape
        self.p = p
        self.axis = axis
        self.keepdim = keepdim

        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.out = ref_p_norm(self.x, self.axis, self.p, self.keepdim)
        if len(self.out.shape) == 0:
            self.out = self.out.reshape([1])
        out_static = self.run_static()
        out_dygraph = self.run_dygraph()
        for out in [out_static, out_dygraph]:
            self.assertEqual(self.out.shape, out.shape)
            self.assertTrue(np.allclose(self.out, out))

    def test_api(self):
        self.api_case()
        self.api_case(dtype='float64')
        self.api_case(shape=[10, 12])
        self.api_case(axis=[1, 2, 3])
        self.api_case(axis=(-2, ))
        self.api_case(keepdim=True)
        self.api_case(p=0, axis=[0, 2])
        self.api_case(p=float('inf'), axis=[-1, -2])
        self.api_case(p=float('-inf'), axis=[0, 1, 2, 3])
        self.api_case(p=1)
        self.api_case(p=2.0)

        paddle.disable_static(self.place)
        x = (np.arange(120) - 60).astype('float32')
        x = paddle.to_tensor(x)
        out1 = paddle.norm(x).numpy()
        out2 = paddle.tensor.norm(x).numpy()
        out3 = paddle.tensor.linalg.norm(x).numpy()
        self.assertTrue(np.allclose(out1, out2))
        self.assertTrue(np.allclose(out1, out3))
        paddle.enable_static()

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x1 = paddle.static.data('x1', [100])
            self.assertRaises(ValueError, paddle.norm, x1, p='f')
            self.assertRaises(TypeError, paddle.norm, 1)
            x2 = paddle.static.data('x2', [100], 'int32')
            self.assertRaises(TypeError, paddle.norm, x2)
            self.assertRaises(AssertionError, paddle.norm, x1, p=-1)
            self.assertRaises(TypeError, paddle.norm, x1, p=2, axis=2.0)
            self.assertRaises(TypeError, paddle.norm, x1, p=2, axis=[2.0])


if __name__ == '__main__':
    unittest.main()
