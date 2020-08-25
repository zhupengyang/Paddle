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
        self.axis = [1, 2]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()

        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_frobenius_norm(x, self.axis, self.keepdim)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all
        }

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestFrobeniusNormOpFloat32(TestFrobeniusNormOp):
    def set_attrs(self):
        self.dtype = 'float32'


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
        self.axis = [0, 1]
        self.reduce_all = True


def ref_p_norm(x, axis, porder, keepdims=False):
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
    elif porder > 0:
        out = np.power(x, porder)
        out = np.sum(out, axis=axis, keepdims=keepdim)
        out = np.power(out, 1.0 / porder)
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

        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x[np.abs(self.x) < 0.05] = 0.05
        self.out = ref_p_norm(self.x, self.axis, self.porder, self.keepdim)
        self.inputs = {'X': self.x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': self.porder
        }
        self.outputs = {'Out': self.out}
        #self.gradient = self.calc_gradient()

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    # def calc_gradient(self):
    #     self.attrs = {
    #         'epsilon': self.epsilon,
    #         'axis': self.axis,
    #         'keepdim': self.keepdim,
    #         'porder': float(self.porder)
    #     }
    #     x = self.inputs["X"]
    #     porder = self.attrs["porder"]
    #     axis = self.attrs["axis"]
    #     if porder == 0:
    #         grad = np.zeros(x.shape).astype(x.dtype)
    #     elif porder in [float("inf"), float("-inf")]:
    #         norm = p_norm(x, axis=axis, porder=porder, keepdims=True)
    #         x_abs = np.abs(x)
    #         grad = np.sign(x)
    #         grad[x_abs != norm] = 0.0
    #     else:
    #         norm = p_norm(x, axis=axis, porder=porder, keepdims=True)
    #         grad = np.power(norm, 1 - porder) * np.power(
    #             np.abs(x), porder - 1) * np.sign(x)

    #     numel = 1
    #     for s in x.shape:
    #         numel *= s
    #     numel /= x.shape[axis]
    #     return [grad.astype(x.dtype) * 1 / numel]


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


class TestPnormOpInf(TestPnormOp):
    def set_attrs(self):
        self.porder = float('inf')


class TestPnormOpNegativeInf(TestPnormOp):
    def set_attrs(self):
        self.porder = float('-inf')


# class TestPnormOp3(TestPnormOp):
#     def init_test_case(self):
#         self.shape = [3, 20, 3]
#         self.axis = 2
#         self.epsilon = 1e-12
#         self.porder = np.inf
#         self.keepdim = True
#         self.dtype = "float32"

#     def test_check_grad(self):
#         self.check_grad(['X'], 'Out', user_defined_grads=self.gradient)

# class TestPnormOp4(TestPnormOp):
#     def init_test_case(self):
#         self.shape = [3, 20, 3]
#         self.axis = 2
#         self.epsilon = 1e-12
#         self.porder = -np.inf
#         self.keepdim = True
#         self.dtype = "float32"

#     def test_check_grad(self):
#         self.check_grad(['X'], 'Out', user_defined_grads=self.gradient)

# class TestPnormOp5(TestPnormOp):
#     def init_test_case(self):
#         self.shape = [3, 20, 3]
#         self.axis = 2
#         self.epsilon = 1e-12
#         self.porder = 0
#         self.keepdim = True
#         self.dtype = "float32"

#     def test_check_grad(self):
#         self.check_grad(['X'], 'Out', user_defined_grads=self.gradient)


class TestNormAPI(unittest.TestCase):
    def init_data(self):
        pass

    def run_static(self):
        pass

    def run_dygraph(self):
        pass

    def api_case(self):
        pass

    def test_api(self):
        pass

    def test_errors(self):
        pass


if __name__ == '__main__':
    unittest.main()
