#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class TestRandnOp(unittest.TestCase):
    def test_api(self):
        x1 = paddle.randn(shape=[1000, 784], dtype='float32')
        x2 = paddle.randn(shape=[1000, 784], dtype='float64')

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(fluid.default_main_program(),
                      feed={},
                      fetch_list=[x1, x2])

        self.assertAlmostEqual(np.mean(res[0]), .0, delta=0.1)
        self.assertAlmostEqual(np.std(res[0]), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res[1]), .0, delta=0.1)
        self.assertAlmostEqual(np.std(res[1]), 1., delta=0.1)


class TestRandnOpForDygraph(unittest.TestCase):
    def run_net(self, use_cuda=False):
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            paddle.randn([3, 4])

            paddle.randn([3, 4], 'float64')

            dim_1 = fluid.layers.fill_constant([1], "int64", 3)
            dim_2 = fluid.layers.fill_constant([1], "int32", 5)
            paddle.randn(shape=[dim_1, dim_2])

            var_shape = fluid.dygraph.to_variable(np.array([3, 4]))
            paddle.randn(var_shape)

    def test_run(self):
        self.run_net(False)
        if core.is_compiled_with_cuda():
            self.run_net(True)


class TestRandnOpError(unittest.TestCase):
    def test_error(self):
        with program_guard(Program(), Program()):
            # The argument shape's size of randn_op should not be 0.
            def test_shape_size():
                paddle.randn(shape=[])

            self.assertRaises(AssertionError, test_shape_size)

            # The argument shape's type of randn_op should be list or tuple.
            def test_shape_type():
                paddle.randn(shape=1)

            self.assertRaises(TypeError, test_shape_type)

            # The argument dtype of randn_op should be float32 or float64.
            def test_dtype_int32():
                paddle.randn(shape=[1, 2], dtype='int32')

            self.assertRaises(TypeError, test_dtype_int32)

            def test_shape_list():
                paddle.randn(shape=[2.])

            self.assertRaises(TypeError, test_shape_list)


if __name__ == "__main__":
    unittest.main()
