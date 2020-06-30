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

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid import Program, program_guard


def check_randperm_out(n, data_np):
    assert isinstance(data_np, np.ndarray), \
        "The input data_np should be np.ndarray."
    gt_sorted = np.arange(n)
    out_sorted = np.sort(data_np)
    return list(gt_sorted == out_sorted)


def error_msg(data_np):
    return "The sorted ground truth and sorted out should " + \
 "be equal, out = " + str(data_np)


def convert_dtype(dtype_str):
    dtype_str_list = ["int32", "int64", "float32", "float64"]
    dtype_num_list = [
        core.VarDesc.VarType.INT32, core.VarDesc.VarType.INT64,
        core.VarDesc.VarType.FP32, core.VarDesc.VarType.FP64
    ]
    assert dtype_str in dtype_str_list, dtype_str + \
        " should in " + str(dtype_str_list)
    return dtype_num_list[dtype_str_list.index(dtype_str)]


class TestRandpermOp(OpTest):
    """ Test randperm op."""

    def setUp(self):
        self.op_type = "randperm"
        self.n = 200
        self.dtype = "int64"

        self.inputs = {}
        self.outputs = {"Out": np.zeros((self.n)).astype(self.dtype)}
        self.init_attrs()
        self.attrs = {
            "n": self.n,
            "dtype": convert_dtype(self.dtype),
        }

    def init_attrs(self):
        pass

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        out_np = np.array(outs[0])
        self.assertTrue(
            check_randperm_out(self.n, out_np), msg=error_msg(out_np))


class TestRandpermOpN(TestRandpermOp):
    def init_attrs(self):
        self.n = 10000


class TestRandpermOpInt32(TestRandpermOp):
    def init_attrs(self):
        self.dtype = "int32"


class TestRandpermOpFloat32(TestRandpermOp):
    def init_attrs(self):
        self.dtype = "float32"


class TestRandpermOpFloat64(TestRandpermOp):
    def init_attrs(self):
        self.dtype = "float64"


class TestRandpermOpError(unittest.TestCase):
    """ Test randperm op for raise error. """

    def test_errors(self):
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_value():
                paddle.randperm(n=-3)

            self.assertRaises(ValueError, test_value)

            def test_dtype():
                paddle.randperm(n=10, dtype='int8')

            self.assertRaises(TypeError, test_dtype)


class TestRandpermDygraphMode(unittest.TestCase):
    def test_check_output(self):
        with fluid.dygraph.guard():
            n = 10
            data_1 = paddle.randperm(n, dtype="int64")
            data_1_np = data_1.numpy()
            self.assertTrue(
                check_randperm_out(n, data_1_np), msg=error_msg(data_1_np))

            data_2 = paddle.randperm(n, dtype="int32")
            data_2_np = data_2.numpy()
            self.assertTrue(
                check_randperm_out(n, data_2_np), msg=error_msg(data_2_np))

            data_3 = paddle.randperm(n, dtype="float32")
            data_3_np = data_3.numpy()
            self.assertTrue(
                check_randperm_out(n, data_3_np), msg=error_msg(data_3_np))

            data_4 = paddle.randperm(n, dtype="float64")
            data_4_np = data_4.numpy()
            self.assertTrue(
                check_randperm_out(n, data_4_np), msg=error_msg(data_4_np))


if __name__ == "__main__":
    unittest.main()
