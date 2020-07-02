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

# TODO: define random functions  

import numpy as np

from ..fluid import core
from ..fluid.framework import device_guard, in_dygraph_mode, _varbase_creator, Variable, convert_np_dtype_to_dtype_
from ..fluid.layers.layer_function_generator import templatedoc
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layers import utils, uniform_random, gaussian_random
from ..fluid.layers.tensor import fill_constant

from ..fluid.io import shuffle  #DEFINE_ALIAS

__all__ = [
    #       'gaussin',
    #       'uniform',
    'shuffle',
    'randn',
    'rand',
    'randint',
    'randperm'
]


def randint(low=0, high=None, shape=[1], dtype=None, name=None):
    """
	:alias_main: paddle.randint
	:alias: paddle.randint,paddle.tensor.randint,paddle.tensor.random.randint

    This function returns a Tensor filled with random integers from the
    "discrete uniform" distribution of the specified data type in the interval
    [low, high). If high is None (the default), then results are from [0, low).

    Args:
        low (int): The lower bound on the range of random values to generate,
            the low is included in the range.(unless high=None, in which case
            this parameter is one above the highest such integer). Default is 0.
        high (int, optional): The upper bound on the range of random values to
            generate, the high is excluded in the range. Default is None(see
            above for behavior if high=None).
        shape (list|tuple|Variable, optional): The shape of the output Tensor,
            if the shape is a list or tuple, its elements can be an integer or
            a Tensor with the shape [1], and the type of the Tensor must be
            int32 or int64. If the shape is a Variable, it is a 1-D Tensor,
            and the type of the Tensor must be int32 or int64. Default is None.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the
            output Tensor which can be int32, int64. If dtype is `None`, the
            data type of created Tensor is `int64`
        name(str, optional): The default value is None.  Normally there is no
            need for user to set this property.  For more information, please
            refer to :ref:`api_guide_Name`.

    Returns: 
        Variable: A Tensor of the specified shape filled with random integers.

    Raises:
        ValueError: Randint's low must less then high.
        TypeError: shape's type must be list, tuple or Variable.
        TypeError: dtype must be int32 or int64.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = paddle.randint(low=-5, high=5, shape=[3, 4], dtype="int64")

            # example 2:
            # attr shape is a list which contains tensor Variable.
            dim_1 = fluid.layers.fill_constant([1],"int64",3)
            dim_2 = fluid.layers.fill_constant([1],"int32",5)
            result_2 = paddle.randint(low=-5, high=5, shape=[dim_1, dim_2], dtype="int32")

            # example 3:
            # attr shape is a Variable, the data type must be int64 or int32.
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = paddle.randint(low=-5, high=5, shape=var_shape, dtype="int32")
            var_shape_int32 = fluid.data(name='var_shape_int32', shape=[2], dtype="int32")
            result_4 = paddle.randint(low=-5, high=5, shape=var_shape_int32, dtype="int64")

            # example 4:
            # Input only one parameter
            # low=0, high=10, shape=[1], dtype='int64'
            result_4 = paddle.randint(10)

    """
    if high is None:
        high = low
        low = 0
    if dtype is None:
        dtype = 'int64'
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils._convert_shape_to_list(shape)
        return core.ops.randint('low', low, 'high', high, 'seed', seed, 'dtype',
                                dtype)

    check_type(shape, 'shape', (list, tuple, Variable), 'randint')
    check_dtype(dtype, 'dtype', ['int32', 'int64'], 'randint')
    if low >= high:
        raise ValueError(
            "randint's low must less then high, but received low = {0}, "
            "high = {1}".format(low, high))

    inputs = dict()
    attrs = {'low': low, 'high': high, 'seed': 0, 'dtype': dtype}
    utils._get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='randint')

    helper = LayerHelper("randint", **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)
    helper.append_op(
        type='randint', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def randn(shape, dtype=None, name=None):
    """
	:alias_main: paddle.randn
	:alias: paddle.randn,paddle.tensor.randn,paddle.tensor.random.randn

    This function returns a tensor filled with random numbers from a normal 
    distribution with mean 0 and standard deviation 1 (also called the standard normal
    distribution).

    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created. The data
            type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
            the elements of it should be integers or Tensors with shape [1]. If
            ``shape`` is a Variable, it should be an 1-D Tensor .
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the output 
            tensor, which can be float32, float64. if dtype is `None` , the data 
            type of output tensor is `float32` . Default is None.
        name(str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
            Default is None.

    Returns:
        Random tensor whose data is drawn from a standard normal distribution,
        dtype: flaot32 or float64 as specified.

    Return type:
        Variable

    Raises:
        TypeError: If the type of `shape` is not Variable, list or tuple.
        TypeError: If the data type of `dtype` is not float32 or float64.
        ValueError: If the length of `shape` is not bigger than 0.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()

        # example 1: attr shape is a list which doesn't contain tensor Variable.
        result_1 = paddle.randn(shape=[2, 3])
        # [[-2.923464    0.11934398 -0.51249987]
        #  [ 0.39632758  0.08177969  0.2692008 ]]

        # example 2: attr shape is a list which contains tensor Variable.
        dim_1 = paddle.fill_constant([1], "int64", 2)
        dim_2 = paddle.fill_constant([1], "int32", 3)
        result_2 = paddle.randn(shape=[dim_1, dim_2, 2])
        # [[[-2.8852394  -0.25898588]
        #   [-0.47420555  0.17683524]
        #   [-0.7989969   0.00754541]]
        #  [[ 0.85201347  0.32320443]
        #   [ 1.1399018   0.48336947]
        #   [ 0.8086993   0.6868893 ]]]

        # example 3: attr shape is a Variable, the data type must be int64 or int32.
        var_shape = paddle.imperative.to_variable(np.array([2, 3]))
        result_3 = paddle.randn(var_shape)
        # [[-2.878077    0.17099959  0.05111201]
        #  [-0.3761474  -1.044801    1.1870178 ]]

    """
    if dtype is None:
        dtype = 'float32'
    return gaussian_random(
        shape=shape, mean=0.0, std=1.0, seed=0, dtype=dtype, name=name)


@templatedoc()
def randperm(n,
             out=None,
             dtype="int64",
             device=None,
             stop_gradient=True,
             seed=0):
    """
	:alias_main: paddle.randperm
	:alias: paddle.randperm,paddle.tensor.randperm,paddle.tensor.random.randperm

    ${comment}

    Args:
        n (int): The upper bound (exclusive), and it should be greater than 0.
        out (Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            If out is None, a new Varibale will be create to store the result. 
            Default: None.
        dtype (np.dtype|core.VarDesc.VarType|str, optional): The type of the 
            output Tensor. Supported data types: int64, int32. Default: int32.
        device (str, optional): Specific the output variable to be saved in cpu
            or gpu memory. Supported None, 'cpu', 'gpu'. If it is None, the output
            variable will be automatically assigned devices.
            Default: None.
        stop_gradient (bool, optional): Whether grad should record operations 
            on the returned tensor. Default: True.
        seed (int, optional): Random seed used for permute samples. If seed is 
            equal to 0, it means use a seed generated by the system. Note that 
            if seed is not 0, this operator will always generate the same random 
            permutation every time. Default: 0.

    Returns:
        ${out_comment}.

    Return Type:
        ${out_type}

    Examples:
        .. code-block:: python

	    import paddle
	    import paddle.fluid as fluid

	    num = 6
	    is_use_gpu = False

	    data_1 = paddle.randperm(num)
	    fluid.layers.Print(data_1)

	    data_2 = paddle.randperm(num, dtype="int32", seed=1)
	    fluid.layers.Print(data_2)

	    data_3 = paddle.randperm(num, stop_gradient=False, device="cpu")
	    fluid.layers.Print(data_3)

	    paddle.randperm(num, out=data_3)
	    fluid.layers.Print(data_3)

	    place = fluid.CUDAPlace(0) if is_use_gpu else fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
	    exe.run()
 
    """

    if n < 1:
        raise ValueError("The input n should be greater than 0 in randperm op.")
    check_dtype(dtype, 'dtype', ['int64', 'int32'], 'randperm')
    dtype = convert_dtype(dtype)
    if device not in [None, 'cpu', 'gpu']:
        raise ValueError("The input device should in [None, 'cpu', 'gpu'].")
    check_type(stop_gradient, 'stop_gradient', bool, 'randperm')

    helper = LayerHelper("randperm", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        check_variable_and_dtype(out, 'out', [dtype], 'randperm')
    if stop_gradient:
        out.stop_gradient = True
    inputs = dict()
    outputs = {'Out': [out]}
    attrs = {'n': n, 'dtype': out.dtype, 'seed': seed}
    with device_guard(device):
        helper.append_op(
            type='randperm', inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def rand(shape, dtype=None, name=None):
    """
	:alias_main: paddle.rand
	:alias: paddle.rand,paddle.tensor.rand,paddle.tensor.random.rand

    This OP initializes a variable with random values sampled from a
    uniform distribution in the range [0, 1).

    Examples:
    ::

        Input:
          shape = [1, 2]

        Output:
          result=[[0.8505902, 0.8397286]]

    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created. The data
            type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
            the elements of it should be integers or Tensors with shape [1]. If
            ``shape`` is a Variable, it should be an 1-D Tensor .
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the
            output tensor which can be float32, float64, if dytpe is `None`,
            the data type of created tensor is `float32`
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Variable: A Tensor of the specified shape filled with random numbers
        from a uniform distribution on the interval [0, 1).

    Raises:
        TypeError: The shape type should be list or tupple or Variable.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()

        # example 1: attr shape is a list which doesn't contain tensor Variable.
        result_1 = paddle.rand(shape=[2, 3])
        # [[0.451152  , 0.55825245, 0.403311  ],
        #  [0.22550228, 0.22106001, 0.7877319 ]]

        # example 2: attr shape is a list which contains tensor Variable.
        dim_1 = paddle.fill_constant([1], "int64", 2)
        dim_2 = paddle.fill_constant([1], "int32", 3)
        result_2 = paddle.rand(shape=[dim_1, dim_2, 2])
        # [[[0.8879919  0.25788337]
        #   [0.28826773 0.9712097 ]
        #   [0.26438272 0.01796806]]
        #  [[0.33633623 0.28654453]
        #   [0.79109055 0.7305809 ]
        #   [0.870881   0.2984597 ]]]

        # example 3: attr shape is a Variable, the data type must be int64 or int32.
        var_shape = paddle.imperative.to_variable(np.array([2, 3]))
        result_3 = paddle.rand(var_shape)
        # [[0.22920267 0.841956   0.05981819]
        #  [0.4836288  0.24573246 0.7516129 ]]

    """
    if dtype is None:
        dtype = 'float32'
    return uniform_random(shape, dtype, min=0.0, max=1.0, name=name)
