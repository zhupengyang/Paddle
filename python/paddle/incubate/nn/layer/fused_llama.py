# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

import paddle
from paddle.fluid import core
from paddle.fluid.core import VarDesc
from paddle.fluid.dygraph import no_grad
from paddle.fluid.framework import _non_static_mode, convert_np_dtype_to_dtype_
from paddle.incubate.nn import functional as incubate_f
from paddle.nn import Layer
from paddle.nn.initializer import Constant
from paddle.nn.layer.transformer import (
    _convert_attention_mask,
    _convert_param_attr_to_list,
)


# for distributed tensor model parallel
def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    if not _non_static_mode():
        # NOTE: use current_block and find_var_recursive to support while_loop
        startup_block = paddle.static.default_startup_program().current_block()
        main_block = paddle.static.default_main_program().current_block()
        startup_block._find_var_recursive(var.name).is_distributed = True
        main_block._find_var_recursive(var.name).is_distributed = True


def _to_dtype(t, dtype):
    # this function is a prune of Layer._transform function to fix fused op under amp.decorator(O2)
    if not paddle.is_floating_point(t):
        return t

    if type(dtype) is not VarDesc.VarType:
        dtype = convert_np_dtype_to_dtype_(dtype)

    if t.place.is_gpu_place():
        size_dtype = core.size_of_dtype(dtype)
        waiting_alloc_memory = (
            ((np.prod(t.shape) * size_dtype) / 256 + 1) * 256 * 1.2
        )
        gpu_memory_available = core.gpu_memory_available()
        if gpu_memory_available < waiting_alloc_memory:
            t_used = t._copy_to(paddle.CPUPlace(), False)
            t.value().get_tensor()._clear()
        else:
            t_used = t
    else:
        t_used = t

    if dtype is not None and dtype != t_used.dtype:
        with paddle.fluid.framework._dygraph_place_guard(place=t_used.place):
            t_casted = t_used.cast(dtype=dtype)
    else:
        t_casted = t_used

    new_t = t_casted

    dst_tensor = t.value().get_tensor()
    src_tensor = new_t.value().get_tensor()
    dst_tensor._share_data_with(src_tensor)

    return t


class FusedLLAMA(Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        normalize_before=True,
        ln_scale_attrs=None,
        qkv_weight_attrs=None,
        linear_weight_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn1_weight_attrs=None,
        ffn2_weight_attrs=None,
        qkv_weight_scale_attrs=None,
        linear_weight_scale_attrs=None,
        ffn1_weight_scale_attrs=None,
        ffn2_weight_scale_attrs=None,
        epsilon=1e-5,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        name=None, 
        quant_weight=False
    ):
        super().__init__()

        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id
        self._quant_weight=quant_weight

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        assert dim_feedforward % nranks == 0
        num_heads = num_heads // nranks
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward

        if isinstance(qkv_weight_attrs, (list, tuple)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.ln_scales = []
        self.qkv_weights = []
        self.linear_weights = []
        self.ffn_ln_scales = []
        self.ffn1_weights = []
        self.ffn2_weights = []
        self.qkv_weights_scales = []
        self.linear_weights_scales = []
        self.ffn1_weights_scales = []
        self.ffn2_weights_scales = []


        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs
        
        def _add_parameter(param):
            assert param.name not in self._parameters
            self._parameters[param.name] = param

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)

            qkv_weight_scale_attr = get_attr(qkv_weight_scale_attrs, i)
            linear_weight_scale_attr = get_attr(linear_weight_scale_attrs, i)
            ffn1_weight_scale_attr = get_attr(ffn1_weight_scale_attrs, i)
            ffn2_weight_scale_attr = get_attr(ffn2_weight_scale_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
            )
            
            if not self._quant_weight:
                qkv_weight = self.create_parameter(
                    shape=[3, num_heads, self.head_dim, embed_dim]
                    if trans_qkvw
                    else [embed_dim, 3, num_heads, self.head_dim],
                    attr=qkv_weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
            else:
                qkv_weight = self.create_parameter(
                    [embed_dim, 3, num_heads, self.head_dim],
                    attr=qkv_weight_attr,
                    dtype='int8',
                    is_bias=False,
                )

            if not self._quant_weight:
                linear_weight = self.create_parameter(
                    shape=[num_heads * self.head_dim, embed_dim],
                    attr=linear_weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
            else:
                linear_weight = self.create_parameter(
                    shape=[num_heads * self.head_dim, embed_dim],
                    attr=linear_weight_attr,
                    dtype='int8',
                    is_bias=False,
                )

            

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
            )

            if not self._quant_weight:
                ffn1_weight = self.create_parameter(
                    shape=[embed_dim, dim_feedforward * 2], # Since LLAMA use GLU Arch, we need double the dimension. 
                    attr=ffn1_weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
            else:
                ffn1_weight = self.create_parameter(
                    shape=[embed_dim, dim_feedforward * 2], # Since LLAMA use GLU Arch, we need double the dimension. 
                    attr=ffn1_weight_attr,
                    dtype='int8',
                    is_bias=False,
                )

            if not self._quant_weight:
                ffn2_weight = self.create_parameter(
                    shape=[dim_feedforward, embed_dim],
                    attr=ffn2_weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
            else: 
                ffn2_weight = self.create_parameter(
                    shape=[dim_feedforward, embed_dim],
                    attr=ffn2_weight_attr,
                    dtype='int8',
                    is_bias=False,
                )

            qkv_weight_scale = self.create_parameter(
                [3, num_heads, self.head_dim],
                attr=qkv_weight_scale_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            linear_weight_scale = self.create_parameter(
                shape=[embed_dim],
                attr=linear_weight_scale_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            ffn1_weight_scale = self.create_parameter(
                shape=[dim_feedforward * 2],
                attr=ffn1_weight_scale_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            ffn2_weight_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn2_weight_scale_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            
            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)

                _set_var_distributed(ffn1_weight)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

                _set_var_distributed(qkv_weight_scale)
                _set_var_distributed(ffn1_weight_scale)
                _set_var_distributed(linear_weight_scale)
                _set_var_distributed(ffn2_weight_scale)


            self.ln_scales.append(ln_scale)
            self.qkv_weights.append(qkv_weight)
            self.linear_weights.append(linear_weight)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn2_weights.append(ffn2_weight)

            self.qkv_weights_scales.append(qkv_weight_scale)
            self.linear_weights_scales.append(linear_weight_scale)
            self.ffn1_weights_scales.append(ffn1_weight_scale)
            self.ffn2_weights_scales.append(ffn2_weight_scale)

            _add_parameter(ln_scale)
            _add_parameter(qkv_weight)
            _add_parameter(linear_weight)

            _add_parameter(ffn_ln_scale)
            _add_parameter(ffn1_weight)
            _add_parameter(ffn2_weight)

            _add_parameter(qkv_weight_scale)
            _add_parameter(linear_weight_scale)
            _add_parameter(ffn1_weight_scale)
            _add_parameter(ffn2_weight_scale)

        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name

    def forward(
        self,
        src,
        attn_mask=None,
        caches=None,
        pre_caches=None,
        rotary_embs=None,
        rotary_emb_dims=0,
        seq_lens=None,
        time_step=None,
    ):
        r"""
        Applies multi transformer layers on the input.

        Parameters:
            src (Tensor): The input of Transformer layers. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float16 or float32.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                `[batch_size, 1, sequence_length, sequence_length]`. It can be
                None when nothing wanted or needed to be prevented attention to.
                Default None.
            caches (list(Tensor)|tuple(Tensor), optional): The cache structure
                tensors for the inference generation model. It is only used for
                inference and should be None for training. The shape is
                `[2, batch_size, num_head, max_seq_len, head_dim]`. Default None.
            pre_caches (list(Tensor)|tuple(Tensor), optional): The prefix caches
                for the generation model. The shape is `[2, bsz, num\_head, cache\_len, head\_dim]`. Default None.
            rotary_embs (Tensor optional): The RoPE embs for the rotary computation. The shape is `[2, bsz, 1, seq\_len, head\_dim]`. Default None.
            rotary_emb_dims (int, optional): The rotary_emb_dims of rotary computation, and it is 0 when rotary_embs is None,
                1 when rotary_embs is not None and pos_extra_ids is None, 2 when rotary_embs and pos_extra_ids are both not None. Default 0.
            seq_lens (Tensor optional): The sequence lengths of this batch. The shape is `[bsz]`. Default None.
            time_step (Tensor, optional): The time step tensor for the generation
                model. Which used in decode stage, to represent the time step,
                that is, the real seq_len of CacheKV. The shape is `[1]`, must be
                in CPUPlace. Default None.

        Returns:
            Tensor|tuple: If `caches` is None, return a tensor that has
            the same shape and data type with `src`, representing the output
            of Transformer layers. If `caches` is not None, return the
            tuple (output, caches), which output is the output of
            Transformer layers, caches is inplace with input `caches`.
        """

        if caches is not None:
            assert len(caches) == len(self.qkv_weights)
        out = incubate_f.fused_llama(
            src,
            self.ln_scales,
            self.qkv_weights,
            self.linear_weights,
            self.ffn_ln_scales,
            self.ffn1_weights,
            self.ffn2_weights,
            self.qkv_weights_scales,
            self.linear_weights_scales,
            self.ffn1_weights_scales,
            self.ffn2_weights_scales,
            epsilon=self._epsilon,
            cache_kvs=caches,
            pre_caches=pre_caches,
            rotary_embs=rotary_embs,
            time_step=time_step,
            seq_lens=seq_lens,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            rotary_emb_dims=rotary_emb_dims,
            activation=self.activation,
            training=self.training,
            mode='upscale_in_train',
            trans_qkvw=self._trans_qkvw,
            ring_id=self._ring_id,
            name=self.name
        )
        return out
