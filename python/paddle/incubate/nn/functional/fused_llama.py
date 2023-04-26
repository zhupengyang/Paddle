#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import _legacy_C_ops
from paddle.fluid import core
from paddle.fluid.data_feeder import check_dtype, check_variable_and_dtype
from paddle.fluid.framework import _non_static_mode, default_main_program
from paddle.fluid.layer_helper import LayerHelper

__all__ = []


def _verify_dropout_rate(dropout_rate):
    if not isinstance(dropout_rate, (float, int)):
        raise TypeError("dropout_rate argument should be a number")
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate argument should between 0 and 1")

def fused_llama(
    x,
    ln_scales,
    qkv_weights,
    linear_weights,
    ffn_ln_scales,
    ffn1_weights,
    ffn2_weights,
    pre_layer_norm=True,
    epsilon=1e-6,
    cache_kvs=None,
    pre_caches=None,
    seq_lens=None,
    rotary_embs=None,
    time_step=None,
    attn_mask=None,
    dropout_rate=0.0,
    rotary_emb_dims=0,
    activation="gelu",
    training=False,
    mode='upscale_in_train',
    trans_qkvw=True,
    ring_id=-1,
    name=None
):
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
        )
    mode = (
        'downgrade_in_infer' if mode == 'downscale_in_infer' else mode
    )  # semantic transfer

    if _non_static_mode():
        cache_kv_out, final_out = _legacy_C_ops.fused_llama(
            x,
            ln_scales,
            qkv_weights,
            cache_kvs,
            pre_caches,
            rotary_embs,
            time_step,
            seq_lens,
            attn_mask,
            linear_weights,
            ffn_ln_scales,
            ffn1_weights,
            ffn2_weights,
            cache_kvs,
            'epsilon',
            epsilon,
            'dropout_rate',
            dropout_rate,
            'rotary_emb_dims',
            rotary_emb_dims,
            'is_test',
            not training,
            'dropout_implementation',
            mode,
            'act_method',
            activation,
            'trans_qkvw',
            trans_qkvw,
            'ring_id',
            ring_id,
        )
        if cache_kvs is not None:
            return final_out, cache_kv_out
        return final_out
    else:
        helper = LayerHelper('fused_llama', **locals())
        dtype = x.dtype
        # check dtypes
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32'], 'fused_llama'
        )
        check_dtype(
            dtype, 'dtype', ['float16', 'float32'], 'fused_llama'
        )

        # set inputs
        inputs = {}
        inputs['X'] = [x]
        inputs['LnScale'] = ln_scales
        inputs['QKVW'] = qkv_weights

        if cache_kvs is not None:
            assert len(cache_kvs) == len(qkv_weights)
            inputs['CacheKV'] = cache_kvs
            if time_step is not None:
                inputs['TimeStep'] = time_step
        if pre_caches is not None:
            inputs['PreCaches'] = pre_caches
        if rotary_emb_dims > 0:
            inputs['RotaryPosEmb'] = rotary_embs
        inputs['SeqLengths'] = seq_lens
        inputs['SrcMask'] = attn_mask
        inputs['OutLinearW'] = linear_weights

        inputs['FFNLnScale'] = ffn_ln_scales
        inputs['FFN1Weight'] = ffn1_weights
        inputs['FFN2Weight'] = ffn2_weights

        # set attrs
        attrs = {
            'epsilon': epsilon,
            'dropout_rate': dropout_rate,
            'rotary_emb_dims': rotary_emb_dims,
            'is_test': not training,
            'dropout_implementation': mode,
            'act_method': activation,
            'trans_qkvw': trans_qkvw,
            'ring_id': ring_id,
        }

        outputs = {}
        final_out = helper.create_variable_for_type_inference(dtype=dtype)
        outputs['Out'] = final_out
        if cache_kvs:
            # NOTE: inplace
            outputs['CacheKVOut'] = cache_kvs

        helper.append_op(
            type='fused_llama',
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

        return (final_out, cache_kvs) if cache_kvs else final_out
