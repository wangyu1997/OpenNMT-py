# -*- coding: utf-8 -*-
"""Average Attention module."""

from abc import ABC

import torch
import torch.nn as nn

from onmt.modules.position_ffn import PositionwiseFeedForward


def _cumulative_average_mask(batch_size, inputs_len, device):
    """
    Builds the mask to compute the cumulative average as described in
    :cite:`DBLP:journals/corr/abs-1805-00631` -- Figure 3

    Args:
        batch_size (int): batch size
        inputs_len (int): length of the inputs
        device

    Returns:
        (FloatTensor):

        * A Tensor of shape ``(batch_size, input_len, input_len)``
    """

    triangle = torch.tril(torch.ones(inputs_len, inputs_len,
                                     dtype=torch.float, device=device))
    weights = torch.ones(1, inputs_len, dtype=torch.float, device=device) \
              / torch.arange(1, inputs_len + 1, dtype=torch.float, device=device)

    # * 使用了broadcast
    # 1
    # 1/2 1/2
    # 1/3 1/3 1/3

    mask = triangle * weights.transpose(0, 1)

    return mask.unsqueeze(0).expand(batch_size, inputs_len, inputs_len)


class AverageAttention(nn.Module, ABC):
    """
    Average Attention module from
    "Accelerating Neural Transformer via an Average Attention Network"
    :cite:`DBLP:journals/corr/abs-1805-00631`.

    Args:
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, model_dim, dropout=0.1, aan_use_ffn=False):
        self.model_dim = model_dim
        self.aan_use_ffn = aan_use_ffn
        super(AverageAttention, self).__init__()
        if aan_use_ffn:
            self.average_layer = PositionwiseFeedForward(model_dim, model_dim,
                                                         dropout)
        self.gating_layer = nn.Linear(model_dim * 2, model_dim * 2)

    @staticmethod
    def cumulative_average(inputs, mask_or_step,
                           layer_cache=None):
        """
        Computes the cumulative average as described in
        :cite:`DBLP:journals/corr/abs-1805-00631` -- Equations (1) (5) (6)

        Args:
            inputs (FloatTensor): sequence to average
                ``(batch_size, input_len, dimension)``
            mask_or_step: if cache is set, this is assumed
                to be the current step of the
                dynamic decoding. Otherwise, it is the mask matrix
                used to compute the cumulative average.
            layer_cache: a dictionary containing the cumulative average
                of the previous step.

        Returns:
            a tensor of the same shape and type as ``inputs``.
        """

        if layer_cache is not None:
            step = mask_or_step
            # (out_3*3+input_4)/4
            # 每次求的都是均值
            average_attention = (inputs + step *
                                 layer_cache["prev_g"]) / (step + 1)
            layer_cache["prev_g"] = average_attention
            return average_attention
        else:
            mask = mask_or_step
            return torch.matmul(mask.to(inputs.dtype), inputs)

    def forward(self, inputs, layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, input_len, model_dim)``
            layer_cache
            step

        Returns:
            (FloatTensor, FloatTensor):

            * gating_outputs ``(batch_size, input_len, model_dim)``
            * average_outputs average attention
                ``(batch_size, input_len, model_dim)``
        """

        batch_size, inputs_len, _ = inputs.size()

        average_outputs = self.cumulative_average(
            inputs, _cumulative_average_mask(batch_size,
                                             inputs_len, inputs.device)
            if layer_cache is None else step, layer_cache=layer_cache)

        if self.aan_use_ffn:
            average_outputs = self.average_layer(average_outputs)

        gating_outputs = self.gating_layer(torch.cat((inputs,
                                                      average_outputs), -1))
        input_gate, forget_gate = torch.chunk(gating_outputs, 2, dim=2)

        gating_outputs = torch.sigmoid(input_gate) * inputs + \
                         torch.sigmoid(forget_gate) * average_outputs

        return gating_outputs, average_outputs
