# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
loss.py
~~~~~~~

Loss functions used to compute loss.
"""

import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.fluid.layers import unsqueeze

from utils.utils import init_scorer, get_self_critical_reward

class RewardCriterion(nn.Layer):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.reshape((-1, ))
        reward = reward.reshape((-1, ))
        mask = paddle.to_tensor((seq > 0), dtype='float32')
        mask = paddle.concat(x=[fluid.layers.fill_constant(shape=[mask.shape[0], 1], value=1, dtype='float32'),
                                mask[:, :-1]], axis=1).reshape((-1, ))
        output = - input * reward * mask
        output = paddle.sum(output) / paddle.sum(mask)

        return output

class XECriterion(nn.Layer):
    def __init__(self):
        super(XECriterion, self).__init__()

    def forward(self, pred, target, mask):
        """Inputs:
         - pred: logits has shape of [batch_size, seq_len, vocab_size].
         - target: [batch_size, seq_len].
         - mask: [batch_size, seq_len].
        """
        # truncate to the same size
        target = target[:, :pred.shape[1]]
        mask = mask[:, :pred.shape[1]]

        loss_ = F.cross_entropy(pred, target, reduction='none')
        loss_ *= mask

        return paddle.sum(loss_) / paddle.sum(mask)

class LossWrapper(nn.Layer):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = XECriterion()
        self.rl_crit = RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices, sc_flag):
        out = {}
        if not sc_flag:
            self.model.train()
            pred = self.model(fc_feats, att_feats, labels, att_masks)
            loss = self.crit(pred, labels[:, 1:], masks[:, 1:])

        else:
            self.model.eval()
            with paddle.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method': 'sample'},
                                                     mode='sample')
            gts = [gts[_].numpy().astype('uint32') for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = paddle.to_tensor(reward, dtype='float32')
            loss = self.rl_crit(sample_logprobs, gen_result, reward)
            out['reward'] = reward[:, 0].mean()
        out['loss'] = loss
        return out
