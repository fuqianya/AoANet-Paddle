"""
utils.py
~~~~~~~~

A module contains common utils used in this project.
"""
import os
import math
import copy
import pickle
import sys
import numpy as np
from collections import OrderedDict

# paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# CIDERD
sys.path.append('cider')
from pyciderevalcap.ciderD.ciderD import CiderD
# BLEU
sys.path.append('coco-caption')
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j-1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words)+flag])
        out.append(txt.replace('@@ ', ''))
    return out

def clones(module, N):
    "Produce N identical layers."
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention.
    Inputs:
     - query: [batch_size, num_heads, num_objects, feat_dim]
     - key: [batch_size, num_heads, num_objects, feat_dim]
     - value: [batch_size, num_heads, num_objects, feat_dim]
     - mask: [batch_size, 1, 1, num_objects]
     - dropout: dropout prob.
    """
    d_k = query.shape[-1]
    # [batch_size, num_heads, num_objects, num_objects]
    scores = paddle.matmul(query, key.transpose([0, 1, 3, 2])) / math.sqrt(d_k)

    if mask is not None:
        scores_mask = paddle.fluid.layers.fill_constant(shape=scores.shape, dtype=scores.dtype, value=-1e9)
        scores = paddle.where(paddle.broadcast_to(mask, shape=scores.shape)!=0, scores, scores_mask)
    p_attn = F.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return paddle.matmul(p_attn, value), p_attn

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
    batch_size = gen_result.shape[0]  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = OrderedDict()

    gen_result = gen_result.numpy()
    greedy_res = greedy_res.numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        #print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        #print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

def save_checkpoint(model, infos, optimizer, opt, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    # if checkpoint_path doesn't exist
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pdparams' %(append))
    paddle.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pdparams' %(append))
    paddle.save(optimizer.state_dict(), optimizer_path)
    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
        pickle.dump(infos, f)
    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            pickle.dump(histories, f)

class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."
    def __init__(self, init_lr, model, opt, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(init_lr, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose)
        self.optimizer = paddle.optimizer.Adam(learning_rate=self.scheduler,
                                               parameters=model.parameters(),
                                               beta1=opt.optim_alpha, beta2=opt.optim_beta,
                                               epsilon=opt.optim_epsilon,
                                               grad_clip=paddle.fluid.clip.ClipGradByValue(opt.grad_clip))
        self.current_lr = self.optimizer.get_lr()

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def clear_grad(self):
        """Clear the grad."""
        self.optimizer.clear_grad()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = self.optimizer.get_lr()

    def state_dict(self):
        return {'current_lr':self.current_lr,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.set_state_dict(state_dict)
            self.optimizer.set_lr(self.current_lr)
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.set_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.set_state_dict(state_dict['optimizer_state_dict'])
