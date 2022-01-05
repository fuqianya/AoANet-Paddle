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
predict.py
~~~~~~~~

A script to run prediction on sigle image.
"""
import json
import os
import argparse
import pickle
import numpy as np

# paddle
import paddle

# config
from config.config import add_eval_options
# model
from model.AoAModel import AoAModel
# dataloader
from model.dataloader import get_dataloaders
# utils
from utils import utils
# eval utils
from utils.eval_utils import eval_split



def main(opt):
    # load infos
    with open(opt.infos_path, 'rb') as f:
        infos = pickle.load(f)

    # collect parameter
    for k in vars(infos['opt']).keys():
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy options from model
    
    # set up dataloader (only for meta data)
    _, _, test_loader = get_dataloaders(opt)
    # when eval using provided pretrained mode, the vocab may be different from what you have in cocotalk.json
    # so make sure to use the vocab in infos file
    test_loader.ix_to_word = infos['vocab']

    # set up model
    vocab = infos['vocab']
    opt.vocab = vocab
    opt.vocab_size = test_loader.dataset.vocab_size
    model = AoAModel(opt)
    del opt.vocab

    # load state_dict
    model.set_state_dict(paddle.load(opt.model))
    # set mode
    model.eval()

    # load pre-extract image features
    feats = np.load('data/573223.npz')
    att_feat = feats['att_feat']
    att_feat = att_feat.reshape(1, -1, att_feat.shape[-1])  # [num_object, feat_dim]
    fc_feat = feats['fc_feat']
    fc_feat = fc_feat.reshape(-1, fc_feat.shape[-1])
    att_feat = paddle.to_tensor(att_feat)
    fc_feat = paddle.to_tensor(fc_feat)

    # forward the model to also get generated samples for each image
    with paddle.no_grad():
        eval_kwargs={'beam_size': 2}
        seq = model(fc_feat, att_feat, att_masks=None, opt=eval_kwargs, mode='sample')[0]
        sents = utils.decode_sequence(test_loader.dataset.get_vocab(), seq)
        print('prediction: ', sents[0])

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./log/log_aoa_rl/model.pdparams',
                        help='path to model to evaluate.')
    parser.add_argument('--infos_path', type=str, default='./log/log_aoa_rl/infos_aoa.pkl',
                        help='path to infos to evaluate.')

    add_eval_options(parser)

    opt = parser.parse_args()

    # call main
    main(opt)
