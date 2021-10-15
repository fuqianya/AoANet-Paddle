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
train.py
~~~~~~~~

A script to eval the captioner.
"""
import json
import os
import argparse
import pickle

# paddle
import paddle

# config
from config.config import add_eval_options
# model
from model.AoAModel import AoAModel
# dataloader
from model.dataloader import get_dataloaders
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

    # set up dataloader
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

    # set crit to None
    # since we do not care about loss when evaluate the model
    crit = None

    _, _, lang_stats = eval_split(model, crit, test_loader, vars(opt))

    if lang_stats:
        # output results
        print(lang_stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./log/log_aoa_rl/model-best.pdparams',
                        help='path to model to evaluate.')
    parser.add_argument('--infos_path', type=str, default='./log/log_aoa_rl/infos_aoa-best.pkl',
                        help='path to infos to evaluate.')

    add_eval_options(parser)

    opt = parser.parse_args()

    # call main
    main(opt)
