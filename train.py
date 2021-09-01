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

A script to train the captioner.
"""
import os
import time
import json
import pickle
import random
import traceback

from tqdm import tqdm

# paddle
import paddle
# options
from config.config import parse_opt
# model
from model.AoAModel import AoAModel
# dataloader
from model.dataloader import get_dataloaders
# criterion
from model.loss import LossWrapper
# utils
from utils import utils, eval_utils

def main(opt):
    # set up loader
    train_loader, val_loader, _ = get_dataloaders(opt)
    opt.vocab_size = train_loader.dataset.vocab_size
    opt.seq_length = train_loader.dataset.seq_length

    # training info and history
    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = pickle.load(f)

        if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
            with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
                histories = pickle.load(f)
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['vocab'] = train_loader.dataset.get_vocab()

    infos['opt'] = opt

    # get training history if opt.start_from else training from scratch
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    # restore best_val_score
    best_val_score = infos.get('best_val_score', None)

    # set up model
    opt.vocab = train_loader.dataset.get_vocab()
    model = AoAModel(opt)
    del opt.vocab

    # restore model and continue to train
    if opt.start_from is not None:
        model.set_state_dict(paddle.load(os.path.join(opt.start_from, 'model.pth')))
        print('Load state dict from %s.' % os.path.join(opt.start_from, 'model.pth'))

    # set up criterion
    criterion = LossWrapper(model, opt)
    model.train()

    # set up optimizer
    if opt.reduce_on_plateau:
        optimizer = utils.ReduceLROnPlateau(opt.learning_rate, model, opt, factor=0.5, patience=3)
    else:
        optimizer = paddle.optimizer.Adam(learning_rate=opt.learning_rate,
                                          parameters=model.parameters(),
                                          beta1=opt.optim_alpha, beta2=opt.optim_beta,
                                          epsilon=opt.optim_epsilon,
                                          grad_clip=paddle.fluid.clip.ClipGradByValue(opt.grad_clip))

    try:
        iteration = 0
        for epoch in range(opt.max_epochs):
            # update state before each epoch
            # update learning rate
            if not opt.reduce_on_plateau:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                optimizer.set_lr(opt.current_lr)
            else:
                opt.current_lr = optimizer.current_lr

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                utils.init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            start = time.time()
            for fc_feats, att_feats, att_masks, labels, masks, gts, _ in train_loader:
                model_out = criterion(fc_feats, att_feats, labels, masks, att_masks, gts,
                                      paddle.arange(0, len(gts)), sc_flag)
                loss = model_out['loss']

                # clear the gard
                optimizer.clear_grad()
                # backward
                loss.backward()
                # step
                optimizer.step()

                # Write the training loss summary
                if iteration % opt.losses_log_every == 0:
                    if not sc_flag:
                        print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                              .format(iteration, epoch, loss.item(), time.time() - start))
                    else:
                        print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                              .format(iteration, epoch, model_out['reward'].mean().item(), time.time() - start))

                    loss_history[iteration] = loss.item() if not sc_flag else model_out['reward'].mean().item()
                    lr_history[iteration] = opt.current_lr
                    ss_prob_history[iteration] = model.ss_prob

                # make evaluation on validation set, and save model
                if ((iteration+1) % opt.save_checkpoint_every == 0):
                    # eval model
                    eval_kwargs = {'split': 'val',
                                   'dataset': opt.input_json}
                    eval_kwargs.update(vars(opt))
                    val_loss, predictions, lang_stats = eval_utils.eval_split(
                        model, criterion.crit, val_loader, eval_kwargs)

                    if opt.reduce_on_plateau:
                        if 'CIDEr' in lang_stats:
                            optimizer.scheduler_step(-lang_stats['CIDEr'])
                        else:
                            optimizer.scheduler_step(val_loss)

                    val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                                     'predictions': predictions}

                    # Save model if is improving on validation result
                    if opt.language_eval == 1:
                        current_score = lang_stats['CIDEr']
                    else:
                        current_score = - val_loss

                    best_flag = False

                    if best_val_score is None or current_score > best_val_score:
                        best_val_score = current_score
                        best_flag = True

                    # Dump miscalleous informations
                    infos['best_val_score'] = best_val_score
                    histories['val_result_history'] = val_result_history
                    histories['loss_history'] = loss_history
                    histories['lr_history'] = lr_history
                    histories['ss_prob_history'] = ss_prob_history

                    utils.save_checkpoint(model, infos, optimizer, opt, histories)
                    if opt.save_history_ckpt:
                        utils.save_checkpoint(model, infos, optimizer, opt, append=str(iteration))

                    if best_flag:
                        utils.save_checkpoint(model, infos, optimizer, opt, append='best')

                # Update the iteration and start time
                iteration += 1
                start = time.time()

                # update infos
                infos['iter'] = iteration
                infos['epoch'] = epoch

    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(model, infos, optimizer, opt)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
