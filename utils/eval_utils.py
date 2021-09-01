"""
eval_utils.py
~~~~~~~~~~~~~

A module that contains utils for evaluation.
"""
import os
import json
import numpy as np

# paddle
import paddle

# utils
from utils import utils

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'coco-caption/f30k_captions4eval.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', False)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 0)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    label_smoothing = eval_kwargs.get('remove_bad_endings', 0.0)
    # Use this nasty way to make other code clean since it's a global configuration
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)

    # set mode
    model.eval()

    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    for fc_feats, att_feats, att_masks, labels, masks, gts, infos in loader:

        if labels is not None and verbose_loss:
            # forward the model to get loss
            with paddle.no_grad():
                logit_pred, prob_pred = model(fc_feats, att_feats, labels, att_masks)
                if label_smoothing > 0:
                    loss = crit(prob_pred, labels[:, 1:], masks[:, 1:]).item()
                else:
                    loss = crit(logit_pred, labels[:, 1:], masks[:, 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [fc_feats[paddle.arange(loader.dataset.batch_size) * loader.dataset.seq_per_img],
               att_feats[paddle.arange(loader.dataset.batch_size) * loader.dataset.seq_per_img],
               att_masks[paddle.arange(loader.dataset.batch_size) * loader.dataset.seq_per_img] if att_masks is not None else None]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        with paddle.no_grad():
            seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0]

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.dataset.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.dataset.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': infos[k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = infos[k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], infos[k]['file_path']) + \
                      '" vis/imgs/img' + str(len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()

    return loss_sum/loss_evals, predictions, lang_stats
