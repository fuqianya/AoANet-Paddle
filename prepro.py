"""
prepro.py
~~~~~~~~~

Preprocessing the COCO2014 dataset for training and evluation.
"""
import os
import sys
import csv
import json
import base64
import argparse
from collections import defaultdict

import h5py
import pickle
import numpy as np
from tqdm import tqdm

def build_vocab(params):
    """Filter the words that appear less than threshold times to build vocabulary."""
    count_thresh = params['word_count_threshold']

    with open(params['coco_caption_file'], 'rb') as f:
        obj = json.load(f)
        images = obj['images']

    # count up the number of words
    counts = {}
    for image in images:
        for sent in image['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1

    # filter the words that occur less than count_thresh times
    vocab = [w for w, n in counts.items() if n > count_thresh]

    bad_words = [w for w, n in counts.items() if n <= count_thresh]
    if len(bad_words) > 0:
        # insert UNK token
        vocab.append('UNK')

        # compute how many words will be replaced with UNK
        bad_count = sum(counts[w] for w in bad_words)
        total_words = sum(counts.values())
        print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    return vocab

def precook(s, n=4, out=False):
    """Takes a string as input and returns an object that can be given to either cook_refs or cook_test.
    This is optional: cook_refs and cook_test can take string arguments as well.

    Inputs:
     - s: string : sentence to be converted into ngrams
     - n: int    : number of ngrams for which representation is calculated

     Returns:
         term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4):
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.

    Inputs:
     - refs: list of string : reference sentences for some image
     - n: int : number of ngrams for which (ngram) representation is calculated

    Returns:
        result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))

    return crefs

def compute_doc_freq(crefs):
    """Compute term frequency for reference data. This will be used to compute idf
    (inverse document frequency later). The term frequency is stored in the object"""
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
    return document_frequency

def build_ngram(params, wtoi):
    """Build ngram words."""
    wtoi['<eos>'] = 0

    with open(params['coco_caption_file'], 'rb') as f:
        obj = json.load(f)
        images = obj['images']

    count_imgs = 0
    refs_words = []
    refs_idxs = []
    for image in images:
        if image['split'] == 'train' or image['split'] == 'restval':
            ref_words = []
            ref_idxs = []
            for sent in image['sentences']:
                tmp_tokens = sent['tokens'] + ['<eos>']
                tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
                ref_words.append(' '.join(tmp_tokens))
                ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
            refs_words.append(ref_words)
            refs_idxs.append(ref_idxs)
            count_imgs += 1
    print('total images: ', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))

    return ngram_words, ngram_idxs, count_imgs

def encode_captions(params, wtoi):
    """Encode captions, i.e., convert captions from string into indices based on the wtoi dict."""
    with open(params['coco_caption_file'], 'rb') as f:
        obj = json.load(f)
        images = obj['images']

    N = len(images)
    M = sum(len(image['sentences']) for image in images)  # total number of captions
    max_length = params['max_length']

    label_arrays = []
    # note: these will be one-indexed
    label_start_ix = np.zeros(N, dtype='uint32')
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1

    for i, image in enumerate(images):
        n = len(image['sentences'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(image['sentences']):
            tokens = s['tokens']
            label_length[caption_counter] = min(max_length, len(tokens))
            caption_counter += 1
            for k, w in enumerate(tokens):
                if k < max_length:
                    if w in wtoi: Li[j, k] = wtoi[w]
                    else: Li[j, k] = wtoi['UNK']

        # note: word indices are 1-indexed,
        # and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'
    print('encoded captions to array of size ', L.shape)

    return L, label_start_ix, label_end_ix, label_length

def process_captions(params):
    """Prepare vocabulary and encoded captions."""
    # build vocab
    vocab = build_vocab(params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # 1-indexed
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    # encode captions into indices
    L, label_start_ix, label_end_ix, label_length = encode_captions(params, wtoi)

    # dump encoded captions into h5 file
    with h5py.File(params['output_h5'], 'w') as f:
        f.create_dataset('labels', dtype='uint32', data=L)  # encoded captions
        # start and end idx for each image
        f.create_dataset('label_start_ix', dtype='uint32', data=label_start_ix)
        f.create_dataset('label_end_ix', dtype='uint32', data=label_end_ix)
        # length of each caption
        f.create_dataset('label_length', dtype='uint32', data=label_length)

    # create output json file
    output_json_obj = {}
    output_json_obj['ix_to_word'] = itow
    output_json_obj['images'] = []

    with open(params['coco_caption_file'], 'rb') as f:
        obj = json.load(f)
        images = obj['images']

    for i, image in enumerate(images):
        img = {}
        img['split'] = image['split']
        img['filename'] = image['filename']
        img['id'] = image['cocoid']
        output_json_obj['images'].append(img)

    # dump into json file
    with open(params['output_json'], 'w') as f:
        json.dump(output_json_obj, f)

def prepare_ngrams(params):
    """Prepare ngrams for calculating cider score."""
    with open(params['output_json'], 'r') as f:
        obj = json.load(f)
        itow = obj['ix_to_word']

    wtoi = {w:i for i, w in itow.items()}
    ngram_words, ngram_idxs, ref_len = build_ngram(params, wtoi)

    # dump into pickle file
    with open(params['output_pkl'] + '-words.pkl', 'wb') as f:
        pickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, f)

    with open(params['output_pkl'] + '-idxs.p', 'wb') as f:
        pickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, f)

def prepare_features(params):
    # using csv to read feature
    csv.field_size_limit(sys.maxsize)

    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    infiles = ['karpathy_test_resnet101_faster_rcnn_genome.tsv',
               'karpathy_val_resnet101_faster_rcnn_genome.tsv',
               'karpathy_train_resnet101_faster_rcnn_genome.tsv.0',
               'karpathy_train_resnet101_faster_rcnn_genome.tsv.1']

    output_feature_dir = params['output_feature_dir']
    if not os.path.isdir(output_feature_dir):
        os.makedirs(output_feature_dir)

    for infile in infiles:
        print('Reading ' + infile)
        with open(os.path.join(params['coco_feature_dir'], infile), "r") as f:
            reader = csv.DictReader(f, delimiter='\t', fieldnames=FIELDNAMES)
            for item in tqdm(reader):
                item['image_id'] = int(item['image_id'])
                item['num_boxes'] = int(item['num_boxes'])

                item['features'] = np.frombuffer(base64.decodebytes(item['features'].encode()), dtype=np.float32). \
                    reshape((item['num_boxes'], -1))

                save_path = os.path.join(output_feature_dir, str(item['image_id']))
                np.savez(save_path, att_feat=item['features'], fc_feat=item['features'].mean(0))

def main(params):
    # prepare captions
    print('Start to process captions ... ')
    process_captions(params)

    # prepare ngrams for calculating cider score
    print('Start to prepare ngram words ... ')
    prepare_ngrams(params)

    # prepare features
    print('Start to process features ... ')
    prepare_features(params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input file
    parser.add_argument('--coco_caption_file', type=str, default='./data/dataset_coco.json',
                        help='path to the caption file of coco2014.')
    parser.add_argument('--coco_feature_dir', type=str, default='./data/trainval/',
                        help='folder to the feature file of coco2014.')

    # output file
    parser.add_argument('--output_json', type=str, default='./data/cocotalk.json',
                        help='output json file which contains')
    parser.add_argument('--output_h5', type=str, default='./data/cocotalk.h5',
                        help='output h5 file which contains encoded captions.')
    parser.add_argument('--output_pkl', type=str, default='./data/cocotalk',
                        help='output pickle file which contains.')
    parser.add_argument('--output_feature_dir', type=str, default='./data/feats',
                        help='folder to be store the preprocessed features.')

    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=4, type=int,
                        help='only words that occur more than this number of times will be put in vocab.')

    opt = parser.parse_args()
    params = vars(opt)  # convert to dictionary

    # call main()
    main(params)