#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Kevin Lee - kevin_lee@claris.com'

import argparse
import numpy as np
import os
import pdb
import torch
from tqdm import tqdm

from classes.dataset.ImageDataLoader import getDataLoader
from classes.model.pix2code import Pix2Code

from classes.Utils import Utils


def main():

    # Model Hyperparams
    embed_size = 1024
    hidden_size = 1024
    num_layers = 2

    # Dataset paths (For testing purposes, we use a pre-split dataset rather than do it here)
    dev_data_dir = os.path.join(opt.dataset, 'eval_set')

    #Determine device
    device = Utils.get_device(opt.gpu_id)

    # Load vocabulary
    vocab = Utils.build_vocab(opt.vocab)

    vocab_size = len(vocab)

    print(vocab.word2idx)

    # Create data loader
    data_loader = getDataLoader(dev_data_dir, vocab, opt.batch_size)

    for (dirpath, _, filenames) in os.walk(opt.weights_dir):

        for f in filenames:
            if f.endswith('.pkl'):
                weights = os.path.join(dirpath, f)
                print('Weights: ', weights)

                #Load data from checkpoint
                checkpoint = torch.load(weights, map_location=device)
                if 'hyp' in checkpoint:
                    embed_size = checkpoint['hyp']['embed_size']
                    hidden_size = checkpoint['hyp']['hidden_size']
                    num_layers = checkpoint['hyp']['num_layers']
                    assert vocab_size == checkpoint['hyp']['vocab_size'], 'incompatible vocab_sizes {} and {}'.format(vocab_size, checkpoint['hyp']['vocab_size'])


                #Create models
                model = Pix2Code(embed_size, hidden_size, vocab_size, num_layers)

                #Convert device
                model = model.to(device)

                # Load trained models
                model.load_state_dict(checkpoint['model'], strict=True)

                # Calculate BLEU score
                with torch.no_grad():
                    bleu, _ = Utils.eval_bleu_score(model, data_loader, vocab, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset path. Must contain training_set and eval_set subdirectories.')
    parser.add_argument('--vocab', '-v', type=str, required=False, default='../bootstrap.vocab', help='*-config.json path')
    parser.add_argument('--weights-dir', '-w', type=str, required=True, help='weights directory to evaluate')
    parser.add_argument('--batch-size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('--gpu-id', type=int, required=False, default=0, help='GPU ID to use')
    opt = parser.parse_args()
    main()

