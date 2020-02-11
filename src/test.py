#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Kevin Lee - kevin_lee@claris.com'

import argparse
import numpy as np
import os
import pdb
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from PIL import Image

from classes.dataset.ImageDataset import *
from classes.model.pix2code import *

from classes.Utils import *


def main():

    # Model Hyperparams
    embed_size = 1024
    hidden_size = 1024
    num_layers = 2

    # Other params
    shuffle = True
    num_workers = 2

    # Dataset paths (For testing purposes, we use a pre-split dataset rather than do it here)
    dev_data_dir = os.path.join(opt.dataset, 'eval_set')

    # DO NOT CHANGE:
    crop_size = 224 # Required by resnet152

    #Determine device
    device = Utils.get_device(opt.gpu_id)

    # Load vocabulary
    vocab = Utils.build_vocab(opt.vocab)

    vocab_size = len(vocab)

    print(vocab.word2idx)

    # See https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
    def collate_fn (data):
        # Sort datalist by caption length; descending order
        data.sort(key = lambda data_pair: len(data_pair[1]), reverse=True)
        images, captions = zip(*data)
        
        # Merge images (from tuple of 3D Tensor to 4D Tensor)
        images = torch.stack(images, 0)
        
        # Merge captions (from tuple of 1D tensor to 2D tensor)
        lengths = [len(caption) for caption in captions] # List of caption lengths
        targets = torch.zeros(len(captions), max(lengths)).long()
        
        for i, caption in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = caption[:end]
            
        return images, targets, lengths

    # Transform to modify images for pre-trained ResNet base
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)), # Match resnet size
        transforms.ToTensor(),
        # See for magic #'s: http://pytorch.org/docs/master/torchvision/models.html
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create data loaders
    dev_img_html_dataset = ImageDataset(data_dir=dev_data_dir, vocab=vocab, transform=transform)
    dev_data_loader = DataLoader(dataset=dev_img_html_dataset,
                            batch_size=opt.batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)         


    for (dirpath, _, filenames) in os.walk(opt.weights_dir):

        for f in filenames:
            if f.endswith('.pkl'):
                weights = os.path.join(dirpath, f)

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
                    bleu, _ = Utils.eval_bleu_score(model, dev_data_loader, vocab, device)
                    print('Weights: ', weights)
                    print('BLEU score: ', bleu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset path. Must contain training_set and eval_set subdirectories.')
    parser.add_argument('--vocab', '-v', type=str, required=False, default='../bootstrap.vocab', help='*-config.json path')
    parser.add_argument('--weights-dir', '-w', type=str, required=True, help='weights directory to evaluate')
    parser.add_argument('--batch-size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('--gpu-id', type=int, required=False, default=0, help='GPU ID to use')
    opt = parser.parse_args()
    main()

