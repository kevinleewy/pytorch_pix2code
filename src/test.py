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


# Hyperparams
batch_size = 32
embed_size = 256
num_epochs = 1000
learning_rate = 0.005
hidden_size = 512
num_layers = 2

# Other params
shuffle = True
num_workers = 2

# Logging/Saving Variables
save_after_x_epochs = 10
log_step = 5

# Paths
data_dir = '../datasets/web/training_set/' # For testing purposes, we use a pre-split dataset rather than do it here
dev_data_dir = '../datasets/web/eval_set/'
model_path = '../weights'
vocab_path = '../bootstrap.vocab'

# DO NOT CHANGE:
crop_size = 224 # Required by resnet152

def main():

    # Load vocabulary
    vocab = build_vocab(vocab_path)

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
    img_html_dataset = ImageDataset(data_dir=data_dir, vocab=vocab, transform=transform)
    data_loader = DataLoader(dataset=img_html_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    dev_img_html_dataset = ImageDataset(data_dir=dev_data_dir, vocab=vocab, transform=transform)
    dev_data_loader = DataLoader(dataset=dev_img_html_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)         

    bleu_scores = []

    # Initialize models
    dev_encoder = EncoderCNN(embed_size).to(device)
    dev_decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    for model_idx, model_name in enumerate(models_to_test):
        
        # Load trained models
        # encoder_model_path = os.path.join(model_path, 'encoder-{}.pkl'.format(model_name))
        # decoder_model_path = os.path.join(model_path, 'decoder-{}.pkl'.format(model_name))
        # dev_encoder.load_state_dict(torch.load(encoder_model_path))
        # dev_decoder.load_state_dict(torch.load(decoder_model_path))

        checkpoint_path = os.path.join(model_path, model_name)
        checkpoint = torch.load(checkpoint_path)
        dev_encoder.load_state_dict(checkpoint['encoder'])
        dev_decoder.load_state_dict(checkpoint['decoder'])

        bleu, count = Utils.eval_bleu_score(dev_encoder, dev_decoder, dev_data_loader, vocab, device)    
        
        bleu_scores.append((model_name, bleu))

        print('done with {} items for model: {}'.format(str(count), model_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', '-o', type=str, required=True, help='output path for saving model weights')
    parser.add_argument('--config', '--cfg', '-c', type=str, required=True, help='*-config.json path')
    parser.add_argument('--load', type=str, help='weights file to preload the model with')
    parser.add_argument('--resume', action='store_true', help='resume training')
    opt = parser.parse_args()
    main()

