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
from torchvision import transforms
from PIL import Image

from classes.dataset.ImageDataset import *
from classes.model.pix2code import *

from classes.Utils import *

def main():
    # Model Hyperparams
    embed_size = 512
    learning_rate = 0.001
    hidden_size = 512
    num_layers = 1

    # DO NOT CHANGE:
    crop_size = 224 # Required by resnet152

    #Determine device
    device = Utils.get_device()

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

    #Load data from checkpoint
    checkpoint = None
    if opt.weights != '':
        checkpoint = torch.load(opt.weights, map_location=device)
        if 'hyp' in checkpoint:
            embed_size = checkpoint['hyp']['embed_size']
            hidden_size = checkpoint['hyp']['hidden_size']
            num_layers = checkpoint['hyp']['num_layers']
            assert vocab_size == checkpoint['hyp']['num_layers'], 'incompatible vocab_size'


    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # Load trained models
    enc_load_succ = encoder.load_state_dict(checkpoint['encoder'])
    print(enc_load_succ)
    dec_load_succ = decoder.load_state_dict(checkpoint['decoder'])
    print(dec_load_succ)

    #Set models to eval mode
    encoder.eval()
    decoder.eval()

    test_data_path = './datasets/web/eval_set'

    output_path = './output'
    model_name = 'best.pkl'

    # Get image from filesystem
    assert opt.input.endswith('.png'), 'unsupported input format'
    image = Image.open(os.path.join(test_data_path, file_name + '.png')).convert('RGB')
    plt.imshow(image);
    image = transform(image)

    image_tensor = Variable(image.unsqueeze(0).cuda())

    features = encoder(image_tensor)

    sampled_ids = decoder.sample(features)
    sampled_ids = sampled_ids.cpu().data.numpy()

    predicted = Utils.transform_idx_to_words(sampled_ids)

    predicted = ''.join(predicted)
    predicted = predicted.replace('{', "{\n").replace('}', "\n}\n").replace('}\n\n}', "}\n}")

    #Strip START and END tokens
    predicted = predicted.replace(START_TOKEN, "").replace(END_TOKEN, "")

    output_path = opt.output
    if not output_path.endswith('.gui'):
        output_path += '.gui'

    #Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Open file and write
    with open(output_path, 'w') as out_f:
        out_f.write(predicted)

    print('done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='input image path')
    parser.add_argument('--output', '-o', type=str, required=True, help='output path (<path>/*.gui)')
    parser.add_argument('--vocab', '-v', type=str, required=False, default='../bootstrap.vocab', help='*-config.json path')
    parser.add_argument('--weights', '-w', type=str, required=True, default='', help='weights to preload into model')
    opt = parser.parse_args()
    main()

