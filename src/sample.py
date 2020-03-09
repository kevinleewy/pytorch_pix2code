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
from torchvision import transforms
from PIL import Image

from .classes.dataset.ImageDataLoader import transform
from .classes.model.pix2code import Pix2Code

from .classes.Utils import Utils
from .classes.Vocabulary import *

def build_model_and_vocab(vocab_path, weights_path, device='cpu'):

    # Model Hyperparams
    embed_size = 1024
    hidden_size = 1024
    num_layers = 2

    # Load vocabulary
    vocab = Utils.build_vocab(vocab_path)

    vocab_size = len(vocab)

    print(vocab.word2idx)

    #Load data from checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    if 'hyp' in checkpoint:
        embed_size = checkpoint['hyp']['embed_size']
        hidden_size = checkpoint['hyp']['hidden_size']
        num_layers = checkpoint['hyp']['num_layers']
        assert vocab_size == checkpoint['hyp']['vocab_size'], 'incompatible vocab_sizes {} and {}'.format(vocab_size, checkpoint['hyp']['vocab_size'])


    # Create models
    model = Pix2Code(embed_size, hidden_size, vocab_size, num_layers)

    # Convert device
    model = model.to(device)

    # Load trained models
    model.load_state_dict(checkpoint['model'], strict=True)

    # Set model to eval mode
    model.eval()

    return model, vocab

def sample(image, model, vocab, device='cpu'):

    image = image.convert('RGB')
    image = transform(image)

    image_tensor = Variable(image.unsqueeze(0).to(device))

    #Sample
    sampled_ids = model.sample(image_tensor)

    #Convert tensor to numpy array
    sampled_ids = sampled_ids.cpu().data.numpy()

    predicted = Utils.transform_idx_to_words(vocab, sampled_ids)

    predicted = ''.join(predicted)
    predicted = predicted.replace('{', "{\n").replace('}', "\n}\n").replace('}\n\n}', "}\n}")

    #Strip START and END tokens
    predicted = predicted.replace(START_TOKEN, "").replace(END_TOKEN, "")

    return predicted

def main():

    #Determine device
    device = Utils.get_device(opt.gpu_id)

    # Load model and vocab
    model, vocab = build_model_and_vocab(opt.vocab, opt.weights, device)

    # Get image from filesystem
    assert opt.input.endswith('.png'), 'unsupported input format'

    image = Image.open(opt.input)

    # Sample
    predicted = sample(image, model, vocab, device)

    if(opt.output):
        output_path = opt.output
        
        if not output_path.endswith('.gui'):
            output_path += '.gui'

        output_dir = os.path.dirname(output_path)
    else:
        output_dir = os.path.dirname(opt.input)
        output_path = opt.input.replace('.png', '.gui')

    #Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Open file and write
    with open(output_path, 'w') as out_f:
        out_f.write(predicted)

    print('done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='input image path')
    parser.add_argument('--output', '-o', type=str, required=False, help='output path (<path>/*.gui)')
    parser.add_argument('--vocab', '-v', type=str, required=True, default='../bootstrap.vocab', help='*-config.json path')
    parser.add_argument('--weights', '-w', type=str, required=True, default='', help='weights to preload into model')
    parser.add_argument('--gpu-id', type=int, required=False, default=0, help='GPU ID to use')
    opt = parser.parse_args()
    main()

