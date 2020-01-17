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

from classes.dataset.ImageHTMLDataSet import *
from classes.model.pix2code import *

from .Utils import *


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

# Vocab
START_TOKEN = '<START>'
END_TOKEN = '<END>'

# Logging/Saving Variables
save_after_x_epochs = 10
log_step = 5

# Paths
data_dir = './datasets/web/training_set/' # For testing purposes, we use a pre-split dataset rather than do it here
dev_data_dir = './datasets/web/eval_set/'
model_path = '/content/gdrive/My Drive/colab/pix2code/weights'
vocab_path = '/content/gdrive/My Drive/colab/pix2code/bootstrap.vocab'

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
    img_html_dataset = ImageHTMLDataSet(data_dir=data_dir, vocab=vocab, transform=transform)
    data_loader = DataLoader(dataset=img_html_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    dev_img_html_dataset = ImageHTMLDataSet(data_dir=dev_data_dir, vocab=vocab, transform=transform)
    dev_data_loader = DataLoader(dataset=dev_img_html_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)         


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr = learning_rate)

    start_epoch = 0
    best_bleu = 0.0
    # best_bleu = 0.5598234212490967

    #Load from checkpoint
    LOAD_CHECKPOINT = False
    RESUME_TRAINING = True
    model_name = 'best.pkl'
    if LOAD_CHECKPOINT:
        checkpoint_path = os.path.join(model_path, model_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load trained models
        encoder.load_state_dict(checkpoint['encoder'], strict=True)
        decoder.load_state_dict(checkpoint['decoder'], strict=True)

        # Load optimizer
        if RESUME_TRAINING:
            if checkpoint['epoch']:
                start_epoch = checkpoint['epoch'] + 1
            if checkpoint['best_bleu']:
                best_bleu = checkpoint['best_bleu']
            if checkpoint['optimizer']:
                optimizer.load_state_dict(checkpoint['optimizer']) 

    batch_count = len(data_loader)

    for epoch in range(start_epoch, num_epochs):
        encoder.train()
        decoder.train()
        with tqdm(enumerate(data_loader), total=batch_count, position=0, leave=True) as pbar: # progress bar
            for i, (images, captions, lengths) in pbar:
                # Shape: torch.Size([batch_size, 3, crop_size, crop_size])
                images = Variable(images.to(device))

                # Shape: torch.Size([batch_size, len(longest caption)])
                captions = Variable(captions.to(device))

                # lengths is a list of how long captions are in descending order (e.g., [77, 77, 75, 25])

                # We remove the paddings from captions that are padded and then pack them into a single sequence
                # Our data loader's collate_fn adds extra zeros to the end of sequences that are too short
                # Shape: torch.Size([sum(lengths)])
                targets = nn.utils.rnn.pack_padded_sequence(input = captions, lengths = lengths, batch_first = True)[0]

                # Zero out buffers
                encoder.zero_grad()
                decoder.zero_grad()

                # Forward, Backward, and Optimize
                features = encoder(images) # Outputs features of torch.Size([batch_size, embed_size])
                outputs = decoder(features, captions, lengths)

                # CrossEntropyLoss is expecting:
                # Input:  (N, C) where C = number of classes
                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                s = ('%10s Loss: %.4f, Perplexity: %5.4f') % ('%g/%g' % (epoch, num_epochs - 1), loss.item(), np.exp(loss.item()))
                pbar.set_description(s)

        # end batch ------------------------------------------------------------------------------------------------

        # Calculate BLEU score
        with torch.no_grad():
            bleu, _ = eval_bleu_score(encoder, decoder, dev_data_loader, device)
            print('BLEU score: ', bleu)
            if(bleu > best_bleu):
                best_bleu = bleu

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_bleu': best_bleu
        }

        # Save last checkpoint
        torch.save(checkpoint, os.path.join(model_path, 'last.pkl'))

        # Save best checkpoint
        if bleu == best_bleu:
            torch.save(checkpoint, os.path.join(model_path, 'best.pkl'))

        # Save backup every 10 epochs (optional)
        if (epoch + 1) % save_after_x_epochs == 0:
            # Save our models
            print('!!! saving models at epoch: ' + str(epoch))
            torch.save(checkpoint, os.path.join(model_path, 'checkpoint-%d-%d.pkl' %(epoch+1, 1)))             

        # Delete checkpoint
        del checkpoint

    print('done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', '-o', type=str, required=True, help='output path for saving model weights')
    parser.add_argument('--config', '--cfg', '-c', type=str, required=True, help='*-config.json path')
    parser.add_argument('--load', type=str, help='weights file to preload the model with')
    parser.add_argument('--resume', action='store_true', help='resume training')
    opt = parser.parse_args()
    main()

