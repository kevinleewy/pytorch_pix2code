from __future__ import absolute_import
__author__ = 'Kevin Lee - kevin_lee@claris.com'

from .Config import *
from .AModel import *

import torch
import torch.nn as nn
import torchvision

class EncoderCNN (nn.Module):
    def __init__ (self, embed_size):
        super(EncoderCNN, self).__init__()
        
        # Load pretrained resnet model
        resnet = torchvision.models.resnet152(pretrained = True)
        
        # Remove the fully connected layers
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Create our replacement layers
        # We reuse the in_feature size of the resnet fc layer for our first replacement layer = 2048 as of creation
        self.linear = nn.Linear(in_features = resnet.fc.in_features, out_features = embed_size)
        self.bn = nn.BatchNorm1d(num_features = embed_size, momentum = 0.01)
        
        print('EncoderCNN created with embed_size: ' + str(embed_size))

    def forward (self, images):
        # Get the expected output from the fully connected layers
        # Fn: AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        # Output: torch.Size([batch_size, 2048, 1, 1])
        features = self.resnet(images)

        # Resize the features for our linear function
        features = features.view(features.size(0), -1)
        
        # Fn: Linear(in_features=2048, out_features=embed_size, bias=True)
        # Output: torch.Size([batch_size, embed_size])
        features = self.linear(features)
        
        # Fn: BatchNorm1d(embed_size, eps=1e-05, momentum=0.01, affine=True)
        # Output: torch.Size([batch_size, embed_size])
        features = self.bn(features)
        
        return features

class DecoderRNN (nn.Module):
    def __init__ (self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        
        # 19 word vocabulary, embed_size dimensional embeddings
        self.embed = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size)

        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)

        self.linear = nn.Linear(in_features = hidden_size, out_features = vocab_size)
        
        # Store the embed size for use when sampling
        self.embed_size = embed_size
        
        print('DecoderRNN created with embed_size: ' + str(embed_size))
        
    def forward (self, features, captions, lengths):
        # 'captions' enters as shape torch.Size([batch_size, len(longest caption)])
        
        # Fn: Embedding(vocab_size, embed_size)
        # Input: LongTensor (N = mini_batch, W = # of indices to extract per mini-batch)
        # Output: (N, W, embedding_dim) => torch.Size([batch_size, len(longest caption), embed_size])
        embeddings = self.embed(captions)
        
        # Match features dimensions to embedding's
        features = features.unsqueeze(1) # torch.Size([4, 128]) => torch.Size([4, 1, 128])
        
        embeddings = torch.cat((features, embeddings), 1)
        
        packed = nn.utils.rnn.pack_padded_sequence(input = embeddings, lengths = lengths, batch_first = True)
        
        # Fn: LSTM(embed_size, hidden_size, batch_first = True)
        hiddens, _ = self.lstm(packed)
        
        outputs = self.linear(hiddens[0])
        
        return outputs
    
    # Sample method used for testing our model
    def sample (self, features, states=None, num_iter=100):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        
        # Put the features input through our decoder for n iterations
        for i in range(num_iter):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(dim = 1, keepdim = True)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.view(-1, 1, self.embed_size)

        sampled_ids = torch.cat(sampled_ids, 1)

        return sampled_ids.squeeze()

class Pix2Code (nn.Module):
    def __init__ (self, embed_size, hidden_size, vocab_size, num_layers):
        super(Pix2Code, self).__init__()
        
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward (self, images, captions, lengths):
        features = self.encoder(images) # Outputs features of torch.Size([batch_size, embed_size])
        outputs = self.decoder(features, captions, lengths)
        return outputs

    # Sample method used for testing our model
    def sample (self, image):

        features = self.encoder(image)
        sampled_ids = decoder.sample(features)

        return sampled_ids
