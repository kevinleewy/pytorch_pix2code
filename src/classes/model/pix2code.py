from __future__ import absolute_import
__author__ = 'Kevin Lee - kevin_lee@claris.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..Utils import Utils
from ..Vocabulary import END_TOKEN

def convert(sampled_ids, vocab):
    #Convert tensor to numpy array
    sampled_ids = sampled_ids.cpu().data.numpy()
    
    sampled_caption = []
        
    for idx in sampled_ids:
        word = vocab.idx2word[idx]
        sampled_caption.append(word)
    
    output = ' '.join(sampled_caption)

    output = output.replace(' ,', ',')

    output = output.split(' ')

    output = ''.join(output)

    return output

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
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        
        # Store the embed size for use when sampling
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
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
    def sample (self, features, vocab, states=None, num_iter=200, beam_size=3):
        
        device = features.device
        k = beam_size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[] for _ in range(k)]).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and their scores
        complete_seqs = []
        complete_seqs_scores = []

        # [1, embed_size] -> [k, 1, embed_size]
        inputs = features.expand(k, -1, -1)

        # Put the features input through our decoder for n iterations
        for i in range(num_iter):
            
            # [k, 1, embed_size] -> [k, 1, hidden_size]
            # state_dim: None or ([2, k, hidden_size], [2, k, hidden_size])
            hiddens, states = self.lstm(inputs, states)

            # outputs: Scores for each word in vocab
            # [k, 1, hidden_size] -> [k, vocab_size]
            outputs = self.linear(hiddens.squeeze(1))

            # [k, vocab_size] -> [k, vocab_size]
            scores = F.log_softmax(outputs, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            # [k, vocab_size] -> [k], [k]
            # top_k_scores: [k] FloatTensor
            # top_k_words : [k] LongTensor
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if i == 0:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0, largest=True, sorted=True)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0, largest=True, sorted=True)

            # Convert unrolled indices to actual indices of scores
            prev_seq_inds = top_k_words / self.vocab_size  # [k]
            next_word_inds = top_k_words % self.vocab_size  # [k]

            # Add new words to sequences, alphas
            # seqs: [k, i+1]
            seqs = torch.cat([seqs[prev_seq_inds], next_word_inds.unsqueeze(1)], dim=1)

            # Which sequences are incomplete (didn't reach <END>)?
            incomplete_seq_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != vocab(END_TOKEN)]
            complete_seq_inds = list(set(range(len(next_word_inds))) - set(incomplete_seq_inds))

            # Set aside complete sequences
            if len(complete_seq_inds) > 0:
                complete_seqs.extend(seqs[complete_seq_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_seq_inds])

            # # Proceed with incomplete sequences
            if len(incomplete_seq_inds) == 0:
                break

            seqs = seqs[incomplete_seq_inds]

            states = (states[0][:, prev_seq_inds[incomplete_seq_inds], :], states[1][:, prev_seq_inds[incomplete_seq_inds], :])

            # Convert sampled ID into embedding vector
            # [k, 1] -> [k, 1, embed_size]
            inputs = self.embed(next_word_inds[incomplete_seq_inds].unsqueeze(1))

            # [k] -> [k, 1]
            top_k_scores = top_k_scores[incomplete_seq_inds].unsqueeze(1)

            # Reshape inputs tensor
            # [k, 1, embed_size] -> [k, 1, embed_size]
            inputs = inputs.view(-1, 1, self.embed_size)

        best_seq_score = max(complete_seqs_scores)
        best_seq_index = complete_seqs_scores.index(best_seq_score)
        best_seq = complete_seqs[best_seq_index]

        # Convert list to tensor
        # [k, num_iter]
        best_seq = torch.Tensor(best_seq, device=device)

        print('Score:', best_seq_score.cpu().numpy())
        return best_seq

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, embed_size, hidden_size, attention_dim):
        """
        :param embed_size: feature size of encoded images
        :param hidden_size: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(in_features=embed_size, out_features=attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(in_features=hidden_size, out_features=attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(in_features=attention_dim, out_features=1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, embed_size)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, hidden_size)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, embed_size)

        return attention_weighted_encoding, alpha

class DecoderWithAttention (nn.Module):
    """
    DecoderWithAttention.
    """

    def __init__ (self, attention_dim, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderWithAttention, self).__init__()
        
        # vocab_size word vocabulary, embed_size dimensional embeddings
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

        # attention network
        self.attention = Attention(embed_size, hidden_size, attention_dim)  

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        
        # Store the embed size for use when sampling
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        print('DecoderWithAttention created with embed_size: ' + str(embed_size))
        
    def forward (self, features, captions, lengths):
        """
        Forward propagation.
        :param features: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

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
    def sample (self, features, vocab, states=None, num_iter=200, beam_size=3):
        
        device = features.device
        k = beam_size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[] for _ in range(k)]).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and their scores
        complete_seqs = []
        complete_seqs_scores = []

        # [1, embed_size] -> [k, 1, embed_size]
        inputs = features.expand(k, -1, -1)

        # Put the features input through our decoder for n iterations
        for i in range(num_iter):
            
            # [k, 1, embed_size] -> [k, 1, hidden_size]
            # state_dim: None or ([2, k, hidden_size], [2, k, hidden_size])
            hiddens, states = self.lstm(inputs, states)

            # outputs: Scores for each word in vocab
            # [k, 1, hidden_size] -> [k, vocab_size]
            outputs = self.linear(hiddens.squeeze(1))

            # [k, vocab_size] -> [k, vocab_size]
            scores = F.log_softmax(outputs, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            # [k, vocab_size] -> [k], [k]
            # top_k_scores: [k] FloatTensor
            # top_k_words : [k] LongTensor
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if i == 0:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0, largest=True, sorted=True)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0, largest=True, sorted=True)

            # Convert unrolled indices to actual indices of scores
            prev_seq_inds = top_k_words / self.vocab_size  # [k]
            next_word_inds = top_k_words % self.vocab_size  # [k]

            # Add new words to sequences, alphas
            # seqs: [k, i+1]
            seqs = torch.cat([seqs[prev_seq_inds], next_word_inds.unsqueeze(1)], dim=1)

            # Which sequences are incomplete (didn't reach <END>)?
            incomplete_seq_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != vocab(END_TOKEN)]
            complete_seq_inds = list(set(range(len(next_word_inds))) - set(incomplete_seq_inds))

            # Set aside complete sequences
            if len(complete_seq_inds) > 0:
                complete_seqs.extend(seqs[complete_seq_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_seq_inds])

            # # Proceed with incomplete sequences
            if len(incomplete_seq_inds) == 0:
                break

            seqs = seqs[incomplete_seq_inds]

            states = (states[0][:, prev_seq_inds[incomplete_seq_inds], :], states[1][:, prev_seq_inds[incomplete_seq_inds], :])

            # Convert sampled ID into embedding vector
            # [k, 1] -> [k, 1, embed_size]
            inputs = self.embed(next_word_inds[incomplete_seq_inds].unsqueeze(1))

            # [k] -> [k, 1]
            top_k_scores = top_k_scores[incomplete_seq_inds].unsqueeze(1)

            # Reshape inputs tensor
            # [k, 1, embed_size] -> [k, 1, embed_size]
            inputs = inputs.view(-1, 1, self.embed_size)

        best_seq_score = max(complete_seqs_scores)
        best_seq_index = complete_seqs_scores.index(best_seq_score)
        best_seq = complete_seqs[best_seq_index]

        # Convert list to tensor
        # [k, num_iter]
        best_seq = torch.Tensor(best_seq, device=device)

        print('Score:', best_seq_score.data())
        return best_seq

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
    def sample (self, image, vocab, beam_size=3):

        features = self.encoder(image)
        sampled_ids = self.decoder.sample(features, vocab, beam_size=beam_size)

        return sampled_ids
