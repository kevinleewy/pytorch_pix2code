from __future__ import absolute_import
__author__ = 'Kevin Lee - kevin_lee@claris.com'

# from keras.layers import Input, Dense, Dropout, \
#                          RepeatVector, LSTM, concatenate, \
#                          Conv2D, MaxPooling2D, Flatten
# from keras.models import Sequential, Model
# from keras.optimizers import RMSprop
# from keras import *
from .Config import *
from .AModel import *

import torch
import torch.nn as nn
import torchvision

class pix2code(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "pix2code"

        print(self.input_shape, output_size) #(256, 256, 3) 19

        self.image_model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=0), #(32, 254, 254)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=0), #(32, 252, 252)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), #(32, 126, 126)
            nn.Dropout(0.25), #(32, 126, 126)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0), #(64, 124, 124)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0), #(64, 122, 122)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), #(64, 61, 61)
            nn.Dropout(0.25), #(64, 61, 61)

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=0), #(128, 59, 59)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=0), #(128, 57, 57)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), #(128, 28, 28)
            nn.Dropout(0.25), #(128, 28, 28)

            nn.Flatten(), #(128*28*28=100352)
            nn.Linear(128*28*28, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            # nn.RepeatVector(CONTEXT_LENGTH)
        )

        # visual_input = Input(shape=input_shape)
        # encoded_image = image_model(visual_input)

        self.language_model = nn.Sequential(
            nn.LSTM(input_size=(CONTEXT_LENGTH, output_size), hidden_size=128),
            nn.LSTM(input_size=128, hidden_size=128)
        )

        # textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        # encoded_text = language_model(textual_input)

        self.decoder = nn.Sequential(
            nn.LSTM(input_size=(CONTEXT_LENGTH, output_size), hidden_size=512),
            nn.LSTM(input_size=512, hidden_size=512)
        )

        # decoder = concatenate([encoded_image, encoded_text])

        # decoder = LSTM(512, return_sequences=True)(decoder)
        # decoder = LSTM(512, return_sequences=False)(decoder)
        # decoder = Dense(output_size, activation='softmax')(decoder)

        # self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

        # optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# class pix2codeKeras(AModel):
#     def __init__(self, input_shape, output_size, output_path):
#         AModel.__init__(self, input_shape, output_size, output_path)
#         self.name = "pix2code"

#         image_model = Sequential()
#         Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
#         Conv2D(32, (3, 3), padding='valid', activation='relu'))
#         MaxPooling2D(pool_size=(2, 2)))
#         Dropout(0.25))

#         Conv2D(64, (3, 3), padding='valid', activation='relu'))
#         Conv2D(64, (3, 3), padding='valid', activation='relu'))
#         MaxPooling2D(pool_size=(2, 2)))
#         Dropout(0.25))

#         Conv2D(128, (3, 3), padding='valid', activation='relu'))
#         Conv2D(128, (3, 3), padding='valid', activation='relu'))
#         MaxPooling2D(pool_size=(2, 2)))
#         Dropout(0.25))

#         Flatten())
#         Dense(1024, activation='relu'))
#         Dropout(0.3))
#         Dense(1024, activation='relu'))
#         Dropout(0.3))

#         RepeatVector(CONTEXT_LENGTH))

#         visual_input = Input(shape=input_shape)
#         encoded_image = image_model(visual_input)

#         language_model = Sequential()
#         language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
#         language_model.add(LSTM(128, return_sequences=True))

#         textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
#         encoded_text = language_model(textual_input)

#         decoder = concatenate([encoded_image, encoded_text])

#         decoder = LSTM(512, return_sequences=True)(decoder)
#         decoder = LSTM(512, return_sequences=False)(decoder)
#         decoder = Dense(output_size, activation='softmax')(decoder)

#         self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

#         optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
#         self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#     def fit(self, images, partial_captions, next_words):
#         self.model.fit([images, partial_captions], next_words, shuffle=False, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
#         self.save()

#     def fit_generator(self, generator, steps_per_epoch):
#         self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
#         self.save()

#     def predict(self, image, partial_caption):
#         return self.model.predict([image, partial_caption], verbose=0)[0]

#     def predict_batch(self, images, partial_captions):
#         return self.model.predict([images, partial_captions], verbose=1)

class EncoderCNN (nn.Module):
    def __init__ (self, embed_size):
        super(EncoderCNN, self).__init__()
        
        # Load pretrained resnet model
        resnet = models.resnet152(pretrained = True)
        
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
    def sample (self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        
        # Put the features input through our decoder for i iterations
        # TODO: Put this range into a parameter?
        for i in range(100):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(dim = 1, keepdim = True)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.view(-1, 1, self.embed_size)

        sampled_ids = torch.cat(sampled_ids, 1)

        return sampled_ids.squeeze()