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

        image_model = nn.Sequential(
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

        language_model = nn.Sequential(
            nn.LSTM(input_size=(CONTEXT_LENGTH, output_size), hidden_size=128),
            nn.LSTM(input_size=128, hidden_size=128)
        )

        # textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        # encoded_text = language_model(textual_input)

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
