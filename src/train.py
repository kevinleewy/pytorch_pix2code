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

from classes.dataset.Generator import *
from classes.model.pix2code import *


def run(input_path, output_path, is_memory_intensive=False, pretrained_model=None):
    np.random.seed(1234)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    if not is_memory_intensive:
        dataset.convert_arrays()

        input_shape = dataset.input_shape
        output_size = dataset.output_size

        print(len(dataset.input_images), len(dataset.partial_sequences), len(dataset.next_words))
        print(dataset.input_images.shape, dataset.partial_sequences.shape, dataset.next_words.shape)
    else:
        gui_paths, img_paths = Dataset.load_paths_only(input_path)

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size / BATCH_SIZE

        voc = Vocabulary()
        voc.retrieve(output_path)

        generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, generate_binary_sequences=True)

    model = pix2code(input_shape, output_size, output_path)

    # if pretrained_model is not None:
    #     model.model.load_weights(pretrained_model)

    # if not is_memory_intensive:
    #     model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)
    # else:
    #     model.fit_generator(generator, steps_per_epoch=steps_per_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', '-o', type=str, required=True, help='output path for saving model weights')
    parser.add_argument('--config', '--cfg', '-c', type=str, required=True, help='*-config.json path')
    parser.add_argument('--load', type=str, help='weights file to preload the model with')
    parser.add_argument('--resume', action='store_true', help='resume training')
    opt = parser.parse_args()
    run('./datasets/web/training_set/', opt.out_dir, is_memory_intensive=False, pretrained_model=opt.load)

