#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Kevin Lee - kevin_lee@claris.com'

import argparse
import imgkit
import hashlib
import json
import numpy as np
import os
import sys

from classes.Compiler import *
from classes.Generator import *
from classes.Utils import *
from compile import compile
from tqdm import tqdm

TRAINING_SET_NAME = "training_set"
EVALUATION_SET_NAME = "eval_set"

def generate(path, count, compiler, generator, hashes):
    
    i = 0
    collisions = 0

    #Create output directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    #Generate samples
    with tqdm(total=count) as pbar:
        while i < count:
            sample = generator.sample()
            content_hash = sample.replace(" ", "").replace("\n", "")
            content_hash = hashlib.sha256(content_hash.encode('utf-8')).hexdigest()

            if content_hash in hashes:
                collisions += 1
                continue

            hashes.add(content_hash)
            input_file_path = os.path.join(path, content_hash + '.gui')
            output_file_path = os.path.join(path, content_hash + '.png')

            with open(input_file_path, 'w') as f:
                f.write(sample)

            compiled_file_path = compile(compiler, input_file_path, '[]', opt.domain)

            options = {
                'format': 'png',
                'quality': 10,
                'quiet': '',
                'width': 1500
            }

            imgkit.from_file(compiled_file_path, output_file_path, options=options)
            os.remove(compiled_file_path)

            i += 1
            pbar.update(1)

    return i, collisions        

def main():

    if opt.seed:
        np.random.seed(opt.seed)

    compiler = Compiler(opt.config)
    generator = Generator(opt.config)
    hashes = set()

    #Create output directory if it doesn't exist
    train_dir = os.path.join(opt.out_dir, opt.domain, TRAINING_SET_NAME)
    eval_dir = os.path.join(opt.out_dir, opt.domain, EVALUATION_SET_NAME)

    #Generate training set
    train_count, train_collisions = generate(train_dir, opt.num_train, compiler, generator, hashes)
    print('Finished generating {} training data. {} collisions.'.format(train_count, train_collisions))

    #Generate eval set
    eval_count, eval_collisions = generate(eval_dir, opt.num_eval, compiler, generator, hashes)
    print('Finished generating {} eval data. {} collisions.'.format(eval_count, eval_collisions))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', '-o', type=str, required=True, help='output path')
    parser.add_argument('--config', '--cfg', '-c', type=str, required=True, help='*-config.json path')
    parser.add_argument('--num-train', type=int, default=10000, help='number of gui files to generate for the training set')
    parser.add_argument('--num-eval', type=int, default=1000, help='number of gui files to generate for the test set')
    parser.add_argument('--domain', '-d', type=str, choices=['android', 'ios', 'web'], default='web', help='web, android or ios')
    parser.add_argument('--seed', type=int, default=1234, help='RNG seed')
    opt = parser.parse_args()
    main()