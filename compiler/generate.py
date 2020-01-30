#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Kevin Lee - kevin_lee@claris.com'

import argparse
import asyncio
import hashlib
import json
import numpy as np
import os
import shutil
import sys

from classes.Compiler import *
from classes.Generator import *
from classes.Utils import *
from compile import compile
from pyppeteer import launch
from tqdm import tqdm

TRAINING_SET_NAME = "training_set"
EVALUATION_SET_NAME = "eval_set"

async def generate(path, count, compiler, generator, hashes, css, page):
    
    i = 0
    collisions = 0

    #Create output directory if it doesn't exist
    assets_path = os.path.join(path, 'assets')
    if not os.path.exists(assets_path):
        os.makedirs(assets_path)

    #Copy assets
    shutil.copyfile(css, os.path.join(assets_path, 'styles.css'))

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

            compiled_file_path = compile(compiler, input_file_path, r'\[\]', opt.domain)

            # Capture screenshot
            await page.goto('file://' + os.path.abspath(compiled_file_path))
            await page.screenshot(
                {
                    'path' : output_file_path,
                    'type' : 'png',
                    'fullPage' : True
                }
            )
            
            # Remove HTML file
            if not opt.keep_compiled:
                os.remove(compiled_file_path)

            i += 1
            pbar.update(1)

    return i, collisions        

async def main():

    if opt.seed:
        np.random.seed(opt.seed)

    compiler = Compiler(opt.config)
    generator = Generator(opt.config)
    hashes = set()

    #Create output directory if it doesn't exist
    train_dir = os.path.join(opt.out_dir, opt.domain, TRAINING_SET_NAME)
    eval_dir = os.path.join(opt.out_dir, opt.domain, EVALUATION_SET_NAME)

    try:
        browser = await launch({'headless': opt.headless})
    except Exception as e:
        print(e)
        browser = await launch(executablePath="/usr/lib/chromium-browser/chromium-browser", headless=opt.headless, args=['--no-sandbox'])

    page = await browser.newPage()
    await page.setViewport({
        'width': 1920,
        'height': 1080,
    })

    #Generate training set
    train_count, train_collisions = await generate(train_dir, opt.num_train, compiler, generator, hashes, opt.css, page)
    print('Finished generating {} training data. {} collisions.'.format(train_count, train_collisions))

    #Generate eval set
    eval_count, eval_collisions = await generate(eval_dir, opt.num_eval, compiler, generator, hashes, opt.css, page)
    print('Finished generating {} eval data. {} collisions.'.format(eval_count, eval_collisions))

    await browser.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', '-o', type=str, required=True, help='output path')
    parser.add_argument('--config', '--cfg', '-c', type=str, required=True, help='*-config.json path')
    parser.add_argument('--css', type=str, required=True, help='.css path')
    parser.add_argument('--num-train', type=int, default=10000, help='number of gui files to generate for the training set')
    parser.add_argument('--num-eval', type=int, default=1000, help='number of gui files to generate for the test set')
    parser.add_argument('--domain', '-d', type=str, choices=['android', 'ios', 'web'], default='web', help='web, android or ios')
    parser.add_argument('--seed', type=int, default=1234, help='RNG seed')
    parser.add_argument('--keep-compiled', action='store_true', help='preserve compiled file (.storyboard, .html, etc.)')
    parser.add_argument('--headless', type=bool, required=False, default=True, help='view screenshots in action')
    opt = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(main())