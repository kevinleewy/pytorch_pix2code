#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Kevin Lee - kevin_lee@claris.com'

import argparse
import json
import re
import sys

from os.path import basename
from .classes.Utils import *
from .classes.Compiler import *

def compile(compiler, input_file, text_placeholder=None, domain='web'):
    def render_content_with_text(key, value, cfg={}):
        if text_placeholder:
            value = re.sub(text_placeholder, lambda _ : Utils.get_random_text(**cfg), value)
        return value

    file_uid = basename(input_file)[:basename(input_file).find(".")]
    path = input_file[:input_file.find(file_uid)]

    input_file_path = "{}{}.gui".format(path, file_uid)
    if domain == 'android':
        ext = 'xml'
    if domain == 'android':
        ext = 'storyboard'    
    elif domain == 'web':
        ext = 'html'
    else:
        raise('Unsupported domain')    
    output_file_path = "{}{}.{}".format(path, file_uid, ext)
    
    compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text, domain=domain)

    return output_file_path

def main():
    compile(Compiler(opt.config), opt.source, opt.text_placeholder if opt.random_text else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '--src', '-s', type=str, required=True, help='*.gui path')
    parser.add_argument('--config', '--cfg', '-c', type=str, required=True, help='*-config.json path')
    parser.add_argument('--random-text', action='store_true', help='insert random text')
    parser.add_argument('--text-placeholder', type=str, default=r'\[\]', help='characters for placeholder text')
    opt = parser.parse_args()
    main()