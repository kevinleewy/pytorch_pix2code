#!/usr/bin/env python
from __future__ import print_function
import os
dirname = os.path.dirname(__file__)
__author__ = 'Kevin Lee - kevin_lee@claris.com'


class Node:
    def __init__(self, key, parent_node, content_holder):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.content_holder = content_holder

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        print(self.key)
        for child in self.children:
            child.show()

    def render(self, mapping, rendering_function=None, domain='web'):
        content = ""
        for child in self.children:
            content += child.render(mapping, rendering_function)

        # value = mapping[self.key]['dsl'][domain]
        value = ""
        with open(os.path.join(dirname, '../templates/{}/{}.html'.format(domain, self.key))) as f:
            value = f.read()

        if rendering_function is not None:
            cfg = {}
            if 'random_text_config' in mapping[self.key]:
                cfg = mapping[self.key]['random_text_config']
            value = rendering_function(self.key, value, cfg)

        if len(self.children) != 0:
            value = value.replace(self.content_holder, content)

        return value
