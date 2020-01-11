#!/usr/bin/env python
__author__ = 'Kevin Lee - kevin_lee@claris.com'

import json
import random
# from classes.Node import *

class Generator:
    def __init__(self, config_path, terminate_prob=0.2):

        with open(config_path) as f:
            self.config = json.load(f)

        self.elements = { v['id'] : v for v in self.config['elements'] }
        self.terminate_prob = terminate_prob

    def sample(self, root='body'):

        elem = self.elements[root]

        if not elem['children']:
            return root if root is not 'body' else ''

        childrenSample = []
        currentWeight = 0
        while currentWeight < elem['min_weight'] or random.random() > self.terminate_prob:
            child=random.choice(elem['children'])
            if currentWeight + child['weight'] > elem['max_weight']:
                continue
            childrenSample.append(self.sample(root=child['id']))
            currentWeight += child['weight']
        
        childrenSample = ', '.join(childrenSample)

        if root is 'body':
            return childrenSample.replace('}, ', '}\n')

        return '{} {{\n{}\n}}'.format(root, childrenSample)
