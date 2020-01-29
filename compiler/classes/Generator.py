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

        if not elem['children_groups']:
            return root if root is not 'body' else ''

        childrenSample = []

        for group in elem['children_groups']:
            currentWeight = 0
            while currentWeight < group['min_weight'] or random.random() > self.terminate_prob:
                child = random.choice(group['children'])
                if currentWeight + child['weight'] > group['max_weight']:
                    continue
                childrenSample.append(self.sample(root=child['id']))
                currentWeight += child['weight']
            
        if len(childrenSample) > 0:
            childrenSample = ', '.join(childrenSample)
        else:
            childrenSample = ''

        if root is 'body':
            return childrenSample.replace('}, ', '}\n')

        return '{} {{\n{}\n}}'.format(root, childrenSample)
