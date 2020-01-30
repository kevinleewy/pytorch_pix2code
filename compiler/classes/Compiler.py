#!/usr/bin/env python
__author__ = 'Kevin Lee - kevin_lee@claris.com'

import json
from classes.Node import *


class Compiler:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

        self.opening_tag = self.config["opening-tag"]
        self.closing_tag = self.config["closing-tag"]
        self.content_holder = self.opening_tag + self.closing_tag
        self.elements = { v['id'] : v for v in self.config["elements"] }

    def compile(self, input_file_path, output_file_path, rendering_function=None, domain='web'):
        
        with open(input_file_path, 'r') as input_file:
            self.root = Node("body", None, self.content_holder)
            current_parent = self.root

            for line in input_file:
                line = line.replace(" ", "").replace("\n", "")
                tokens = line.split(",")

                for token in tokens:
                    if token.find(self.opening_tag) != -1:
                        token = token.replace(self.opening_tag, "")

                        element = Node(token, current_parent, self.content_holder)
                        current_parent.add_child(element)
                        current_parent = element
                    elif token.find(self.closing_tag) != -1:
                        current_parent = current_parent.parent
                    else:
                        element = Node(token, current_parent, self.content_holder)
                        current_parent.add_child(element)

        output_html = self.root.render(self.elements, rendering_function=rendering_function, domain='web')
        with open(output_file_path, 'w') as output_file:
            output_file.write(output_html)
