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
        dsl_file = open(input_file_path)
        self.root = Node("body", None, self.content_holder)
        current_parent = self.root

        for token in dsl_file:
            token = token.replace(" ", "").replace("\n", "")

            if token.find(self.opening_tag) != -1:
                token = token.replace(self.opening_tag, "")

                element = Node(token, current_parent, self.content_holder)
                current_parent.add_child(element)
                current_parent = element
            elif token.find(self.closing_tag) != -1:
                current_parent = current_parent.parent
            else:
                tokens = token.split(",")
                for t in tokens:
                    element = Node(t, current_parent, self.content_holder)
                    current_parent.add_child(element)

        output_html = self.root.render(self.elements, rendering_function=rendering_function, domain='web')
        with open(output_file_path, 'w') as output_file:
            output_file.write(output_html)
