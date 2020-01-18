__author__ = 'Kevin Lee - kevin_lee@claris.com'

import numpy as np
# from .Vocabulary import *

class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")

    # @staticmethod
    # def load_doc(filename):
    #     file = open(filename, 'r')
    #     text = file.read()
    #     file.close()
    #     return text

    # @staticmethod
    # def build_vocab (vocab_file_path):
    #     vocab = Vocabulary()

    #     # Load the vocab file (super basic split())
    #     words_raw = Utils.load_doc(vocab_file_path)
    #     words = words_raw.split(' ')
        
    #     for i, word in enumerate(words):
    #         vocab.add_word(word)

    #     vocab.add_word(' ')
    #     vocab.add_word('<unk>') # If we find an unknown word
        
    #     print('Created vocabulary of ' + str(len(vocab)) + ' items from ' + vocab_file_path)
        
    #     return vocab    
