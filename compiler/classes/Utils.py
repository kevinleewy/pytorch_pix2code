__author__ = 'Kevin Lee - kevin_lee@claris.com'

import string
import random


class Utils:
    @staticmethod
    def get_random_text(max_text_length=10, min_word_length=2, max_word_length=6, with_upper_case=True):

        def get_random_word(word_length, with_upper_case=False):
            chars=[random.choice(string.ascii_letters[:26]) for _ in range(word_length)]

            if with_upper_case:
                chars[0] = chars[0].upper()    

            return ''.join(chars)

        words = []
        text_length = -1
        while text_length < max_text_length:
            max_word_length = min(max_word_length, max_text_length - text_length)
            if max_word_length < min_word_length:
                break
            word_length = random.randint(min_word_length, max_word_length)
            word = get_random_word(word_length, with_upper_case and len(words) == 0)
            words.append(word)
            text_length += len(word) + 1

        return ' '.join(words)
        

        # chars = []
        # word_len = 0
        # while len(chars) < length_text:

        #     if word_len >= max_word_length:
        #         char = ' '
        #         word_len = 0

        #     elif word_len < min_word_length:
        #         char = random.choice(string.ascii_letters[:26])
        #         word_len += 1

        #     else:
        #         char = random.choice(string.ascii_letters[:26] + ' ')
        #         word_len += 1
            
        #     chars.append(char)

        # if with_upper_case:
        #     chars[0] = chars[0].upper()    

        # return ''.join(chars)

        # results = []
        # while len(results) < length_text:
        #     char = random.choice(string.ascii_letters[:26])
        #     results.append(char)
        # if with_upper_case:
        #     results[0] = results[0].upper()

        # current_spaces = []
        # while len(current_spaces) < space_number:
        #     space_pos = random.randint(2, length_text - 3)
        #     if space_pos in current_spaces:
        #         break
        #     results[space_pos] = " "
        #     if with_upper_case:
        #         results[space_pos + 1] = results[space_pos - 1].upper()

        #     current_spaces.append(space_pos)

        # return ''.join(results)

    @staticmethod
    def get_ios_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.digits + string.ascii_letters)
            results.append(char)

        results[3] = "-"
        results[6] = "-"

        return ''.join(results)

    @staticmethod
    def get_android_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.ascii_letters)
            results.append(char)

        return ''.join(results)
