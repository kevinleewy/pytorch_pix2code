import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from ..Utils import *
from ..Vocabulary import *

class ImageDataset (Dataset):
    def __init__ (self, data_dir, vocab, transform):
        self.data_dir = data_dir
        self.vocab = vocab
        self.transform = transform
        
        self.raw_image_names = []
        self.raw_captions = []
        
        # Fetch all files from our data directoruy
        self.filenames = os.listdir(data_dir)
        self.filenames.sort()
        
        # Sort files based on their filetype
        # Assume associated training examples have same filenames
        for filename in self.filenames:
            if filename[-3:] == 'png':
                # Store image filename
                self.raw_image_names.append(filename)
            elif filename[-3:] == 'gui':
                # Load .gui file
                data = Utils.load_doc(os.path.join(data_dir, filename))
                self.raw_captions.append(data)
                
        print('Created dataset of ' + str(len(self)) + ' items from ' + data_dir)

    def __len__ (self):
        return len(self.raw_image_names)
    
    def __getitem__ (self, idx):
        img_path, raw_caption = self.raw_image_names[idx], self.raw_captions[idx]
        
        # Get image from filesystem
        image = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
        image = self.transform(image)
        
        # Convert caption (string) to list of vocab ID's
        caption = []
        caption.append(self.vocab(START_TOKEN))
        
        # Remove newlines, separate words with spaces
        tokens = ' '.join(raw_caption.split())

        # Add space after each comma
        tokens = tokens.replace(',', ' ,')
        
        # Split into words
        tokens = tokens.split(' ')
        
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(END_TOKEN))
        
        target = torch.Tensor(caption)
        
        return image, target