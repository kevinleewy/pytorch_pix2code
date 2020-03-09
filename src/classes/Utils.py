__author__ = 'Kevin Lee - kevin_lee@claris.com'

from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from .Vocabulary import *

class Utils:
    # @staticmethod
    # def sparsify(label_vector, output_size):
    #     sparse_vector = []

    #     for label in label_vector:
    #         sparse_label = np.zeros(output_size)
    #         sparse_label[label] = 1

    #         sparse_vector.append(sparse_label)

    #     return np.array(sparse_vector)

    # @staticmethod
    # def get_preprocessed_img(img_path, image_size):
    #     import cv2
    #     img = cv2.imread(img_path)
    #     img = cv2.resize(img, (image_size, image_size))
    #     img = img.astype('float32')
    #     img /= 255
    #     return img

    # @staticmethod
    # def show(image):
    #     import cv2
    #     cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow("view", image)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow("view")

    @staticmethod
    def get_device(gpu_id=None):

        #CPU
        device = 'cpu'

        #GPU
        if torch.cuda.is_available():

            device = 'cuda'

            c = 1024 ** 2  # bytes to MB
            ng = torch.cuda.device_count()
            x = [torch.cuda.get_device_properties(i) for i in range(ng)]
            cuda_str = 'Using CUDA '
            for i in range(0, ng):

                if i == gpu_id:
                    device += ':' + str(i)

                if i == 1:
                    cuda_str = ' ' * len(cuda_str)
                print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                    (cuda_str, i, x[i].name, x[i].total_memory / c))
                
        print('Device:', device)
        return device

    @staticmethod
    def load_doc(filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
        return text

    @staticmethod
    def build_vocab (vocab_file_path):
        vocab = Vocabulary()

        # Load the vocab file (super basic split())
        words_raw = Utils.load_doc(vocab_file_path)
        words = words_raw.split(' ')
        
        for i, word in enumerate(words):
            vocab.add_word(word)

        vocab.add_word(' ')
        vocab.add_word('<unk>') # If we find an unknown word
        
        print('Created vocabulary of ' + str(len(vocab)) + ' items from ' + vocab_file_path)
        
        return vocab

    @staticmethod
    def transform_idx_to_words (vocab, input):
        sampled_caption = []
        
        for idx in input:
            word = vocab.idx2word[idx]
            sampled_caption.append(word)

            if word == '<END>':
                break

        output = ' '.join(sampled_caption[1:-1])

        output = output.replace(' ,', ',')

        return output.split(' ')

    @staticmethod
    def eval_bleu_score(model, data_loader, vocab, device):
        
        #Set model into eval mode
        model.eval()

        data_count = len(data_loader.dataset)

        predicted, actual = [], []

        with tqdm(enumerate(data_loader.dataset), total=data_count) as pbar: # progress bar
            
            pbar.set_description('    BLEU score: Computing...')
            
            for i, (image, caption) in pbar:
                image_tensor = Variable(image.unsqueeze(0).to(device))

                #Sample
                sampled_ids = model.sample(image_tensor)

                #Convert tensor to numpy array
                sampled_ids = sampled_ids.cpu().data.numpy()

                predicted.append(sampled_ids)
                actual.append(caption.numpy())

            predicted = [Utils.transform_idx_to_words(vocab, item) for item in predicted]
            actual = [[Utils.transform_idx_to_words(vocab, item)] for item in actual]
            
            bleu = corpus_bleu(actual, predicted)
            pbar.set_description('    BLEU score: %.12f' % (bleu))

        return bleu, len(predicted)

    @staticmethod
    def compute_saliency_maps(X, y, model, criterion, optimizer, device):
        """
        Compute a class saliency map using the model for images X and labels y.
        Input:
        - X: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the saliency map.
        Returns:
        - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input images.
        """
        # Make sure the model is in "test" mode
        model.eval()
        
        #Move tensors to correct device
        X = X.to(device)
        y = y.to(device)
        
        # Make input tensor require gradient
        X.requires_grad_()
    
        #Forward pass 
        outputs = model(X)
        loss = criterion(outputs, y)
        
        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #To compute the saliency map, we take the absolute value
        #of this gradient, then take the maximum value over the
        #3 input channels
        saliency, _ = X.grad.abs().max(dim=1)
        return saliency, outputs.cpu().detach().numpy()

