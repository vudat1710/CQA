import numpy as np 
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
import re

from keras.callbacks import Callback

class Glove:
    def __init__(self, N = 300, file_path='vectors.txt'):
        self.N = N
        self.g = dict()
        self.path_to_file = file_path

        with open(file_path, 'r') as f:
            for line in f.readlines():
                temp = line.split()
                word = temp[0]
                self.g[word] = np.array(temp[1:]).astype(float)
    
    def main(self):
        print(self.g["the"])
if __name__=="__main__":
    a = Glove()
    a.main()

    # give a list of tokens, return a list of glove embeddings or a single BOW average if ndims == 1
    # very rare words are map to 0-vectors 
    # def map_tokens(self, token_list, ndims = 2):
    #     glove_tokens = []
    #     for token in token_list:
    #         if token in self.g:
    #             glove_tokens.append(self.g[token])
    #     if not glove_tokens:
    #         if ndims == 2:
    #             return np.zeros(1, self.N)
    #         else:
    #             return np.zeros(self.N)
    #     glove_tokens = np.array(glove_tokens)
    #     if ndims == 2:
    #         return glove_tokens
    #     else:
    #         return glove_tokens.mean(axis = 0)
    
    # #map tokens on a set of sentences
    # def map_set(self, sentences, ndims = 2):
    #     return [self.map_tokens(sentence, ndims = ndims) for sentence in sentences]
    
    # #Given a set of sentences transformed into per-word embeddings as above
    # #convert them into 3D matrix with fixed sentence sizes - padded or trimed to spad embeddings per sentence
    # #output is a tensor of shape(len(sentences), spad, N)
    # #to determine spad, use st like np.sort([np.shape(s) for s in s0], axis=0)[-1000]
    # #so that typically everything fits
    # def pad_set(self, sentences, spad, N = None):
    #     sentences_temp = []
    #     if N is None:
    #         N = self.N
    #     for sentence in sentences:
    #         if spad > sentence.shape[0]:
    #             if sentence.ndims == 2:
    #                 sentence = np.vstack((sentence, np.zeros(spad - sentence.shape[0], N))) #vstack: concat vectors vertically
    #             else:
    #                 sentence = np.hstack((sentence, np.zeros(spad - sentence.shape[0]))) #hstack: concat vectors horizontally
    #         elif spad < sentence.shape[0]:
    #             sentence = sentence[:spad]
    #         sentences_temp.append(sentence)
    #     return np.array(sentences_temp)

    

