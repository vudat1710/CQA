import logging
from gensim.models import Word2Vec
import pickle
import preprocess
import numpy as np 
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
from tensorflow.contrib import learn
import random

class Embedding_Data:
    def build_corpus(self, FILE_PATH):
        prep = preprocess.PreprocessData()
        data_processed = prep.get_modified_data(FILE_PATH)
        questions = []
        answers = []
        labels = []
        for i in range (len(data_processed)):
            questions.extend([data_processed[i][0]])
            answers.append([data_processed[i][1]])
            labels.append(int(data_processed[i][2]))
        return (questions, answers, labels)
    
    def pad(self, data, length):
        return pad_sequences(data, maxlen=length, padding='post', truncating='post', value=0)
    
    def sentence_to_vec(self, sentence, model):
        temp_list = sentence.split(' ')
        result = np.zeros([len(temp_list),], dtype=int)
        for i in range(len(temp_list)):
            if temp_list[i] in model.wv.vocab:
                result[i] = self.get_index(temp_list[i], model)
            else:
                result[i] = random.randint(0, 44604)
        return result

    def turn_to_vector(self, list_to_transform, model):
        # vocab_size = 44604
        pad = 150
        encoded_list = [self.sentence_to_vec(str(d), model) for d in list_to_transform]
        padded_list = self.pad(encoded_list, pad)
        return padded_list
        
    def get_index(self, word, model):
        return model.wv.vocab[word].index

    def main(self):
        FILE_PATH = '/home/vudat1710/Downloads/NLP/CQA/dev.txt'
        questions, answers = self.build_corpus(FILE_PATH)
        model = pickle.load(open('/home/vudat1710/Downloads/NLP/CQA/skipgram_model.pkl','rb'))
        questions = self.turn_to_vector(questions, model)
        answers = self.turn_to_vector(answers, model)
        print (questions[23])

        
if __name__ == "__main__":
    a = Embedding_Data()
    a.main()
