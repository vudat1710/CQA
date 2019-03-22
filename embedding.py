import logging
from gensim.models import Word2Vec
import pickle
import preprocess
import numpy as np 
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
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
        return pad_sequences(data, maxlen=length, padding='post', truncating='post')
    
    def sentence_to_vec(self, sentence, vocab):
        splited_sentence = sentence.split(' ')
        result = np.zeros([len(splited_sentence),], dtype=int)
        for i in range(len(splited_sentence)):
            if splited_sentence[i] in vocab:
                result[i] = self.get_index(splited_sentence[i], vocab)
            else:
                result[i] = random.randint(0, 44603)
        return result

    def turn_to_vector(self, list_to_transform, vocab):
        # vocab_size = 44604
        pad = 150
        encoded_list = [self.sentence_to_vec(str(d), vocab) for d in list_to_transform]
        padded_list = self.pad(encoded_list, pad)
        return padded_list
        
    def get_index(self, word, vocab):
        return vocab[word]

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
