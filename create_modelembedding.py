import logging
from gensim.models import Word2Vec
import pickle
import preprocess
import numpy as np
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class EmbeddingMatrix:
    global g
    def create_model(self, FILE_PATH):
        prep = preprocess.PreprocessData()
        data_processed = prep.get_modified_data(FILE_PATH)
        train_data = []
        for data_point in data_processed:
            train_data.append(data_point[0])
            train_data.append(data_point[1])
        # print(train_data)
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = Word2Vec(train_data, size=100, window=10,
                         min_count=1, workers=4, sg=0)
        filename = 'skipgram_model.pkl'
        pickle.dump(model, open(filename, 'wb'))

    def create_vocab_with_index(self, model):
        with open('vocab_all.txt', 'w') as f:
            vocab = model.wv.vocab
            for key, _ in vocab.items():
                index = vocab[key].index
                f.write(str(index) + '\t' + key)
                f.write('\n')
        f.close()
    
    def create_vocab_dict(self):
        vocab = {}
        with open('vocab_all.txt', 'r') as f:
            for line in f.readlines():
                temp = line.split('\t')
                vocab[temp[1].strip()] = temp[0].strip()
        f.close()
        return vocab

    def get_glove_vectors(self):
        N = 300
        g = dict()
        file_path='vectors.txt'

        with open(file_path, 'r') as f:
            for line in f.readlines():
                temp = line.split()
                word = temp[0]
                g[word] = np.array(temp[1:]).astype(float)
        return g

    def embmatrix(self, g, vocab):
        embedding_weights = np.zeros((len(vocab)+1, 300), dtype=float)
        for word in vocab:
            if word in g:
                embedding_weights[int(vocab[word]), :] = np.array(g[word])
            else:
                embedding_weights[int(vocab[word]), :] = np.random.uniform(-0.25, 0.25, 300)
        return embedding_weights

    def turn_to_vector(self, questions, answers, tok):
        max_len = 150
        vocab_size = len(tok.word_index) + 1

        q_vec = tok.texts_to_sequences(questions)
        a_vec = tok.texts_to_sequences(answers)
        q_vec_pad = pad_sequences(q_vec, maxlen=max_len, padding='post', truncating='post')
        a_vec_pad = pad_sequences(a_vec, maxlen=max_len, padding='post', truncating='post')

        return (q_vec_pad, a_vec_pad)


    def main(self):
        g = self.get_glove_vectors()
        # self.create_model('all_file.txt')
        # model = pickle.load(open('skipgram_model.pkl', 'rb'))
        # self.create_vocab_with_index(model)
        # vocab1 = self.create_vocab_dict()
        # weights1 = self.embmatrix(g, vocab1)
        # print (np.shape(weights1))
        
        tok = Tokenizer()
        data = []
        with open('all_file.txt', 'r') as f:
            for line in f.readlines():
                temp = line.split('\t')
                if temp[0] not in data:
                    data.append(temp[0])
                data.append(temp[1])
        f.close()
        print (tok.word_docs)
        
        # weights = self.embmatrix(g, vocab)
        # print (np.shape(weights))
        # tok.word_docs

if __name__ == "__main__":
    a = EmbeddingMatrix()
    a.main()
