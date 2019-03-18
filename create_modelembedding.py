import logging
from gensim.models import Word2Vec
import pickle
import preprocess
import numpy as np
import random


class EmbeddingMatrix:
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
                         min_count=1, workers=4, sg=1)
        filename = 'skipgram_model.pkl'
        pickle.dump(model, open(filename, 'wb'))

    def create_vocab_with_index(self, model):
        with open('vocab_all.txt', 'w') as f:
            vocab = model.wv.vocab
            for key, _ in vocab.items():
                index = vocab[key].index + 1
                f.write(str(index) + '\t' + key)
                f.write('\n')
        f.close()
    
    def create_vocab_dict(self):
        vocab = {}
        with open('vocab_all.txt', 'r') as f:
            for line in f.readlines():
                temp = line.split('\t')
                vocab[temp[1]] = temp[0]
        f.close()
        return vocab

    def get_glove_vectors(self, N=300, file_path='vectors.txt'):
        self.N = N
        self.g = dict()
        self.path_to_file = file_path

        with open(file_path, 'r') as f:
            for line in f.readlines():
                temp = line.split()
                word = temp[0]
                self.g[word] = np.array(temp[1:]).astype(float)
        return self.g

    def embmatrix(self, vocab):
        embedding_weights = np.zeros((len(vocab), 300), dtype=float)
        for word, index in vocab.items():
            try:
                embedding_weights[index, :] = self.g[word]
            except KeyError:
                if index == 0:
                    embedding_weights[index, :] = np.zeros(300)
                else:
                    embedding_weights[random.randint(0,44604), :] = np.random.uniform(-0.25, 0.25, 300)
        return embedding_weights

    def main(self):
        FILE_PATH = '/home/vudat1710/Downloads/NLP/CQA/train.txt'
        # self.create_model(FILE_PATH)
        # prep = preprocess.PreprocessData()
        # data_processed = prep.get_modified_data(FILE_PATH)
        # print (data_processed)
        loaded_model = pickle.load(
            open('/home/vudat1710/Downloads/NLP/CQA/skipgram_model.pkl', 'rb'))
        vocab = loaded_model.wv.vocab
        vector_to_dump = [loaded_model[key] for key, _ in vocab.items()]
        with open('word_100_dim.embedding', 'wb') as f:
            np.save(f, vector_to_dump)
        f.close()

        # print (loaded_model["work"])
        # print(type(vocab))
        # print (type(loaded_model.wv.vocab["work"].index))
        # self.create_vocab_with_index(loaded_model)


if __name__ == "__main__":
    a = EmbeddingMatrix()
    a.main()
