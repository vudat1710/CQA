from keras import backend as K
from keras.layers import Embedding
from keras.layers import LSTM, Input, dot, Lambda, GlobalMaxPool1D, Dense, Dropout, concatenate
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import numpy as np
import embedding
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from create_modelembedding import EmbeddingMatrix
from keras.callbacks import EarlyStopping


class ModelTraining:

    def get_cosine_similarity(self):
        def dot(a, b): return K.batch_dot(a, b, axes=1)
        return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())

    def ranknet(self, y_true, y_pred):
        return K.mean(K.log(1. + K.exp(-(y_true * y_pred - (1 - y_true) * y_pred))), axis=-1)

    def map_score(self, s1s_dev, s2s_dev, y_pred, labels_dev):
        QA_pairs = {}
        for i in range(len(s1s_dev)):
            pred = y_pred[i]

            s1 = " ".join(s1s_dev[i])
            s2 = " ".join(s2s_dev[i])
            if s1 in QA_pairs:
                QA_pairs[s1].append((s2, labels_dev[i], pred[1]))
            else:
                QA_pairs[s1] = [(s2, labels_dev[i], pred[1])]

        MAP, MRR = 0, 0
        num_q = len(QA_pairs.keys())
        for s1 in QA_pairs.keys():
            p, AP = 0, 0
            MRR_check = False

            QA_pairs[s1] = sorted(
                QA_pairs[s1], key=lambda x: x[-1], reverse=True)

            for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
                if int(label) == 1:
                    if not MRR_check:
                        MRR += 1 / (idx + 1)
                        MRR_check = True

                    p += 1
                    AP += p / (idx + 1)
            if(p == 0):
                AP = 0
            else:
                AP /= p
            MAP += AP
        MAP /= num_q
        MRR /= num_q
        return MAP, MRR

    def get_bilstm_model(self, vocab_size, vocab):
        margin = 0.05
        enc_timesteps = 150
        dec_timesteps = 150
        hidden_dim = 128

        # initialize the question and answer shapes and datatype
        question = Input(shape=(enc_timesteps,),
                         dtype='int32', name='question_base')
        answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')

        embed = EmbeddingMatrix()
        g = embed.get_glove_vectors()
        weights = embed.embmatrix(g, vocab)
        # weights = embedding_file
        qa_embedding = Embedding(
            input_dim=vocab_size, input_length=150, output_dim=weights.shape[1], mask_zero=False, weights=[weights])
        bi_lstm = Bidirectional(
            LSTM(activation='tanh', dropout=0.2, units=hidden_dim, return_sequences=False))
        # maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))

        # embed the question and pass it through bilstm
        question_embedding = qa_embedding(question)
        question_enc_1 = bi_lstm(question_embedding)
        # question_enc_1 = Dropout(0.2)(question_enc_1)

        # embed the answer and pass it through bilstm
        answer_embedding = qa_embedding(answer)
        answer_enc_1 = bi_lstm(answer_embedding)
        # answer_enc_1 = Dropout(0.2)(answer_enc_1)

        # qa_merged = dot([question_enc_1, answer_enc_1], axes=1, normalize=True)
        qa_merged = concatenate([question_enc_1, answer_enc_1])
        qa_merged = Dense(1, activation='sigmoid')(qa_merged)
        lstm_model = Model(name="bi_lstm", inputs=[
                           question, answer], outputs=qa_merged)
        output = lstm_model([question, answer])
        # # lam = Lambda(lambda x: x)
        # # loss = lam(similarity)
        training_model = Model(
            inputs=[question, answer], outputs=output, name='training_model')
        training_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        # lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
        return training_model

    def main(self):
        eb = embedding.Embedding_Data()
        embed = EmbeddingMatrix()
        # embedding_file = 'word_100_dim.embedding'
        vocab = embed.create_vocab_dict()
        vocab_len = len(vocab)
        training_model = self.get_bilstm_model(vocab_len, vocab)
        epoch = 1
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=2)
        for i in range(epoch):
            print("Training epoch: ", i)
            FILE_PATH = 'train.txt'
            questions, answers, labels = eb.build_corpus(FILE_PATH)
            # print (np.shape(questions))
            # print (np.shape(answers))
            # model = pickle.load(open('skipgram_model.pkl', 'rb'))
            questions = eb.turn_to_vector(questions, vocab)
            answers = eb.turn_to_vector(answers, vocab)
            # Y = np.zeros(np.shape(labels))
            Y = np.array(labels)
            # Y = Y.reshape(1, -1)
            # print(questions[1])
            # print(answers[1])
            training_model.fit(
                [questions, answers],
                Y,
                epochs=100,
                batch_size=64,
                validation_split=0.1,
                verbose=1,
                callbacks=[es]
            )
            training_model.save_weights(
                'train_weights_epoch_' + str(epoch) + '.h5', overwrite=True)
        training_model.summary()

        training_model.load_weights('train_weights_epoch_1.h5')
        questions, answers, labels = eb.build_corpus('test.txt')
        # model = pickle.load(open('skipgram_model.pkl', 'rb'))
        questions = eb.turn_to_vector(questions, vocab)
        answers = eb.turn_to_vector(answers, vocab)
        # print(questions[1])
        # print(answers[1])
        sims = training_model.predict([questions, answers])
        # # # c = 0
        # # for i in range(len(sims)):
        # #     if (sims[i] > 0):
        # #         sims[i] = 1
        # #     else: sims[i] = 0
        # # for i in range(len(sims)):
        # #     if sims[i] == labels[i]:
        # #         c = c + 1
        # # print(c/len(sims))
        # # max_r = np.argmax(sims)
        # # print(max_r)
        with open('sim2.txt', 'w') as f:
            for i in range(len(sims)):
                # max_r = np.argmax(sims[i])
                f.write(str(sims[i]))
                f.write('\n')
            # print('\n')
        f.close()


if __name__ == "__main__":
    a = ModelTraining()
    a.main()
