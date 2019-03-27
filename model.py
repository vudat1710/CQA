from keras import backend as K
from keras.layers import Embedding
from keras.layers import LSTM, Input, dot, Lambda, Dense, Dropout, concatenate, CuDNNLSTM, BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import numpy as np
import embedding
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from create_modelembedding import EmbeddingMatrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from callback import AnSelCB
from keras import regularizers


class ModelTraining:
    def ranknet(self, y_true, y_pred):
        return K.mean(K.log(1. + K.exp(-(y_true * y_pred - (1 - y_true) * y_pred))), axis=-1)

    def map_score(self, s1s_dev, s2s_dev, y_pred, labels_dev):
        QA_pairs = {}
        for i in range(len(s1s_dev)):
            pred = y_pred[i]

            s1 = str(s1s_dev[i])
            s2 = str(s2s_dev[i])
            if s1 in QA_pairs:
                QA_pairs[s1].append((s2, labels_dev[i], pred))
            else:
                QA_pairs[s1] = [(s2, labels_dev[i], pred)]

        MAP, MRR = 0, 0
        num_q = len(QA_pairs.keys())
        for s1 in QA_pairs.keys():
            p, AP = 0, 0
            MRR_check = False

            QA_pairs[s1] = sorted(
                QA_pairs[s1], key=lambda x: x[-1], reverse=True)

            for idx, (s2, label, _) in enumerate(QA_pairs[s1]):
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
        enc_timesteps = 150
        dec_timesteps = 150
        hidden_dim = 128

        question = Input(shape=(enc_timesteps,),
                         dtype='int32', name='question_base')
        answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')

        embed = EmbeddingMatrix()
        g = embed.get_glove_vectors()
        weights = embed.embmatrix(g, vocab)

        qa_embedding = Embedding(
            input_dim=vocab_size + 1, input_length=150, output_dim=weights.shape[1], mask_zero=False, weights=[weights])
        bi_lstm = Bidirectional(
            CuDNNLSTM(units=hidden_dim, return_sequences=False))
        # bi_lstm = Bidirectional(
        #     LSTM(activation="tanh", dropout=0.5, units=hidden_dim, return_sequences=False))

        question_embedding = qa_embedding(question)
        question_embedding = Dropout(0.5)(question_embedding)
        question_enc_1 = bi_lstm(question_embedding)
        question_enc_1 = Dropout(0.5)(question_enc_1)
        question_enc_1 = BatchNormalization()(question_enc_1)

        answer_embedding = qa_embedding(answer)
        answer_embedding = Dropout(0.5)(answer_embedding)
        answer_enc_1 = bi_lstm(answer_embedding)
        answer_enc_1 = Dropout(0.5)(answer_enc_1)
        answer_enc_1 = BatchNormalization()(answer_enc_1)

        # qa_merged = dot([question_enc_1, answer_enc_1], axes=1, normalize=True)
        qa_merged = concatenate([question_enc_1, answer_enc_1])
        qa_merged = Dense(64, activation='relu')(qa_merged)
        qa_merged = Dropout(0.5)(qa_merged)
        qa_merged = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l1(0.001))(qa_merged)
        lstm_model = Model(name="bi_lstm", inputs=[
                           question, answer], outputs=qa_merged)
        output = lstm_model([question, answer])
        training_model = Model(
            inputs=[question, answer], outputs=output, name='training_model')
        opt = Adam(lr=0.0001)
        training_model.compile(loss='binary_crossentropy', optimizer=opt)
        # lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
        return training_model

    def main(self):
        eb = embedding.Embedding_Data()
        embed = EmbeddingMatrix()
        vocab = embed.create_vocab_dict()
        vocab_len = len(vocab)

        questions, answers, labels = eb.build_corpus('train.txt')
        questions = eb.turn_to_vector(questions, vocab)
        answers = eb.turn_to_vector(answers, vocab)
        Y = np.array(labels)
        q_dev, a_dev, l_dev = eb.build_corpus('dev.txt')
        q_dev_eb = eb.turn_to_vector(q_dev, vocab)
        a_dev_eb = eb.turn_to_vector(a_dev, vocab)

        training_model = self.get_bilstm_model(vocab_len, vocab)
        epoch = 1
        callback_list = [AnSelCB(q_dev, a_dev, l_dev, [q_dev_eb, a_dev_eb]),
                         ModelCheckpoint('model_improvement-{epoch:02d}-{map:.2f}.h5', monitor='map', verbose=1, save_best_only=True, mode='max'),
                         EarlyStopping(monitor='map', mode='max', patience=5)]

        training_model.fit(
            [questions, answers],
            Y,
            epochs=100,
            batch_size=64,
            validation_data=([q_dev, a_dev], l_dev),
            verbose=1,
            callbacks=callback_list
        )
        training_model.save_weights(
            'train_weights_epoch_' + str(epoch) + '.h5', overwrite=True)
        training_model.summary()

        # training_model.load_weights('model_CuDNNimprovement-07-0.73.h5')
        # questions, answers, labels = eb.build_corpus('test.txt')
        # questions_eb = eb.turn_to_vector(questions, vocab)
        # answers_eb = eb.turn_to_vector(answers, vocab)
        # print(questions_eb[0])
        # Y = np.array(labels)
        # sims = training_model.predict([questions_eb, answers_eb])
        # MAP, MRR = self.map_score(questions, answers, sims, Y)
        # print("MAP: ", MAP)
        # print("MRR: ", MRR)
        # res = training_model.evaluate([questions, answers], Y, verbose=1)
        # print (res)

        # with open('sim2.txt', 'w') as f:
        #     for i in range(len(sims)):
        #         # max_r = np.argmax(sims[i])
        #         f.write(str(sims[i]))
        #         f.write('\n')
        #     # print('\n')
        # f.close()


if __name__ == "__main__":
    a = ModelTraining()
    a.main()
