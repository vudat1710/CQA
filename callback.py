from keras.callbacks import Callback

def map_score(s1s_dev, s2s_dev, y_pred, labels_dev):
        QA_pairs = {}
        for i in range(len(s1s_dev)):
            pred = y_pred[i]

            s1 = " ".join(str(s1s_dev[i]))
            s2 = " ".join(str(s2s_dev[i]))
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

class AnSelCB(Callback):
    def __init__(self, val_q, val_s, y, inputs):
        self.val_q = val_q
        self.val_s = val_s
        self.val_y = y
        self.val_inputs = inputs
    
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.val_inputs)
        map__, mrr__ = map_score(self.val_q, self.val_s, pred, self.val_y)
        print('val MRR %f; val MAP %f' % (mrr__, map__))
        logs['mrr'] = mrr__
        logs['map'] = map__
