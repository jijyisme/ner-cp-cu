"""
Keras Model
"""

from keras.models import Sequential
from keras.layers import Embedding, LSTM, TimeDistributed, Dense
from keras.layers.wrappers import Bidirectional
from keras import metrics
from NER import constant

from keras_contrib.layers import CRF

class Model(object):
    def __init__(self):

        self.model_name = 'bi-lstm-crf'
        

        model = Sequential()
        # Random embedding

        model.add(Embedding(constant.WORD_INDEXER_SIZE, constant.EMBEDDING_SIZE, mask_zero=True))  # Random embedding
        model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))
        model.add(TimeDistributed(Dense(constant.NUM_TAGS, activation='softmax')))
        crf = CRF(constant.NUM_TAGS)
        model.add(crf)
        model.summary()

        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        self.model = model
        print(self.model_name)


def load_model(model_path):
    from keras_contrib.utils import save_load_utils
    model_architecture = Model()
    model = model_architecture.model
    save_load_utils.load_all_weights(model, model_path)
    return model


def save_model(model_path):
    pass