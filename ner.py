import csv
import gc
import glob
import json
import os
import shutil
import sys
import warnings
from collections import Counter

from multiprocessing import Process, Queue
from pprint import pprint

# Prevent Keras info message; "Using TensorFlow backend."
STDERR = sys.stderr
sys.stderr = open(os.devnull, "w")
sys.stderr = STDERR

import numpy as np
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning
import sklearn.metrics

from NER import constant
from NER.model import load_model, save_model, Model
from NER.metric import custom_metric
from NER.callback import CustomCallback

from keras_contrib.layers import CRF
from types import SimpleNamespace

class NamedEntityRecognizer(object):

    def __init__(self, model_path=None, new_model=False):

        self.new_model = new_model
        self.model_path = model_path
        self.model_architecture = Model()
        self.model = None

        if not self.new_model:
            if model_path is not None:
                self.model = load_model(model_path)
            else:
                self.model = load_model(constant.DEFAULT_MODEL_PATH)

    def evaluate(self, x_true, y_true):
        if self.new_model:
            print("model is not trained")
            return

        model = self.model

        pred = model.predict(x_true)
        print('pred shape is ', pred.shape)
        amax_pred = np.argmax(pred, axis=2)
        amax_true = np.argmax(y_true, axis=2)
        pred_flat = amax_pred.flatten()
        true_flat = amax_true.flatten()

        scores = custom_metric(true_flat, pred_flat)

        for score in scores:
            print(score,": ",scores[score])
        return scores


    def train(self, x_true, y_true, train_name, model_path=None, num_step=60, valid_split=0.1,
              initial_epoch=None, epochs=100, batch_size=32, learning_rate=0.001,
              shuffle=False, model= None):
        """Train model"""
        if(train_name==''):
            train_name = model.model_name

        # Create new model or load model
        if self.new_model:
            if model == None:
                initial_epoch = 0
                model = self.model_architecture.model
        else:
            if not model_path:
                raise Exception("Model path is not defined.")

            if initial_epoch is None:
                raise Exception("Initial epoch is not defined.")

            model = load_model(model_path)

        # Display model summary before train
        model.summary()

        callbacks = CustomCallback(train_name).callbacks
        self.model = model

        # Train model
        model.fit(x_true, y_true, validation_split=valid_split,
                  initial_epoch=initial_epoch, epochs=epochs,
                  batch_size=batch_size, shuffle=shuffle ,callbacks=callbacks)

        self.new_model = False

    def save(self, path, name):
        # Save model architecture to file
        with open(os.path.join(path, name+".json"), "w") as file:
            file.write(self.model.to_json())

        # Save model config to file
        with open(os.path.join(path, name+"_config.txt"), "w") as file:
            pprint(self.model.get_config(), stream=file)

        self.model.save(os.path.join(path,name+'.hdf5'))

    def predict(self, x_vector):
        if self.new_model:
            print("model is not trained")
            return
        model = self.model
        print('make prediction')
        per = model.predict(x_vector)
        print('flatten data')
        amax = np.argmax(per, axis=2)
        predict_y = amax.flatten()
        x_flat = x_vector.flatten()
        print('return')
        return dict({
            'x': x_flat,
            'ner_tag': predict_y
        })