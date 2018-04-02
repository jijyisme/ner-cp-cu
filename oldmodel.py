from loader import prepare_train_data
import numpy as np

from keras.models import load_model
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

class NameEntityRecognizer(object):
	def __init__(self,model_path=None, max_num_words=300, word_vec_length=100):
		self.model_status = ''
		if model_path is None:
			self.mode = 'new_model'
			self.initialize_model(max_num_words, word_vec_length)
		else:
			self.mode = 'existing_model'
			self.model = load_model(model_path)

	def initialize_model(self, max_num_words, word_vec_length):
		self.model = Sequential()
		lstm = LSTM(64,input_shape=(max_num_words,word_vec_length),return_sequences=True)

		self.model.add(Bidirectional(lstm,
		            input_shape=(max_num_words, word_vec_length)))

	def add_lstm_layer(self,input_shape, bidirectional=True):
		if(self.mode == 'existing_model'):
			print('The loaded model cannot be changed')
			return
		else:
			lstm = LSTM(64,input_shape=input_shape,return_sequences=True)

			self.model.add(Bidirectional(lstm,
			            input_shape=input_shape))

	def add_dense_layer(self, output_length, activation='softmax'):
		if(self.mode == 'existing_model'):
			print('The loaded model cannot be changed')
			return
		else:
			self.model.add(Dense(output_length, activation=activation))

	def compile(self):
		if(self.mode == 'existing_model'):
			print('The loaded model cannot be changed')
			return
		else:
			self.model.compile(optimizer='adam',
			              loss='categorical_crossentropy',
			              metrics=['accuracy'])
			print(self.model.summary())

	def train(self, x_train, y_train):
		return self.model.fit(x_train, y_train, epochs=1, batch_size = 128)		

	def predict(self, data):
		return self.model.predict(data)

	def evaluate(self, ):
