from loader import prepare_train_data
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

class NameEntityRecognizer(object):
	def __init__(self,max_num_words,WORD_VEC_SIZE,tag_list):
		self.max_num_words = max_num_words
		self.tag_list = tag_list
		self.model = Sequential()
		lstm = LSTM(64,input_shape=(max_num_words,WORD_VEC_SIZE),return_sequences=True)
		self.model.add(Bidirectional(lstm,
		            input_shape=(max_num_words, WORD_VEC_SIZE)))
		self.model.add(Bidirectional(lstm,
		            input_shape=(max_num_words, WORD_VEC_SIZE)))
		self.model.add(Bidirectional(lstm,
		            input_shape=(max_num_words, WORD_VEC_SIZE)))
		self.model.add(Dense(len(tag_list), activation='softmax'))

		self.model.compile(optimizer='adam',
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])
		print(self.model.summary())
		
	def tag_decode(self, y_dummy):
		y_pred = []	
		for post in y_dummy:
			temp = []
			for i in range(len(post)):
				temp.append(self.tag_list[np.argmax(y_dummy[0][i])])
			y_pred.append(temp)
		return y_pred

	def train(self, x_train, y_train):
		return self.model.fit(x_train, y_train, epochs=1, batch_size = 128)		

	def predict(self, data):
		y_dummy = self.model.predict(data)
		return self.tag_decode(y_dummy)