import numpy as np
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
import pybrain as pb
from pybrain.datasets import ClassificationDataSet

class model_manager(object):
	def __init__(self):
		self.models = {}

	def build_model(self, move_number):
		X_train, y_train, X_test, y_test = self.load_data(move_number)
		RFC = RandomForestClassifier()
		RFC_model = RFC.fit(X_train, y_train)
		pickle.dump( self.models, open( "model"+str(move_number)+".pkl", "wb" ))
		self.models[move_number] = RFC_model
		print "===========COMPLETE==========="

	def build_neural_network(self, move_number):
		X_train, y_train, X_test, y_test = self.load_data(move_number)

		y_train = y_train.astype(int)
		y_test = y_test.astype(int)

		DS = ClassificationDataSet(len(X_train[0]), nb_classes=2, class_labels=['black','white'])


	def build_many_models(self, start, end):
		for move_number in range(start, end):
			self.build_model(move_number)

	def load_data(self, move_number):
		X_train = np.array( pickle.load( open( "X_train"+str(move_number)+".pkl", "rb" )))
		y_train = np.array( pickle.load( open( "y_train"+str(move_number)+".pkl", "rb" )))
		X_test = np.array( pickle.load( open( "X_test"+str(move_number)+".pkl", "rb" )))
		y_test = np.array( pickle.load( open( "y_test"+str(move_number)+".pkl", "rb" )))
		return X_train, y_train, X_test, y_test

	def load_model(self, move_number):
		model = pickle.load( open( "model"+str(move_number)+".pkl", "rb" ) ) 
		self.models[move_number] = model

	def load_many_models(self, start, end):
		for move_number in range(start, end):
			self.load_model(move_number)

	def evaluate_position(self, npboard, move_number):
		return self.models[move_number].predict_proba(npboard)

	def evaluate_next_move():
		pass