import numpy as np
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sknn.mlp import Classifier, Layer

class model_manager(object):
	def __init__(self):
		self.models = {}

	def build_model(self, move_number):
		X_train, y_train, X_test, y_test = self.load_data(move_number)
		model = RandomForestClassifier(verbose = 2, n_estimators = 100, n_jobs = -1)
		#model = GradientBoostingClassifier()
		nnet_layers = [Layer("Maxout", units=100, pieces=2), Layer('Softmax')]
		model = Classifier(layers=nnet_layers, learning_rate=0.001, n_iter=25)

		#model = model.fit(X_train, y_train.astype(int))
		model.verbose = 0
		pickle.dump( model, open( "model"+str(move_number)+".pkl", "wb" ))
		#self.models[move_number] = RFC_model
		print "==========="+str(move_number)+" COMPLETE==========="

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
		if not move_number in self.models:
			model = pickle.load( open( "model"+str(move_number)+".pkl", "rb" ) ) 
			print "model", model
			model.verbose = 0
			self.models[move_number] = model

	def load_many_models(self, start, end):
		for move_number in range(start, end):
			print "loading model for move", move_number, "..."
			self.load_model(move_number)

	def evaluate_position(self, most_popular_moves, npboard, move_number):
		return self.models[move_number].predict_proba(npboard)

	def guess_list(self, npboard, move_number, dm_board):
		prob_board = []
		pop_board = dm_board.popularity_boards[move_number+1]
		most_popular_moves = []
		pop_sorted = sorted(pop_board, reverse=True)
		for p in pop_sorted:
		    itemindex = np.where(pop_board==p)[0][0]
		    location = dm_board.columns[itemindex]
		    most_popular_moves.append(location)


		for i in range(len(npboard)):
			if dm_board.columns[i] in most_popular_moves[:50]:
				if npboard[i] == .5:
					#for the top 12 most common moves
					if move_number % 2 == 0:  # blacks turn
						npboard[i] = 0.
						prob = self.evaluate_position(most_popular_moves, npboard, move_number=move_number+1)[0][0]
					else:  # whites turn
						npboard[i] = 1.
						prob = self.evaluate_position(most_popular_moves, npboard, move_number=move_number+1)[0][1]
					prob_board.append(prob)
					npboard[i] = .5
				else:
					prob_board.append(0.)
					if npboard[i] == 0.:
						print 'black:', dm_board.columns[i]
					if npboard[i] == 1.:
						print 'white:', dm_board.columns[i]
			else:
				prob_board.append(0.)
				if npboard[i] == 0.:
						print 'black:', dm_board.columns[i]
				if npboard[i] == 1.:
					print 'white:', dm_board.columns[i]

		pop_prob_board = np.array(prob_board)# * np.power(pop_board,1./(17.*move_number ))
		prob_board_sorted = sorted(pop_prob_board, reverse=True)
		suggested_move_list = []
		for p in prob_board_sorted:
		    itemindex = np.where(pop_prob_board==p)[0][0]
		    location = dm_board.columns[itemindex]
		    suggested_move_list.append(location)
		if move_number % 2 == 0:
		    print 'black' 
		else:
		    print 'white'
		return suggested_move_list
		#print dm_board.columns