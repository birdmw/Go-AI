import re
import numpy as np
import pandas as pd
import cPickle as pickle
from os import walk
from os.path import splitext, join
from random import random
from gomill import sgf, sgf_moves
from gomill.boards import Board
from copy import deepcopy
from time import time
import os

class data_manager(object):
	def __init__(self):
		self.properties = {'BR':'Black Rank', 'HA':'Handicap', 'KM':'Komi','RE':'Result', 'SZ':'Size', 'WR':'White Rank'}
		self.locations = {}
		for ix in range(19):
			for iy in range(19):
				self.locations[(ix,iy)] = .5
		self.universal_columns = sorted(self.properties.keys()) + sorted(self.locations.keys())
		self.prop_list = self.universal_columns[:6]
		self.columns = self.universal_columns[6:]
		self.default_path = "/home/birdmw/Desktop/final_project/"
	
	def select_files(self, root, files):
		selected_files = []
		for file in files:
			full_path = join(root, file)
			ext = splitext(file)[1]
			if ext == ".sgf":
				selected_files.append(full_path)
		return selected_files

	def build_recursive_dir_tree(self, path):
		selected_files = []

		for root, dirs, files in walk(path):
			selected_files += self.select_files(root, files)
		return selected_files

	def get_properties(self, game_string):
		split_data = game_string.split(';')
		game_info = split_data[1].strip()
		game_data = np.char.strip( np.array(game_string[2:]), chars='\n\r\t)')
		prop_dict = {}
		for k, v in self.properties.iteritems():
			m = re.search(k+"(\[.+?\])", game_info)
			if m is not None:
				prop_dict[k] = m.groups()[0].strip("[]")
			else:
				prop_dict[k] = ''
		return prop_dict

	def get_more_properties(self, properties, sgf_game):
		properties['SZ'] = sgf_game.get_size()
		if pd.isnull(properties['SZ']):
			properties['SZ'] = 19
		properties['KM'] = sgf_game.get_komi()
		if pd.isnull(properties['KM']):
			properties['KM'] = 0.0
		properties['HA'] = sgf_game.get_handicap()
		if pd.isnull(properties['HA']):
			properties['HA'] = 0.0
		properties['RE'] = sgf_game.get_winner()
		if pd.isnull(properties['RE']):
			properties['RE'] = 'b' 
		return properties

	def game_path_to_npboard(self, move_number, game_path=None):
		if game_path == None:
			game_path = "/home/birdmw/Desktop/final_project/pro_SGFs/1997/7/ChangHao-LeeChangho22917.sgf"
		game_string = ''
		with open(game_path) as p:
			game_string = p.read()
		npboard = self.game_string_to_npboard(game_string, move_number=move_number)


		for i in range(len(npboard))[6:]:
			if npboard[i] =='w':
				npboard[i] = 1. 
			elif npboard[i] =='b':
				npboard[i] = 0. 
			else:
				npboard[i] = .5
		return npboard

	def game_string_to_npboard(self, game_string, move_number):
		sgf_game = sgf.Sgf_game.from_string(game_string)
		properties = self.get_properties(game_string)
		properties = self.get_more_properties(properties, sgf_game)
		board = sgf_moves.get_setup_and_moves(sgf_game, Board(sgf_game.get_size()))
		npboard = None
		if board[1][0][0] == 'b':
			for move in board[1][:move_number]:
				color, coordinates = move
				x, y = coordinates
				board[0].play(x, y, color)
				board_locations = deepcopy(self.locations)
			for moves in board[0].list_occupied_points():
				board_locations[moves[1]] = moves[0]
			npboard = []
			for column in self.universal_columns[:6]:
				npboard.append(properties[column])
			for column in self.universal_columns[6:]:
				npboard.append(board_locations[column])
		return npboard

	def process_train_test_data(self, train, test):
		X_train = np.array(train)[:,6:]
		X_test = np.array(test)[:,6:]
		y_train = np.array(train)[:,self.prop_list.index('RE')]
		y_test = np.array(test)[:,self.prop_list.index('RE')]

		y_train[y_train=='w']=1.
		y_train[y_train=='b']=0.
		y_test[y_test=='w']=1.
		y_test[y_test=='b']=0.

		for row in X_train:
			row[row=='w'] = 1.
			row[row=='0'] = .5
			row[row=='b'] = 0.
			row = row.astype(float)

		for row in X_test:
			row[row=='w'] = 1.
			row[row=='0'] = .5
			row[row=='b'] = 0.
			row = row.astype(float)

		X_train=X_train.astype(float)
		X_test=X_test.astype(float)

		y_train=y_train.astype(float)
		y_test=y_test.astype(float)

		return X_train, y_train, X_test, y_test

	def build_game_data(self, start_move, end_move, games_to_load = -1, base_path = None, symmetry = 0):
		if base_path == None:
			base_path = self.default_path
		path_list = self.build_recursive_dir_tree(base_path)
		total_games = len(path_list[:games_to_load]) * (end_move-start_move)
		start_time, time_left, game_count = time(), 0, 1
		for move_number in range(start_move, end_move):
			print "move number", move_number, "of", end_move-1
			train, test= [], []
			for path in path_list[:games_to_load]:
				data_type = 'train' if random() >= .1 else 'test'
				with open(path) as p:
					try:	
						if (game_count % 500) == 0:
							t_passed = (time()-start_time)/60.
							total_time = t_passed * total_games / float(game_count)
							now_time_left = int(total_time - t_passed)
							if now_time_left != time_left:
								if now_time_left > 60:
									print now_time_left/60., "hours remaining" 
								else:
									print now_time_left, "minutes remaining"
								time_left = now_time_left
						game_string = p.read() 
						npboard = self.game_string_to_npboard(game_string, move_number)
						if npboard != None:
							train.append(npboard) if data_type == 'train' else test.append(npboard)
					except:
						pass
					game_count += 1
			X_train, y_train, X_test, y_test = self.process_train_test_data(train, test)
			if symmetry == 1:
				train_length = X_train.shape[0]
				for i in xrange(train_length):
					if i%1000==0:
						print i
					x = X_train[i].reshape(19,19)
					for j in xrange(1,3):
						m_rot = np.rot90(x, j).reshape(361,)
						np.append(X_train, m_rot)
						np.append(y_train, y_train[i])
			pickle.dump( X_train, open( "X_train"+str(move_number)+".pkl", "wb" ) )
			pickle.dump(  y_train, open( "y_train" +str(move_number)+".pkl", "wb" ) )
			pickle.dump( X_test, open( "X_test"+str(move_number)+".pkl", "wb" ) )
			pickle.dump(  y_test, open( "y_test" +str(move_number)+".pkl", "wb" ) )
			print "move number", move_number, "pickled"
		print "===========COMPLETE==========="

	def location_to_npboard_index(self, location):
		return self.columns.index(location)

	def build_popularity_boards(self, games_to_load = -1, base_path = "/home/birdmw/Desktop/final_project/"):
		self.popularity_boards = []
		for i in range(362):
			self.popularity_boards.append([0]*len(self.columns)) 
		path_list = self.build_recursive_dir_tree(base_path)
		for path in path_list[:games_to_load]:
			try:
				with open(path) as p:
					game_string = p.read()
					sgf_game = sgf.Sgf_game.from_string(game_string)
					board = sgf_moves.get_setup_and_moves(sgf_game, Board(sgf_game.get_size()))
					if board[1][0][0] == 'b': # no handicap
						for i in range(len(board[1])):
							location = board[1][i][1]
							if location != None:
								npb_index = self.location_to_npboard_index(location)
								self.popularity_boards[i][ npb_index ] += 1
					else:
						os.remove(path)
			except:
				os.remove(path)
		for pb in self.popularity_boards:
			sumVal = sum(pb)
			for loc in range(len(pb)):
				pb[loc] = round(pb[loc] / float(sumVal),3)
		self.popularity_boards = np.array(self.popularity_boards)
		pickle.dump(  self.popularity_boards, open( "popularity_boards.pkl", "wb" ) )
		print "==========COMPLETE========="
		return self.popularity_boards

	def load_popularity_boards(self):
		self.popularity_boards = pickle.load( open( "popularity_boards.pkl", "rb" ))

	def npboard_to_gomillboard(self):
		pass

	def millboard_to_sgf_string(self):
		pass

	def sgf_string_to_sgf_file(self):
		pass