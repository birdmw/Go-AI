from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split
import cPickle as pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from gomill import sgf, sgf_moves
from gomill.ascii_boards import render_board
from gomill.boards import Board
from sklearn import svm
from copy import deepcopy
import pandas as pd
import numpy as np
from time import time
import re
from os import walk
from os.path import splitext, join
import matplotlib.pyplot as plt
import resource
from random import random
def select_files(root, files):
    """
    simple logic here to filter out interesting files
    .py files in this example
    """

    selected_files = []
    for file in files:
        full_path = join(root, file)
        ext = splitext(file)[1]
        if ext == ".sgf":
            selected_files.append(full_path)

    return selected_files

def build_recursive_dir_tree(path):
    """
    path    -    where to begin folder scan
    """
    selected_files = []

    for root, dirs, files in walk(path):
        selected_files += select_files(root, files)

    return selected_files
FULL_GAME_PATH = "/home/birdmw/Desktop/final_project"
root_dir = FULL_GAME_PATH
path_list = build_recursive_dir_tree(root_dir)
properties = {'BR':'Black Rank', 'HA':'Handicap', 'KM':'Komi','RE':'Result', 'SZ':'Size', 'WR':'White Rank'}
points = {}
for i in range(19):
            for j in range(19):
                points[(i, j)] = '0'
universal_cols = properties.keys()+points.keys()
def build_pickle_game_depth_range(a, b):
    for master_index in range(a,b):  
        def get_properties(game_string, properties):
            split_data = game_string.split(';')
            game_info = split_data[1].strip()
            game_data = np.char.strip( np.array(game_string[2:]), chars='\n\r\t)')
            prop_dict = {}
            for k, v in properties.iteritems():
                m = re.search(k+"(\[.+?\])", game_info)
                if m is not None:
                        prop_dict[k] = m.groups()[0].strip("[]")
                else:
                    prop_dict[k] = ''
            return prop_dict
        universal_index = 0
        values = []
        train = []
        test = []
        game_count = 1
        fail_count = 0
        for path in path_list:
            if random()<=.2:
                data_type = 'test'
            else:
                data_type = 'train'
            with open(path) as p:
                    try:
                        if game_count%5000==0:
                                print game_count, path
                                print master_index
                        game_string = p.read()
                        game = sgf.Sgf_game.from_string(game_string)
                        #=======data=========#
                        props = get_properties(game_string, properties)
                        #====================#
                        props['SZ'] = game.get_size()
                        if pd.isnull(props['SZ']):
                            props['SZ'] = 19
                        props['KM'] = game.get_komi()
                        if pd.isnull(props['KM']):
                            props['KM'] = 0.0
                        props['HA'] = game.get_handicap()
                        if pd.isnull(props['HA']):
                            props['HA'] = 0.0
                        props['RE'] = game.get_winner()
                        if pd.isnull(props['RE']):
                            props['RE'] = 'b' 
                        empty_board = Board(game.get_size())
                        board = sgf_moves.get_setup_and_moves(game, empty_board)  #(Board, list of tuples (colour, move))
                        for move in board[1][:master_index]:
                            colour, coords = move
                            x, y = coords
                            board[0].play(x,y,colour)
                            board_points = points
                            h = []
                            for moves in board[0].list_occupied_points():
                                board_points[moves[1]] = moves[0]
                            for v in props.values():
                                h.append(v)
                            for v in board_points.values():
                                h.append(v)
                            universal_index += 1
                            #values.append(h)
                        if data_type == 'train':
                            train.append(h)
                        else:  # data_type = 'test'
                            test.append(h)
                        game_count += 1
                    except:
                        print "failed", data_type, fail_count
                        fail_count += 1
        pickle.dump( train, open( "trainmove"+str(master_index)+".pkl", "wb" ) )
        pickle.dump( test, open( "testmove"+str(master_index)+".pkl", "wb" ) )
        print "Done!", master_index
build_pickle_game_depth_range(15, 20)   
print "===========COMPLETE==========="