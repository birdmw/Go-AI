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

properties = {'BR':'Black Rank', 'HA':'Handicap', 'KM':'Komi','RE':'Result', 'SZ':'Size', 'WR':'White Rank'}
points = {}
npboard_table = []
for i in range(19):
            for j in range(19):
                points[(i, j)] = '0'
universal_cols = sorted(properties.keys())+sorted(points.keys())
print universal_cols[:40]

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

def build_game_data(a, b):
    base_path = "/home/birdmw/Desktop/final_project/"
    path_list = build_recursive_dir_tree(base_path)
    for master_index in range(a,b):  
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
                        board = sgf_moves.get_setup_and_moves(game, empty_board)
                        for move in board[1][:master_index]:
                            colour, coords = move
                            x, y = coords
                            board[0].play(x,y,colour)
                            board_points = deepcopy(points)
                        h = []
                        for moves in board[0].list_occupied_points():
                            board_points[moves[1]] = moves[0]

                        for c in universal_cols[:6]:
                            h.append(props[c])
                        for c in universal_cols[6:]:
                            h.append(board_points[c])

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
    print "===========COMPLETE==========="
#build_game_data(20, 25)   


def build_models(a, b):
    for move_number in range(a,b):
        print "building model for move number", move_number,"..."

        train = np.array( pickle.load( open( "trainmove"+str(move_number)+".pkl", "rb" ) ) )
        test = np.array( pickle.load( open( "testmove"+str(move_number)+".pkl", "rb" ) ) )
        columns = universal_cols[6:]
        X_train = train[:,6:]
        X_test = test[:,6:]
        y_train = train[:,universal_cols.index('RE')]
        y_test = test[:,universal_cols.index('RE')]

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

        RFC = RandomForestClassifier(n_jobs=-1)
        GBC = GradientBoostingClassifier()
        LR = LogisticRegression()
        ADA = AdaBoostClassifier()

        RFC_model = RFC.fit(X_train, y_train)
        print "RFC fit"
        GBC_model = GBC.fit(X_train, y_train)
        print "GBC fit"
        LR_model = LR.fit(X_train, y_train)
        print "LR fit"
        ADA_model = ADA.fit(X_train, y_train)
        print "ADA fit"

        print "===== models for", move_number,"COMPLETE ====="

        RFC_y_pred = RFC_model.predict(X_test)
        GBC_y_pred = GBC_model.predict(X_test)
        LR_y_pred = LR_model.predict(X_test)
        ADA_y_pred = ADA_model.predict(X_test)

        RFC_roc_auc = roc_auc_score(y_test, RFC_y_pred)
        GBC_roc_auc = roc_auc_score(y_test, GBC_y_pred)
        LR_roc_auc  = roc_auc_score(y_test, LR_y_pred)
        ADA_roc_auc = roc_auc_score(y_test, ADA_y_pred)

        print "RandomForestClassifier roc_auc:", RFC_roc_auc
        print "GradientBoostingClassifier roc_auc:", GBC_roc_auc
        print "LogisticRegression roc_auc:", LR_roc_auc
        print "AdaBoostClassifier roc_auc:", ADA_roc_auc

        roc_aucs = [RFC_roc_auc, GBC_roc_auc, LR_roc_auc, ADA_roc_auc]
        models = [RFC_model, GBC_model, LR_model, ADA_model]

        print "pickling..."
        pickle.dump( roc_aucs, open( "roc_auc"+str(move_number)+".pkl", "wb" ) )
        pickle.dump( models, open( "models"+str(move_number)+".pkl", "wb" ) )
        print "pickling complete", move_number
        
def unpickle_models(a,b):
    print "unpickling models", a, "to", b
    m = []
    ra = []
    for i in range(a,b):
        m.append( pickle.load( open( "models"+str(i+1)+".pkl", "rb" ) ) )
        ra.append( np.array( pickle.load( open( "roc_auc"+str(i+1)+".pkl", "rb" ) ) ) )
        print i
    print "unpickled"
    return m, ra

def model_predict(npboard, move_number):
    '''
    takes an npboard array and a move number and returns the likelyhood of white winning
    '''
    a = []
    for i in range(len(models[move_number])):
        a.append(models[move_number][i].predict(npboard)[0])
    predictions = np.array(a)
    roc_auc_sum = np.sum( roc_auc[move_number] )
    print roc_auc_sum
    auc_scaled = np.sum(predictions * roc_auc[move_number]) / (roc_auc_sum)
    return auc_scaled

def f(x):
    if x == 'w':
        return 1.
    elif x == 'b':
        return 0.
    else:
        return .5

def path_to_npboard(game_path, move_number):
    columns = universal_cols[6:]
    #print type(columns[0][0])
    game_state = []
    with open(game_path) as p:
        game_string = p.read()
        game = sgf.Sgf_game.from_string(game_string)
        props = get_properties(game_string, properties)
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
        board = sgf_moves.get_setup_and_moves(game, empty_board)
        for move in board[1][:move_count]:
            colour, coords = move
            x, y = coords
            board[0].play(x,y,colour)
        board_points = points
        h = []
        for moves in board[0].list_occupied_points():
            board_points[moves[1]] = moves[0]
        for c in universal_cols[:6]:
            h.append(props[c])
        for c in universal_cols[6:]:
            h.append(board_points[c])
        game_state = np.array(h[6:])
    v_f = np.vectorize(f)
    game_state = v_f(game_state)
    return game_state

def unpickle_models(a,b):
    move_count = a
    move_count_end = b  # last item not included
    m = [0] * 400
    ra = [0] * 400
    models[move_count:move_count_end], roc_auc[move_count:move_count_end] = unpickle_models(move_count, move_count_end)
    return models, roc_auc

def game_path_to_stats_board(move_count, game_path = "/home/birdmw/Desktop/final_project/pro_SGFs/1941/12/HayashiYutaro-MaedaNobuaki4362.sgf"):

    stats_board_if_w_plays = [0.5]*361
    stats_board_if_b_plays = [0.5]*361

    npboard = path_to_npboard(game_path, move_count)
    columns = universal_cols[6:]
    print columns

    for i in range(len(npboard)):
        if npboard[i] == .5:
            npboard[i] = 1.
            stats_board_if_w_plays[i] = model_predict(npboard, move_count+1)
            npboard[i] = 0.
            stats_board_if_b_plays[i] = model_predict(npboard, move_count+1)
            npboard[i] = .5
        else:
            pass
    return np.round(np.array(stats_board_if_b_plays),2), np.round(np.array(stats_board_if_w_plays),2)

#build_models(1,10)
#models, roc_auc = unpickle_models(1,5)
#print game_path_to_stats_board(move_count =1)
