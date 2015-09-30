# Go-AI
A depth-sliced learning program for playing Go

This project is composed of three important files:
board_manager.py - the highest level where boards are built and played on
model_manager.py - the middle level pieces for building prediction models
data_manager.py - the lowest level piece for manipulating SGF files

Motivation:

Go is a different style of game than chess. Where alpha-beta pruning is sufficient for modern computers to beat humans at chess, Go presents a unique challenge. In Go, there is a large set of possible games (10^761). To make a competitive Go playing algorithms, we will need to appraoch Go in a more creative way, different from the exhaustive and Monte-Carlo techniques used in Chess.

Technique:   Exhaustive   Monte-Carlo   Depth-sliced modeling
Complexity:  Checkers     Chess         Go

Abstract:

For this project we are divising a new way to approach playing Go with an AI. With current Go playing programs we see AI getting left in the dust in the early abstract stages of gameplay known as
"Fuseki". The computer is then left to fight from a disadvantaged position.

In order to remedy this situation, we can devise a new way of thinking about Go by slicing vertically across all games at a single depth (or "move number") and train a model upon each slice. In other words, take a training set of games, extract the board position at move N for each game, and then let that group of board positions be the training set for a model at move N.

Usage:

You will need a collection of .sgf files to train on: http://senseis.xmp.net/?GoDatabases

For the training data, data_manager.py expects your dataset to be a collection of .SGF files held in a base_path (or any sub directory as it will crawl recursively from the base path up). Open up Data Manager and change self.default_path to be the location of your SGF database. You can convert your .sgf files into training sets for each move_number by calling the following method:

data_manager.build_game_data(start_move, end_move, [games_to_load = -1, base_path])
	start_move = 1, end_move = 15 (these are reasonable #'s for a laptop)
	games to load = -1 means "all games"
	base_path = "my/path/to/SGF/files/here"

With your game files built and ready to train on, start by generating a popularity board, and a few models:

data_manager.build_popularity_boards()
model_manager.build_many_models(start, end)
	start = 1, end = 15 # for our example

Finally let's load up the models we built:

model_manager.load_many_models(1,15)

Next open board_manager.py and set the moves_to_play to be 15 and run board_manager.py

Play:

Your terminal should launch an ascii interface as follows: 

X . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . + . . . . . + . . . . . + . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . + . . . . . + . . . . . + . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . + . . . . . + . . . . . + . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
move #: 0
Black: 0 <===> White: 0
(1, 1)'s prediction None's Black's move... 


Use W-A-S-D to move your cursor 'X' around the board. Space bar places a stone, colors should alternate between white and black.

If you want the computer to play, simply press 'c'. The computer will think for a moment, and then return a move on the board as well as a list of other highly favorable moves.

use 'U' and 'R' to undo and redo moves.

