from data_manager import data_manager
class board_manager(object):
	def __init__(self):
		self.data_manager = data_manager()
		self.properties = self.data_manager.properties
		self.columns = self.data_manager.columns
	
	def path_to_npboard(self, move_number, game_path = "/home/birdmw/Desktop/final_project/pro_SGFs/1997/7/ChangHao-LeeChangho22917.sgf"):
		npboard = None
		with open(game_path) as p:
			game_string = p.read()
			npboard = self.data_manager.game_string_to_npboard(game_string, move_number)
		return npboard

	def npboard_to_stats_board(self, npboard, move_number):
		stats_board = [0.5]*361
		next_move_number = move_number + 1
		next_color = 1. if next_move_number % 2 == 0 else 0.

		for location in npboard:
			if location == .5:
			    location = next_color
			    stats_board[i] = self.data_manager.evaluate_position(npboard, next_move_number)
			    location = .5
			else:
			    pass
		return stats_board

	def npboard_to_board(self, npboard):
	    board = Board(19)
	    w_moves = self.columns[npboard==1.]
	    b_moves = self.columns[npboard==0.]
	    for w in w_moves:
	        board.play(w[0],w[1],'w')
	    for b in b_moves:
	        board.play(b[0],b[1],'b') 
	    return board 

	def top_moves_by_model(self, move_number):
		pass