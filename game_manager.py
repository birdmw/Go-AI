import board_manager as bm
bm = reload(bm)
bm.main()

#TASKS:
'''
1. add function to board_manager.py def get_board() that returns 
	the board as a npboard.
2. have board_manager import model_manager.
3. add function to model_manager def get_move_list(npboard, move_number) 
	that game_manager (this file) can call and check for legal moves and 
	then play it on behalf of the computer
'''