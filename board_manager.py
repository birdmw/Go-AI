import data_manager
import model_manager
import numpy as np
import argparse
import sys
from go import Board, BoardError, View, clear, getch

def main(moves_to_play = 3):
    # Get arguments
    parser = argparse.ArgumentParser(description='Starts a game of go in the terminal.')
    parser.add_argument('-s', '--size', type=int, default=19, help='size of board')

    args = parser.parse_args()

    if args.size < 7 or args.size > 19:
        sys.stdout.write('Board size must be between 7 and 19!\n')
        sys.exit(0)

    # Initialize board and view
    board = Board(args.size)
    view = View(board)
    err = None
    global move_count, prev_move_count, dm, mm, npboard, pred
    pred = (1,1)
    dm = data_manager.data_manager()
    dm.load_popularity_boards()
    mm = model_manager.model_manager()
    mm.load_many_models(1,moves_to_play)
    move_count = 0
    prev_move_count = move_count 

    #actions

    def goboard_to_npboard(goboard):
        global move_count
    	goboard_np = np.array(goboard)
        goboard_array = []
        for i in range(19):
            goboard_array.append([.5]*19)
        i,j=0,0
        for row in goboard:
            if j >18:
                j=0
            for col in row:
                if i >18:
                    i=0
                if col._type == 'white':
                    goboard_array[i][j] = 1.0
                elif col._type == 'black':
                    goboard_array[i][j] = 0.0
                else:
                    goboard_array[i][j] = 0.5
                i+=1
            j+=1
        for i in range(len(goboard_array)):
            goboard_array[i] = goboard_array[i][::-1]
        goboard_array = np.array(goboard_array).T
        return np.concatenate(goboard_array)
    	#return npboard
    
    def cpu_play():
        global mm, move_count, npboard
        global pred
        if move_count > 0:
            if (move_count % 2) == 0:
                color = 'b'
            else:
                color = 'w'
            predictions = mm.guess_list(npboard, move_count, dm)
            x, y = predictions[0]
            pred = predictions[0]
            move = (y+1, 18-x+1)
            board.move(move[0], move[1])
            view.redraw()

    def move():
        """
        Makes a move at the current position of the cursor for the current
        turn.
        """
        board.move(*view.cursor)
        view.redraw()

    def undo():
        """
        Undoes the last move.
        """
        board.undo()
        view.redraw()

    def redo():
        """
        Redoes an undone move.
        """
        board.redo()
        view.redraw()

    def exit():
        """
        Exits the game.
        """
        sys.exit(0)

    # Action keymap
    KEYS = {
        'w': view.cursor_up,
        's': view.cursor_down,
        'a': view.cursor_left,
        'd': view.cursor_right,
        ' ': move,
        'u': undo,
        'r': redo,
        'c': cpu_play,
        '\x1b': exit,
    }

    # Main loop
    while True:
        clear()
        global pred
        sys.stdout.write('{0}\n'.format(view))
        print "move #:", move_count
        sys.stdout.write('Black: {black} <===> White: {white}\n'.format(**board.score))
        sys.stdout.write('{0}\'s prediction '.format(pred))
        sys.stdout.write('{0}\'s '.format(mm.most_popular_moves))
        sys.stdout.write('{0}\'s move... '.format(board.turn))
        if err:
            sys.stdout.write('\n' + err + '\n')
            err = None

        # Get action key
        c = getch()
        global move_count, prev_move_count
        change_flag = 0
        try:
            # Execute selected action
            KEYS[c]()
            prev_move_count = move_count
            if c == ' ' or c == 'r' or c == 'c':
                move_count += 1
                change_flag = 1 
            elif c == 'u':
                move_count = max( [0, move_count-1] )
                change_flag = 1 

        except BoardError as be:
            # Board error (move on top of other piece, suicidal move, etc.)
            if change_flag == 1:
                move_count = prev_move_count
            change_flag = 1 
            err = be.message
        except KeyError:
            # Action not found, do nothing
            pass
        if change_flag == 1: # update global npboard
            global npboard
            npboard = goboard_to_npboard(board._state.board)
            #print board._state.board
            #print npboard


if __name__ == '__main__':
	main()
