"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def sum_empty(board):
    """
    Returns the number of empty in the board.
    """
    num_empty = 0;
    for lst in board:
        num_empty += sum(map(lambda x: x == EMPTY, lst))
    
    return num_empty


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_empty = sum_empty(board)

    if num_empty % 2 == 1:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible.add((i, j))

    return possible


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i = action[0]
    j = action[1]
    if i < 0 or i > 2 or j < 0 or j > 2 or board[i][j] != EMPTY:
        raise ValueError("Invalid move!")
    new = copy.deepcopy(board)
    current = player(new)
    new[i][j] = current
    return new

def row_win(board, row, p):
    """
    True if the row full of `p (current player)
    """
    return sum(map(lambda x: x == p, board[row])) == 3

def col_win(board, col, p):
    """
    True if the column full of `p
    """
    return board[0][col] == board[1][col] == board[2][col] == p

def has_won(board, p):
    """
    True if p has won the game
    """
    for i in range(3):
        if row_win(board, i, p) or col_win(board, i, p):
            return True
        
    return board[0][0] == board[1][1] == board[2][2] == p or \
        board[0][2] == board[1][1] == board[2][0] == p

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if has_won(board, X):
        return X
    if has_won(board, O):
        return O


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return sum_empty(board) == 0 or winner(board) != None


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return
    
    current = player(board)
    choices = actions(board)
    choices_rank = []
    for c in choices:
        choices_rank.append((c, estimate(board, current, c)))

    if (current == X):
        optiaml = max
    else:
        optiaml = min

    return optiaml(choices_rank, key=lambda x: x[1])[0]
    
    
def other(current):
    """
    Return the player other than the current one.
    """
    if (current == X):
        return O
    return X

def estimate(board, current, action):
    """
    Return the estimated score can get from the action.
    """
    new_board = result(board, action)
    if (terminal(new_board)):
        return utility(new_board)
    choices = actions(new_board)

    outcomes = set()
    for choice in choices:
        outcomes.add(estimate(new_board, other(current), choice))
    
    if current == X:
        return min(outcomes)
    return max(outcomes)

    

