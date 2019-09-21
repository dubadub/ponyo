import time
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import random

# Initialize the board with starting positions
def init_board(ponyo_position, my_board):
    my_board[ponyo_position[0], ponyo_position[1]] = 1
    return my_board


# Input variables for the board
boardsize = 50        # board will be X by X where X = boardsize

# Initialize the board
ponyo_position = [45, 5]
my_board = np.zeros((boardsize, boardsize))
my_board = init_board(ponyo_position, my_board)

##### Animate the board #####

# Initialize the plot of the board that will be used for animation
fig = plt.gcf()
# Show first image - which is the initial board
im = plt.imshow(my_board)

def move_ponyo(ponyo_position, board):
    delta_x = random.randint(-1, 1)
    delta_y = random.randint(-1, 1)

    if ponyo_position[0] + delta_x < 0 or ponyo_position[0] + delta_x >= boardsize:
        delta_x = 0

    if ponyo_position[1] + delta_y < 0 or ponyo_position[1] + delta_y >= boardsize:
        delta_y = 0

    ponyo_position_new = [ponyo_position[0] + delta_x, ponyo_position[1] + delta_y]

    board[ponyo_position_new[0], ponyo_position_new[1]] = 1
    board[ponyo_position[0], ponyo_position[1]] = 0

    ponyo_position[0] = ponyo_position_new[0]
    ponyo_position[1] = ponyo_position_new[1]


def update_board(ponyo_position, board):
    move_ponyo(ponyo_position, board)

    return board


# Helper function that updates the board and returns a new image of
# the updated board animate is the function that FuncAnimation calls
def update(frame):
    im.set_data(update_board(ponyo_position, my_board))

    return im,


ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128), blit=True)
plt.show()
