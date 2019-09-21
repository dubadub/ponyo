import time
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import random

# Initialize the board with starting positions
def init_board(ponyo_position, shark_position, my_board):
    my_board[ponyo_position[0], ponyo_position[1]] = 1
    my_board[shark_position[0], shark_position[1]] = 2

    return my_board


# Input variables for the board
boardsize = 50        # board will be X by X where X = boardsize

# Initialize the board
ponyo_position = [45, 5]
shark_position = [5, 45]
epoch = 0
my_board = np.zeros((boardsize, boardsize))
my_board = init_board(ponyo_position, shark_position, my_board)

##### Animate the board #####

# Initialize the plot of the board that will be used for animation
fig = plt.gcf()
# Show first image - which is the initial board
im = plt.imshow(my_board)

def move_ponyo():
    delta_x = random.randint(-3, 3)
    delta_y = random.randint(-3, 3)

    if ponyo_position[0] + delta_x < 0 or ponyo_position[0] + delta_x >= boardsize:
        delta_x = 0

    if ponyo_position[1] + delta_y < 0 or ponyo_position[1] + delta_y >= boardsize:
        delta_y = 0

    ponyo_position_new = [ponyo_position[0] + delta_x, ponyo_position[1] + delta_y]

    my_board[ponyo_position_new[0], ponyo_position_new[1]] = 1
    my_board[ponyo_position[0], ponyo_position[1]] = 0

    ponyo_position[0] = ponyo_position_new[0]
    ponyo_position[1] = ponyo_position_new[1]

def move_shark():
    delta_x = 1 if ponyo_position[0] - shark_position[0] > 0 else -1
    delta_y = 1 if ponyo_position[1] - shark_position[1] > 0 else -1

    if shark_position[0] + delta_x < 0 or shark_position[0] + delta_x >= boardsize:
        delta_x = 0

    if shark_position[1] + delta_y < 0 or shark_position[1] + delta_y >= boardsize:
        delta_y = 0

    shark_position_new = [shark_position[0] + delta_x, shark_position[1] + delta_y]

    my_board[shark_position_new[0], shark_position_new[1]] = 2
    my_board[shark_position[0], shark_position[1]] = 0

    shark_position[0] = shark_position_new[0]
    shark_position[1] = shark_position_new[1]

finished = False

def gen():
    global finished
    global epoch

    while not finished:
        epoch += 1
        yield epoch

def update_board():
    global finished

    move_ponyo()
    move_shark()

    if shark_position[0] == ponyo_position[0] and shark_position[1] == ponyo_position[1]:
        finished = True

    return my_board


# Helper function that updates the board and returns a new image of
# the updated board animate is the function that FuncAnimation calls
def update(frame):
    im.set_data(update_board())
    global epoch
    plt.title(epoch)
    return im,


ani = animation.FuncAnimation(fig, update, frames=gen, blit=True, repeat = False)
plt.show()
