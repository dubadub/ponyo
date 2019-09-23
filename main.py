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
ponyo_position = [25, 25]
shark_position = [5, 45]
epoch = 0
my_board = np.zeros((boardsize, boardsize))
my_board = init_board(ponyo_position, shark_position, my_board)

class Ponyo:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, delta_x, delta_y):
        self.x = self.x + delta_x
        self.y = self.y + delta_y


class Shark:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, delta_x, delta_y):
        self.x = self.x + delta_x
        self.y = self.y + delta_y


class Board:

    def __init__(self, size):
        self.size = size
        self.values = np.zeros((size, size))

    def set(self, x, y, value):
        self.values[x,y] = value


class Game:

    def __init__(self, board, ponyo, shark):
        self.ponyo = ponyo
        self.shark = shark
        self.board = board
        self.finished = False
        self.frame = 0


    def generator(self):
        while not self.finished:
            self.frame += 1
            yield self.frame

    def tick(self):
        self.move_ponyo()
        self.move_shark()

        if self.shark.x == self.ponyo.x and self.shark.y == self.ponyo.y:
            self.finished = True

    def move_ponyo(self):
        delta_x = random.randint(-3, 3)
        delta_y = random.randint(-3, 3)

        if self.ponyo.x + delta_x < 0 or self.ponyo.x + delta_x >= self.board.size:
            delta_x = 0

        if self.ponyo.y + delta_y < 0 or self.ponyo.y + delta_y >= self.board.size:
            delta_y = 0

        ponyo_position_new = [self.ponyo.x + delta_x, self.ponyo.y + delta_y]

        self.board.set(ponyo_position_new[0], ponyo_position_new[1], 1)
        self.board.set(self.ponyo.x, self.ponyo.y, 0)

        self.ponyo.x = ponyo_position_new[0]
        self.ponyo.y = ponyo_position_new[1]

    def move_shark(self):
        delta_x = 1 if self.ponyo.x - self.shark.x > 0 else -1
        delta_y = 1 if self.ponyo.y - self.shark.y > 0 else -1

        if self.shark.x + delta_x < 0 or self.shark.x + delta_x >= self.board.size:
            delta_x = 0

        if self.shark.y + delta_y < 0 or self.shark.y + delta_y >= self.board.size:
            delta_y = 0

        shark_position_new = [self.shark.x + delta_x, self.shark.y + delta_y]

        self.board.set(shark_position_new[0], shark_position_new[1], 2)
        self.board.set(self.shark.x, self.shark.y, 0)

        self.shark.x = shark_position_new[0]
        self.shark.y = shark_position_new[1]

##### Animate the board #####

# Initialize the plot of the board that will be used for animation
fig = plt.gcf()
# Show first image - which is the initial board
im = plt.imshow(my_board)

game = Game(Board(50), Ponyo(25,25), Shark(5, 45))
# Helper function that updates the board and returns a new image of
# the updated board animate is the function that FuncAnimation calls
def update(frame):
    game.tick()
    im.set_data(game.board.values)

    plt.title(game.frame)
    return im,


ani = animation.FuncAnimation(fig, update, frames=game.generator, blit=True, repeat = False)
plt.show()

