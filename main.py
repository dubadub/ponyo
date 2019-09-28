import time
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import random
import os
import neat

class Ponyo:

    def __init__(self, x, y, genome, config):
        self.vision_size = 4
        self.max_energy = 3
        self.energy = 2
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.x = x
        self.y = y

    def move(self, delta_x, delta_y):
        self.x = self.x + delta_x
        self.y = self.y + delta_y

    def increase_energy(self):
        self.energy += 1
        if self.energy > self.max_energy:
            self.energy = self.max_energy

    def decrease_energy(self, delta):
        self.energy -= delta
        if self.energy < 0:
            self.energy = 0


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
        self.board.values[ponyo.x, ponyo.y] = 1
        self.board.values[shark.x, shark.y] = 2
        self.finished = False
        self.frame = 0


    def generator(self):
        while not self.finished:
            self.frame += 1
            yield self.frame

    def move_ponyo(self):
        visible = tuple(self.ponyo_vision().reshape(1, -1)[0])

        output = self.ponyo.net.activate(visible)

        self.ponyo.increase_energy()

        delta_x = int(round(self.ponyo.energy * output[0]))
        delta_y = int(round(self.ponyo.energy * output[1]))



        if self.ponyo.x + delta_x < 0 or self.ponyo.x + delta_x >= self.board.size:
            delta_x = 0

        if self.ponyo.y + delta_y < 0 or self.ponyo.y + delta_y >= self.board.size:
            delta_y = 0

        self.ponyo.decrease_energy(max(abs(delta_x), abs(delta_y)))

        ponyo_position_new = [self.ponyo.x + delta_x, self.ponyo.y + delta_y]

        self.board.set(self.ponyo.x, self.ponyo.y, 0)
        self.board.set(ponyo_position_new[0], ponyo_position_new[1], 1)

        self.ponyo.x = ponyo_position_new[0]
        self.ponyo.y = ponyo_position_new[1]

    def move_shark(self):
        delta_x = self.ponyo.x - self.shark.x
        delta_y = self.ponyo.y - self.shark.y

        if delta_x > 0:
            delta_x = 1
        if delta_x < 0:
            delta_x = -1

        if delta_y > 0:
            delta_y = 1
        if delta_y < 0:
            delta_y = -1

        if self.shark.x + delta_x < 0 or self.shark.x + delta_x >= self.board.size:
            delta_x = 0

        if self.shark.y + delta_y < 0 or self.shark.y + delta_y >= self.board.size:
            delta_y = 0

        shark_position_new = [self.shark.x + delta_x, self.shark.y + delta_y]

        self.board.set(shark_position_new[0], shark_position_new[1], 2)
        self.board.set(self.shark.x, self.shark.y, 0)

        self.shark.x = shark_position_new[0]
        self.shark.y = shark_position_new[1]

    def ponyo_vision(self):
        def neighbors(arr, x, y, N):
            left = max(0, x - N)
            right = min(arr.shape[0], x + N + 1)
            top = max(0, y - N)
            bottom = min(arr.shape[1], y + N + 1)

            window = arr[left:right,top:bottom]
            fillval = -1

            result = np.empty((2*N+1, 2*N+1))
            result[:] = fillval

            ll = N - x
            tt = N - y
            result[ll+left:ll+right,tt+top:tt+bottom] = window

            return result


        return neighbors(self.board.values, self.ponyo.x, self.ponyo.y, self.ponyo.vision_size)


    def catched(self):
        return self.shark.x == self.ponyo.x and self.shark.y == self.ponyo.y



gen = 0

def eval_genomes(genomes, config):

    global gen

    gen += 1

    games = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0

        games.append(Game(Board(50), Ponyo(25, 25, genome, config), Shark(20, 30)))
        games.append(Game(Board(50), Ponyo(25, 25, genome, config), Shark(30, 20)))
        games.append(Game(Board(50), Ponyo(25, 25, genome, config), Shark(30, 30)))
        games.append(Game(Board(50), Ponyo(25, 25, genome, config), Shark(20, 20)))
        games.append(Game(Board(50), Ponyo(25, 25, genome, config), Shark(20, 25)))
        games.append(Game(Board(50), Ponyo(25, 25, genome, config), Shark(25, 20)))
        games.append(Game(Board(50), Ponyo(25, 25, genome, config), Shark(30, 25)))
        games.append(Game(Board(50), Ponyo(25, 25, genome, config), Shark(25, 30)))


    frame = 0
    while len(games) > 0:

        frame += 1

        if frame == 500:
            break

        for x, game in enumerate(games):
            game.ponyo.genome.fitness += 0.1

            game.move_ponyo()
            game.move_shark()


        for x, game in enumerate(games):
            if game.catched():
                game.ponyo.genome.fitness -= 1
                games.pop(x)




def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    # Initialize the plot of the board that will be used for animation
    fig = plt.gcf()

    game = Game(Board(50), Ponyo(25, 25, winner, config), Shark(15, 40))

    im = plt.imshow(game.board.values)
    # Helper function that updates the board and returns a new image of
    # the updated board animate is the function that FuncAnimation calls
    def update(frame):
        game.move_ponyo()
        game.move_shark()

        if game.catched():
            game.finished = True
        im.set_data(game.board.values)

        plt.title(f'frame: {game.frame} energy: {game.ponyo.energy}')
        return im,


    ani = animation.FuncAnimation(fig, update, frames=game.generator, blit=True, repeat = False)
    plt.show()



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
