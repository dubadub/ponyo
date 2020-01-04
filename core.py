import numpy as np
import neat

class Ponyo:

    def __init__(self, genome, config):
        self.vision_size = 7
        self.max_energy = 3
        self.energy = 2
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)


class Shark:

    def __init__(self):
        self


class Game:

    def __init__(self, ponyo, shark, shark_position):
        self.ponyo = ponyo
        self.shark = shark
        self.board_size = 2 * ponyo.vision_size + 1
        self.finished = False
        self.frame = 0
        self.shark_position = shark_position


    def generator(self):
        while not self.finished:
            self.frame += 1
            yield self.frame

    def move_ponyo(self):
        visible = tuple(self.ponyo_vision().reshape(1, -1)[0])

        output = self.ponyo.net.activate(visible)

        delta_x = int(round(2 * output[0]))
        delta_y = int(round(2 * output[1]))

        # print((delta_x, delta_y))

        self.shark_position = (self.shark_position[0] - delta_x, self.shark_position[1] - delta_y)

    def move_shark(self):
        delta_x = self.shark_position[0]
        delta_y = self.shark_position[1]

        if self.shark_position[0] > 0:
            delta_x = -1
        if self.shark_position[0] < 0:
            delta_x = 1

        if self.shark_position[1] > 0:
            delta_y = -1
        if self.shark_position[1] < 0:
            delta_y = 1

        self.shark_position = (self.shark_position[0] + delta_x, self.shark_position[1] + delta_y)

    def ponyo_vision(self):
        return self.board(self.board_size)


    def board(self, size):
        visible_area = np.zeros((size, size))
        centre = size // 2

        visible_area[centre + 1, centre + 1] = 2

        if abs(self.shark_position[0]) <= centre and abs(self.shark_position[1]) <= centre:
            visible_area[centre + self.shark_position[0], centre + self.shark_position[1]] = 3

        return visible_area


    def catched(self):
        return self.shark_position[0] == 0 and self.shark_position[1] == 0

    def escaped(self):
        return abs(self.shark_position[0]) > self.ponyo.vision_size and abs(self.shark_position[1]) > self.ponyo.vision_size

