import numpy as np
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



class Game:

    def __init__(self, board_size, ponyo, shark):
        self.ponyo = ponyo
        self.shark = shark
        self.board_size = board_size        
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

        if self.ponyo.x + delta_x < 0 or self.ponyo.x + delta_x >= self.board_size:
            delta_x = 0

        if self.ponyo.y + delta_y < 0 or self.ponyo.y + delta_y >= self.board_size:
            delta_y = 0

        self.ponyo.decrease_energy(max(abs(delta_x), abs(delta_y)))        

        self.ponyo.x = self.ponyo.x + delta_x
        self.ponyo.y = self.ponyo.y + delta_y

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

        if self.shark.x + delta_x < 0 or self.shark.x + delta_x >= self.board_size:
            delta_x = 0

        if self.shark.y + delta_y < 0 or self.shark.y + delta_y >= self.board_size:
            delta_y = 0
        

        self.shark.x = self.shark.x + delta_x
        self.shark.y = self.shark.y + delta_y

    def ponyo_vision(self):
        return self.board(self.ponyo.x, self.ponyo.y, self.ponyo.vision_size)


    def catched(self):
        return self.shark.x == self.ponyo.x and self.shark.y == self.ponyo.y



    def board(self, x = None, y = None, N = None):
        if x == None:
            x = (self.board_size + 1) // 2 - 1 

        if y == None:
            y = (self.board_size + 1) // 2 - 1

        if N == None:
            N = (self.board_size + 1) // 2 - 1
        
        left = max(0, x - N)
        right = min(self.board_size, x + N + 1)
        top = max(0, y - N)
        bottom = min(self.board_size, y + N + 1)
        
        window = np.zeros((right - left, bottom - top))
        
        if self.ponyo.x >= left and self.ponyo.x < right and self.ponyo.y >= top and self.ponyo.y < bottom:
            window[self.ponyo.x - left, self.ponyo.y - top] = 1

        if self.shark.x >= left and self.shark.x < right and self.shark.y >= top and self.shark.y < bottom:
            window[self.shark.x - left, self.shark.y - top] = 2

        result = np.empty((2*N+1, 2*N+1))
        result[:] = -1

        ll = N - x
        tt = N - y

        result[ll+left:ll+right, tt+top:tt+bottom] = window
        
        return result   

