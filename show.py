import pickle
import time
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core import Game, Ponyo, Shark

def display(genome, config):
    
    fig = plt.gcf()

    game = Game(49, Ponyo(25, 25, genome, config), Shark(20, 30))

    im = plt.imshow(game.board())
    # Helper function that updates the board and returns a new image of
    # the updated board animate is the function that FuncAnimation calls
    def update(frame):
        game.move_ponyo()
        game.move_shark()

        if game.catched():
            game.finished = True
        im.set_data(game.board())

        plt.title(f'frame: {game.frame} energy: {game.ponyo.energy}')
        return im,


    ani = animation.FuncAnimation(fig, update, frames=game.generator, blit=True, repeat = False)
    plt.show()


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory. 
    with open('winner.pickle', 'rb') as f:
        winner = pickle.load(f)

    with open('config.pickle', 'rb') as f:
        config = pickle.load(f)

    display(winner, config)