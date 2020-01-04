import pickle
import time
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core import Game, Ponyo, Shark

levels = [0, 1, 2, 3, 4]
colors = ['blue', 'purple', 'red', 'yellow']
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)

def display(genome, config):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    game = Game(Ponyo(genome, config), Shark(), (-15, 10))

    im1 = ax1.imshow(game.board(100), cmap=cmap, norm=norm, interpolation='none')
    im2 = ax2.imshow(game.ponyo_vision(), cmap=cmap, norm=norm, interpolation='none')
    # Helper function that updates the board and returns a new image of
    # the updated board animate is the function that FuncAnimation calls
    def update(frame):
        game.move_ponyo()
        game.move_shark()

        print(game.shark_position)

        if game.catched():
            game.finished = True
        im1.set_data(game.board(50))
        im2.set_data(game.ponyo_vision())

        ax1.title.set_text(f'frame: {game.frame} energy: {game.ponyo.energy}')
        return ax1, ax2


    ani = animation.FuncAnimation(fig, update, frames=game.generator, blit=False, repeat = False)
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
