import numpy as np
import os
import neat
import pickle

from core import Game, Board, Ponyo, Shark

def eval_genomes(genomes, config):    

    games = []

    for genome_id, genome in genomes:
        genome.fitness = 0  

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
    winner = p.run(eval_genomes, 5)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    # Initialize the plot of the board that will be used for animation
    with open('winner.pickle', 'wb') as f:
        pickle.dump(winner, f)

    with open('config.pickle', 'wb') as f:
        pickle.dump(config, f)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
