import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from ReG_Arena import Arena
from ReG_MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning for single-player games.
    
    It uses the functions defined in Game and NeuralNet. args are specified in main.py.
    For single-player games, the value assignment logic is simplified - values are
    not negated based on player since there's only one player.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play for a single-player game.
        
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        For single-player games:
        - Player is always 1 (no alternation)
        - Value v is the game result directly: +1 if solved (win), -1 if lost (max steps)
        - No need to negate values based on player perspective

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, pi, v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the puzzle was eventually solved, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        # For single-player games, player is always 1
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            r = self.game.getGameEnded(board, curPlayer)

            if r != 0:
                # For single-player games, assign the result directly (no negation)
                # r is already 1 for win, -1 for loss
                return [(x[0], x[1], r) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            # For single-player games, we compare how many puzzles each model solves
            # Test previous model
            arena_prev = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                              lambda x: None, self.game)  # player2 not used
            prev_solved, prev_lost, prev_draws = arena_prev.playGames(self.args.arenaCompare)
            
            # Test new model
            arena_new = Arena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                             lambda x: None, self.game)  # player2 not used
            new_solved, new_lost, new_draws = arena_new.playGames(self.args.arenaCompare)

            log.info('PREV/NEW SOLVED : %d / %d ; PREV/NEW LOST : %d / %d' % 
                    (prev_solved, new_solved, prev_lost, new_lost))
            
            # Accept new model if it solves more puzzles (or same but fewer losses)
            total_games = prev_solved + prev_lost + prev_draws
            if total_games == 0:
                log.info('REJECTING NEW MODEL (no games played)')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                # Calculate win rate (solved / total)
                prev_win_rate = prev_solved / total_games if total_games > 0 else 0
                new_win_rate = new_solved / total_games if total_games > 0 else 0
                improvement = new_win_rate - prev_win_rate
                
                if improvement >= (self.args.updateThreshold - 0.5):  # Adjust threshold for single-player
                    log.info('ACCEPTING NEW MODEL (improvement: %.2f%%)' % (improvement * 100))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                else:
                    log.info('REJECTING NEW MODEL (improvement too small: %.2f%%)' % (improvement * 100))
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
